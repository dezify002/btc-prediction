"""
Shared prediction logic, used by the Flask web app (webapp/app.py).
Factored out so the web frontend and the original CLI scripts
(predict_now.py, price_target_probability.py) can both call the same,
already-tested logic without duplicating it.
"""

import math
import os
import pickle
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from indicators import add_baseline_features  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# -- NEW: Volatility regime thresholds from Phase 2/3 analysis --
# If vol_zscore_20 exceeds this, the model's edge is unproven.
# These are placeholders -- update with your actual Phase 2 quartile cutoffs.
VOLATILITY_REGIME_THRESHOLDS = {
    "high_vol_cutoff": 1.0,      # vol_zscore_20 > 1.0 = elevated regime
    "extreme_vol_cutoff": 2.0,   # vol_zscore_20 > 2.0 = model untested
}


def load_model():
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train_final_model.py first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)
    return model, calibrator, feature_columns


# Tried in order -- some cloud hosts (e.g. Railway's default US region) get a
# 451 "restricted location" response from Binance's main site, since Binance
# blocks connections from US-based servers by policy. Falling back to other
# exchanges keeps the app working regardless of where it's hosted.
EXCHANGE_FALLBACK_ORDER = [
    ("binance", "BTC/USDT"),
    ("coinbase", "BTC/USD"),
    ("kraken", "BTC/USD"),
]


def _try_exchanges(action):
    """Tries each exchange in EXCHANGE_FALLBACK_ORDER, returning the first
    successful result of action(exchange, symbol). Raises the last error
    if all of them fail."""
    import ccxt

    last_error = None
    for exchange_id, symbol in EXCHANGE_FALLBACK_ORDER:
        try:
            exchange_cls = getattr(ccxt, exchange_id)
            exchange = exchange_cls({"enableRateLimit": True})
            return action(exchange, symbol), exchange_id
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Could not reach any exchange (tried {[e for e, _ in EXCHANGE_FALLBACK_ORDER]}). "
        f"Last error: {last_error}"
    )


def fetch_recent_data(limit=1500):
    def action(exchange, symbol):
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
        if not ohlcv:
            raise RuntimeError("No data returned.")
        return ohlcv

    ohlcv, exchange_used = _try_exchanges(action)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.attrs["exchange_used"] = exchange_used
    return df


def fetch_order_book_signal():
    """Live-only, NOT part of the validated/backtested model (see predict_now.py notes)."""
    def action(exchange, symbol):
        book = exchange.fetch_order_book(symbol, limit=50)
        bid_volume = sum(qty for _, qty in book["bids"])
        ask_volume = sum(qty for _, qty in book["asks"])
        total = bid_volume + ask_volume
        if total == 0:
            raise RuntimeError("Empty order book.")
        imbalance = (bid_volume - ask_volume) / total
        return {"bid_volume": bid_volume, "ask_volume": ask_volume, "imbalance": imbalance}

    try:
        result, _ = _try_exchanges(action)
        return result
    except Exception:
        return None


def parse_target_time(time_str: str, now_utc: datetime) -> datetime:
    """Forgiving parser: '10:00', '10;00', '10.00', '1000', '10am', '2:30pm', etc."""
    s = time_str.strip().lower().replace(" ", "")
    is_pm = "pm" in s
    is_am = "am" in s
    s = s.replace("am", "").replace("pm", "")
    s = re.sub(r"[;.,]", ":", s)

    if ":" in s:
        parts = s.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    elif s.isdigit() and len(s) <= 2:
        hour, minute = int(s), 0
    elif s.isdigit() and len(s) in (3, 4):
        minute = int(s[-2:])
        hour = int(s[:-2])
    else:
        raise ValueError(f"Unrecognized time format: {time_str!r}")

    if is_pm and hour < 12:
        hour += 12
    if is_am and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Time out of range: {time_str!r}")

    target = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now_utc:
        target += timedelta(days=1)
    return target


def get_current_prediction():
    """Returns a dict for the 'right now, next 15 minutes' prediction."""
    model, calibrator, feature_columns = load_model()

    df = fetch_recent_data(limit=200)
    featured = add_baseline_features(df)
    complete = featured.dropna(subset=feature_columns)
    if len(complete) == 0:
        raise RuntimeError("Not enough recent data to compute indicators. Try again shortly.")

    latest = complete.iloc[-1]
    X_latest = latest[feature_columns].values.reshape(1, -1).astype(float)
    raw_prob = model.predict_proba(X_latest)[0, 1]
    p_up = float(calibrator.transform([raw_prob])[0])

    ob_signal = fetch_order_book_signal()

    # -- NEW: Volatility regime check --
    vol_z = float(latest.get("vol_zscore_20", float("nan")))
    regime_warning = None
    if not np.isnan(vol_z):
        if vol_z > VOLATILITY_REGIME_THRESHOLDS["extreme_vol_cutoff"]:
            regime_warning = (
                f"EXTREME volatility detected (vol_zscore={vol_z:.2f}). "
                f"The model was NOT tested in this regime. Prediction is unreliable."
            )
        elif vol_z > VOLATILITY_REGIME_THRESHOLDS["high_vol_cutoff"]:
            regime_warning = (
                f"Elevated volatility detected (vol_zscore={vol_z:.2f}). "
                f"Model edge is weaker here per Phase 2/3. Treat with caution."
            )

    return {
        "timestamp": latest["timestamp"].isoformat(),
        "price": float(latest["close"]),
        "p_up": p_up,
        "p_down": 1 - p_up,
        "rsi": float(latest.get("rsi_14", float("nan"))),
        "ema_dist": float(latest.get("ema_dist", float("nan"))),
        "ret_5": float(latest.get("ret_5", float("nan"))),
        "ret_15": float(latest.get("ret_15", float("nan"))),
        "vol_z": vol_z,
        "order_book": ob_signal,
        "regime_warning": regime_warning,
    }


def analyze_price_target(target_price: float, target_time_str: str):
    """Returns a dict for the 'will BTC hit $X by time Y' analysis."""
    model, calibrator, feature_columns = load_model()

    df = fetch_recent_data(limit=1500)
    featured = add_baseline_features(df)
    complete = featured.dropna(subset=feature_columns)
    if len(complete) == 0:
        raise RuntimeError("Not enough recent data to compute indicators. Try again shortly.")

    latest = complete.iloc[-1]
    current_price = float(latest["close"])
    now_utc = latest["timestamp"].to_pydatetime()

    target_time = parse_target_time(target_time_str, now_utc)
    minutes_ahead = (target_time - now_utc).total_seconds() / 60.0

    X_latest = latest[feature_columns].values.reshape(1, -1).astype(float)
    raw_prob = model.predict_proba(X_latest)[0, 1]
    p_up_15min = float(calibrator.transform([raw_prob])[0])
    p_up_15min = min(max(p_up_15min, 0.01), 0.99)

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    sigma_per_minute = log_returns.std()
    if sigma_per_minute == 0 or np.isnan(sigma_per_minute):
        raise RuntimeError("Could not estimate volatility from recent data -- try again.")

    # -- FIX: Time-decay drift + volatility regime adjustment --
    # The model's 15-min signal decays as we extrapolate further.
    # We apply a time-decay factor: signal strength halves every 15 minutes.
    # Also scale up volatility estimate in elevated regimes.
    time_decay_factor = max(0.3, math.exp(-0.046 * minutes_ahead / 15))
    # ^ decay factor: 1.0 at 0 min, ~0.63 at 15 min, ~0.40 at 30 min, ~0.25 at 60 min

    # Volatility regime adjustment: if vol_zscore is elevated, sigma is likely
    # to stay elevated or spike further. Scale up the estimate.
    vol_z = float(latest.get("vol_zscore_20", 0.0))
    vol_regime_multiplier = 1.0
    if vol_z > VOLATILITY_REGIME_THRESHOLDS["extreme_vol_cutoff"]:
        vol_regime_multiplier = 1.5
    elif vol_z > VOLATILITY_REGIME_THRESHOLDS["high_vol_cutoff"]:
        vol_regime_multiplier = 1.2

    sigma_15min = sigma_per_minute * math.sqrt(15)
    implied_mu_15min = sigma_15min * norm.ppf(p_up_15min)
    implied_mu_per_minute = implied_mu_15min / 15.0

    # Apply time decay to the drift rate
    decayed_mu_per_minute = implied_mu_per_minute * time_decay_factor

    mu_total = decayed_mu_per_minute * minutes_ahead
    sigma_horizon = sigma_per_minute * math.sqrt(minutes_ahead) * vol_regime_multiplier

    log_target_ratio = math.log(target_price / current_price)
    z = (log_target_ratio - mu_total) / sigma_horizon
    prob_at_or_above = float(1 - norm.cdf(z))
    prob_below = 1 - prob_at_or_above

    verdict = "YES" if prob_at_or_above >= 0.5 else "NO"
    confidence = max(prob_at_or_above, prob_below)

    # -- NEW: Regime warning for the target analysis too --
    regime_warning = None
    if not np.isnan(vol_z):
        if vol_z > VOLATILITY_REGIME_THRESHOLDS["extreme_vol_cutoff"]:
            regime_warning = (
                f"EXTREME volatility (vol_zscore={vol_z:.2f}). Model untested here."
            )
        elif vol_z > VOLATILITY_REGIME_THRESHOLDS["high_vol_cutoff"]:
            regime_warning = (
                f"Elevated volatility (vol_zscore={vol_z:.2f}). Edge is weaker."
            )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "prob_at_or_above": prob_at_or_above,
        "prob_below": prob_below,
        "current_price": current_price,
        "target_price": target_price,
        "now_utc": now_utc.isoformat(),
        "target_time_utc": target_time.isoformat(),
        "minutes_ahead": minutes_ahead,
        "required_move_pct": (target_price / current_price - 1) * 100,
        "sigma_per_minute_pct": sigma_per_minute * 100,
        "implied_drift_per_minute_pct": decayed_mu_per_minute * 100,
        "time_decay_factor": time_decay_factor,
        "vol_regime_multiplier": vol_regime_multiplier,
        "rsi": float(latest.get("rsi_14", float("nan"))),
        "ema_dist": float(latest.get("ema_dist", float("nan"))),
        "ret_5": float(latest.get("ret_5", float("nan"))),
        "ret_15": float(latest.get("ret_15", float("nan"))),
        "vol_z": vol_z,
        "p_up_15min": p_up_15min,
        "extrapolation_warning": minutes_ahead > 30,
        "regime_warning": regime_warning,
    }