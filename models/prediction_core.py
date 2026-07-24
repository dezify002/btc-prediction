"""
Shared prediction logic with Bitget API integration.

Uses Bitget REST API for live price data instead of ccxt multi-exchange fallback.
"""

import math
import os
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from indicators import add_baseline_features  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

VOLATILITY_REGIME_THRESHOLDS = {
    "high_vol_cutoff": 1.0,
    "extreme_vol_cutoff": 2.0,
}

# Bitget API endpoints
BITGET_BASE = "https://api.bitget.com"
BITGET_TICKER = "/api/spot/v1/market/ticker?symbol=BTCUSDT_SPBL"
BITGET_CANDLES = "/api/spot/v1/market/candles?symbol=BTCUSDT_SPBL&granularity=60&limit={limit}"


def load_model(horizon: str = "15m"):
    """Load a specific horizon model. Falls back to legacy single model."""
    model_path = os.path.join(ARTIFACTS_DIR, f"model_{horizon}.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, f"calibrator_{horizon}.pkl"), "rb") as f:
            calibrator = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"), "rb") as f:
            feature_columns = pickle.load(f)
        return model, calibrator, feature_columns, horizon

    legacy_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
            calibrator = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"), "rb") as f:
            feature_columns = pickle.load(f)
        return model, calibrator, feature_columns, "15m (legacy)"

    raise FileNotFoundError("No model found. Run train_multi_horizon.py or train_final_model.py first.")


def select_horizon(minutes_ahead: float) -> str:
    if minutes_ahead <= 22.5:
        return "15m"
    elif minutes_ahead <= 150:
        return "1h"
    else:
        return "4h"


def fetch_bitget_ticker():
    """Fetch live BTC price from Bitget."""
    try:
        resp = requests.get(BITGET_BASE + BITGET_TICKER, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget API error: {data}")

        ticker = data["data"][0]
        return {
            "price": float(ticker["close"]),
            "bid": float(ticker["bidPr"]),
            "ask": float(ticker["askPr"]),
            "high_24h": float(ticker["high24h"]),
            "low_24h": float(ticker["low24h"]),
            "volume_24h": float(ticker["baseVol"]),
            "timestamp": int(ticker["ts"]),
            "exchange_used": "bitget",
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Bitget ticker: {e}")


def fetch_bitget_candles(limit=1500):
    """Fetch 1-minute candles from Bitget for feature calculation.

    Bitget granularity: 60 = 1 minute
    """
    try:
        url = BITGET_BASE + BITGET_CANDLES.format(limit=min(limit, 200))  # Bitget max 200 per request
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget API error: {data}")

        candles = data["data"]
        # Bitget format: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.attrs["exchange_used"] = "bitget"
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Bitget candles: {e}")


def fetch_order_book_signal():
    """Bitget order book (live-only, not backtested)."""
    try:
        url = BITGET_BASE + "/api/spot/v1/market/depth?symbol=BTCUSDT_SPBL&limit=50"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "00000":
            return None

        book = data["data"]
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        bid_volume = sum(float(qty) for _, qty, _ in bids) if bids else 0
        ask_volume = sum(float(qty) for _, qty, _ in asks) if asks else 0
        total = bid_volume + ask_volume

        if total == 0:
            return None

        imbalance = (bid_volume - ask_volume) / total
        return {
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance": imbalance,
        }
    except Exception:
        return None


def parse_target_time(time_str: str, now_utc: datetime) -> datetime:
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
    """Returns prediction using Bitget live ticker + recent candles for features."""
    model, calibrator, feature_columns, horizon_used = load_model("15m")

    live_ticker = fetch_bitget_ticker()
    df = fetch_bitget_candles(limit=200)

    featured = add_baseline_features(df)
    complete = featured.dropna(subset=feature_columns)
    if len(complete) == 0:
        raise RuntimeError("Not enough recent data to compute indicators.")

    latest = complete.iloc[-1]
    X_latest = latest[feature_columns].values.reshape(1, -1).astype(float)
    raw_prob = model.predict_proba(X_latest)[0, 1]
    p_up = float(calibrator.transform([raw_prob])[0])

    ob_signal = fetch_order_book_signal()
    vol_z = float(latest.get("vol_zscore_20", float("nan")))

    regime_warning = None
    if not np.isnan(vol_z):
        if vol_z > VOLATILITY_REGIME_THRESHOLDS["extreme_vol_cutoff"]:
            regime_warning = f"EXTREME volatility (vol_zscore={vol_z:.2f}). Model untested here."
        elif vol_z > VOLATILITY_REGIME_THRESHOLDS["high_vol_cutoff"]:
            regime_warning = f"Elevated volatility (vol_zscore={vol_z:.2f}). Edge is weaker."

    return {
        "timestamp": datetime.fromtimestamp(live_ticker["timestamp"] / 1000, tz=timezone.utc).isoformat(),
        "price": live_ticker["price"],
        "p_up": p_up,
        "p_down": 1 - p_up,
        "rsi": float(latest.get("rsi_14", float("nan"))),
        "ema_dist": float(latest.get("ema_dist", float("nan"))),
        "ret_5": float(latest.get("ret_5", float("nan"))),
        "ret_15": float(latest.get("ret_15", float("nan"))),
        "vol_z": vol_z,
        "order_book": ob_signal,
        "regime_warning": regime_warning,
        "model_used": horizon_used,
        "exchange_used": "bitget",
        "data_source": "bitget_api",
    }


def analyze_price_target(target_price: float, target_time_str: str):
    """Uses multi-horizon model selection based on target time."""
    live_ticker = fetch_bitget_ticker()
    df = fetch_bitget_candles(limit=200)

    featured = add_baseline_features(df)

    horizon = select_horizon(0)
    model, calibrator, feature_columns, _ = load_model(horizon)

    complete = featured.dropna(subset=feature_columns)
    if len(complete) == 0:
        raise RuntimeError("Not enough recent data to compute indicators.")

    latest = complete.iloc[-1]

    current_price = live_ticker["price"]
    now_utc = datetime.now(timezone.utc)

    target_time = parse_target_time(target_time_str, now_utc)
    minutes_ahead = (target_time - now_utc).total_seconds() / 60.0

    horizon = select_horizon(minutes_ahead)
    model, calibrator, feature_columns, _ = load_model(horizon)

    X_latest = latest[feature_columns].values.reshape(1, -1).astype(float)
    raw_prob = model.predict_proba(X_latest)[0, 1]
    p_up = float(calibrator.transform([raw_prob])[0])
    p_up = min(max(p_up, 0.01), 0.99)

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    sigma_per_minute = log_returns.std()
    if sigma_per_minute == 0 or np.isnan(sigma_per_minute):
        raise RuntimeError("Could not estimate volatility from recent data.")

    trained_horizon = {"15m": 15, "1h": 60, "4h": 240}[horizon]
    excess_minutes = max(0, minutes_ahead - trained_horizon)
    time_decay_factor = max(0.3, math.exp(-0.046 * excess_minutes / trained_horizon))

    vol_z = float(latest.get("vol_zscore_20", 0.0))
    vol_regime_multiplier = 1.0
    if vol_z > VOLATILITY_REGIME_THRESHOLDS["extreme_vol_cutoff"]:
        vol_regime_multiplier = 1.5
    elif vol_z > VOLATILITY_REGIME_THRESHOLDS["high_vol_cutoff"]:
        vol_regime_multiplier = 1.2

    sigma_trained = sigma_per_minute * math.sqrt(trained_horizon)
    implied_mu_trained = sigma_trained * norm.ppf(p_up)
    implied_mu_per_minute = implied_mu_trained / trained_horizon

    decayed_mu_per_minute = implied_mu_per_minute * time_decay_factor
    mu_total = decayed_mu_per_minute * minutes_ahead
    sigma_horizon = sigma_per_minute * math.sqrt(minutes_ahead) * vol_regime_multiplier

    log_target_ratio = math.log(target_price / current_price)
    z = (log_target_ratio - mu_total) / sigma_horizon
    prob_at_or_above = float(1 - norm.cdf(z))
    prob_below = 1 - prob_at_or_above

    verdict = "YES" if prob_at_or_above >= 0.5 else "NO"
    confidence = max(prob_at_or_above, prob_below)

    regime_warning = None
    if not np.isnan(vol_z):
        if vol_z > VOLATILITY_REGIME_THRESHOLDS["extreme_vol_cutoff"]:
            regime_warning = f"EXTREME volatility (vol_zscore={vol_z:.2f}). Model untested here."
        elif vol_z > VOLATILITY_REGIME_THRESHOLDS["high_vol_cutoff"]:
            regime_warning = f"Elevated volatility (vol_zscore={vol_z:.2f}). Edge is weaker."

    extrapolation_warning = minutes_ahead > trained_horizon * 2

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
        "p_up_trained_horizon": p_up,
        "model_used": horizon,
        "trained_horizon_min": trained_horizon,
        "extrapolation_warning": extrapolation_warning,
        "regime_warning": regime_warning,
        "exchange_used": "bitget",
        "data_source": "bitget_api",
    }