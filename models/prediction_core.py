"""
Integrated Prediction Core — fetches live data, runs bot + XGBoost,
calls decision engine, logs result, returns full context to UI.
"""

import os
import sys
import json
import pickle
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "labels"))

from indicators import add_baseline_features, FEATURE_COLUMNS
from triple_barrier import triple_barrier_labels

# Decision engine
sys.path.append(os.path.dirname(__file__))
from decision_engine import decide, classify_regime

# Logger
from logger import log_prediction

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# ── Configuration ──────────────────────────────────────────
EXCHANGE_FALLBACK_ORDER = [
    ("bitget", "BTC/USDT"),
    ("binance", "BTC/USDT"),
    ("coinbase", "BTC/USD"),
    ("kraken", "BTC/USD"),
]

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ── Load Models ──────────────────────────────────────────────
_model_cache = {}


def load_model(name="model"):
    """Load a pickled model. Caches after first load."""
    if name in _model_cache:
        return _model_cache[name]

    path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        obj = pickle.load(f)

    _model_cache[name] = obj
    return obj


def load_artifacts():
    """Load all model artifacts."""
    artifacts = {}

    # Original bot model
    artifacts["model"] = load_model("model")
    artifacts["calibrator"] = load_model("calibrator")
    artifacts["feature_columns"] = load_model("feature_columns")

    # XGBoost model
    artifacts["xgb_model"] = load_model("xgboost_model")
    artifacts["xgb_calibrator"] = load_model("xgboost_calibrator")
    artifacts["xgb_threshold"] = 0.5

    thresh_path = os.path.join(ARTIFACTS_DIR, "xgboost_threshold.txt")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            artifacts["xgb_threshold"] = float(f.read().strip())

    return artifacts

# ── Data Fetching ──────────────────────────────────────────

def fetch_live_data():
    """Fetch live BTC price + recent OHLCV via ccxt with fallback."""
    if not HAS_CCXT:
        raise RuntimeError("ccxt not installed. Run: pip install ccxt")

    errors = []

    for exchange_name, symbol in EXCHANGE_FALLBACK_ORDER:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({"enableRateLimit": True})

            # Live ticker price
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker["last"])
            bid = float(ticker.get("bid", price))
            ask = float(ticker.get("ask", price))
            timestamp = ticker.get("timestamp")

            if timestamp:
                ts = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            # Recent OHLCV (1m, last 200 bars for features)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=200)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            return {
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "spread_pct": (ask - bid) / price * 100,
                "timestamp": ts.isoformat(),
                "exchange": exchange_name,
                "ohlcv": df,
            }

        except Exception as e:
            errors.append(f"{exchange_name}: {str(e)[:60]}")
            continue

    raise RuntimeError(f"All exchanges failed: {' | '.join(errors)}")


# ── Feature Engineering ────────────────────────────────────

def compute_features(df: pd.DataFrame) -> dict:
    """Compute features from recent OHLCV data."""
    # Add label placeholder (needed by add_baseline_features)
    df["label"] = 0

    featured = add_baseline_features(df)
    latest = featured.iloc[-1]

    features = {col: latest[col] for col in FEATURE_COLUMNS if col in latest}

    # Add derived features for decision engine
    features["hour"] = pd.to_datetime(latest.get("timestamp", datetime.now(timezone.utc))).hour if "timestamp" in latest else datetime.now(timezone.utc).hour
    features["dayofweek"] = pd.to_datetime(latest.get("timestamp", datetime.now(timezone.utc))).dayofweek if "timestamp" in latest else datetime.now(timezone.utc).weekday()

    return features


# ── Prediction ─────────────────────────────────────────────

def predict_bot(features: dict, artifacts: dict) -> dict:
    """Run the original bot model."""
    model = artifacts.get("model")
    calibrator = artifacts.get("calibrator")
    feature_cols = artifacts.get("feature_columns", FEATURE_COLUMNS)

    if model is None:
        return {"pred": "UNKNOWN", "confidence": 0.5, "prob": 0.5}

    X = np.array([[features.get(c, 0) for c in feature_cols]])
    prob = model.predict_proba(X)[0, 1]

    if calibrator:
        prob = float(calibrator.transform(np.array([[prob]]))[0, 0])

    pred = "UP" if prob >= 0.5 else "DOWN"
    confidence = prob if pred == "UP" else 1 - prob

    return {"pred": pred, "confidence": round(confidence, 4), "prob": round(prob, 4)}


def predict_xgb(features: dict, artifacts: dict) -> dict:
    """Run the XGBoost model."""
    model = artifacts.get("xgb_model")
    calibrator = artifacts.get("xgb_calibrator")
    threshold = artifacts.get("xgb_threshold", 0.5)

    if model is None:
        return {"pred": "UNKNOWN", "confidence": 0.5, "prob": 0.5}

    X = np.array([[features.get(c, 0) for c in FEATURE_COLUMNS]])
    prob = model.predict_proba(X)[0, 1]

    if calibrator:
        prob = float(calibrator.transform(np.array([[prob]]))[0, 0])

    pred = "UP" if prob >= threshold else "DOWN"
    confidence = prob if pred == "UP" else 1 - prob

    return {"pred": pred, "confidence": round(confidence, 4), "prob": round(prob, 4)}


# ── Main Entry Point ────────────────────────────────────────

def get_full_prediction() -> dict:
    """
    Full pipeline: fetch data → features → bot + XGB → decision engine → log → return.
    """
    # 1. Fetch live data
    data = fetch_live_data()

    # 2. Compute features
    features = compute_features(data["ohlcv"])

    # 3. Load models
    artifacts = load_artifacts()

    # 4. Run both models
    bot_result = predict_bot(features, artifacts)
    xgb_result = predict_xgb(features, artifacts)

    # 5. Decision engine
    decision = decide(
        bot_pred=bot_result["pred"],
        bot_confidence=bot_result["confidence"],
        xgb_prob=xgb_result["prob"],
        features=features,
    )

    # 6. Log prediction
    try:
        log_prediction(
            current_price=data["price"],
            prediction=decision["ensemble_verdict"],
            confidence=decision["trust_score"],
            xgb_prob=xgb_result["prob"],
            xgb_pred=xgb_result["pred"],
            bot_pred=bot_result["pred"],
            bot_confidence=bot_result["confidence"],
            features=features,
            market_regime=decision["market_regime"],
            ensemble_verdict=decision["ensemble_verdict"],
            trust_score=decision["trust_score"],
        )
    except Exception as e:
        # Don't crash if logging fails
        print(f"Logging warning: {e}")

    # 7. Build response
    response = {
        "timestamp": data["timestamp"],
        "price": round(data["price"], 2),
        "bid": round(data["bid"], 2),
        "ask": round(data["ask"], 2),
        "spread_pct": round(data["spread_pct"], 4),
        "exchange": data["exchange"],
        "features": {k: round(v, 6) if isinstance(v, float) else v for k, v in features.items()},
        "bot": bot_result,
        "xgboost": xgb_result,
        "decision": decision,
    }

    return response


# ── Target Analysis (price target probability) ─────────────

def analyze_price_target(target_price: float, target_time_str: str) -> dict:
    """Analyze probability of hitting a price target by a given time."""
    data = fetch_live_data()
    current_price = data["price"]

    # Parse target time
    try:
        target_time = datetime.fromisoformat(target_time_str.replace("Z", "+00:00"))
    except ValueError:
        # Try common formats
        for fmt in ["%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%H:%M"]:
            try:
                target_time = datetime.strptime(target_time_str, fmt)
                target_time = target_time.replace(tzinfo=timezone.utc)
                if target_time < datetime.now(timezone.utc):
                    target_time += timedelta(days=1)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse time: {target_time_str}")

    minutes_ahead = max(1, int((target_time - datetime.now(timezone.utc)).total_seconds() / 60))

    # Get model for appropriate horizon
    artifacts = load_artifacts()

    # Use XGBoost probability as base
    features = compute_features(data["ohlcv"])
    xgb_result = predict_xgb(features, artifacts)
    p_up_15m = xgb_result["prob"]

    # Volatility estimate
    returns = data["ohlcv"]["close"].pct_change().dropna()
    sigma_per_minute = returns.std()
    sigma_horizon = sigma_per_minute * math.sqrt(minutes_ahead)

    # Time decay
    time_decay = math.exp(-minutes_ahead / 30.0)  # half-life ~30 min

    # Vol regime multiplier
    vol_zscore = features.get("vol_zscore_20", 0)
    if vol_zscore > 2.0:
        vol_mult = 1.3
    elif vol_zscore > 1.0:
        vol_mult = 1.1
    elif vol_zscore < 0.3:
        vol_mult = 0.9
    else:
        vol_mult = 1.0

    sigma_horizon *= vol_mult

    # Drift from model signal
    from scipy.stats import norm
    implied_mu_15min = sigma_per_minute * math.sqrt(15) * norm.ppf(max(0.001, min(0.999, p_up_15m)))
    implied_mu_per_minute = implied_mu_15min / 15.0
    mu_total = implied_mu_per_minute * minutes_ahead * time_decay

    # Price target probability
    log_target = math.log(target_price / current_price)
    z = (log_target - mu_total) / max(sigma_horizon, 1e-8)
    p_above = 1 - norm.cdf(z)
    p_below = norm.cdf(z)

    # Decision
    if p_above >= 0.65:
        verdict = "YES"
        verdict_conf = p_above
    elif p_below >= 0.65:
        verdict = "NO"
        verdict_conf = p_below
    else:
        verdict = "UNCERTAIN"
        verdict_conf = max(p_above, p_below)

    warnings = []
    if minutes_ahead > 60:
        warnings.append(f"Target is {minutes_ahead} min away — model trained for 15-30 min")
    if vol_zscore > 2.0:
        warnings.append("High volatility — probability estimates unreliable")
    if time_decay < 0.5:
        warnings.append("Signal has decayed significantly")

    return {
        "current_price": round(current_price, 2),
        "target_price": round(target_price, 2),
        "target_time": target_time.isoformat(),
        "minutes_ahead": minutes_ahead,
        "probability_above": round(p_above, 4),
        "probability_below": round(p_below, 4),
        "verdict": verdict,
        "verdict_confidence": round(verdict_conf, 4),
        "model_p_up_15m": round(p_up_15m, 4),
        "time_decay_factor": round(time_decay, 4),
        "vol_regime_multiplier": round(vol_mult, 2),
        "sigma_horizon": round(sigma_horizon, 6),
        "warnings": warnings,
    }


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "target":
        if len(sys.argv) < 4:
            print("Usage: python prediction_core.py target <price> <time>")
            sys.exit(1)
        result = analyze_price_target(float(sys.argv[2]), sys.argv[3])
        print(json.dumps(result, indent=2))
    else:
        result = get_full_prediction()
        print(json.dumps(result, indent=2))