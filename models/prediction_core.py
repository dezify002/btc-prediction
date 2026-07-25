"""
Integrated Prediction Core — fetches live data, runs bot + XGBoost,
calls decision engine, logs result, returns full context to UI.

PHASE 4 UPDATE: Now logs target_price, prediction_window, and
returns log_timestamp for outcome tracking.
"""

import os
import sys
import json
import pickle
import math
import re
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

# Logger — Phase 4 enhanced
from logger import log_prediction

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# ── NumPy → Native Type Converter ──────────────────────────

def _to_native(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


# ── Configuration ──────────────────────────────────────────
EXCHANGE_FALLBACK_ORDER = [
    ("bitget", "BTC/USDT"),
    ("binance", "BTC/USDT"),
    ("coinbase", "BTC/USD"),
    ("kraken", "BTC/USD"),
]

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Default prediction parameters (match your triple barrier)
DEFAULT_PROFIT_PCT = 0.006   # +0.6%
DEFAULT_STOP_PCT = 0.003     # -0.3%
DEFAULT_WINDOW = "15m"

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
        calibrated = calibrator.transform(np.array([[prob]]))
        if calibrated.ndim == 2:
            prob = float(calibrated[0, 0])
        else:
            prob = float(calibrated[0])

    pred = "UP" if prob >= 0.5 else "DOWN"
    confidence = float(prob) if pred == "UP" else float(1 - prob)

    return {"pred": pred, "confidence": round(confidence, 4), "prob": round(float(prob), 4)}


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
        calibrated = calibrator.transform(np.array([[prob]]))
        if calibrated.ndim == 2:
            prob = float(calibrated[0, 0])
        else:
            prob = float(calibrated[0])

    pred = "UP" if prob >= threshold else "DOWN"
    confidence = float(prob) if pred == "UP" else float(1 - prob)

    return {"pred": pred, "confidence": round(confidence, 4), "prob": round(float(prob), 4)}


# ── Main Entry Point ────────────────────────────────────────

def get_full_prediction(prediction_window: str = DEFAULT_WINDOW,
                         profit_pct: float = DEFAULT_PROFIT_PCT,
                         stop_pct: float = DEFAULT_STOP_PCT) -> dict:
    """
    Full pipeline: fetch data → features → bot + XGB → decision engine → log → return.

    PHASE 4: Now logs target_price, prediction_window, and returns log_timestamp.
    """
    # 1. Fetch live data
    data = fetch_live_data()
    current_price = data["price"]

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

    # 6. Compute target price from prediction direction
    if decision["ensemble_verdict"] and "UP" in decision["ensemble_verdict"]:
        target_price = current_price * (1 + profit_pct)
    elif decision["ensemble_verdict"] and "DOWN" in decision["ensemble_verdict"]:
        target_price = current_price * (1 - stop_pct)
    else:
        target_price = current_price

    # 7. Log prediction (Phase 4 enhanced)
    log_timestamp = None
    try:
        log_record = log_prediction(
            current_price=current_price,
            target_price=round(target_price, 2),
            prediction_window=prediction_window,
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
            decision=decision.get("recommendation", "UNKNOWN"),
            risk_level=decision.get("risk", "UNKNOWN"),
            reasons=decision.get("reasons", []),
        )
        log_timestamp = log_record.get("timestamp")
    except Exception as e:
        # Don't crash if logging fails
        print(f"Logging warning: {e}")

    # 8. Build response
    response = {
        "timestamp": data["timestamp"],
        "price": round(float(data["price"]), 2),
        "bid": round(float(data["bid"]), 2),
        "ask": round(float(data["ask"]), 2),
        "spread_pct": round(float(data["spread_pct"]), 4),
        "exchange": str(data["exchange"]),
        "features": {k: round(float(v), 6) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else bool(v) if isinstance(v, np.bool_) else v for k, v in features.items()},
        "bot": bot_result,
        "xgboost": xgb_result,
        "decision": decision,
        "target_price": round(float(target_price), 2),
        "prediction_window": prediction_window,
        "log_timestamp": log_timestamp,  # PHASE 4: for outcome tracking
    }

    # Convert any remaining NumPy types to native Python types for JSON
    return _to_native(response)


# ── Target Analysis (price target probability) ─────────────

def _normalize_time_string(time_str: str) -> str:
    """
    Normalize messy time inputs into something parseable.
    Handles: 2;00am, 2:00am, 2.00am, 2am, 14:30, 2:00 pm, 12:00pm, etc.
    """
    if not time_str:
        return time_str

    s = time_str.strip().lower()

    # Replace common separators with colon
    s = s.replace(";", ":").replace(".", ":")

    # Remove spaces around am/pm
    s = s.replace(" am", "am").replace(" pm", "pm")
    s = s.replace("a.m", "am").replace("p.m", "pm")

    # If it looks like "2am" or "2pm" (no minutes), insert :00
    s = re.sub(r"^(\d{1,2})(am|pm)$", r"\1:00\2", s)

    # If it looks like "2:30am" or "12:00pm" — already good

    return s


def _parse_target_time(time_str: str) -> datetime:
    """
    Parse a target time string into a datetime.

    Supports:
        - ISO format: 2026-07-25T14:30:00 or 2026-07-25 14:30
        - Time only: 14:30, 2:30pm, 2;30am, 2.00pm, 2am, 12:00pm
        - Common formats: 07/25/2026 14:30, 25-07-2026 14:30
    """
    original = time_str
    time_str = _normalize_time_string(time_str)
    now = datetime.now(timezone.utc)

    # Try ISO format first
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Try formats with date + time
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M%p",
        "%m/%d/%Y %I:%M %p",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # Try time-only formats (assume today, or tomorrow if passed)
    # Try lowercase am/pm first
    time_formats_lower = [
        "%H:%M",
        "%H:%M:%S",
        "%I:%M%p",      # 12:00pm, 2:30pm
        "%I:%M %p",     # 12:00 pm, 2:30 pm
        "%I%p",         # 12pm, 2pm
    ]
    for fmt in time_formats_lower:
        try:
            dt = datetime.strptime(time_str, fmt)
            dt = dt.replace(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
            if dt < now:
                dt += timedelta(days=1)
            return dt
        except ValueError:
            pass

    # Try uppercase AM/PM as fallback
    time_str_upper = time_str.upper()
    time_formats_upper = [
        "%I:%M%p",      # 12:00PM
        "%I:%M %p",     # 12:00 PM
        "%I%p",         # 12PM
    ]
    for fmt in time_formats_upper:
        try:
            dt = datetime.strptime(time_str_upper, fmt)
            dt = dt.replace(year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc)
            if dt < now:
                dt += timedelta(days=1)
            return dt
        except ValueError:
            pass

    raise ValueError(f"Cannot parse time: '{original}' (normalized: '{time_str}')")


def analyze_price_target(target_price: float, target_time_str: str) -> dict:
    """Analyze probability of hitting a price target by a given time."""
    data = fetch_live_data()
    current_price = data["price"]

    # Parse target time with robust handling
    target_time = _parse_target_time(target_time_str)

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