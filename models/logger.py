"""
Prediction Logger — Phase 4 Compatible
Logs every prediction with all metadata needed by the Learning Engine.
Writes to JSON Lines format in reports/logs/ for daily rotation.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any


LOG_DIR = Path("reports/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _today_file() -> Path:
    """Get today's log file path."""
    today = datetime.utcnow().strftime("%Y%m%d")
    return LOG_DIR / f"predictions_{today}.jsonl"


def log_prediction(
    current_price: float,
    prediction: str,
    confidence: float,
    xgb_prob: float,
    xgb_pred: str,
    bot_pred: str,
    bot_confidence: float,
    features: dict,
    market_regime: str,
    ensemble_verdict: str,
    trust_score: float,
    # Phase 4 additions
    target_price: Optional[float] = None,
    prediction_window: str = "15m",
    decision: Optional[str] = None,
    risk_level: Optional[str] = None,
    reasons: Optional[list] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log a single prediction with all Phase 4 fields.

    Returns the logged record (including timestamp for later outcome update).
    """
    ts = timestamp or datetime.utcnow().isoformat()

    # Extract individual features for Phase 4 analysis
    record = {
        "timestamp": ts,
        "current_price": float(current_price),
        "target_price": float(target_price) if target_price else float(current_price) * 1.006,
        "prediction_window": prediction_window,
        "prediction": prediction,
        "confidence": float(confidence) * 100 if confidence <= 1.0 else float(confidence),
        "xgb_probability": float(xgb_prob),
        "xgb_pred": xgb_pred,
        "bot_pred": bot_pred,
        "bot_confidence": float(bot_confidence),
        "trust_score": float(trust_score),
        "ensemble_verdict": ensemble_verdict,
        "market_regime": market_regime,
        "decision": decision or "UNKNOWN",
        "risk_level": risk_level or "UNKNOWN",
        "reasons": reasons or [],
        # Individual features
        "rsi": _safe_get(features, "rsi_14"),
        "atr": _safe_get(features, "atr_14"),
        "ema_distance": _safe_get(features, "dist_ema50"),
        "volatility": _safe_get(features, "realized_vol_20"),
        "volume": _safe_get(features, "dollar_vol"),
        "hour": datetime.utcnow().hour,
        "dayofweek": datetime.utcnow().weekday(),
        # Outcome fields — filled later
        "actual_result": None,
        "correct": None,
        "outcome_timestamp": None,
    }

    filepath = _today_file()
    with open(filepath, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return record


def update_outcome(timestamp_str: str, actual_result: str, correct: bool) -> bool:
    """
    Update a logged prediction with its actual outcome.

    Searches today's and yesterday's log files for the matching timestamp.
    """
    correct = bool(correct)

    # Search today and yesterday
    dates = [
        datetime.utcnow().strftime("%Y%m%d"),
        (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
    ]

    for date_str in dates:
        filepath = LOG_DIR / f"predictions_{date_str}.jsonl"
        if not filepath.exists():
            continue

        updated = False
        lines = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("timestamp") == timestamp_str:
                    rec["actual_result"] = actual_result
                    rec["correct"] = correct
                    rec["outcome_timestamp"] = datetime.utcnow().isoformat()
                    updated = True
                lines.append(rec)

        if updated:
            with open(filepath, "w") as f:
                for rec in lines:
                    f.write(json.dumps(rec, default=str) + "\n")
            return True

    return False


def get_recent_predictions(n: int = 20) -> List[Dict]:
    """Get the N most recent predictions (newest first)."""
    predictions = []
    files = sorted(LOG_DIR.glob("predictions_*.jsonl"), reverse=True)

    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
        if len(predictions) >= n:
            break

    return predictions[-n:][::-1]  # Return oldest-to-newest of the batch


def get_stats() -> Dict[str, Any]:
    """Quick stats about logged predictions."""
    files = list(LOG_DIR.glob("predictions_*.jsonl"))
    total = 0
    with_outcomes = 0
    correct = 0

    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                if rec.get("correct") is not None:
                    with_outcomes += 1
                    if rec["correct"]:
                        correct += 1

    accuracy = (correct / with_outcomes * 100) if with_outcomes > 0 else 0

    return {
        "total_predictions": total,
        "with_outcomes": with_outcomes,
        "pending_outcomes": total - with_outcomes,
        "correct": correct,
        "incorrect": with_outcomes - correct,
        "accuracy_pct": round(accuracy, 1),
        "log_files": len(files),
        "log_dir": str(LOG_DIR.absolute()),
    }


def _safe_get(d: dict, key: str, default=None):
    """Safely get a value from dict, coercing to float."""
    val = d.get(key, default)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Backwards compatibility ─────────────────────────────────

class PredictionLogger:
    """Class wrapper for backwards compatibility."""

    def __init__(self, log_dir: str = "reports/logs"):
        global LOG_DIR
        LOG_DIR = Path(log_dir)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, **kwargs):
        return log_prediction(**kwargs)

    def update_outcome(self, timestamp_str: str, actual_result: str, correct: bool):
        return update_outcome(timestamp_str, actual_result, correct)

    def get_recent_predictions(self, n: int = 20):
        return get_recent_predictions(n)

    def get_stats(self):
        return get_stats()


if __name__ == "__main__":
    # Quick test
    print("Logger test:")
    rec = log_prediction(
        current_price=67250.20,
        prediction="UP",
        confidence=0.82,
        xgb_prob=0.74,
        xgb_pred="UP",
        bot_pred="UP",
        bot_confidence=0.81,
        features={"rsi_14": 61.3, "atr_14": 0.48, "dist_ema50": 0.22, 
                  "realized_vol_20": 0.31, "dollar_vol": 145000},
        market_regime="uptrend_low_vol",
        ensemble_verdict="UP (Strong)",
        trust_score=0.85,
        target_price=67500.00,
        prediction_window="15m",
        decision="TAKE_TRADE",
        risk_level="LOW",
        reasons=["Both models agree", "Low volatility"]
    )
    print(f"  Logged: {rec['timestamp']}")

    # Update outcome
    update_outcome(rec["timestamp"], "UP", True)
    print(f"  Updated outcome: correct=True")

    print(f"  Stats: {get_stats()}")