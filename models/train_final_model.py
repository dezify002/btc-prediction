"""
Trains a final model on ALL available historical data and saves it to disk,
so predict_now.py doesn't need to retrain every time you want a prediction.

This is deliberately different from baseline_xgb.py's purged walk-forward
CV -- that script exists to HONESTLY MEASURE how good the model is (using
held-out folds). This script exists to produce the actual model you'll use
day-to-day, trained on as much data as possible. Run baseline_xgb.py first
to know whether this model is worth trusting at all; run this script only
after you've decided the honest evaluation looks reasonable.

Usage:
    python train_final_model.py
"""

import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "labels"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))

from triple_barrier import triple_barrier_labels          # noqa: E402
from indicators import add_baseline_features, FEATURE_COLUMNS  # noqa: E402

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier


def make_model():
    if HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )


def main():
    print("Loading historical BTC data...")
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.csv"))

    # -- FIX: Sync barrier mode with Phase 2 baseline (pct is now default) --
    print("Applying triple-barrier labeling (fixed-percentage mode)...")
    labeled = triple_barrier_labels(
        raw,
        barrier_mode="pct",       # <-- synced with baseline_xgb.py default
        tp_pct=0.006,
        sl_pct=0.003,
        max_holding=15,
    )

    print("Computing features...")
    featured = add_baseline_features(labeled)

    data = featured.dropna(subset=["label"] + FEATURE_COLUMNS).copy()
    data = data[data["label"] != 0].reset_index(drop=True)
    data["target"] = (data["label"] == 1).astype(int)

    print(f"Training on {len(data)} labeled samples...")

    X = data[FEATURE_COLUMNS].values
    y = data["target"].values

    # Hold out the most recent 15% for calibration.
    cal_cutoff = int(len(X) * 0.85)

    model = make_model()
    model.fit(X[:cal_cutoff], y[:cal_cutoff])

    cal_probs_fit = model.predict_proba(X[cal_cutoff:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_probs_fit, y[cal_cutoff:])

    # -- FIX: Retrain on full dataset for the final deployed model.
    # The calibrator was fit on predictions from a model that did NOT see
    # the calibration holdout. final_model is stronger (trained on all data).
    # This is conservative -- the calibrator may slightly under-correct,
    # which is safer than over-correcting. If you want tighter calibration,
    # train final_model on X[:cal_cutoff] instead (matching calibrator strength).
    print("Retraining on full dataset for the final deployed model...")
    final_model = make_model()
    final_model.fit(X, y)

    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # -- NEW: Save feature hash for version tracking --
    feature_hash = hashlib.sha256(str(FEATURE_COLUMNS).encode()).hexdigest()[:8]
    print(f"Feature set hash: {feature_hash}")

    with open(os.path.join(artifacts_dir, "model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    with open(os.path.join(artifacts_dir, "calibrator.pkl"), "wb") as f:
        pickle.dump(iso, f)

    with open(os.path.join(artifacts_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(FEATURE_COLUMNS, f)

    # Save metadata for tracking
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_hash": feature_hash,
        "n_samples": len(data),
        "barrier_mode": "pct",
        "tp_pct": 0.006,
        "sl_pct": 0.003,
        "max_holding": 15,
    }
    with open(os.path.join(artifacts_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nSaved model, calibrator, feature list, and metadata to {artifacts_dir}/")
    print("You can now run predict_now.py to get live predictions.")


if __name__ == "__main__":
    main()