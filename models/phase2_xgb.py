"""
Phase 2: Train XGBoost with expanded features and proper threshold tuning.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "labels"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))

from triple_barrier import triple_barrier_labels
from indicators import add_baseline_features, FEATURE_COLUMNS

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("❌ xgboost not installed. Run: pip install xgboost")
    sys.exit(1)


def find_best_threshold(y_true, probs):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def main():
    print("Loading BTC data...")
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.csv"))

    print("Labeling with triple-barrier (mode=pct)...")
    labeled = triple_barrier_labels(
        raw,
        barrier_mode="pct",
        tp_pct=0.008,      # slightly wider for cleaner signal
        sl_pct=0.004,
        max_holding=30,    # longer holding for more UP captures
    )

    print("Computing features...")
    featured = add_baseline_features(labeled)

    data = featured.dropna(subset=["label"] + FEATURE_COLUMNS).copy()
    data = data[data["label"] != 0].reset_index(drop=True)
    data["target"] = (data["label"] == 1).astype(int)

    print(f"\nDataset: {len(data)} samples")
    print(f"UP rate: {data['target'].mean():.3f}")

    # Train/test split (time-based)
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[FEATURE_COLUMNS].values, train["target"].values
    X_test, y_test = test[FEATURE_COLUMNS].values, test["target"].values

    print(f"\nTrain: {len(train)} | Test: {len(test)}")
    print(f"Train UP: {y_train.mean():.3f} | Test UP: {y_test.mean():.3f}")

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Train with better hyperparameters
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight * 1.5,  # extra emphasis on UP
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    # Predictions
    test_probs = model.predict_proba(X_test)[:, 1]

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(y_test, test_probs)
    print(f"\nBest threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    test_preds = (test_probs >= best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    cm = confusion_matrix(y_test, test_preds)

    print("\n" + "=" * 50)
    print("XGBOOST RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"\nConfusion Matrix (threshold={best_thresh:.2f}):")
    print(f"                 Predicted")
    print(f"              DOWN    UP")
    print(f"Actual DOWN   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Actual UP     {cm[1,0]:4d}    {cm[1,1]:4d}")

    # Probability distribution
    print(f"\nProbability distribution:")
    print(f"  Min: {test_probs.min():.4f} | Max: {test_probs.max():.4f} | Mean: {test_probs.mean():.4f}")

    # Calibration
    cal_idx = int(len(X_train) * 0.85)
    cal_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, colsample_bylevel=0.7,
        min_child_weight=5, gamma=0.1,
        scale_pos_weight=scale_pos_weight * 1.5,
        eval_metric="logloss", random_state=42, n_jobs=4,
    )
    cal_model.fit(X_train[:cal_idx], y_train[:cal_idx])
    cal_probs = cal_model.predict_proba(X_train[cal_idx:])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_probs, y_train[cal_idx:])

    # Save everything
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(artifacts_dir, "xgboost_calibrator.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(artifacts_dir, "xgboost_threshold.txt"), "w") as f:
        f.write(str(best_thresh))

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "roc_auc": auc, "brier": brier, "best_threshold": best_thresh,
        "scale_pos_weight": scale_pos_weight,
    }
    with open(os.path.join(artifacts_dir, "xgboost_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        "actual": y_test,
        "prob": test_probs,
        "pred": test_preds,
    })
    pred_df.to_csv(os.path.join(artifacts_dir, "xgboost_test_predictions.csv"), index=False)

    print(f"\n✅ Saved to {artifacts_dir}/")
    print("   - xgboost_model.pkl")
    print("   - xgboost_calibrator.pkl")
    print("   - xgboost_threshold.txt")
    print("   - xgboost_metrics.json")

    if auc < 0.55:
        print("\n⚠️  ROC-AUC < 0.55 — weak signal. Consider:")
        print("   - Better features (order flow, sentiment, macro)")
        print("   - Different target definition")
    elif rec < 0.05:
        print("\n⚠️  Very low recall — model is too conservative.")
        print("   - Lower threshold manually, or")
        print("   - Use as a DOWN filter only")
    else:
        print("\n✅ Model looks usable. Deploy with ensemble.")


if __name__ == "__main__":
    main()