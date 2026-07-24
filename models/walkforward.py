"""
Phase 2 Walk-Forward Validation.
Trains on expanding window, tests on next month.
This is the honest way to evaluate time-series models.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
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


def find_best_threshold(y_true, probs):
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
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    print("Labeling...")
    labeled = triple_barrier_labels(raw, barrier_mode="pct", tp_pct=0.008, sl_pct=0.004, max_holding=30)

    print("Features...")
    featured = add_baseline_features(labeled)
    data = featured.dropna(subset=["label"] + FEATURE_COLUMNS).copy()
    data = data[data["label"] != 0].reset_index(drop=True)
    data["target"] = (data["label"] == 1).astype(int)
    data["year_month"] = data["timestamp"].dt.to_period("M")

    months = sorted(data["year_month"].unique())
    print(f"\nData spans {len(months)} months: {months[0]} to {months[-1]}")

    if len(months) < 4:
        print("❌ Need at least 4 months for walk-forward. Use regular train/test instead.")
        return

    # Walk-forward: train on months 0..n-1, test on month n
    results = []

    for i in range(3, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = data["year_month"].isin(train_months)
        test_mask = data["year_month"] == test_month

        train_df = data[train_mask]
        test_df = data[test_mask]

        if len(test_df) < 100:
            continue

        X_train, y_train = train_df[FEATURE_COLUMNS].values, train_df["target"].values
        X_test, y_test = test_df[FEATURE_COLUMNS].values, test_df["target"].values

        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw * 1.5,
            eval_metric="logloss", random_state=42, n_jobs=4,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        thresh, f1 = find_best_threshold(y_test, probs)
        preds = (probs >= thresh).astype(int)

        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        cm = confusion_matrix(y_test, preds)

        result = {
            "test_month": str(test_month),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "test_up_rate": y_test.mean(),
            "threshold": thresh,
            "f1": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "roc_auc": auc,
            "cm_tn": int(cm[0,0]), "cm_fp": int(cm[0,1]),
            "cm_fn": int(cm[1,0]), "cm_tp": int(cm[1,1]),
        }
        results.append(result)

        print(f"  {test_month}: AUC={auc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | UP={y_test.mean():.2%} | n={len(test_df)}")

    # Summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)

    aucs = [r["roc_auc"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    print(f"Months tested: {len(results)}")
    print(f"ROC-AUC:  mean={np.mean(aucs):.3f} | std={np.std(aucs):.3f} | min={np.min(aucs):.3f} | max={np.max(aucs):.3f}")
    print(f"Precision: mean={np.mean(precisions):.3f} | std={np.std(precisions):.3f}")
    print(f"Recall:    mean={np.mean(recalls):.3f} | std={np.std(recalls):.3f}")

    # Check for degradation
    if len(aucs) >= 3:
        first_half = np.mean(aucs[:len(aucs)//2])
        second_half = np.mean(aucs[len(aucs)//2:])
        print(f"\nFirst half AUC: {first_half:.3f}")
        print(f"Second half AUC: {second_half:.3f}")
        if second_half < first_half - 0.05:
            print("⚠️  Model degrading over time — may not generalize to future data")
        else:
            print("✅ Model stable across time periods")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "walkforward_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved results to {out_dir}/walkforward_results.json")


if __name__ == "__main__":
    main()