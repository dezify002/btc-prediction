"""
Diagnose why XGBoost predicts 100% DOWN.
Run this BEFORE training to understand your data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "labels"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))

from triple_barrier import triple_barrier_labels
from indicators import add_baseline_features, FEATURE_COLUMNS

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def main():
    print("=" * 60)
    print("XGBOOST DIAGNOSTICS")
    print("=" * 60)

    # Load data
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.csv"))
    print(f"\nRaw data: {len(raw)} rows")
    print(f"Price range: ${raw['close'].min():.2f} - ${raw['close'].max():.2f}")

    # Label
    labeled = triple_barrier_labels(raw, barrier_mode="pct", tp_pct=0.006, sl_pct=0.003, max_holding=15)
    print(f"\nAfter labeling: {len(labeled)} rows")

    # Features
    featured = add_baseline_features(labeled)
    data = featured.dropna(subset=["label"] + FEATURE_COLUMNS).copy()
    data = data[data["label"] != 0].reset_index(drop=True)
    data["target"] = (data["label"] == 1).astype(int)

    print(f"After dropna: {len(data)} rows")
    print(f"UP rate: {data['target'].mean():.3f} ({data['target'].sum()} UP, {(data['target']==0).sum()} DOWN)")

    # Split
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[FEATURE_COLUMNS].values, train["target"].values
    X_test, y_test = test[FEATURE_COLUMNS].values, test["target"].values

    print(f"\nTrain: {len(train)} | Test: {len(test)}")
    print(f"Train UP rate: {y_train.mean():.3f} | Test UP rate: {y_test.mean():.3f}")

    # Train model
    if not HAS_XGB:
        print("\n❌ xgboost not installed. Install with: pip install xgboost")
        return

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict probabilities
    test_probs = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("PROBABILITY DISTRIBUTION")
    print("=" * 60)
    print(f"Min prob:   {test_probs.min():.4f}")
    print(f"Max prob:   {test_probs.max():.4f}")
    print(f"Mean prob:  {test_probs.mean():.4f}")
    print(f"Median:     {np.median(test_probs):.4f}")
    print(f"Std dev:    {test_probs.std():.4f}")
    print(f"\nProb < 0.3:  {(test_probs < 0.3).sum()} ({(test_probs < 0.3).mean()*100:.1f}%)")
    print(f"Prob 0.3-0.5: {((test_probs >= 0.3) & (test_probs < 0.5)).sum()}")
    print(f"Prob 0.5-0.7: {((test_probs >= 0.5) & (test_probs < 0.7)).sum()}")
    print(f"Prob > 0.7:   {(test_probs > 0.7).sum()}")

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (top 20)")
    print("=" * 60)
    importance = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:20]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:25s} {imp:.4f} {bar}")

    # Save plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Probability histogram
    axes[0].hist(test_probs, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[0].axvline(test_probs.mean(), color="green", linestyle="--", label=f"mean={test_probs.mean():.3f}")
    axes[0].set_xlabel("Predicted Probability (UP)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Probability Distribution on Test Set")
    axes[0].legend()

    # Feature importance plot
    top_feats = feat_imp[:15]
    axes[1].barh([f[0] for f in top_feats][::-1], [f[1] for f in top_feats][::-1])
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top 15 Feature Importances")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "artifacts", "diagnostics.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n📊 Saved diagnostic plot to: {out_path}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if test_probs.max() < 0.5:
        print("⚠️  Model NEVER predicts UP (max prob < 0.5)")
        print("   → Features may be too weak, or target is too hard")
        print("   → Try: wider barriers (tp_pct=0.01), longer holding (30-60m)")
        print("   → Try: add more features (order flow, sentiment, macro)")
    elif test_probs.mean() < 0.3:
        print("⚠️  Model is very conservative (mean prob < 0.3)")
        print("   → This is OK for a filter, but not for standalone trading")
    else:
        print("✅ Model produces a range of probabilities — usable")

    if max(importance) < 0.05:
        print("⚠️  No single feature is important (all < 0.05)")
        print("   → Your features may not contain predictive signal")
    else:
        print(f"✅ Top feature importance: {max(importance):.4f} — signal exists")

    print("\nDone. Check the plot and decide if features need work.")


if __name__ == "__main__":
    main()