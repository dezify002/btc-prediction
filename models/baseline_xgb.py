"""
Phase 2: baseline model.

Trains an XGBoost classifier on the triple-barrier labels using purged
walk-forward CV, calibrates the raw probabilities, and reports the
metrics that actually gate whether Phase 3 is worth doing:
  - ROC-AUC (discrimination: can it separate long-hits from short-hits at all?)
  - Brier score (calibration: are the probabilities honest?)
  - Reliability curve data (bucketed predicted vs actual)

This deliberately only predicts LONG vs SHORT (drops timeout bars) to
keep the baseline simple -- a natural first extension is a 3-class or
two-model (P(long), P(short)) setup, but get this working first.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "labels"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "validation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from triple_barrier import triple_barrier_labels          # noqa: E402
from indicators import add_baseline_features, FEATURE_COLUMNS  # noqa: E402
from purged_cv import purged_walk_forward_splits           # noqa: E402

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier


def make_model(params=None):
    """XGBoost if available (recommended), else sklearn GBM as a fallback
    so this pipeline still runs somewhere without network/package access.
    Pass `params` (a dict) to override the defaults, e.g. from a
    hyperparameter search."""
    default_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )
    if params:
        default_params.update(params)

    if HAS_XGB:
        return xgb.XGBClassifier(
            eval_metric="logloss", random_state=42, **default_params
        )
    # GradientBoostingClassifier doesn't accept colsample_bytree -- drop it
    sk_params = {k: v for k, v in default_params.items() if k != "colsample_bytree"}
    return GradientBoostingClassifier(random_state=42, **sk_params)


def prepare_dataset(raw_df: pd.DataFrame, max_holding: int = 15,
                     barrier_mode: str = "pct", tp_pct: float = 0.006,
                     sl_pct: float = 0.003, tp_mult: float = 2.0, sl_mult: float = 1.0):
    """
    barrier_mode="pct" (new default): fixed-percentage barriers, sized to
        clear a realistic fee/slippage budget by design (see labels/triple_barrier.py).
    barrier_mode="atr": original volatility-relative barriers -- kept
        available for comparison against the earlier Phase 2/3 results.
    """
    print(f"  Applying triple-barrier labeling (mode={barrier_mode})...")
    if barrier_mode == "pct":
        labeled = triple_barrier_labels(
            raw_df, barrier_mode="pct", tp_pct=tp_pct, sl_pct=sl_pct, max_holding=max_holding
        )
    else:
        labeled = triple_barrier_labels(
            raw_df, barrier_mode="atr", atr_window=14, tp_mult=tp_mult, sl_mult=sl_mult,
            max_holding=max_holding
        )
    print(f"  Labeling done ({len(labeled)} rows). Computing features...")
    featured = add_baseline_features(labeled)
    print("  Features done. Dropping incomplete rows and timeouts...")

    data = featured.dropna(subset=["label"] + FEATURE_COLUMNS).copy()
    data = data[data["label"] != 0].reset_index(drop=True)
    data["target"] = (data["label"] == 1).astype(int)
    return data


def reliability_table(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": float(np.mean(y_prob[mask])),
            "actual_rate": float(np.mean(y_true[mask])),
        })
    return pd.DataFrame(rows)


def confidence_breakdown(timestamps, atr_values, y_true, y_prob, threshold=0.5):
    """
    Shows WHERE (in time and in volatility) the confident (prob > threshold)
    predictions occur, and whether they actually pay off there. This is the
    check that tells you whether an apparent edge is broad-based or is really
    just one market regime / time period masquerading as a general pattern.
    """
    confident = y_prob > threshold
    n_confident = confident.sum()
    n_total = len(y_prob)

    print(f"\n-- Confidence breakdown (predictions > {threshold}) --")
    print(f"Confident predictions: {n_confident} / {n_total} "
          f"({n_confident / n_total * 100:.2f}% of all test bars)")

    if n_confident == 0:
        print("No predictions exceeded the threshold -- nothing to break down.")
        return

    ts = pd.to_datetime(timestamps.iloc[np.where(confident)[0]] if hasattr(timestamps, "iloc")
                         else timestamps[confident])
    months = ts.dt.to_period("M")

    print("\nBy month (where do confident predictions cluster in time?):")
    month_counts = months.value_counts().sort_index()
    for month, count in month_counts.items():
        mask_month = confident & (pd.to_datetime(timestamps).dt.to_period("M") == month).values
        hit_rate = y_true[mask_month].mean() if mask_month.sum() > 0 else float("nan")
        print(f"  {month}: n={count:5d}  actual_hit_rate={hit_rate:.3f}")

    print("\nBy volatility quartile (does the edge depend on market conditions?):")
    atr_series = pd.Series(atr_values)
    quartiles = pd.qcut(atr_series, 4, labels=["Q1_low_vol", "Q2", "Q3", "Q4_high_vol"], duplicates="drop")
    for q in quartiles.cat.categories:
        mask_q = confident & (quartiles == q).values
        n_q = mask_q.sum()
        hit_rate = y_true[mask_q].mean() if n_q > 0 else float("nan")
        print(f"  {q}: n={n_q:5d}  actual_hit_rate={hit_rate:.3f}")

    print("\nInterpretation: if confident predictions and their hit rates are spread "
          "fairly evenly across months and volatility quartiles, that supports a "
          "genuine, broad-based edge. If they're concentrated in 1-2 months or one "
          "volatility regime, treat the edge as regime-dependent, not general -- "
          "it may not transfer to different future market conditions.")


def run_baseline(data: pd.DataFrame, n_folds: int = 5, label_horizon: int = 15, model_params=None):
    if not HAS_XGB:
        print("[note: xgboost not available in this environment, "
              "falling back to sklearn GradientBoostingClassifier for this demo run. "
              "Install xgboost locally for the real thing -- it's faster and usually "
              "performs at least as well on tabular data like this.]\n")

    X = data[FEATURE_COLUMNS].values
    y = data["target"].values
    has_timestamp = "timestamp" in data.columns
    has_atr = "atr_14" in data.columns

    fold_aucs, fold_briers = [], []
    all_test_probs, all_test_y = [], []
    all_test_timestamps, all_test_atr = [], []

    for fold_i, (train_idx, test_idx) in enumerate(
            purged_walk_forward_splits(len(data), n_folds, label_horizon, embargo=label_horizon), start=1):
        if len(train_idx) < 200 or len(test_idx) < 20:
            print(f"Fold {fold_i}: skipped (insufficient samples)")
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # -- FIX: Use the SAME model for predictions AND calibration --
        # Train on the full training set first
        model = make_model(model_params)
        model.fit(X_train, y_train)
        raw_probs = model.predict_proba(X_test)[:, 1]

        # Calibrate using predictions from THIS SAME model on a held-out slice
        cal_cutoff = int(len(X_train) * 0.85)
        cal_probs_fit = model.predict_proba(X_train[cal_cutoff:])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_probs_fit, y_train[cal_cutoff:])

        calibrated_probs = iso.transform(raw_probs)

        auc = roc_auc_score(y_test, raw_probs)
        brier = brier_score_loss(y_test, calibrated_probs)

        fold_aucs.append(auc)
        fold_briers.append(brier)
        all_test_probs.append(calibrated_probs)
        all_test_y.append(y_test)
        if has_timestamp:
            all_test_timestamps.append(data["timestamp"].values[test_idx])
        if has_atr:
            all_test_atr.append(data["atr_14"].values[test_idx])

        print(f"Fold {fold_i}: train={len(train_idx):6d} test={len(test_idx):5d}  "
              f"AUC={auc:.4f}  Brier(calibrated)={brier:.4f}")
        print(f"  -- Fold {fold_i} reliability (calibrated) --")
        rt = reliability_table(y_test, calibrated_probs)
        print("  " + rt.to_string(index=False).replace("\n", "\n  "))
        print()

    print()
    print(f"Mean AUC across folds:   {np.mean(fold_aucs):.4f}  (0.50 = no edge)")
    print(f"AUC std across folds:    {np.std(fold_aucs):.4f}  "
          f"(high std relative to the mean means the edge is not stable/repeatable)")
    print(f"Mean Brier (calibrated): {np.mean(fold_briers):.4f}  (lower is better; "
          f"0.25 = naive 50/50 baseline)")

    if np.mean(fold_aucs) < 0.53:
        print("\nKILL CRITERION HIT: mean AUC below ~0.53 -- no meaningful discrimination "
              "found. Do not proceed to Phase 3 on this feature set; revisit features "
              "or reconsider the labeling before adding complexity.")
    elif np.std(fold_aucs) > 0.03:
        print("\nCAUTION: AUC varies substantially across folds (std > 0.03). "
              "The apparent edge may be concentrated in one time period rather than "
              "a stable, generalizable pattern. Look at which fold(s) are carrying "
              "the mean before treating this as real edge.")
    else:
        print("\nAUC clears the discrimination floor and is reasonably stable across "
              "folds -- reasonable to proceed to the pooled reliability check and, "
              "if that also looks sane, Phase 3 simulation.")

    all_probs = np.concatenate(all_test_probs)
    all_y = np.concatenate(all_test_y)
    print("\n-- Reliability table (calibrated probabilities, all folds pooled) --")
    print(reliability_table(all_y, all_probs).to_string(index=False))

    if has_timestamp and has_atr:
        all_timestamps = np.concatenate(all_test_timestamps)
        all_atr = np.concatenate(all_test_atr)
        confidence_breakdown(pd.Series(all_timestamps), all_atr, all_y, all_probs, threshold=0.5)
    else:
        print("\n(Skipping confidence breakdown -- timestamp/atr_14 not found in dataset.)")


if __name__ == "__main__":
    print("Loading real BTC data...\n")
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.csv"))
    data = prepare_dataset(raw)
    print(f"Dataset ready: {len(data)} labeled long/short samples "
          f"(timeouts dropped for this baseline)\n")

    run_baseline(data, n_folds=5, label_horizon=15)