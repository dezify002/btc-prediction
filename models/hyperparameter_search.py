"""
Hyperparameter search: tries several model configurations honestly, using
the SAME purged walk-forward CV as baseline_xgb.py -- no shortcuts that
would make results look better than they'll be in practice.

Uses the new percentage-based labels by default (barrier_mode="pct"),
since that fixes the fee-vs-ATR mismatch found in Phase 3 testing.

This will take a while -- each config trains 5 folds on your full
dataset. With ~5 configs x 5 folds on ~500k+ rows, expect this to run
for a meaningful chunk of time (many minutes depending on your machine).

Usage:
    python hyperparameter_search.py
"""

import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from baseline_xgb import prepare_dataset, run_baseline  # noqa: E402


# a small, deliberately compact grid -- enough to see real differences
# without taking hours to run
PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.6},
]


def run_baseline_quiet(data, n_folds, label_horizon, model_params):
    """Runs the fold loop and returns just (mean_auc, std_auc) without all
    of run_baseline's printed diagnostics -- keeps the search's output readable."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_baseline(data, n_folds=n_folds, label_horizon=label_horizon, model_params=model_params)

    output = buf.getvalue()
    mean_auc = None
    std_auc = None
    for line in output.splitlines():
        if line.startswith("Mean AUC across folds:"):
            mean_auc = float(line.split(":")[1].strip().split()[0])
        if line.startswith("AUC std across folds:"):
            std_auc = float(line.split(":")[1].strip().split()[0])
    return mean_auc, std_auc


def main():
    print("Loading real BTC data...")
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.csv"))

    print("Preparing dataset with fixed-percentage labels (recommended fix from Phase 3)...")
    data = prepare_dataset(raw, barrier_mode="pct", tp_pct=0.006, sl_pct=0.003)
    print(f"Dataset ready: {len(data)} labeled samples\n")

    results = []
    for i, params in enumerate(PARAM_GRID, start=1):
        print(f"[{i}/{len(PARAM_GRID)}] Testing config: {params}")
        start = time.time()
        mean_auc, std_auc = run_baseline_quiet(data, n_folds=5, label_horizon=15, model_params=params)
        elapsed = time.time() - start
        print(f"    -> Mean AUC: {mean_auc:.4f}  |  Std: {std_auc:.4f}  |  ({elapsed:.0f}s)\n")
        results.append({"params": params, "mean_auc": mean_auc, "std_auc": std_auc})

    results.sort(key=lambda r: r["mean_auc"], reverse=True)

    print("=" * 70)
    print("HYPERPARAMETER SEARCH RESULTS (ranked by mean AUC)")
    print("=" * 70)
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. AUC={r['mean_auc']:.4f} (std={r['std_auc']:.4f})  {r['params']}")

    best = results[0]
    print("\n" + "=" * 70)
    print("BEST CONFIG FOUND")
    print("=" * 70)
    print(best["params"])
    print(f"\nTo use this, update train_final_model.py's make_model() call to pass "
          f"these params, e.g.:\n    model = make_model({best['params']})")
    print("\nCAUTION: only trust this ranking if the AUC differences between the top")
    print("few configs are larger than their std -- small differences within noise")
    print("aren't a real reason to prefer one config over another.")


if __name__ == "__main__":
    main()