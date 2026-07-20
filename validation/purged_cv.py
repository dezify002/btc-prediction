"""
Purged, embargoed walk-forward cross-validation.
"""

import numpy as np
import pandas as pd


def purged_walk_forward_splits(n_samples: int, n_folds: int = 5,
                                label_horizon: int = 15, embargo: int = 15):
    fold_size = n_samples // (n_folds + 1)

    for fold in range(1, n_folds + 1):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n_samples)

        train_end_raw = test_start
        purge_start = max(0, train_end_raw - label_horizon)

        train_idx = np.arange(0, purge_start)
        test_idx = np.arange(test_start, test_end)

        embargo_end = min(test_end + embargo, n_samples)

        yield train_idx, test_idx


def summarize_folds(n_samples, n_folds=5, label_horizon=15, embargo=15):
    rows = []
    for i, (tr, te) in enumerate(purged_walk_forward_splits(
            n_samples, n_folds, label_horizon, embargo), start=1):
        rows.append({
            "fold": i,
            "train_size": len(tr),
            "test_size": len(te),
        })
    return pd.DataFrame(rows)