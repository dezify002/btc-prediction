"""
Phase 1 deliverable: label distribution report.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "labels"))
from triple_barrier import triple_barrier_labels  # noqa: E402


def label_report(df: pd.DataFrame, true_regime_col: str = None) -> None:
    labeled = df.dropna(subset=["label"]).copy()
    n = len(labeled)

    print("=" * 60)
    print("PHASE 1 LABEL REPORT")
    print("=" * 60)
    print(f"Total labeled bars: {n} (of {len(df)} total bars)")
    print()

    print("-- Label distribution --")
    counts = labeled["label"].value_counts().sort_index()
    for lbl, cnt in counts.items():
        name = {1.0: "LONG (TP hit)", -1.0: "SHORT (SL hit)", 0.0: "TIMEOUT"}.get(lbl, str(lbl))
        print(f"  {name:20s}: {cnt:7d}  ({cnt / n * 100:5.1f}%)")

    timeout_pct = (labeled["label"] == 0).mean() * 100
    print()
    if timeout_pct > 85:
        print(f"WARNING: {timeout_pct:.1f}% timeout labels -- barriers may be too wide.")
    elif timeout_pct < 15:
        print(f"WARNING: only {timeout_pct:.1f}% timeout labels -- barriers may be too tight.")
    else:
        print(f"Timeout rate ({timeout_pct:.1f}%) is in a workable range.")

    print()
    print("-- Time to barrier hit (bars) --")
    hit_only = labeled[labeled["label"] != 0]
    if len(hit_only) > 0:
        print(f"  mean:   {hit_only['bars_to_hit'].mean():.2f}")
        print(f"  median: {hit_only['bars_to_hit'].median():.2f}")
        print(f"  p90:    {hit_only['bars_to_hit'].quantile(0.9):.2f}")

    print()
    print("-- Hit type breakdown --")
    print(labeled["hit_type"].value_counts().to_string())

    if true_regime_col and true_regime_col in labeled.columns:
        print()
        print(f"-- Label distribution by regime ({true_regime_col}) --")
        for regime_val, group in labeled.groupby(true_regime_col):
            gc = group["label"].value_counts(normalize=True).sort_index()
            row = "  ".join(f"{lbl:+.0f}:{pct*100:5.1f}%" for lbl, pct in gc.items())
            print(f"  regime {regime_val}: n={len(group):6d}  {row}")

    print()
    print("-- ATR / volatility quartile breakdown --")
    labeled["atr_quartile"] = pd.qcut(labeled["atr"], 4, labels=["Q1_low_vol", "Q2", "Q3", "Q4_high_vol"])
    for q, group in labeled.groupby("atr_quartile", observed=True):
        gc = group["label"].value_counts(normalize=True).sort_index()
        row = "  ".join(f"{lbl:+.0f}:{pct*100:5.1f}%" for lbl, pct in gc.items())
        print(f"  {q}: n={len(group):6d}  {row}")

    print("=" * 60)


if __name__ == "__main__":
    print("Loading real BTC data...\n")
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "btc_1m.csv"))
    print("Applying volatility-adjusted triple-barrier labeling...\n")
    labeled = triple_barrier_labels(
        raw, atr_window=14, tp_mult=2.0, sl_mult=1.0, max_holding=15
    )

    label_report(labeled, true_regime_col="regime")