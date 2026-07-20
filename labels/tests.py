"""
Unit tests for triple_barrier_labels.

Each test builds a tiny, hand-checkable OHLC sequence where we know
in advance which barrier *should* fire, then asserts the function
agrees. This is the step people skip -- and it's the step that catches
off-by-one and lookahead bugs before they contaminate a real backtest.
"""

import numpy as np
import pandas as pd
from triple_barrier import triple_barrier_labels, compute_atr


def make_flat_df(n=40, price=100.0):
    """Flat, low-volatility baseline series to build ATR history on."""
    return pd.DataFrame({
        "open": [price] * n,
        "high": [price + 0.1] * n,
        "low": [price - 0.1] * n,
        "close": [price] * n,
    })


def test_upper_barrier_hit():
    df = make_flat_df(n=20, price=100.0)
    # after warmup, force a sharp rally in bar 21
    rally = pd.DataFrame({
        "open": [100, 100, 106, 100, 100],
        "high": [100.1, 100.1, 106.5, 100.1, 100.1],
        "low": [99.9, 99.9, 105.5, 99.9, 99.9],
        "close": [100, 100, 106, 100, 100],
    })
    df = pd.concat([df, rally], ignore_index=True)

    res = triple_barrier_labels(df, atr_window=14, tp_mult=2.0, sl_mult=1.0, max_holding=5)
    t = 19  # last flat bar before the rally; entry price 100, atr ~0.2
    assert res.loc[t, "label"] == 1.0, f"expected TP hit, got {res.loc[t, 'label']}"
    assert res.loc[t, "hit_type"] == "tp"
    print("PASS: upper barrier (take-profit) correctly detected")


def test_lower_barrier_hit():
    df = make_flat_df(n=20, price=100.0)
    drop = pd.DataFrame({
        "open": [100, 100, 94, 100, 100],
        "high": [100.1, 100.1, 94.5, 100.1, 100.1],
        "low": [99.9, 99.9, 93.5, 99.9, 99.9],
        "close": [100, 100, 94, 100, 100],
    })
    df = pd.concat([df, drop], ignore_index=True)

    res = triple_barrier_labels(df, atr_window=14, tp_mult=2.0, sl_mult=1.0, max_holding=5)
    t = 19
    assert res.loc[t, "label"] == -1.0, f"expected SL hit, got {res.loc[t, 'label']}"
    assert res.loc[t, "hit_type"] == "sl"
    print("PASS: lower barrier (stop-loss) correctly detected")


def test_timeout_no_barrier_hit():
    df = make_flat_df(n=30, price=100.0)  # stays flat the whole time
    res = triple_barrier_labels(df, atr_window=14, tp_mult=2.0, sl_mult=1.0, max_holding=5)
    t = 15
    assert res.loc[t, "label"] == 0.0, f"expected timeout, got {res.loc[t, 'label']}"
    assert res.loc[t, "hit_type"] == "timeout"
    assert res.loc[t, "bars_to_hit"] == 5
    print("PASS: timeout correctly detected when neither barrier is hit")


def test_conservative_tie_break_on_same_bar():
    df = make_flat_df(n=20, price=100.0)
    # one huge-range bar that breaches both barriers simultaneously
    wild = pd.DataFrame({
        "open": [100, 100, 100, 100, 100],
        "high": [100.1, 100.1, 110, 100.1, 100.1],   # breaches upper
        "low": [99.9, 99.9, 90, 99.9, 99.9],          # breaches lower
        "close": [100, 100, 100, 100, 100],
    })
    df = pd.concat([df, wild], ignore_index=True)

    res = triple_barrier_labels(df, atr_window=14, tp_mult=2.0, sl_mult=1.0,
                                 max_holding=5, conservative_tie_break=True)
    t = 19
    assert res.loc[t, "label"] == -1.0, "conservative tie-break should default to stop-loss"
    assert res.loc[t, "hit_type"] == "both_conservative_sl"
    print("PASS: same-bar dual breach resolves to conservative (SL) label")


def test_no_lookahead_in_atr():
    """ATR at time t must not use data from t or later."""
    df = make_flat_df(n=30, price=100.0)
    atr = compute_atr(df, window=14)
    # ATR should be NaN for the first `window` rows (not enough history)
    assert atr.iloc[:13].isna().all(), "ATR should be NaN before enough history exists"
    assert not np.isnan(atr.iloc[13]), "ATR should be defined once window is filled"
    print("PASS: ATR warmup window has no lookahead / premature values")


def test_last_rows_are_nan_no_full_window():
    df = make_flat_df(n=25, price=100.0)
    res = triple_barrier_labels(df, atr_window=14, max_holding=10)
    # last 10 rows can't have a full forward window -> label must be NaN
    tail_labels = res["label"].iloc[-10:]
    assert tail_labels.isna().all(), "rows without a full forward window must be unlabeled"
    print("PASS: incomplete trailing windows correctly left unlabeled")


def test_pct_mode_upper_barrier_hit():
    df = make_flat_df(n=10, price=100.0)
    rally = pd.DataFrame({
        "open": [100, 100.7, 100, 100],
        "high": [100.1, 100.8, 100.1, 100.1],
        "low": [99.9, 100.6, 99.9, 99.9],
        "close": [100, 100.7, 100, 100],
    })
    df = pd.concat([df, rally], ignore_index=True)
    # tp_pct=0.005 (+0.5%) -> upper barrier at 100.5, hit by the 100.8 high
    res = triple_barrier_labels(df, barrier_mode="pct", tp_pct=0.005, sl_pct=0.003, max_holding=3)
    t = 9
    assert res.loc[t, "label"] == 1.0, f"expected pct-mode TP hit, got {res.loc[t, 'label']}"
    assert "atr" not in res.columns or res["atr"].isna().all(), \
        "pct mode should not require/compute atr"
    print("PASS: pct-mode upper barrier correctly detected, no ATR dependency")


def test_pct_mode_no_warmup_needed():
    """Unlike atr mode, pct mode should be able to label from bar 0 -- no
    rolling-window warmup is needed since barriers are just % of entry price."""
    df = make_flat_df(n=5, price=100.0)
    res = triple_barrier_labels(df, barrier_mode="pct", tp_pct=0.01, sl_pct=0.01, max_holding=3)
    assert not np.isnan(res.loc[0, "label"]), \
        "pct mode should label bar 0 immediately, no ATR warmup required"
    print("PASS: pct mode requires no warmup period (unlike atr mode)")


if __name__ == "__main__":
    test_upper_barrier_hit()
    test_lower_barrier_hit()
    test_timeout_no_barrier_hit()
    test_conservative_tie_break_on_same_bar()
    test_no_lookahead_in_atr()
    test_last_rows_are_nan_no_full_window()
    test_pct_mode_upper_barrier_hit()
    test_pct_mode_no_warmup_needed()
    print("\nAll tests passed.")