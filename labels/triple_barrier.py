"""
Volatility-adjusted OR fixed-percentage triple-barrier labeling.

Two barrier modes:
  - "atr" (original): barriers scale with recent volatility (ATR).
    Good for finding statistical edge, but -- as Phase 3 testing showed --
    can produce "wins" so small in dollar terms that trading fees erase
    them entirely in calm markets.
  - "pct" (new): barriers are a FIXED percentage of entry price, e.g.
    tp_pct=0.006 means "take profit at +0.6%" regardless of recent
    volatility. This guarantees every labeled "win" is large enough to
    plausibly clear a realistic fee/slippage budget, which "atr" mode
    does not guarantee.

For each bar t, define three barriers looking forward:
  - Upper (profit-take)
  - Lower (stop-loss)
  - Time (timeout): max_holding bars ahead

Label:
   1  -> upper barrier hit first
  -1  -> lower barrier hit first
   0  -> neither hit before timeout

Assumptions (document these, they matter):
  - We only have OHLC, not tick data, so if BOTH barriers are breached
    within the same bar we can't know which happened first. Default
    behavior here is conservative: treat it as the STOP-LOSS hitting
    first (worst case). This avoids overstating edge. You can flip
    `conservative_tie_break` off if you have intrabar (tick) data later.
  - In "atr" mode, ATR is computed using data available strictly BEFORE
    bar t (no lookahead into the label window itself).
  - Labels are undefined (NaN) for the last `max_holding` bars of the
    series, since there isn't a full forward window to evaluate.
"""

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, window: int = 14,
                 high_col: str = "high", low_col: str = "low",
                 close_col: str = "close") -> pd.Series:
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def triple_barrier_labels(
    df: pd.DataFrame,
    barrier_mode: str = "atr",       # "atr" or "pct"
    atr_window: int = 14,
    tp_mult: float = 2.0,            # used when barrier_mode == "atr"
    sl_mult: float = 1.0,            # used when barrier_mode == "atr"
    tp_pct: float = 0.006,           # used when barrier_mode == "pct" (0.6%)
    sl_pct: float = 0.003,           # used when barrier_mode == "pct" (0.3%)
    max_holding: int = 15,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    conservative_tie_break: bool = True,
) -> pd.DataFrame:
    """
    Returns a copy of df with added columns:
      atr (if barrier_mode == "atr"), upper_barrier, lower_barrier,
      label, bars_to_hit, hit_type
    """
    if barrier_mode not in ("atr", "pct"):
        raise ValueError(f"barrier_mode must be 'atr' or 'pct', got {barrier_mode!r}")

    out = df.copy().reset_index(drop=True)

    if barrier_mode == "atr":
        out["atr"] = compute_atr(out, atr_window, high_col, low_col, price_col)
        atr = out["atr"].values
    else:
        atr = None  # not used in pct mode

    n = len(out)
    labels = np.full(n, np.nan)
    bars_to_hit = np.full(n, np.nan)
    hit_type = np.array([None] * n, dtype=object)
    upper_arr = np.full(n, np.nan)
    lower_arr = np.full(n, np.nan)

    close = out[price_col].values
    high = out[high_col].values
    low = out[low_col].values

    for t in range(n - max_holding):
        entry_price = close[t]

        if barrier_mode == "atr":
            if np.isnan(atr[t]):
                continue  # not enough history yet to size barriers
            upper = entry_price + tp_mult * atr[t]
            lower = entry_price - sl_mult * atr[t]
        else:  # pct mode -- fixed percentage of entry price, no warmup needed
            upper = entry_price * (1 + tp_pct)
            lower = entry_price * (1 - sl_pct)

        upper_arr[t] = upper
        lower_arr[t] = lower

        label = 0.0
        bars = max_holding
        h_type = "timeout"

        for k in range(1, max_holding + 1):
            idx = t + k
            hit_upper = high[idx] >= upper
            hit_lower = low[idx] <= lower

            if hit_upper and hit_lower:
                if conservative_tie_break:
                    label, h_type = -1.0, "both_conservative_sl"
                else:
                    label, h_type = 1.0, "both_optimistic_tp"
                bars = k
                break
            elif hit_upper:
                label, h_type = 1.0, "tp"
                bars = k
                break
            elif hit_lower:
                label, h_type = -1.0, "sl"
                bars = k
                break

        labels[t] = label
        bars_to_hit[t] = bars
        hit_type[t] = h_type

    out["upper_barrier"] = upper_arr
    out["lower_barrier"] = lower_arr
    out["label"] = labels
    out["bars_to_hit"] = bars_to_hit
    out["hit_type"] = hit_type

    return out