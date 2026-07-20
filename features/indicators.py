"""
Phase 2 baseline features.

Deliberately minimal -- the point of a baseline is to prove the pipeline
(data -> features -> model -> calibration -> validation) works honestly
before adding complexity. Every feature here uses only information
available strictly at or before bar t (no lookahead).
"""

import numpy as np
import pandas as pd


def add_baseline_features(df: pd.DataFrame,
                           close_col: str = "close",
                           high_col: str = "high",
                           low_col: str = "low",
                           volume_col: str = "volume") -> pd.DataFrame:
    out = df.copy()
    close = out[close_col]

    # -- trend / momentum --
    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()
    out["ema_dist"] = (out["ema_9"] - out["ema_21"]) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # -- volatility --
    out["atr_14"] = out["atr"] if "atr" in out.columns else _atr(out, 14, high_col, low_col, close_col)
    out["realized_vol_20"] = close.pct_change().rolling(20).std()
    # raw=True passes a numpy array instead of a pandas Series -- avoids
    # object creation overhead per window, dramatically faster on large data
    out["atr_pct_rank_100"] = out["atr_14"].rolling(100).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )

    # -- price structure --
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)
    rolling_high_20 = out[high_col].rolling(20).max()
    rolling_low_20 = out[low_col].rolling(20).min()
    out["dist_from_high_20"] = (rolling_high_20 - close) / close
    out["dist_from_low_20"] = (close - rolling_low_20) / close

    # -- volume --
    if volume_col in out.columns:
        out["vol_zscore_20"] = (
            (out[volume_col] - out[volume_col].rolling(20).mean())
            / out[volume_col].rolling(20).std()
        )

    # -- time of day (BTC trades 24/7 but session effects still exist) --
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"])
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    return out


def _atr(df, window, high_col, low_col, close_col):
    high, low, close = df[high_col], df[low_col], df[close_col]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


FEATURE_COLUMNS = [
    "ema_dist", "rsi_14",
    "ret_1", "ret_5", "ret_15", "dist_from_high_20", "dist_from_low_20",
    "vol_zscore_20", "hour_sin", "hour_cos",
]