"""
Phase 2+ features — expanded set with stronger predictive signals.
Every feature uses only information strictly at or before bar t (no lookahead).
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
    high = out[high_col]
    low = out[low_col]

    # -- trend / momentum --
    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_dist"] = (out["ema_9"] - out["ema_21"]) / close
    out["dist_ema50"] = (close - out["ema_50"]) / close

    # Wilder's RSI (canonical)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI slope (momentum of momentum)
    out["rsi_slope_3"] = out["rsi_14"].diff(3)

    # -- volatility --
    out["atr_14"] = out["atr"] if "atr" in out.columns else _atr(out, 14, high_col, low_col, close_col)
    out["atr_pct"] = out["atr_14"] / close
    out["realized_vol_20"] = close.pct_change().rolling(20).std()
    out["atr_pct_rank_100"] = out["atr_14"].rolling(100).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )

    # -- candle structure --
    body = (close - out["open"]).abs() if "open" in out.columns else (close - close.shift(1)).abs()
    rng = high - low
    out["body_pct"] = body / rng.replace(0, np.nan)
    out["upper_wick"] = (high - close) / rng.replace(0, np.nan)
    out["lower_wick"] = (close - low) / rng.replace(0, np.nan)

    # -- price structure --
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_15"] = close.pct_change(15)
    out["ret_30"] = close.pct_change(30)

    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    out["dist_from_high_20"] = (rolling_high_20 - close) / close
    out["dist_from_low_20"] = (close - rolling_low_20) / close

    # -- volume --
    if volume_col in out.columns:
        vol = out[volume_col]
        vol_ma20 = vol.rolling(20).mean()
        vol_std20 = vol.rolling(20).std()
        out["vol_zscore_20"] = (vol - vol_ma20) / vol_std20.replace(0, np.nan)
        out["vol_ratio_5"] = vol / vol.rolling(5).mean().replace(0, np.nan)
        out["dollar_vol"] = vol * close

    # -- trend strength --
    out["trend_30m"] = np.sign(close.pct_change(30))
    out["consecutive_up"] = _consecutive(close, "up")
    out["consecutive_down"] = _consecutive(close, "down")

    # -- time features --
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"])
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        out["dayofweek_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        out["dayofweek_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

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


def _consecutive(close, direction):
    """Count consecutive bars in same direction."""
    ret = close.diff()
    mask = ret > 0 if direction == "up" else ret < 0
    grp = (~mask).cumsum()
    return mask.groupby(grp).cumcount()


FEATURE_COLUMNS = [
    "ema_dist", "dist_ema50", "rsi_14", "rsi_slope_3",
    "atr_14", "atr_pct", "realized_vol_20", "atr_pct_rank_100",
    "body_pct", "upper_wick", "lower_wick",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_15", "ret_30",
    "dist_from_high_20", "dist_from_low_20",
    "vol_zscore_20", "vol_ratio_5", "dollar_vol",
    "trend_30m", "consecutive_up", "consecutive_down",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
]