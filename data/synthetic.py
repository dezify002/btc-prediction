"""
Synthetic BTC-like 1-minute OHLCV generator.
"""

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(n_bars: int = 20000, start_price: float = 65000.0,
                              seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regimes = np.zeros(n_bars, dtype=int)
    regime_vol = {0: 0.0006, 1: 0.0015, 2: 0.004}
    regime_drift = {0: 0.0, 1: 0.00002, 2: -0.00001}

    state = 1
    i = 0
    while i < n_bars:
        run_len = rng.integers(200, 1500)
        regimes[i:i + run_len] = state
        i += run_len
        state = rng.choice([0, 1, 2], p=[0.4, 0.45, 0.15])

    closes = np.empty(n_bars)
    closes[0] = start_price
    for t in range(1, n_bars):
        r = regimes[t]
        shock = rng.normal(regime_drift[r], regime_vol[r])
        closes[t] = closes[t - 1] * (1 + shock)

    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    intrabar_range = np.abs(rng.normal(0, 1, n_bars)) * (closes * 0.0008) + 1e-6
    highs = np.maximum(opens, closes) + intrabar_range * rng.uniform(0.3, 1.0, n_bars)
    lows = np.minimum(opens, closes) - intrabar_range * rng.uniform(0.3, 1.0, n_bars)
    volume = rng.lognormal(mean=2.0, sigma=0.6, size=n_bars)

    ts = pd.date_range("2024-01-01", periods=n_bars, freq="1min")

    df = pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
        "regime": regimes,
    })
    return df


if __name__ == "__main__":
    df = generate_synthetic_ohlcv()
    df.to_csv("synthetic_btc_1m.csv", index=False)
    print(f"Generated {len(df)} synthetic 1m bars -> synthetic_btc_1m.csv")
    print(df.head())