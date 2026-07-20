"""
Fetches the most recent BTC price data directly from Binance (via ccxt)
and prints a probability that BTC closes higher over the next ~15 minutes,
using the model trained by train_final_model.py.

IMPORTANT CAVEATS (read before trusting the output):
  - This is a PROBABILITY, not a prediction of certainty. A 58% reading
    means "58 out of 100 similar setups historically went up" -- not
    "BTC will go up."
  - Per Phase 2/3 testing on 2023-2024 data, this model's edge is modest
    overall (~53-54% AUC) and concentrated almost entirely in calm,
    low-volatility market conditions. In choppy/volatile markets, treat
    any prediction here with much less confidence.
  - Phase 3 testing showed this specific edge is NOT profitable after
    realistic trading fees/slippage at the position sizes tested -- this
    tool is informational only, not a trading signal to act on directly.
  - Markets change over time (regime drift). A model trained on 2023-2024
    data may not reflect current conditions. Retrain periodically.

Usage:
    python predict_now.py
"""

import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from indicators import add_baseline_features  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def fetch_latest_data():
    """
    Pulls recent 1-minute BTC/USDT bars directly from Binance via ccxt --
    the same source your historical training data came from. Fetches the
    last 200 bars, far more than enough history for this model's rolling
    window features (largest window used is 20 bars).
    """
    try:
        import ccxt
    except ImportError:
        print("ccxt is not installed. Run: pip install ccxt --break-system-packages")
        sys.exit(1)

    exchange = ccxt.binance({"enableRateLimit": True})

    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=200)
    except Exception as e:
        print(f"Failed to fetch live data from Binance: {e}")
        print("Check your internet connection, or the exchange may be temporarily "
              "unreachable/rate-limiting.")
        sys.exit(1)

    if not ohlcv:
        print("No data returned from Binance -- try again in a moment.")
        sys.exit(1)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_order_book_signal():
    """
    Pulls the LIVE order book and computes a simple bid/ask imbalance signal.

    IMPORTANT: this is NOT part of the trained/validated model. Binance's
    historical data archive (what your model was trained on) never recorded
    order book snapshots for 2023-2024, so there's no way to backtest this
    signal the way everything else in this pipeline was backtested. Treat
    it as extra, unvalidated context -- not something with a measured AUC
    or a Phase 2/3-style track record behind it.
    """
    try:
        import ccxt
    except ImportError:
        return None

    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        book = exchange.fetch_order_book("BTC/USDT", limit=50)
    except Exception:
        return None

    bid_volume = sum(qty for _, qty in book["bids"])
    ask_volume = sum(qty for _, qty in book["asks"])
    total = bid_volume + ask_volume
    if total == 0:
        return None

    imbalance = (bid_volume - ask_volume) / total  # -1 (all asks) to +1 (all bids)
    return {
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "imbalance": imbalance,
    }


def main():
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run train_final_model.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

    print("Fetching latest BTC price data from Binance...")
    raw = fetch_latest_data()

    print("Computing features...")
    featured = add_baseline_features(raw)

    complete = featured.dropna(subset=feature_columns)
    if len(complete) == 0:
        print("Not enough recent data to compute all features (need ~20+ bars of history). "
              "Try again -- this can happen right after Yahoo's feed has a gap.")
        sys.exit(1)

    latest = complete.iloc[-1]
    X_latest = latest[feature_columns].values.reshape(1, -1).astype(float)

    raw_prob = model.predict_proba(X_latest)[0, 1]
    calibrated_prob = calibrator.transform([raw_prob])[0]

    latest_time = latest["timestamp"]
    latest_price = latest["close"]

    print("\n" + "=" * 50)
    print("BTC 15-MINUTE DIRECTIONAL PROBABILITY")
    print("=" * 50)
    print(f"As of:         {latest_time}")
    print(f"Current price: ${latest_price:,.2f}")
    print(f"P(up):         {calibrated_prob*100:.1f}%")
    print(f"P(down):       {(1-calibrated_prob)*100:.1f}%")
    print("=" * 50)

    ob_signal = fetch_order_book_signal()
    if ob_signal is not None:
        imb = ob_signal["imbalance"]
        lean = "buy-side (more resting bids)" if imb > 0 else "sell-side (more resting asks)"
        print("\n" + "-" * 50)
        print("LIVE ORDER BOOK (extra context -- NOT part of the validated model)")
        print("-" * 50)
        print(f"Bid volume (top 50 levels): {ob_signal['bid_volume']:.2f} BTC")
        print(f"Ask volume (top 50 levels): {ob_signal['ask_volume']:.2f} BTC")
        print(f"Imbalance:                  {imb:+.3f}  (leaning {lean})")
        print("NOTE: this signal was NOT backtested -- no historical order book data")
        print("exists to validate it against. Treat as informational color only,")
        print("not a measured, trustworthy edge like the probability above.")

    print("\nReminder: this is a probability estimate based on a modest, narrow")
    print("historical edge (see Phase 2/3 testing notes at the top of this file).")
    print("It is NOT a trading signal and was found unprofitable after realistic")
    print("fees in backtesting. Treat as informational only.")


if __name__ == "__main__":
    main()