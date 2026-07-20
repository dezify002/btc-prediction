"""
BTC/USDT 1-minute OHLCV downloader.
Requires: pip install ccxt pandas
"""

import argparse
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd


def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1000):
    all_rows = []
    cursor = since_ms
    ms_per_bar = exchange.parse_timeframe(timeframe) * 1000

    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + ms_per_bar
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="btc_1m.csv")
    args = parser.parse_args()

    exchange_cls = getattr(ccxt, args.exchange)
    exchange = exchange_cls({"enableRateLimit": True})

    since_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime.strptime(args.end, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)

    print(f"Fetching {args.symbol} {args.timeframe} from {args.start} to {args.end} on {args.exchange}...")
    rows = fetch_ohlcv_range(exchange, args.symbol, args.timeframe, since_ms, until_ms)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} bars to {args.out}")

    gaps = df["timestamp"].diff().dt.total_seconds().dropna()
    expected = pd.Timedelta(args.timeframe.replace("m", "min")).total_seconds()
    n_gaps = (gaps > expected * 1.5).sum()
    print(f"Sanity check: {n_gaps} gaps larger than expected bar interval.")


if __name__ == "__main__":
    main()