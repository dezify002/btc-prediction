"""
BTC/USDT 1-minute OHLCV downloader using Binance's public historical
data archive (data.binance.vision) instead of the live REST API.
"""

import argparse
import io
import zipfile
from datetime import datetime

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year_month}.zip"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def month_range(start: str, end: str):
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(symbol, year_month):
    url = BASE_URL.format(symbol=symbol, year_month=year_month)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [skip] {year_month}: HTTP {resp.status_code}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="YYYY-MM")
    parser.add_argument("--end", required=True, help="YYYY-MM")
    parser.add_argument("--out", default="btc_1m.csv")
    args = parser.parse_args()

    all_dfs = []
    for ym in month_range(args.start, args.end):
        print(f"Fetching {args.symbol} {ym}...")
        df = fetch_month(args.symbol, ym)
        if df is not None:
            print(f"  got {len(df)} bars")
            all_dfs.append(df)

    if not all_dfs:
        print("No data retrieved -- check symbol name and date range.")
        return

    full = pd.concat(all_dfs, ignore_index=True)

    ot = full["open_time"]
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    full["timestamp"] = pd.to_datetime(ot, unit=unit, utc=True)

    out_df = full[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out_df = out_df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} bars to {args.out}")

    gaps = out_df["timestamp"].diff().dt.total_seconds().dropna()
    n_gaps = (gaps > 90).sum()
    print(f"Sanity check: {n_gaps} gaps larger than expected 1-minute interval.")


if __name__ == "__main__":
    main()