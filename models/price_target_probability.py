"""
Asks for a target BTC price and a target time (UTC, 12-hour). Fetches live price
data, runs it through the SAME technical analysis used by predict_now.py
(EMA, RSI, momentum, distance from recent highs/lows, volume, time-of-day
-- your trained Phase 2 model), and combines that directional read with
recent volatility to estimate the probability of hitting your target.

METHOD (read this before trusting the output):
  1. Pull recent 1-minute bars from Binance and compute the same technical
     features used to train your model.
  2. Run those features through your trained, calibrated model to get
     P(BTC up in the next 15 minutes) -- this IS real chart analysis,
     not a guess.
  3. Convert that probability into an implied drift (using the inverse
     normal distribution) rather than assuming no trend at all.
  4. Combine that drift with recent volatility to project a probability
     distribution over price at your target time, and read off the
     probability of reaching your target.

HONEST LIMITS -- this still isn't a crystal ball:
  - Your model was trained and validated specifically for a 15-MINUTE
    horizon. Its directional read is real, but extrapolating that same
    drift rate out to 30, 60+ minutes is an assumption this script makes,
    not something the model itself was tested on. The further your
    target time is from ~15 minutes away, the more speculative this gets
    -- the script flags this explicitly in its output.
  - Per your own Phase 2/3 testing, this model's edge is modest overall
    (~53-54% AUC) and strongest in calm, low-volatility conditions.
  - Assumes volatility stays roughly constant between now and the target
    time -- a real news/liquidation-driven spike would invalidate this.

Usage:
    python price_target_probability.py
"""

import math
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from indicators import add_baseline_features  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def fetch_recent_data(limit=1500):
    """Pulls recent 1-minute bars from Binance -- enough history for both
    the technical indicators AND a reasonable volatility estimate."""
    try:
        import ccxt
    except ImportError:
        print("ccxt is not installed. Run: pip install ccxt --break-system-packages")
        sys.exit(1)

    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=limit)
    except Exception as e:
        print(f"Failed to fetch data from Binance: {e}")
        sys.exit(1)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_model():
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
    return model, calibrator, feature_columns


def parse_target_time(time_str: str, now_utc: datetime) -> datetime:
    """Forgiving parser: '10:00', '10;00', '10.00', '1000', '10am', '2:30pm', etc."""
    import re

    s = time_str.strip().lower().replace(" ", "")
    is_pm = "pm" in s
    is_am = "am" in s
    s = s.replace("am", "").replace("pm", "")
    s = re.sub(r"[;.,]", ":", s)

    if ":" in s:
        parts = s.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    elif s.isdigit() and len(s) <= 2:
        hour, minute = int(s), 0
    elif s.isdigit() and len(s) in (3, 4):
        minute = int(s[-2:])
        hour = int(s[:-2])
    else:
        raise ValueError(f"Unrecognized time format: {time_str!r}")

    if is_pm and hour < 12:
        hour += 12
    if is_am and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Time out of range: {time_str!r}")

    target = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now_utc:
        target += timedelta(days=1)
    return target


def fmt_time_12h(dt: datetime, with_date: bool = False) -> str:
    """Formats a datetime as 12-hour time (e.g. '5:30 PM'), optionally with date."""
    time_part = dt.strftime("%I:%M %p").lstrip("0")
    if with_date:
        return f"{dt.strftime('%Y-%m-%d')} {time_part}"
    return time_part


def main():
    print("=" * 60)
    print("BTC PRICE TARGET PROBABILITY -- CHART-ANALYSIS-BASED")
    print("=" * 60)
    print("(Combines your trained model's real technical analysis with")
    print(" recent volatility -- see the caveats at the top of this file.)\n")

    target_price_str = input("Target BTC price (e.g. 64000): ").strip().replace(",", "").replace("$", "")
    target_time_str = input("Target time, UTC, 12-hour (e.g. 10:00am, 2:30pm): ").strip()

    try:
        target_price = float(target_price_str)
    except ValueError:
        print("Couldn't parse that price. Please enter a plain number, e.g. 64000")
        sys.exit(1)

    model, calibrator, feature_columns = load_model()

    print("\nFetching live BTC data and running technical analysis...")
    df = fetch_recent_data(limit=1500)
    featured = add_baseline_features(df)
    complete = featured.dropna(subset=feature_columns)

    if len(complete) == 0:
        print("Not enough recent data to compute indicators. Try again shortly.")
        sys.exit(1)

    latest = complete.iloc[-1]
    current_price = latest["close"]
    now_utc = latest["timestamp"].to_pydatetime() if hasattr(latest["timestamp"], "to_pydatetime") \
        else latest["timestamp"]

    try:
        target_time = parse_target_time(target_time_str, now_utc)
    except Exception:
        print("Couldn't parse that time. Try formats like: 10:00, 10am, 2:30pm, 14:30, 1000")
        sys.exit(1)

    minutes_ahead = (target_time - now_utc).total_seconds() / 60.0

    # -- the actual chart analysis: run the current setup through your trained model --
    X_latest = latest[feature_columns].values.reshape(1, -1).astype(float)
    raw_prob = model.predict_proba(X_latest)[0, 1]
    p_up_15min = float(calibrator.transform([raw_prob])[0])
    # clip away from exact 0/1 -- avoids -inf/+inf when converting to a z-score next
    p_up_15min = min(max(p_up_15min, 0.01), 0.99)

    # -- recent volatility, same as before --
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    sigma_per_minute = log_returns.std()
    if sigma_per_minute == 0 or np.isnan(sigma_per_minute):
        print("Could not estimate volatility from recent data -- try again.")
        sys.exit(1)

    # -- convert the model's 15-minute directional probability into an implied
    #    drift, using the fact that under a normal model,
    #    P(up) = CDF(mu / sigma), so mu = sigma * inverse_CDF(P(up)) --
    sigma_15min = sigma_per_minute * math.sqrt(15)
    implied_mu_15min = sigma_15min * norm.ppf(p_up_15min)
    implied_mu_per_minute = implied_mu_15min / 15.0

    # extrapolate that drift rate across the full horizon to the target time.
    # this is the part that's speculative beyond ~15 minutes -- flagged below.
    mu_total = implied_mu_per_minute * minutes_ahead
    sigma_horizon = sigma_per_minute * math.sqrt(minutes_ahead)

    log_target_ratio = math.log(target_price / current_price)
    z = (log_target_ratio - mu_total) / sigma_horizon
    prob_at_or_above = 1 - norm.cdf(z)
    prob_below = 1 - prob_at_or_above

    # -- show the actual indicator readings so this doesn't feel like a black box --
    rsi = latest.get("rsi_14", float("nan"))
    ema_dist = latest.get("ema_dist", float("nan"))
    ret_5 = latest.get("ret_5", float("nan"))
    ret_15 = latest.get("ret_15", float("nan"))
    vol_z = latest.get("vol_zscore_20", float("nan"))

    print("\n" + "=" * 60)
    print("TECHNICAL READ (what the model is actually seeing right now)")
    print("=" * 60)
    print(f"RSI (14):                {rsi:.1f}  "
          f"({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})")
    print(f"EMA(9) vs EMA(21):       {'above' if ema_dist > 0 else 'below'} "
          f"({ema_dist*100:+.3f}% distance) -- {'bullish' if ema_dist > 0 else 'bearish'} short-term trend")
    print(f"5-minute momentum:       {ret_5*100:+.3f}%")
    print(f"15-minute momentum:      {ret_15*100:+.3f}%")
    print(f"Volume vs recent avg:    {vol_z:+.2f} std devs")
    print(f"Model's P(up, next 15m): {p_up_15min*100:.1f}%  <- this is your trained, "
          f"validated model's actual output")

    verdict = "YES" if prob_at_or_above >= 0.5 else "NO"
    confidence = max(prob_at_or_above, prob_below) * 100

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if verdict == "YES":
        print(f">>> YES -- BTC is more likely to be AT OR ABOVE ${target_price:,.2f} "
              f"by {fmt_time_12h(target_time)} UTC")
    else:
        print(f">>> NO -- BTC is more likely to be BELOW ${target_price:,.2f} "
              f"by {fmt_time_12h(target_time)} UTC")
    print(f"    (confidence: {confidence:.0f}%)")

    print("\n" + "=" * 60)
    print("RESULT (details)")
    print("=" * 60)
    print(f"Current time (UTC):    {fmt_time_12h(now_utc, with_date=True)}")
    print(f"Target time (UTC):     {fmt_time_12h(target_time, with_date=True)}")
    print(f"Time remaining:        {minutes_ahead:.0f} minutes")
    print(f"Current BTC price:     ${current_price:,.2f}")
    print(f"Target BTC price:      ${target_price:,.2f}")
    print(f"Required move:         {(target_price/current_price - 1)*100:+.2f}%")
    print(f"Recent volatility:     {sigma_per_minute*100:.4f}% per minute")
    print(f"Model-implied drift:   {implied_mu_per_minute*100:+.5f}% per minute "
          f"(extrapolated from the 15-min signal above)")
    print("-" * 60)
    print(f"P(BTC >= ${target_price:,.2f} by {fmt_time_12h(target_time)} UTC): "
          f"{prob_at_or_above*100:.1f}%")
    print(f"P(BTC <  ${target_price:,.2f} by {fmt_time_12h(target_time)} UTC): "
          f"{prob_below*100:.1f}%")
    print("=" * 60)

    if minutes_ahead > 30:
        print(f"\nNOTE: your target is {minutes_ahead:.0f} minutes away, but the model's "
              f"directional signal was only trained/validated on a 15-minute horizon. "
              f"Extrapolating it this far out is a real assumption, not a tested fact -- "
              f"treat this result with proportionally more skepticism the further out you go.")
    print("\nRemember: this uses your validated model's real read of the current chart, "
          "but per Phase 2/3 testing that edge is modest and was not profitable after fees. "
          "Treat this as an informed estimate, not a guarantee.")


if __name__ == "__main__":
    main()