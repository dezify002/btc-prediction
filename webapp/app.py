"""
Local web frontend for your BTC prediction tools.

Run this, then open http://localhost:5000 in your browser. It wraps the
same trained model and analysis used by predict_now.py and
price_target_probability.py -- no new logic, just a browser UI on top
of prediction_core.py.

Requires: pip install flask --break-system-packages

Usage:
    python app.py
"""

import os
import sys

from flask import Flask, jsonify, render_template_string, request

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
import prediction_core as core  # noqa: E402

app = Flask(__name__)


INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0B0E14;
    --surface: #131824;
    --surface-2: #171D2B;
    --border: #232B3B;
    --text: #E8ECF2;
    --muted: #7C8797;
    --up: #3ECF8E;
    --down: #FF5C7A;
    --accent: #F0A94E;
    --mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace;
    --sans: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: var(--sans);
    padding: 24px 16px 64px;
  }
  .wrap { max-width: 720px; margin: 0 auto; }
  header { margin-bottom: 28px; }
  header .eyebrow {
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.12em;
    color: var(--accent); text-transform: uppercase; margin-bottom: 6px;
  }
  header h1 { font-size: 22px; font-weight: 700; margin: 0; }
  header p { color: var(--muted); font-size: 14px; margin: 6px 0 0; }

  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 20px;
  }
  .panel h2 {
    font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 16px;
  }

  .price-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px; }
  .price { font-family: var(--mono); font-size: 28px; font-weight: 700; }
  .timestamp { font-family: var(--mono); font-size: 12px; color: var(--muted); }

  .split-bar {
    height: 40px; border-radius: 8px; overflow: hidden; display: flex;
    border: 1px solid var(--border); margin-bottom: 10px;
  }
  .split-bar .up { background: var(--up); display: flex; align-items: center; justify-content: center; }
  .split-bar .down { background: var(--down); display: flex; align-items: center; justify-content: center; }
  .split-bar span { font-family: var(--mono); font-weight: 700; font-size: 14px; color: #0B0E14; }

  .indicator-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;
    font-family: var(--mono); font-size: 13px; margin-top: 16px;
  }
  .indicator-grid .label { color: var(--muted); }
  .indicator-grid .value { text-align: right; }

  .refresh-btn, .submit-btn {
    background: var(--accent); color: #0B0E14; border: none; border-radius: 6px;
    font-family: var(--sans); font-weight: 600; font-size: 14px;
    padding: 10px 18px; cursor: pointer; margin-top: 4px;
  }
  .refresh-btn:hover, .submit-btn:hover { opacity: 0.88; }
  .refresh-btn:focus-visible, .submit-btn:focus-visible, input:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }

  form { display: flex; flex-direction: column; gap: 12px; }
  .field-row { display: flex; gap: 12px; }
  .field { flex: 1; }
  .field label {
    display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;
    font-family: var(--mono);
  }
  .field input {
    width: 100%; background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text); font-family: var(--mono); font-size: 15px;
    padding: 10px 12px; border-radius: 6px;
  }

  .verdict-block { margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--border); }
  .verdict-headline {
    font-family: var(--mono); font-size: 32px; font-weight: 700; margin-bottom: 4px;
  }
  .verdict-headline.yes { color: var(--up); }
  .verdict-headline.no { color: var(--down); }
  .verdict-sub { color: var(--muted); font-size: 14px; margin-bottom: 16px; }

  .warning {
    background: #2A2110; border: 1px solid #4A3A18; color: #E8B860;
    border-radius: 6px; padding: 10px 12px; font-size: 13px; margin-top: 14px;
  }
  .error {
    background: #2A1414; border: 1px solid #4A1F1F; color: #FF9B9B;
    border-radius: 6px; padding: 10px 12px; font-size: 13px; margin-top: 14px;
  }
  .muted-note { color: var(--muted); font-size: 12px; margin-top: 16px; line-height: 1.5; }
  .loading { color: var(--muted); font-family: var(--mono); font-size: 13px; }

  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">BTC / USDT</div>
    <h1>Prediction Terminal</h1>
    <p>Backed by your trained model &mdash; not a trading signal, an informed estimate.</p>
  </header>

  <div class="panel">
    <h2>Right now &middot; next 15 minutes</h2>
    <div id="now-content"><div class="loading">Loading&hellip;</div></div>
    <button class="refresh-btn" onclick="loadNow()">Refresh</button>
  </div>

  <div class="panel">
    <h2>Ask about a target</h2>
    <form id="target-form" onsubmit="submitTarget(event)">
      <div class="field-row">
        <div class="field">
          <label for="price">Target price ($)</label>
          <input type="text" id="price" placeholder="64000" required>
        </div>
        <div class="field">
          <label for="time">Target time (UTC, 12-hour)</label>
          <input type="text" id="time" placeholder="4:30pm" required>
        </div>
      </div>
      <button type="submit" class="submit-btn">Analyze</button>
    </form>
    <div id="target-result"></div>
  </div>

  <p class="muted-note">
    Per Phase 2/3 testing on 2023&ndash;2024 data, this model's edge is modest overall
    and was not profitable after realistic trading fees. Treat every result on this
    page as informational context, not a guarantee.
  </p>
</div>

<script>
function fmtPct(x) { return (x * 100).toFixed(1) + '%'; }
function fmtSigned(x, digits) { digits = digits || 3; return (x >= 0 ? '+' : '') + x.toFixed(digits) + '%'; }

async function loadNow() {
  const el = document.getElementById('now-content');
  el.innerHTML = '<div class="loading">Loading&hellip;</div>';
  try {
    const res = await fetch('/api/now');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    const upPct = Math.round(data.p_up * 100);
    const downPct = 100 - upPct;

    let obHtml = '';
    if (data.order_book) {
      const lean = data.order_book.imbalance > 0 ? 'buy-side' : 'sell-side';
      obHtml = `
        <div class="indicator-grid" style="margin-top:14px; border-top:1px solid var(--border); padding-top:14px;">
          <div class="label">Order book imbalance</div><div class="value">${data.order_book.imbalance.toFixed(3)} (${lean})</div>
        </div>
        <div class="muted-note" style="margin-top:8px;">Order book signal is live-only context, not backtested against history.</div>
      `;
    }

    el.innerHTML = `
      <div class="price-row">
        <div class="price">$${data.price.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</div>
        <div class="timestamp">${new Date(data.timestamp + 'Z').toUTCString()}</div>
      </div>
      <div class="split-bar">
        <div class="up" style="width:${upPct}%"><span>${upPct >= 15 ? upPct + '% UP' : ''}</span></div>
        <div class="down" style="width:${downPct}%"><span>${downPct >= 15 ? downPct + '% DOWN' : ''}</span></div>
      </div>
      <div class="indicator-grid">
        <div class="label">RSI (14)</div><div class="value">${data.rsi.toFixed(1)}</div>
        <div class="label">EMA(9) vs EMA(21)</div><div class="value">${fmtSigned(data.ema_dist*100)}</div>
        <div class="label">5-min momentum</div><div class="value">${fmtSigned(data.ret_5*100)}</div>
        <div class="label">15-min momentum</div><div class="value">${fmtSigned(data.ret_15*100)}</div>
        <div class="label">Volume vs avg</div><div class="value">${data.vol_z.toFixed(2)} std</div>
      </div>
      ${obHtml}
    `;
  } catch (err) {
    el.innerHTML = `<div class="error">Couldn't load a prediction: ${err.message}. Check that train_final_model.py has been run, and that this machine can reach Binance.</div>`;
  }
}

async function submitTarget(event) {
  event.preventDefault();
  const price = document.getElementById('price').value;
  const time = document.getElementById('time').value;
  const resultEl = document.getElementById('target-result');
  resultEl.innerHTML = '<div class="loading" style="margin-top:16px;">Analyzing&hellip;</div>';

  try {
    const res = await fetch('/api/target', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({price, time})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    const isYes = data.verdict === 'YES';
    let warningHtml = '';
    if (data.extrapolation_warning) {
      warningHtml = `<div class="warning">Your target is ${Math.round(data.minutes_ahead)} minutes away, but the model's directional signal was only trained/validated on a 15-minute horizon. Treat this with proportionally more skepticism the further out you go.</div>`;
    }

    resultEl.innerHTML = `
      <div class="verdict-block">
        <div class="verdict-headline ${isYes ? 'yes' : 'no'}">${data.verdict}</div>
        <div class="verdict-sub">BTC is more likely to be ${isYes ? 'AT OR ABOVE' : 'BELOW'} $${Number(data.target_price).toLocaleString()} &middot; confidence ${fmtPct(data.confidence)}</div>
        <div class="indicator-grid">
          <div class="label">Current price</div><div class="value">$${data.current_price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
          <div class="label">Required move</div><div class="value">${fmtSigned(data.required_move_pct, 2)}</div>
          <div class="label">Time remaining</div><div class="value">${Math.round(data.minutes_ahead)} min</div>
          <div class="label">Model P(up, 15m)</div><div class="value">${fmtPct(data.p_up_15min)}</div>
          <div class="label">RSI (14)</div><div class="value">${data.rsi.toFixed(1)}</div>
          <div class="label">Recent volatility</div><div class="value">${data.sigma_per_minute_pct.toFixed(4)}%/min</div>
        </div>
        ${warningHtml}
      </div>
    `;
  } catch (err) {
    resultEl.innerHTML = `<div class="error">${err.message}</div>`;
  }
}

loadNow();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/now")
def api_now():
    try:
        return jsonify(core.get_current_prediction())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/target", methods=["POST"])
def api_target():
    body = request.get_json(force=True)
    try:
        price = float(str(body.get("price", "")).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return jsonify({"error": "Couldn't parse that price. Enter a plain number, e.g. 64000"}), 400

    time_str = str(body.get("time", "")).strip()
    if not time_str:
        return jsonify({"error": "Enter a target time, e.g. 4:30pm"}), 400

    try:
        result = core.analyze_price_target(price, time_str)
        return jsonify(result)
    except ValueError:
        return jsonify({"error": "Couldn't parse that time. Try formats like 10am, 2:30pm, 14:30"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Starting BTC prediction terminal...")
    print(f"Open http://localhost:{port} in your browser.")
    app.run(debug=False, host="0.0.0.0", port=port)