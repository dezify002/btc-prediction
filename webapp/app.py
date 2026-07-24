"""
Local web frontend for BTC prediction tools with auto-refresh.
"""

import os
import sys
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string, request

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
import prediction_core as core

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
    --warning: #E8B860;
    --danger: #FF5C7A;
    --info: #5B8DEF;
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
  .timestamp { font-family: var(--mono); font-size: 12px; }
  .timestamp.stale { color: var(--danger); }
  .timestamp.fresh { color: var(--up); }
  .timestamp.warn { color: var(--warning); }

  .live-indicator {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 11px; color: var(--up);
    margin-bottom: 10px;
  }
  .live-indicator .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--up); animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

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

  .model-badge {
    display: inline-block; background: var(--surface-2); border: 1px solid var(--border);
    color: var(--info); font-family: var(--mono); font-size: 11px;
    padding: 3px 8px; border-radius: 4px; margin-bottom: 12px;
  }

  .warning {
    background: #2A2110; border: 1px solid #4A3A18; color: var(--warning);
    border-radius: 6px; padding: 10px 12px; font-size: 13px; margin-top: 14px;
  }
  .error {
    background: #2A1414; border: 1px solid #4A1F1F; color: #FF9B9B;
    border-radius: 6px; padding: 10px 12px; font-size: 13px; margin-top: 14px;
  }
  .danger {
    background: #2A1414; border: 1px solid #4A1F1F; color: var(--danger);
    border-radius: 6px; padding: 10px 12px; font-size: 13px; margin-top: 14px;
  }
  .muted-note { color: var(--muted); font-size: 12px; margin-top: 16px; line-height: 1.5; }
  .loading { color: var(--muted); font-family: var(--mono); font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">BTC / USDT</div>
    <h1>Prediction Terminal</h1>
    <p>Auto-refreshing every 10s &mdash; powered by Bitget API.</p>
  </header>

  <div class="panel">
    <h2>Right now &middot; next 15 minutes</h2>
    <div class="live-indicator"><div class="dot"></div> LIVE &mdash; auto-refreshing</div>
    <div id="now-content"><div class="loading">Loading&hellip;</div></div>
    <button class="refresh-btn" onclick="loadNow()">Refresh Now</button>
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

function parseDate(isoString) {
  if (!isoString) return null;
  try {
    let d = new Date(isoString);
    if (isNaN(d.getTime())) d = new Date(isoString + 'Z');
    return isNaN(d.getTime()) ? null : d;
  } catch (e) { return null; }
}

function formatTimestamp(isoString) {
  const d = parseDate(isoString);
  if (!d) return 'Invalid Date';
  return d.toUTCString();
}

function getAgeSeconds(isoString) {
  const d = parseDate(isoString);
  if (!d) return 9999;
  return (Date.now() - d.getTime()) / 1000;
}

function getAgeClass(ageSec) {
  if (ageSec < 60) return 'fresh';
  if (ageSec < 300) return 'warn';
  return 'stale';
}

function getAgeLabel(ageSec) {
  if (ageSec < 60) return '';
  if (ageSec < 300) return ' (' + Math.round(ageSec) + 's old)';
  return ' (STALE: ' + Math.round(ageSec) + 's old)';
}

let autoRefreshInterval = null;

async function loadNow() {
  const el = document.getElementById('now-content');
  // Only show loading on manual refresh, not auto-refresh
  if (!autoRefreshInterval) el.innerHTML = '<div class="loading">Loading&hellip;</div>';

  try {
    const res = await fetch('/api/now');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    const upPct = Math.round(data.p_up * 100);
    const downPct = 100 - upPct;
    const tsStr = formatTimestamp(data.timestamp);
    const ageSec = getAgeSeconds(data.timestamp);
    const ageClass = getAgeClass(ageSec);
    const ageLabel = getAgeLabel(ageSec);

    let warningHtml = '';
    if (data.regime_warning) {
      const isExtreme = data.regime_warning.includes('EXTREME');
      warningHtml = `<div class="${isExtreme ? 'danger' : 'warning'}">${data.regime_warning}</div>`;
    }

    let obHtml = '';
    if (data.order_book) {
      const lean = data.order_book.imbalance > 0 ? 'buy-side' : 'sell-side';
      obHtml = `
        <div class="indicator-grid" style="margin-top:14px; border-top:1px solid var(--border); padding-top:14px;">
          <div class="label">Order book imbalance</div><div class="value">${data.order_book.imbalance.toFixed(3)} (${lean})</div>
        </div>
        <div class="muted-note" style="margin-top:8px;"><strong>Experimental:</strong> Order book signal is live-only context, not backtested against history. Do not trade on it.</div>
      `;
    }

    let staleWarning = '';
    if (ageSec > 300) {
      staleWarning = `<div class="danger" style="margin-top:8px;">Data is ${Math.round(ageSec)} seconds old. Price may be stale.</div>`;
    }

    el.innerHTML = `
      <div class="model-badge">Model: ${data.model_used || '15m'} | Source: ${data.exchange_used || 'unknown'}</div>
      <div class="price-row">
        <div class="price">$${data.price.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</div>
        <div class="timestamp ${ageClass}">${tsStr}${ageLabel}</div>
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
      ${warningHtml}
      ${obHtml}
      ${staleWarning}
    `;
  } catch (err) {
    el.innerHTML = `<div class="error">Couldn't load a prediction: ${err.message}. Check that models are trained and Bitget API is reachable.</div>`;
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
      warningHtml += `<div class="warning">Your target is ${Math.round(data.minutes_ahead)} minutes away, but the model's directional signal was only trained/validated on a ${data.trained_horizon_min}-minute horizon. Treat this with proportionally more skepticism the further out you go.</div>`;
    }
    if (data.regime_warning) {
      const isExtreme = data.regime_warning.includes('EXTREME');
      warningHtml += `<div class="${isExtreme ? 'danger' : 'warning'}" style="margin-top:8px;">${data.regime_warning}</div>`;
    }

    resultEl.innerHTML = `
      <div class="verdict-block">
        <div class="model-badge">Model: ${data.model_used} (trained for ${data.trained_horizon_min}m)</div>
        <div class="verdict-headline ${isYes ? 'yes' : 'no'}">${data.verdict}</div>
        <div class="verdict-sub">BTC is more likely to be ${isYes ? 'AT OR ABOVE' : 'BELOW'} $${Number(data.target_price).toLocaleString()} &middot; confidence ${fmtPct(data.confidence)}</div>
        <div class="indicator-grid">
          <div class="label">Current price</div><div class="value">$${data.current_price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
          <div class="label">Required move</div><div class="value">${fmtSigned(data.required_move_pct, 2)}</div>
          <div class="label">Time remaining</div><div class="value">${Math.round(data.minutes_ahead)} min</div>
          <div class="label">Model P(up, ${data.trained_horizon_min}m)</div><div class="value">${fmtPct(data.p_up_trained_horizon)}</div>
          <div class="label">RSI (14)</div><div class="value">${data.rsi.toFixed(1)}</div>
          <div class="label">Recent volatility</div><div class="value">${data.sigma_per_minute_pct.toFixed(4)}%/min</div>
          <div class="label">Time-decay factor</div><div class="value">${data.time_decay_factor.toFixed(2)}</div>
          <div class="label">Vol regime mult</div><div class="value">${data.vol_regime_multiplier.toFixed(1)}x</div>
        </div>
        ${warningHtml}
      </div>
    `;
  } catch (err) {
    resultEl.innerHTML = `<div class="error">${err.message}</div>`;
  }
}

// AUTO-REFRESH: poll every 10 seconds
function startAutoRefresh() {
  loadNow();
  autoRefreshInterval = setInterval(loadNow, 10000); // 10 seconds
}

startAutoRefresh();
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
        result = core.get_current_prediction()
        result["server_time"] = datetime.now(timezone.utc).isoformat()
        return jsonify(result)
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
        result["server_time"] = datetime.now(timezone.utc).isoformat()
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