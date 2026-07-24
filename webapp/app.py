"""
BTC Prediction Bot — Web UI with Ensemble Display
Shows: live price, bot prediction, XGBoost prediction, AI Reviewer verdict
"""

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from flask import Flask, jsonify, request, render_template_string
from prediction_core import get_full_prediction, analyze_price_target

app = Flask(__name__)

# ── HTML Template ──────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Prediction Bot — Ensemble</title>
<style>
:root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --up: #3fb950; --down: #f85149; --warn: #d29922;
    --strong: #a371f7;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 20px;
}
.container { max-width: 900px; margin: 0 auto; }
h1 { text-align:center; margin-bottom: 8px; font-size: 1.4rem; }
.subtitle { text-align:center; color: var(--muted); font-size: 0.8rem; margin-bottom: 24px; }

.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 16px;
}
.card h2 { font-size: 1rem; margin-bottom: 12px; color: var(--accent); }

/* Price display */
.price-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
.price-main { font-size: 2.2rem; font-weight: bold; }
.price-change { font-size: 0.85rem; }
.meta { color: var(--muted); font-size: 0.75rem; }

/* LIVE indicator */
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: var(--up); border-radius: 50%;
    margin-right: 6px; animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.3); }
    100% { opacity: 1; transform: scale(1); }
}

/* Ensemble grid */
.ensemble-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.model-box {
    background: rgba(0,0,0,0.2); border-radius: 8px; padding: 14px;
    border: 1px solid var(--border);
}
.model-box h3 { font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; }
.model-pred { font-size: 1.3rem; font-weight: bold; }
.model-conf { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }

/* Verdict */
.verdict-box {
    text-align: center; padding: 16px; border-radius: 10px;
    margin: 16px 0; border: 2px solid;
}
.verdict-strong { border-color: var(--up); background: rgba(63,185,80,0.1); }
.verdict-moderate { border-color: var(--accent); background: rgba(88,166,255,0.1); }
.verdict-weak { border-color: var(--warn); background: rgba(210,153,34,0.1); }
.verdict-uncertain { border-color: var(--muted); background: rgba(139,148,158,0.1); }
.verdict-skip { border-color: var(--down); background: rgba(248,81,73,0.1); }

.verdict-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 4px; }
.verdict-sub { font-size: 0.85rem; color: var(--muted); }

/* Trust score bar */
.trust-bar-container {
    width: 100%; height: 20px; background: rgba(0,0,0,0.3);
    border-radius: 10px; overflow: hidden; margin: 12px 0;
}
.trust-bar {
    height: 100%; border-radius: 10px;
    transition: width 0.5s ease;
}
.trust-high { background: linear-gradient(90deg, #238636, #3fb950); }
.trust-medium { background: linear-gradient(90deg, #9e6a03, #d29922); }
.trust-low { background: linear-gradient(90deg, #da3633, #f85149); }

/* Reasons */
.reasons { list-style: none; }
.reasons li {
    padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.8rem; display: flex; align-items: center;
}
.reasons li::before {
    content: "→"; margin-right: 8px; color: var(--accent);
}
.reasons li.warning::before { content: "⚠"; color: var(--warn); }
.reasons li.danger::before { content: "✕"; color: var(--down); }
.reasons li.good::before { content: "✓"; color: var(--up); }

/* Risk badge */
.risk-badge {
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    font-size: 0.75rem; font-weight: bold; text-transform: uppercase;
}
.risk-low { background: rgba(63,185,80,0.2); color: var(--up); }
.risk-medium { background: rgba(88,166,255,0.2); color: var(--accent); }
.risk-elevated { background: rgba(210,153,34,0.2); color: var(--warn); }
.risk-high { background: rgba(248,81,73,0.2); color: var(--down); }

/* Features table */
.features-table { width: 100%; font-size: 0.75rem; border-collapse: collapse; }
.features-table td { padding: 4px 8px; border-bottom: 1px solid var(--border); }
.features-table td:first-child { color: var(--muted); width: 50%; }

/* Target analysis */
.target-form { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.target-form input {
    background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 8px 12px; border-radius: 6px;
    font-family: inherit; font-size: 0.85rem; flex: 1;
}
.target-form button {
    background: var(--accent); color: #fff; border: none;
    padding: 8px 16px; border-radius: 6px; cursor: pointer;
    font-family: inherit; font-weight: bold;
}
.target-form button:hover { opacity: 0.9; }

/* Refresh */
.refresh-row { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
.refresh-btn {
    background: transparent; border: 1px solid var(--border);
    color: var(--text); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-family: inherit; font-size: 0.8rem;
}
.refresh-btn:hover { border-color: var(--accent); }

/* Auto-refresh indicator */
.auto-refresh-status {
    font-size: 0.7rem; color: var(--muted);
    display: flex; align-items: center; gap: 6px;
}

/* Debug log */
#debugLog {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px; font-size: 0.7rem;
    color: var(--muted); max-height: 120px; overflow-y: auto;
    margin-top: 8px; display: none;
}

.up { color: var(--up); }
.down { color: var(--down); }
.warn { color: var(--warn); }
</style>
</head>
<body>
<div class="container">
    <h1>🪙 BTC Prediction Bot</h1>
    <p class="subtitle">Ensemble: Your Bot + XGBoost + AI Reviewer</p>

    <!-- Live Price -->
    <div class="card">
        <div class="price-row">
            <div>
                <div class="price-main" id="priceDisplay">—</div>
                <div class="meta" id="priceMeta">Loading...</div>
            </div>
            <div style="text-align:right">
                <div class="auto-refresh-status">
                    <span class="live-dot"></span>
                    <span id="refreshStatus">LIVE</span>
                </div>
                <div class="meta" id="exchangeBadge">—</div>
            </div>
        </div>
        <div class="refresh-row">
            <span class="meta" id="lastUpdate">—</span>
            <button class="refresh-btn" onclick="loadNow()">Refresh Now</button>
        </div>
        <div id="debugLog"></div>
    </div>

    <!-- Ensemble Predictions -->
    <div class="card">
        <h2>🔮 Ensemble Predictions</h2>
        <div class="ensemble-grid">
            <div class="model-box">
                <h3>YOUR BOT</h3>
                <div class="model-pred" id="botPred">—</div>
                <div class="model-conf" id="botConf">—</div>
            </div>
            <div class="model-box">
                <h3>XGBOOST</h3>
                <div class="model-pred" id="xgbPred">—</div>
                <div class="model-conf" id="xgbConf">—</div>
            </div>
        </div>

        <!-- Verdict -->
        <div class="verdict-box" id="verdictBox">
            <div class="verdict-title" id="verdictTitle">—</div>
            <div class="verdict-sub" id="verdictSub">—</div>
        </div>

        <!-- Trust Score -->
        <div style="margin: 12px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="meta">Trust Score</span>
                <span class="meta" id="trustScore">—</span>
            </div>
            <div class="trust-bar-container">
                <div class="trust-bar" id="trustBar" style="width:0%"></div>
            </div>
        </div>

        <!-- Risk & Recommendation -->
        <div style="display:flex; gap:12px; align-items:center; margin: 12px 0;">
            <span class="meta">Risk:</span>
            <span class="risk-badge" id="riskBadge">—</span>
            <span class="meta">|</span>
            <span class="meta" id="recommendation">—</span>
        </div>

        <!-- Reasons -->
        <h3 style="font-size:0.8rem; color:var(--muted); margin:16px 0 8px;">AI Reviewer Reasons</h3>
        <ul class="reasons" id="reasonsList">
            <li>Loading...</li>
        </ul>
    </div>

    <!-- Market Regime -->
    <div class="card">
        <h2>📊 Market Regime</h2>
        <div class="meta" id="marketRegime">—</div>
        <table class="features-table" id="featuresTable">
            <tr><td>Loading features...</td><td>—</td></tr>
        </table>
    </div>

    <!-- Target Analysis -->
    <div class="card">
        <h2>🎯 Target Analysis</h2>
        <div class="target-form">
            <input type="number" id="targetPrice" placeholder="Target price (e.g. 67000)" step="1">
            <input type="text" id="targetTime" placeholder="Time (e.g. 14:30 or 2026-07-25 14:30)">
            <button onclick="analyzeTarget()">Analyze</button>
        </div>
        <div id="targetResult"></div>
    </div>
</div>

<script>
let autoRefreshInterval = null;
let lastData = null;

function log(msg) {
    const el = document.getElementById("debugLog");
    el.style.display = "block";
    const time = new Date().toLocaleTimeString();
    el.innerHTML += `[${time}] ${msg}<br>`;
    el.scrollTop = el.scrollHeight;
}

function getVerdictClass(verdict) {
    if (verdict.includes("Strong")) return "verdict-strong";
    if (verdict.includes("Moderate")) return "verdict-moderate";
    if (verdict.includes("Weak")) return "verdict-weak";
    if (verdict.includes("UNCERTAIN")) return "verdict-uncertain";
    if (verdict.includes("SKIP")) return "verdict-skip";
    return "verdict-uncertain";
}

function getTrustClass(score) {
    if (score >= 0.7) return "trust-high";
    if (score >= 0.4) return "trust-medium";
    return "trust-low";
}

function getRiskClass(risk) {
    const map = { "LOW": "risk-low", "MEDIUM": "risk-medium", "ELEVATED": "risk-elevated", "HIGH": "risk-high" };
    return map[risk] || "risk-high";
}

function formatPrice(p) {
    return "$" + parseFloat(p).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function loadNow() {
    log("Fetching /api/now...");
    try {
        const res = await fetch("/api/now");
        log(`Response status: ${res.status}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        log("Data received successfully");
        render(data);
        lastData = data;
    } catch (e) {
        log(`ERROR: ${e.message}`);
        console.error(e);
    }
}

function render(data) {
    // Price
    document.getElementById("priceDisplay").textContent = formatPrice(data.price);
    document.getElementById("priceMeta").innerHTML = 
        `Spread: <span class="${data.spread_pct > 0.05 ? 'warn' : ''}">${data.spread_pct.toFixed(4)}%</span>`;
    document.getElementById("exchangeBadge").textContent = `Source: ${data.exchange}`;

    const ts = data.timestamp ? new Date(data.timestamp + 'Z') : new Date();
    document.getElementById("lastUpdate").textContent = ts.toUTCString();

    // Bot
    const bot = data.bot || {};
    document.getElementById("botPred").textContent = bot.pred || "—";
    document.getElementById("botPred").className = "model-pred " + (bot.pred === "UP" ? "up" : bot.pred === "DOWN" ? "down" : "");
    document.getElementById("botConf").textContent = bot.confidence ? `Confidence: ${(bot.confidence * 100).toFixed(1)}%` : "—";

    // XGBoost
    const xgb = data.xgboost || {};
    document.getElementById("xgbPred").textContent = xgb.pred || "—";
    document.getElementById("xgbPred").className = "model-pred " + (xgb.pred === "UP" ? "up" : xgb.pred === "DOWN" ? "down" : "");
    document.getElementById("xgbConf").textContent = xgb.prob ? `P(UP): ${(xgb.prob * 100).toFixed(1)}%` : "—";

    // Decision
    const dec = data.decision || {};
    const vBox = document.getElementById("verdictBox");
    vBox.className = "verdict-box " + getVerdictClass(dec.ensemble_verdict || "");
    document.getElementById("verdictTitle").textContent = dec.ensemble_verdict || "—";
    document.getElementById("verdictSub").textContent = dec.recommendation || "—";

    // Trust score
    const ts_score = dec.trust_score || 0;
    document.getElementById("trustScore").textContent = (ts_score * 100).toFixed(1) + "%";
    const bar = document.getElementById("trustBar");
    bar.style.width = (ts_score * 100) + "%";
    bar.className = "trust-bar " + getTrustClass(ts_score);

    // Risk
    document.getElementById("riskBadge").textContent = dec.risk_level || "—";
    document.getElementById("riskBadge").className = "risk-badge " + getRiskClass(dec.risk_level);
    document.getElementById("recommendation").textContent = dec.recommendation || "—";

    // Reasons
    const reasons = dec.reasons || ["No reasons available"];
    document.getElementById("reasonsList").innerHTML = reasons.map(r => {
        let cls = "";
        if (r.includes("DISAGREEMENT") || r.includes("dangerous") || r.includes("risky") || r.includes("unreliable")) cls = "danger";
        else if (r.includes("High") || r.includes("spike") || r.includes("avoid")) cls = "warning";
        else if (r.includes("agree") || r.includes("favorable") || r.includes("Low volatility")) cls = "good";
        return `<li class="${cls}">${r}</li>`;
    }).join("");

    // Regime
    document.getElementById("marketRegime").textContent = dec.market_regime || "—";

    // Features table
    const feats = data.features || {};
    const featRows = Object.entries(feats).slice(0, 15).map(([k, v]) => {
        const val = typeof v === "number" ? v.toFixed(4) : v;
        return `<tr><td>${k}</td><td>${val}</td></tr>`;
    }).join("");
    document.getElementById("featuresTable").innerHTML = featRows || "<tr><td>No features</td></tr>";
}

async function analyzeTarget() {
    const price = document.getElementById("targetPrice").value;
    const time = document.getElementById("targetTime").value;
    if (!price || !time) {
        document.getElementById("targetResult").innerHTML = '<p class="warn">Enter both price and time</p>';
        return;
    }
    document.getElementById("targetResult").innerHTML = '<p class="meta">Analyzing...</p>';
    try {
        const res = await fetch(`/api/target?price=${price}&time=${encodeURIComponent(time)}`);
        const data = await res.json();
        if (data.error) {
            document.getElementById("targetResult").innerHTML = `<p class="down">Error: ${data.error}</p>`;
            return;
        }
        let html = `
            <div style="margin-top:12px; padding:12px; background:rgba(0,0,0,0.2); border-radius:8px;">
                <div style="font-size:1.3rem; font-weight:bold; margin-bottom:8px;">
                    ${data.verdict} — ${(data.verdict_confidence * 100).toFixed(1)}%
                </div>
                <div class="meta">Current: ${formatPrice(data.current_price)} → Target: ${formatPrice(data.target_price)}</div>
                <div class="meta">Horizon: ${data.minutes_ahead} min | P(above): ${(data.probability_above * 100).toFixed(1)}%</div>
                <div class="meta">Time decay: ${(data.time_decay_factor * 100).toFixed(0)}% | Vol mult: ${data.vol_regime_multiplier}x</div>
        `;
        if (data.warnings && data.warnings.length > 0) {
            html += `<div style="margin-top:8px; color:var(--warn); font-size:0.8rem;">`;
            data.warnings.forEach(w => html += `⚠ ${w}<br>`);
            html += `</div>`;
        }
        html += `</div>`;
        document.getElementById("targetResult").innerHTML = html;
    } catch (e) {
        document.getElementById("targetResult").innerHTML = `<p class="down">Error: ${e.message}</p>`;
    }
}

// Auto-refresh every 15 seconds
function startAutoRefresh() {
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    autoRefreshInterval = setInterval(loadNow, 15000);
    document.getElementById("refreshStatus").textContent = "AUTO-REFRESH ON";
}

// Init
loadNow();
startAutoRefresh();
</script>
</body>
</html>
"""

# ── API Routes ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/now")
def api_now():
    try:
        result = get_full_prediction()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/target")
def api_target():
    try:
        price = float(request.args.get("price", 0))
        time_str = request.args.get("time", "")
        if not price or not time_str:
            return jsonify({"error": "Missing price or time"}), 400
        result = analyze_price_target(price, time_str)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)