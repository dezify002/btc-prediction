"""
BTC Prediction Bot — Web UI with Ensemble + Phase 4 Learning AI
Shows: live price, bot prediction, XGBoost prediction, AI Reviewer verdict,
       AND Learning AI insights (calibration, drift, recommendations)
"""

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from flask import Flask, jsonify, request, render_template_string
from prediction_core import get_full_prediction, analyze_price_target

# Phase 4 imports
from learning_engine import LearningEngine
from logger import update_outcome, get_stats as get_log_stats

app = Flask(__name__)

# ── HTML Template ──────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Prediction Bot — Ensemble + Learning AI</title>
<style>
:root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --up: #3fb950; --down: #f85149; --warn: #d29922;
    --strong: #a371f7; --learning: #f778ba;
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
.card h2.learning { color: var(--learning); }

/* Price display */
.price-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
.price-main { font-size: 2.2rem; font-weight: bold; }
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

.auto-refresh-status {
    font-size: 0.7rem; color: var(--muted);
    display: flex; align-items: center; gap: 6px;
}

#debugLog {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px; font-size: 0.7rem;
    color: var(--muted); max-height: 120px; overflow-y: auto;
    margin-top: 8px; display: none;
}

/* ── Phase 4 Learning AI Styles ───────────────────────── */

.learning-card {
    border-color: rgba(247, 120, 186, 0.3);
    background: linear-gradient(180deg, rgba(247,120,186,0.03) 0%, var(--card) 100%);
}

.learning-metric {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid rgba(247,120,186,0.1);
}
.learning-metric:last-child { border-bottom: none; }
.learning-label { color: var(--muted); font-size: 0.8rem; }
.learning-value { font-weight: 600; font-size: 0.85rem; }

.learning-bar {
    height: 8px; background: rgba(247,120,186,0.1);
    border-radius: 4px; overflow: hidden; margin-top: 4px;
}
.learning-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.5s ease;
}
.learning-bar-green { background: var(--up); }
.learning-bar-yellow { background: var(--warn); }
.learning-bar-red { background: var(--down); }

.recommendation-item {
    padding: 10px 14px; border-radius: 8px; margin-bottom: 8px;
    font-size: 0.8rem; border-left: 3px solid;
}
.rec-critical { background: rgba(231,76,60,0.1); border-color: #e74c3c; }
.rec-high { background: rgba(240,165,0,0.1); border-color: #f0a500; }
.rec-medium { background: rgba(0,212,170,0.08); border-color: #00d4aa; }
.rec-low { background: rgba(136,136,136,0.08); border-color: #888; }

.pattern-item {
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 0.78rem;
}
.pattern-item:last-child { border-bottom: none; }
.pattern-impact-high { color: #e74c3c; }
.pattern-impact-medium { color: #f0a500; }
.pattern-impact-low { color: #00d4aa; }

.calibration-table {
    width: 100%; border-collapse: collapse; font-size: 0.75rem;
}
.calibration-table th, .calibration-table td {
    padding: 6px 8px; text-align: left;
    border-bottom: 1px solid rgba(247,120,186,0.1);
}
.calibration-table th {
    color: var(--muted); font-weight: 500;
    text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.5px;
}
.gap-over { color: #e74c3c; }
.gap-under { color: #00d4aa; }

.drift-stable { color: var(--up); }
.drift-warning { color: var(--warn); }
.drift-declining { color: var(--down); }

.tabs {
    display: flex; gap: 4px; margin-bottom: 16px;
    border-bottom: 1px solid var(--border); padding-bottom: 8px;
}
.tab-btn {
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-family: inherit; font-size: 0.75rem;
}
.tab-btn.active {
    background: rgba(247,120,186,0.15); border-color: var(--learning);
    color: var(--learning);
}
.tab-btn:hover:not(.active) { border-color: var(--muted); color: var(--text); }

.tab-content { display: none; }
.tab-content.active { display: block; }

.up { color: var(--up); }
.down { color: var(--down); }
.warn { color: var(--warn); }
</style>
</head>
<body>
<div class="container">
    <h1>🪙 BTC Prediction Bot</h1>
    <p class="subtitle">Ensemble: Your Bot + XGBoost + AI Reviewer + Learning AI</p>

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

        <div class="verdict-box" id="verdictBox">
            <div class="verdict-title" id="verdictTitle">—</div>
            <div class="verdict-sub" id="verdictSub">—</div>
        </div>

        <div style="margin: 12px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="meta">Trust Score</span>
                <span class="meta" id="trustScore">—</span>
            </div>
            <div class="trust-bar-container">
                <div class="trust-bar" id="trustBar" style="width:0%"></div>
            </div>
        </div>

        <div style="display:flex; gap:12px; align-items:center; margin: 12px 0;">
            <span class="meta">Risk:</span>
            <span class="risk-badge" id="riskBadge">—</span>
            <span class="meta">|</span>
            <span class="meta" id="recommendation">—</span>
        </div>

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

    <!-- ═══════════════════════════════════════════════════════ -->
    <!-- PHASE 4 — LEARNING AI                                   -->
    <!-- ═══════════════════════════════════════════════════════ -->
    <div class="card learning-card">
        <h2 class="learning">🧠 Learning AI — Performance Insights</h2>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tab-summary', this)">Summary</button>
            <button class="tab-btn" onclick="switchTab('tab-calibration', this)">Calibration</button>
            <button class="tab-btn" onclick="switchTab('tab-patterns', this)">Patterns</button>
            <button class="tab-btn" onclick="switchTab('tab-recommendations', this)">Recommendations</button>
        </div>

        <!-- Tab: Summary -->
        <div class="tab-content active" id="tab-summary">
            <div id="learning-summary-loading" class="meta">Loading insights...</div>
            <div id="learning-summary-content" style="display:none;">
                <div class="learning-metric">
                    <span class="learning-label">Total Predictions Logged</span>
                    <span class="learning-value" id="l4-total">—</span>
                </div>
                <div class="learning-metric">
                    <span class="learning-label">Overall Accuracy</span>
                    <span class="learning-value" id="l4-accuracy">—</span>
                </div>
                <div class="learning-bar"><div class="learning-bar-fill" id="l4-acc-bar" style="width:0%"></div></div>
                <div class="learning-metric" style="margin-top:8px;">
                    <span class="learning-label">Last 7 Days</span>
                    <span class="learning-value" id="l4-last7">—</span>
                </div>
                <div class="learning-metric">
                    <span class="learning-label">Last 30 Days</span>
                    <span class="learning-value" id="l4-last30">—</span>
                </div>
                <div class="learning-metric" style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(247,120,186,0.1);">
                    <span class="learning-label">Model Drift</span>
                    <span class="learning-value" id="l4-drift-status">—</span>
                </div>
                <div class="learning-metric">
                    <span class="learning-label">Drift Amount</span>
                    <span class="learning-value" id="l4-drift-pct">—</span>
                </div>
                <p class="meta" style="margin-top:8px;" id="l4-drift-rec">—</p>
            </div>
        </div>

        <!-- Tab: Calibration -->
        <div class="tab-content" id="tab-calibration">
            <div id="learning-cal-loading" class="meta">Loading calibration data...</div>
            <div id="learning-cal-content" style="display:none;">
                <p class="meta" style="margin-bottom:12px;" id="l4-cal-assessment">—</p>
                <table class="calibration-table">
                    <thead>
                        <tr><th>Bin</th><th>Count</th><th>Avg Conf</th><th>Actual</th><th>Gap</th></tr>
                    </thead>
                    <tbody id="l4-cal-body"></tbody>
                </table>
                <div style="margin-top:16px; padding:12px; background:rgba(0,0,0,0.2); border-radius:8px;">
                    <div class="meta" style="margin-bottom:8px;">Calibrate Current Confidence</div>
                    <div style="display:flex; gap:8px;">
                        <input type="number" id="calibrateInput" placeholder="e.g. 82" min="0" max="100" style="flex:1; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px 12px; border-radius:6px; font-family:inherit;">
                        <button onclick="calibrateConfidence()" style="background:var(--learning); color:#fff; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-family:inherit; font-weight:bold;">Calibrate</button>
                    </div>
                    <div id="calibrateResult" style="margin-top:8px; font-size:0.8rem;"></div>
                </div>
            </div>
        </div>

        <!-- Tab: Patterns -->
        <div class="tab-content" id="tab-patterns">
            <div id="learning-patterns-loading" class="meta">Discovering patterns...</div>
            <div id="learning-patterns-content" style="display:none;">
                <div id="l4-patterns-list"></div>
            </div>
        </div>

        <!-- Tab: Recommendations -->
        <div class="tab-content" id="tab-recommendations">
            <div id="learning-recs-loading" class="meta">Generating recommendations...</div>
            <div id="learning-recs-content" style="display:none;">
                <div id="l4-recs-list"></div>
            </div>
        </div>
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

function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
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

function getLearningBarClass(pct) {
    if (pct >= 70) return "learning-bar-green";
    if (pct >= 55) return "learning-bar-yellow";
    return "learning-bar-red";
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
    loadPhase4();
}

function render(data) {
    document.getElementById("priceDisplay").textContent = formatPrice(data.price);
    document.getElementById("priceMeta").innerHTML = 
        `Spread: <span class="${data.spread_pct > 0.05 ? 'warn' : ''}">${data.spread_pct.toFixed(4)}%</span>`;
    document.getElementById("exchangeBadge").textContent = `Source: ${data.exchange}`;

    const ts = data.timestamp ? new Date(data.timestamp + 'Z') : new Date();
    document.getElementById("lastUpdate").textContent = ts.toUTCString();

    const bot = data.bot || {};
    document.getElementById("botPred").textContent = bot.pred || "—";
    document.getElementById("botPred").className = "model-pred " + (bot.pred === "UP" ? "up" : bot.pred === "DOWN" ? "down" : "");
    document.getElementById("botConf").textContent = bot.confidence ? `Confidence: ${(bot.confidence * 100).toFixed(1)}%` : "—";

    const xgb = data.xgboost || {};
    document.getElementById("xgbPred").textContent = xgb.pred || "—";
    document.getElementById("xgbPred").className = "model-pred " + (xgb.pred === "UP" ? "up" : xgb.pred === "DOWN" ? "down" : "");
    document.getElementById("xgbConf").textContent = xgb.prob ? `P(UP): ${(xgb.prob * 100).toFixed(1)}%` : "—";

    const dec = data.decision || {};
    const vBox = document.getElementById("verdictBox");
    vBox.className = "verdict-box " + getVerdictClass(dec.ensemble_verdict || "");
    document.getElementById("verdictTitle").textContent = dec.ensemble_verdict || "—";
    document.getElementById("verdictSub").textContent = dec.recommendation || "—";

    const ts_score = dec.trust_score || 0;
    document.getElementById("trustScore").textContent = (ts_score * 100).toFixed(1) + "%";
    const bar = document.getElementById("trustBar");
    bar.style.width = (ts_score * 100) + "%";
    bar.className = "trust-bar " + getTrustClass(ts_score);

    document.getElementById("riskBadge").textContent = dec.risk_level || "—";
    document.getElementById("riskBadge").className = "risk-badge " + getRiskClass(dec.risk_level);
    document.getElementById("recommendation").textContent = dec.recommendation || "—";

    const reasons = dec.reasons || ["No reasons available"];
    document.getElementById("reasonsList").innerHTML = reasons.map(r => {
        let cls = "";
        if (r.includes("DISAGREEMENT") || r.includes("dangerous") || r.includes("risky")) cls = "danger";
        else if (r.includes("High") || r.includes("spike") || r.includes("avoid")) cls = "warning";
        else if (r.includes("agree") || r.includes("favorable")) cls = "good";
        return `<li class="${cls}">${r}</li>`;
    }).join("");

    document.getElementById("marketRegime").textContent = dec.market_regime || "—";

    const feats = data.features || {};
    const featRows = Object.entries(feats).slice(0, 15).map(([k, v]) => {
        const val = typeof v === "number" ? v.toFixed(4) : v;
        return `<tr><td>${k}</td><td>${val}</td></tr>`;
    }).join("");
    document.getElementById("featuresTable").innerHTML = featRows || "<tr><td>No features</td></tr>";
}

// ── Phase 4 Learning AI ───────────────────────────────────

async function loadPhase4() {
    try {
        const [summaryRes, driftRes, patternsRes, recsRes] = await Promise.all([
            fetch("/api/learning/summary"),
            fetch("/api/learning/drift"),
            fetch("/api/learning/patterns"),
            fetch("/api/learning/recommendations?limit=5"),
        ]);

        const summary = await summaryRes.json();
        const drift = await driftRes.json();
        const patterns = await patternsRes.json();
        const recs = await recsRes.json();

        // Summary Tab
        document.getElementById("learning-summary-loading").style.display = "none";
        document.getElementById("learning-summary-content").style.display = "block";

        if (summary.has_data) {
            const ov = summary.overall || {};
            document.getElementById("l4-total").textContent = (ov.total_predictions || 0).toLocaleString();
            document.getElementById("l4-accuracy").textContent = (ov.accuracy_pct || 0) + "%";
            const accBar = document.getElementById("l4-acc-bar");
            accBar.style.width = (ov.accuracy_pct || 0) + "%";
            accBar.className = "learning-bar-fill " + getLearningBarClass(ov.accuracy_pct || 0);
            document.getElementById("l4-last7").textContent = (ov.last_7_days_accuracy || "—") + "%";
            document.getElementById("l4-last30").textContent = (ov.last_30_days_accuracy || "—") + "%";
        } else {
            document.getElementById("learning-summary-content").innerHTML = 
                `<p class="meta">${summary.error || "No prediction data yet. Make some predictions first."}</p>`;
        }

        if (drift.has_data) {
            const statusEl = document.getElementById("l4-drift-status");
            statusEl.textContent = drift.status.toUpperCase();
            statusEl.className = "learning-value drift-" + drift.status;
            const pct = drift.drift_pct || 0;
            document.getElementById("l4-drift-pct").textContent = (pct > 0 ? "+" : "") + pct.toFixed(1) + "%";
            document.getElementById("l4-drift-rec").textContent = drift.recommendation || "";
        } else {
            document.getElementById("l4-drift-status").textContent = "N/A";
            document.getElementById("l4-drift-pct").textContent = "—";
            document.getElementById("l4-drift-rec").textContent = drift.error || "Need 200+ predictions with outcomes";
        }

        // Calibration Tab
        loadCalibration();

        // Patterns Tab
        document.getElementById("learning-patterns-loading").style.display = "none";
        document.getElementById("learning-patterns-content").style.display = "block";
        if (patterns.has_data && patterns.patterns && patterns.patterns.length > 0) {
            document.getElementById("l4-patterns-list").innerHTML = patterns.patterns.map(p => {
                const icon = p.impact === 'high' ? '🔴' : p.impact === 'medium' ? '🟡' : '🟢';
                return `<div class="pattern-item">
                    <span class="pattern-impact-${p.impact}">${icon} <strong>${p.category}</strong></span>
                    <div style="color:#aaa; margin-top:2px;">${p.finding}</div>
                    <div style="color:var(--learning); margin-top:2px;">→ ${p.recommendation}</div>
                </div>`;
            }).join("");
        } else {
            document.getElementById("l4-patterns-list").innerHTML = 
                `<p class="meta">${patterns.error || "No patterns discovered yet."}</p>`;
        }

        // Recommendations Tab
        document.getElementById("learning-recs-loading").style.display = "none";
        document.getElementById("learning-recs-content").style.display = "block";
        if (recs.has_data && recs.recommendations && recs.recommendations.length > 0) {
            const priorityMap = { "critical": "rec-critical", "high": "rec-high", "medium": "rec-medium", "low": "rec-low" };
            document.getElementById("l4-recs-list").innerHTML = recs.recommendations.map(r => {
                return `<div class="recommendation-item ${priorityMap[r.priority] || 'rec-low'}">
                    <strong>${r.title}</strong><br>
                    <span style="color:#888;">${r.action}</span>
                </div>`;
            }).join("");
        } else {
            document.getElementById("l4-recs-list").innerHTML = 
                `<p class="meta">${recs.error || "✅ No critical recommendations. Model is performing well."}</p>`;
        }

    } catch (e) {
        console.error("Phase 4 load error:", e);
    }
}

async function loadCalibration() {
    try {
        const res = await fetch("/api/learning/calibration");
        const cal = await res.json();
        document.getElementById("learning-cal-loading").style.display = "none";
        document.getElementById("learning-cal-content").style.display = "block";

        if (cal.has_data && cal.bins) {
            document.getElementById("l4-cal-assessment").textContent = cal.overall_assessment || "";
            document.getElementById("l4-cal-body").innerHTML = cal.bins.map(b => {
                const gapClass = b.calibration_gap > 5 ? 'gap-over' : b.calibration_gap < -5 ? 'gap-under' : '';
                const gapStr = b.calibration_gap > 0 ? '+' + b.calibration_gap.toFixed(1) : b.calibration_gap.toFixed(1);
                return `<tr><td>${b.bin_label}</td><td>${b.count}</td><td>${b.avg_confidence.toFixed(1)}%</td><td>${b.actual_accuracy.toFixed(1)}%</td><td class="${gapClass}">${gapStr}%</td></tr>`;
            }).join("");
        } else {
            document.getElementById("l4-cal-content").innerHTML = 
                `<p class="meta">${cal.error || "Need 50+ predictions with outcomes for calibration."}</p>`;
        }
    } catch (e) {
        console.error("Calibration load error:", e);
    }
}

async function calibrateConfidence() {
    const input = document.getElementById("calibrateInput");
    const conf = parseFloat(input.value);
    if (isNaN(conf)) {
        document.getElementById("calibrateResult").innerHTML = '<span class="warn">Enter a number</span>';
        return;
    }
    document.getElementById("calibrateResult").innerHTML = '<span class="meta">Calibrating...</span>';
    try {
        const res = await fetch("/api/learning/calibrate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({confidence: conf})
        });
        const data = await res.json();
        if (data.method === "historical_lookup") {
            document.getElementById("calibrateResult").innerHTML = 
                `<span class="up">Raw ${conf}% → Calibrated ${data.calibrated_confidence}%</span> 
                 <span class="meta">(based on ${data.sample_size} similar predictions)</span>`;
        } else {
            document.getElementById("calibrateResult").innerHTML = 
                `<span class="meta">Not enough data. Raw ${conf}% kept.</span>`;
        }
    } catch (e) {
        document.getElementById("calibrateResult").innerHTML = `<span class="down">Error: ${e.message}</span>`;
    }
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
        let html = `<div style="margin-top:12px; padding:12px; background:rgba(0,0,0,0.2); border-radius:8px;">
            <div style="font-size:1.3rem; font-weight:bold; margin-bottom:8px;">
                ${data.verdict} — ${(data.verdict_confidence * 100).toFixed(1)}%
            </div>
            <div class="meta">Current: ${formatPrice(data.current_price)} → Target: ${formatPrice(data.target_price)}</div>
            <div class="meta">Horizon: ${data.minutes_ahead} min | P(above): ${(data.probability_above * 100).toFixed(1)}%</div>`;
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

function startAutoRefresh() {
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    autoRefreshInterval = setInterval(loadNow, 15000);
    document.getElementById("refreshStatus").textContent = "AUTO-REFRESH ON";
}

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


# ═════════════════════════════════════════════════════════════
# PHASE 4 — LEARNING AI ROUTES
# ═════════════════════════════════════════════════════════════

@app.route("/api/learning/summary")
def api_learning_summary():
    try:
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs(days=30)
        if not engine.has_data():
            return jsonify({"has_data": False, "error": "No predictions logged yet"})
        return jsonify({"has_data": True, **engine.daily_summary()})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/weekly")
def api_learning_weekly():
    try:
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        if not engine.has_data(50):
            return jsonify({"has_data": False, "error": "Need 50+ predictions with outcomes"})
        return jsonify({"has_data": True, **engine.weekly_report()})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/drift")
def api_learning_drift():
    try:
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        if not engine.has_data(200):
            return jsonify({"has_data": False, "error": "Need 200+ predictions for drift detection"})
        return jsonify({"has_data": True, **engine.detect_drift()})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/calibration")
def api_learning_calibration():
    try:
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        if not engine.has_data(50):
            return jsonify({"has_data": False, "error": "Need 50+ predictions for calibration"})
        return jsonify({"has_data": True, **engine.calibrate_confidence()})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/patterns")
def api_learning_patterns():
    try:
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        if not engine.has_data(50):
            return jsonify({"has_data": False, "error": "Need 50+ predictions for pattern discovery"})
        return jsonify({"has_data": True, "patterns": engine.discover_patterns()})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/recommendations")
def api_learning_recommendations():
    try:
        limit = request.args.get("limit", 10, type=int)
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        if not engine.has_data():
            return jsonify({"has_data": False, "error": "No data available"})
        return jsonify({"has_data": True, "recommendations": engine.generate_recommendations()[:limit]})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/features")
def api_learning_features():
    try:
        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        if not engine.has_data(20):
            return jsonify({"has_data": False, "error": "Need 20+ predictions"})
        return jsonify({"has_data": True, "features": engine.feature_effectiveness()})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e)}), 500


@app.route("/api/learning/calibrate", methods=["POST"])
def api_calibrate_confidence():
    try:
        data = request.get_json() or {}
        confidence = data.get("confidence", 50)
        window = data.get("window")
        regime = data.get("regime")

        engine = LearningEngine(log_dir="reports/logs", artifacts_dir="reports")
        engine.load_logs()
        result = engine.get_calibrated_confidence(confidence, window, regime)
        return jsonify(result)
    except Exception as e:
        return jsonify({"calibrated_confidence": 50, "method": "raw", "sample_size": 0}), 500


@app.route("/api/learning/stats")
def api_learning_stats():
    return jsonify(get_log_stats())


@app.route("/api/outcome", methods=["POST"])
def api_record_outcome():
    try:
        data = request.get_json() or {}
        timestamp = data.get("timestamp")
        actual = data.get("actual_result")
        correct = data.get("correct")

        if not timestamp or actual is None or correct is None:
            return jsonify({"success": False, "error": "Missing timestamp, actual_result, or correct"}), 400

        success = update_outcome(timestamp, actual, correct)
        return jsonify({"success": success, "timestamp": timestamp, "actual": actual, "correct": correct})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)