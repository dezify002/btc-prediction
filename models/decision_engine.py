"""
Decision Engine v2 — fixes RSI overbought/oversold override.
"""

import json


def classify_regime(features: dict) -> str:
    vol_zscore = features.get("vol_zscore_20", 0)
    atr_pct = features.get("atr_pct", 0)
    ema_dist = features.get("ema_dist", 0)

    vol_regime = "high_vol" if (vol_zscore > 2.0 or atr_pct > 0.015) else "low_vol" if vol_zscore < 0.5 else "normal_vol"
    trend = "uptrend" if ema_dist > 0.005 else "downtrend" if ema_dist < -0.005 else "ranging"
    return f"{trend}_{vol_regime}"


def decide(bot_pred: str, bot_confidence: float, xgb_prob: float, features: dict) -> dict:
    xgb_pred = "UP" if xgb_prob >= 0.5 else "DOWN"
    regime = classify_regime(features)

    # Base confidence
    bot_conf = min(bot_confidence, 1.0)
    xgb_conf = max(xgb_prob, 1 - xgb_prob)
    avg_conf = (bot_conf + xgb_conf) / 2

    # Agreement
    if bot_pred == xgb_pred:
        agreement_boost = 0.15
        agreement_reason = f"Both models agree ({bot_pred})"
    else:
        agreement_boost = -0.20
        agreement_reason = f"DISAGREEMENT: bot={bot_pred}, xgb={xgb_pred}"

    # Regime penalties
    regime_penalty = 0
    regime_reasons = []

    vol_zscore = features.get("vol_zscore_20", 0)
    atr_pct = features.get("atr_pct", 0)
    rsi = features.get("rsi_14", 50)
    dist_high = features.get("dist_from_high_20", 0)
    dist_low = features.get("dist_from_low_20", 0)

    if vol_zscore > 2.0:
        regime_penalty -= 0.20
        regime_reasons.append("HIGH VOLATILITY — model unreliable")
    elif vol_zscore < 0.3:
        regime_penalty += 0.05
        regime_reasons.append("Low volatility — favorable regime")

    if atr_pct > 0.02:
        regime_penalty -= 0.15
        regime_reasons.append("ATR spike — avoid new positions")

    # RSI extremes — STRONG override
    if rsi > 80 and bot_pred == "UP":
        regime_penalty -= 0.30
        regime_reasons.append("RSI > 80 OVERBOUGHT — UP prediction dangerous")
    elif rsi > 70 and bot_pred == "UP":
        regime_penalty -= 0.15
        regime_reasons.append("RSI > 70 overbought — UP prediction risky")
    elif rsi < 20 and bot_pred == "DOWN":
        regime_penalty -= 0.30
        regime_reasons.append("RSI < 20 OVERSOLD — DOWN prediction dangerous")
    elif rsi < 30 and bot_pred == "DOWN":
        regime_penalty -= 0.15
        regime_reasons.append("RSI < 30 oversold — DOWN prediction risky")

    # Distance from extremes
    if bot_pred == "UP" and dist_high < 0.005:
        regime_penalty -= 0.15
        regime_reasons.append("Near 20-period high — limited upside")
    elif bot_pred == "DOWN" and dist_low < 0.005:
        regime_penalty -= 0.15
        regime_reasons.append("Near 20-period low — limited downside")

    # Compute trust score
    trust_score = avg_conf + agreement_boost + regime_penalty
    trust_score = max(0.0, min(1.0, trust_score))

    # Risk & recommendation
    if trust_score >= 0.80:
        risk = "LOW"
        recommendation = "TAKE TRADE"
    elif trust_score >= 0.60:
        risk = "MEDIUM"
        recommendation = "SMALL POSITION"
    elif trust_score >= 0.40:
        risk = "ELEVATED"
        recommendation = "WAIT"
    else:
        risk = "HIGH"
        recommendation = "SKIP TRADE"

    # Verdict
    if trust_score < 0.40:
        verdict = "UNCERTAIN"
    elif bot_pred == xgb_pred and trust_score >= 0.75:
        verdict = f"{bot_pred} (Strong)"
    elif bot_pred == xgb_pred and trust_score >= 0.55:
        verdict = f"{bot_pred} (Moderate)"
    else:
        verdict = f"{bot_pred} (Weak — high risk)"

    reasons = [agreement_reason]
    reasons.extend(regime_reasons)
    reasons.append(f"Base confidence: {avg_conf:.1%}")
    reasons.append(f"Regime: {regime}")

    return {
        "bot_prediction": bot_pred,
        "bot_confidence": round(bot_conf, 4),
        "xgb_prediction": xgb_pred,
        "xgb_probability": round(xgb_prob, 4),
        "ensemble_verdict": verdict,
        "trust_score": round(trust_score, 4),
        "risk_level": risk,
        "recommendation": recommendation,
        "market_regime": regime,
        "reasons": reasons,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("DECISION ENGINE v2 TEST")
    print("=" * 50)

    test_cases = [
        ("Strong UP (agree, low vol)", "UP", 0.85, 0.72, {"rsi_14": 58, "atr_pct": 0.005, "ema_dist": 0.008, "vol_zscore_20": 0.4, "dist_from_high_20": 0.03, "dist_from_low_20": 0.01}),
        ("Disagreement", "UP", 0.75, 0.35, {"rsi_14": 62, "atr_pct": 0.012, "ema_dist": 0.002, "vol_zscore_20": 1.5, "dist_from_high_20": 0.02, "dist_from_low_20": 0.02}),
        ("High volatility", "UP", 0.80, 0.65, {"rsi_14": 55, "atr_pct": 0.025, "ema_dist": 0.005, "vol_zscore_20": 2.5, "dist_from_high_20": 0.01, "dist_from_low_20": 0.05}),
        ("RSI overbought (82)", "UP", 0.90, 0.80, {"rsi_14": 82, "atr_pct": 0.006, "ema_dist": 0.010, "vol_zscore_20": 0.6, "dist_from_high_20": 0.002, "dist_from_low_20": 0.08}),
        ("RSI oversold (18) DOWN", "DOWN", 0.85, 0.75, {"rsi_14": 18, "atr_pct": 0.005, "ema_dist": -0.008, "vol_zscore_20": 0.5, "dist_from_high_20": 0.10, "dist_from_low_20": 0.002}),
    ]

    for name, bot_pred, bot_conf, xgb_prob, feats in test_cases:
        print(f"\n{'─' * 50}")
        print(f"TEST: {name}")
        print(f"{'─' * 50}")
        r = decide(bot_pred, bot_conf, xgb_prob, feats)
        print(f"Verdict:     {r['ensemble_verdict']}")
        print(f"Trust Score: {r['trust_score']:.1%}")
        print(f"Risk:        {r['risk_level']}")
        print(f"Action:      {r['recommendation']}")
        print(f"Reasons:")
        for reason in r["reasons"]:
            print(f"  • {reason}")

    print(f"\n{'=' * 50}")
    print("TEST COMPLETE")
    print("=" * 50)