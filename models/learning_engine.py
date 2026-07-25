"""
Phase 4 — Learning Engine
Analyzes prediction logs to generate insights, calibrate confidence,
detect drift, discover patterns, and recommend improvements.
"""

import json
import os
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


class LearningEngine:
    """
    Self-improvement engine for the BTC prediction bot.

    Reads logged predictions, analyzes performance across multiple dimensions,
    and generates actionable recommendations.
    """

    def __init__(self, log_dir: str = "logs", artifacts_dir: str = "artifacts"):
        self.log_dir = Path(log_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.calibration_model: Optional[Dict] = None
        self.drift_baseline: Optional[Dict] = None

    # ─────────────────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────────────────

    def load_logs(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        Load prediction logs from JSON lines files.

        Args:
            days: Only load logs from last N days (None = all)
        """
        records = []
        cutoff = datetime.utcnow() - timedelta(days=days) if days else None

        log_files = list(self.log_dir.glob("predictions_*.jsonl"))
        log_files += list(self.log_dir.glob("predictions_*.json"))
        log_files += list(self.log_dir.glob("*.csv"))

        for filepath in log_files:
            try:
                if filepath.suffix == ".csv":
                    df_temp = pd.read_csv(filepath)
                    records.extend(df_temp.to_dict("records"))
                elif filepath.suffix == ".json":
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            records.extend(data)
                        else:
                            records.append(data)
                else:  # .jsonl
                    with open(filepath, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                records.append(json.loads(line))
            except Exception as e:
                print(f"⚠️  Skipped {filepath}: {e}")
                continue

        if not records:
            # Create empty DataFrame with expected columns
            self.df = pd.DataFrame(columns=[
                "timestamp", "current_price", "target_price", "prediction_window",
                "prediction", "confidence", "xgb_probability", "trust_score",
                "rsi", "atr", "ema_distance", "volatility", "volume",
                "market_regime", "actual_result", "correct", "hour", "dayofweek"
            ])
            return self.df

        self.df = pd.DataFrame(records)

        # Parse timestamps
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")
            self.df = self.df.sort_values("timestamp")

        # Ensure boolean correct column
        if "correct" in self.df.columns:
            self.df["correct"] = self.df["correct"].astype(bool)

        # Derive time features if missing
        if "timestamp" in self.df.columns and "hour" not in self.df.columns:
            self.df["hour"] = self.df["timestamp"].dt.hour
        if "timestamp" in self.df.columns and "dayofweek" not in self.df.columns:
            self.df["dayofweek"] = self.df["timestamp"].dt.dayofweek

        # Ensure numeric columns
        numeric_cols = ["confidence", "xgb_probability", "trust_score", 
                        "rsi", "atr", "ema_distance", "volatility", "volume"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if cutoff is not None:
            self.df = self.df[self.df["timestamp"] >= cutoff]

        return self.df

    def has_data(self, min_records: int = 10) -> bool:
        """Check if we have enough data to analyze."""
        return self.df is not None and len(self.df) >= min_records and "correct" in self.df.columns

    # ─────────────────────────────────────────────────────────
    # BASIC METRICS
    # ─────────────────────────────────────────────────────────

    def overall_accuracy(self) -> Dict[str, Any]:
        """Overall accuracy stats."""
        if not self.has_data():
            return {"error": "Not enough data"}

        df = self.df.dropna(subset=["correct"])
        total = len(df)
        correct = df["correct"].sum()
        accuracy = correct / total if total > 0 else 0

        # Recent trends
        last_7 = df.tail(int(total * 0.1)) if total > 100 else df.tail(7)
        last_30 = df.tail(int(total * 0.3)) if total > 100 else df.tail(30)

        return {
            "total_predictions": int(total),
            "correct": int(correct),
            "accuracy": round(accuracy, 4),
            "accuracy_pct": round(accuracy * 100, 1),
            "last_7_days_accuracy": round(last_7["correct"].mean() * 100, 1) if len(last_7) > 0 else None,
            "last_30_days_accuracy": round(last_30["correct"].mean() * 100, 1) if len(last_30) > 0 else None,
            "date_range": {
                "from": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
                "to": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None,
            }
        }

    def accuracy_by_window(self) -> Dict[str, Dict]:
        """Accuracy broken down by prediction window (15m, 1h, 4h)."""
        if not self.has_data() or "prediction_window" not in self.df.columns:
            return {"error": "No prediction_window data"}

        results = {}
        for window, group in self.df.groupby("prediction_window"):
            group = group.dropna(subset=["correct"])
            if len(group) < 5:
                continue
            acc = group["correct"].mean()
            results[window] = {
                "count": len(group),
                "correct": int(group["correct"].sum()),
                "accuracy": round(acc, 4),
                "accuracy_pct": round(acc * 100, 1)
            }
        return results

    def accuracy_by_regime(self) -> Dict[str, Dict]:
        """Accuracy by market regime."""
        if not self.has_data() or "market_regime" not in self.df.columns:
            return {"error": "No market_regime data"}

        results = {}
        for regime, group in self.df.groupby("market_regime"):
            group = group.dropna(subset=["correct"])
            if len(group) < 5:
                continue
            acc = group["correct"].mean()
            results[regime] = {
                "count": len(group),
                "correct": int(group["correct"].sum()),
                "accuracy": round(acc, 4),
                "accuracy_pct": round(acc * 100, 1)
            }

        # Sort by accuracy descending
        results = dict(sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True))
        return results

    # ─────────────────────────────────────────────────────────
    # CONFIDENCE CALIBRATION
    # ─────────────────────────────────────────────────────────

    def calibrate_confidence(self, bins: int = 5) -> Dict[str, Any]:
        """
        Compare predicted confidence to actual accuracy in bins.

        Returns calibration data showing how well confidence
        matches reality. Use this to adjust displayed confidence.
        """
        if not self.has_data() or "confidence" not in self.df.columns:
            return {"error": "No confidence data"}

        df = self.df.dropna(subset=["confidence", "correct"]).copy()
        if len(df) < 20:
            return {"error": f"Need >=20 records with confidence, have {len(df)}"}

        # Create confidence bins
        df["conf_bin"] = pd.cut(df["confidence"], bins=bins, precision=0)

        calibration = []
        for interval, group in df.groupby("conf_bin", observed=False):
            if len(group) < 3:
                continue
            actual_acc = group["correct"].mean()
            avg_conf = group["confidence"].mean()
            calibration.append({
                "bin_label": f"{int(interval.left)}-{int(interval.right)}%",
                "bin_range": [round(interval.left, 1), round(interval.right, 1)],
                "count": len(group),
                "avg_confidence": round(avg_conf, 1),
                "actual_accuracy": round(actual_acc * 100, 1),
                "calibration_gap": round(avg_conf - actual_acc * 100, 1),
                "reliability": "Overconfident" if avg_conf > actual_acc * 100 + 5 else 
                              "Underconfident" if avg_conf < actual_acc * 100 - 5 else "Well-calibrated"
            })

        # Compute overall calibration score (Brier-like)
        df["prob"] = df["confidence"] / 100.0
        brier = ((df["prob"] - df["correct"].astype(int)) ** 2).mean()

        # Build lookup table: given a confidence, what was historical accuracy?
        lookup = {}
        for item in calibration:
            lookup[item["bin_label"]] = item["actual_accuracy"]

        self.calibration_model = {
            "bins": calibration,
            "lookup": lookup,
            "brier_score": round(brier, 4),
            "overall_assessment": self._assess_calibration(calibration)
        }

        return self.calibration_model

    def _assess_calibration(self, calibration: List[Dict]) -> str:
        """Generate human-readable calibration assessment."""
        over = [c for c in calibration if c["reliability"] == "Overconfident"]
        under = [c for c in calibration if c["reliability"] == "Underconfident"]

        if len(over) > len(under):
            return f"Model is overconfident in {len(over)} of {len(calibration)} bins. Reduce displayed confidence."
        elif len(under) > len(over):
            return f"Model is underconfident in {len(under)} bins. You can trust predictions more."
        else:
            return "Model confidence is reasonably well-calibrated."

    def get_calibrated_confidence(self, confidence: float, 
                                   prediction_window: Optional[str] = None,
                                   market_regime: Optional[str] = None) -> Dict:
        """
        Get historically-calibrated confidence for a specific prediction.

        Looks up similar historical predictions and returns their actual accuracy.
        """
        if not self.has_data() or "confidence" not in self.df.columns:
            return {"calibrated_confidence": confidence, "method": "raw", "sample_size": 0}

        df = self.df.dropna(subset=["confidence", "correct"]).copy()

        # Filter by regime and window for more precise calibration
        if market_regime and "market_regime" in df.columns:
            df = df[df["market_regime"] == market_regime]
        if prediction_window and "prediction_window" in df.columns:
            df = df[df["prediction_window"] == prediction_window]

        if len(df) < 10:
            # Fall back to all data
            df = self.df.dropna(subset=["confidence", "correct"]).copy()

        # Find predictions with similar confidence (±5%)
        similar = df[(df["confidence"] >= confidence - 5) & (df["confidence"] <= confidence + 5)]

        if len(similar) < 5:
            # Widen the net
            similar = df[(df["confidence"] >= confidence - 10) & (df["confidence"] <= confidence + 10)]

        if len(similar) < 3:
            return {"calibrated_confidence": confidence, "method": "raw", "sample_size": 0}

        actual_acc = similar["correct"].mean()
        return {
            "calibrated_confidence": round(actual_acc * 100, 1),
            "method": "historical_lookup",
            "sample_size": len(similar),
            "confidence_range": f"{confidence-5:.0f}-{confidence+5:.0f}%",
            "original_confidence": confidence
        }

    # ─────────────────────────────────────────────────────────
    # FEATURE EFFECTIVENESS
    # ─────────────────────────────────────────────────────────

    def feature_effectiveness(self) -> Dict[str, Dict]:
        """
        Analyze which features correlate with prediction correctness.

        Returns importance scores based on how well each feature
        discriminates between correct and incorrect predictions.
        """
        if not self.has_data():
            return {"error": "No data"}

        features = ["rsi", "atr", "ema_distance", "volatility", "volume", 
                    "xgb_probability", "trust_score"]
        available = [f for f in features if f in self.df.columns]

        if not available:
            return {"error": "No feature columns available"}

        df = self.df.dropna(subset=["correct"] + available)
        if len(df) < 20:
            return {"error": "Not enough complete records"}

        correct = df[df["correct"] == True]
        wrong = df[df["correct"] == False]

        results = {}
        for feat in available:
            c_vals = correct[feat].dropna()
            w_vals = wrong[feat].dropna()

            if len(c_vals) < 5 or len(w_vals) < 5:
                continue

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((c_vals.var() + w_vals.var()) / 2)
            cohens_d = abs(c_vals.mean() - w_vals.mean()) / pooled_std if pooled_std > 0 else 0

            # Mann-Whitney U test
            try:
                statistic, pvalue = stats.mannwhitneyu(c_vals, w_vals, alternative="two-sided")
            except:
                pvalue = 1.0

            # Point-biserial correlation
            corr = df[feat].corr(df["correct"].astype(int))

            results[feat] = {
                "effect_size": round(cohens_d, 3),
                "correlation": round(corr, 3) if not pd.isna(corr) else 0,
                "p_value": round(pvalue, 4),
                "significant": pvalue < 0.05,
                "correct_mean": round(c_vals.mean(), 4),
                "wrong_mean": round(w_vals.mean(), 4),
                "importance": round(abs(corr) if not pd.isna(corr) else cohens_d * 0.5, 3)
            }

        # Sort by importance
        results = dict(sorted(results.items(), key=lambda x: x[1]["importance"], reverse=True))
        return results

    # ─────────────────────────────────────────────────────────
    # DRIFT DETECTION
    # ─────────────────────────────────────────────────────────

    def detect_drift(self, window_size: int = 100, 
                       reference_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect if model performance is degrading over time.

        Compares recent accuracy to baseline (older data).
        """
        if not self.has_data(window_size * 2):
            return {"error": f"Need >= {window_size * 2} records for drift detection"}

        df = self.df.dropna(subset=["correct"]).copy()
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        ref_size = reference_window or window_size * 2

        # Baseline: earliest N predictions
        baseline = df.head(ref_size)
        # Recent: latest N predictions
        recent = df.tail(window_size)

        baseline_acc = baseline["correct"].mean()
        recent_acc = recent["correct"].mean()

        # Statistical test
        baseline_successes = int(baseline["correct"].sum())
        baseline_trials = len(baseline)
        recent_successes = int(recent["correct"].sum())
        recent_trials = len(recent)

        # Two-proportion z-test
        p1 = baseline_successes / baseline_trials
        p2 = recent_successes / recent_trials
        p_pool = (baseline_successes + recent_successes) / (baseline_trials + recent_trials)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/baseline_trials + 1/recent_trials))
        z = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        drift_pct = (recent_acc - baseline_acc) * 100

        status = "stable"
        if p_value < 0.05 and drift_pct < -5:
            status = "declining"
        elif p_value < 0.05 and drift_pct > 5:
            status = "improving"
        elif p_value < 0.1 and drift_pct < -3:
            status = "warning"

        self.drift_baseline = {
            "status": status,
            "drift_pct": round(drift_pct, 2),
            "baseline_accuracy": round(baseline_acc * 100, 1),
            "recent_accuracy": round(recent_acc * 100, 1),
            "baseline_window": baseline_trials,
            "recent_window": recent_trials,
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "recommendation": self._drift_recommendation(status, drift_pct)
        }

        return self.drift_baseline

    def _drift_recommendation(self, status: str, drift_pct: float) -> str:
        if status == "declining":
            return f"🚨 Performance dropped {abs(drift_pct):.1f}%. Consider retraining with recent data."
        elif status == "warning":
            return f"⚠️  Performance trending down ({drift_pct:.1f}%). Monitor closely."
        elif status == "improving":
            return f"✅ Performance improving (+{drift_pct:.1f}%). Current model is adapting well."
        else:
            return "✅ Performance stable. No action needed."

    # ─────────────────────────────────────────────────────────
    # PATTERN DISCOVERY
    # ─────────────────────────────────────────────────────────

    def discover_patterns(self) -> List[Dict]:
        """
        Automatically discover patterns in prediction performance.

        Tests various dimensions and returns significant findings.
        """
        if not self.has_data(50):
            return [{"error": "Need >=50 records for pattern discovery"}]

        patterns = []
        df = self.df.dropna(subset=["correct"]).copy()

        # 1. Time of day patterns
        if "hour" in df.columns:
            hour_perf = df.groupby("hour")["correct"].agg(["mean", "count"])
            hour_perf = hour_perf[hour_perf["count"] >= 5]
            if len(hour_perf) > 1:
                best_hour = hour_perf["mean"].idxmax()
                worst_hour = hour_perf["mean"].idxmin()
                best_acc = hour_perf.loc[best_hour, "mean"]
                worst_acc = hour_perf.loc[worst_hour, "mean"]

                if best_acc - worst_acc > 0.15:
                    patterns.append({
                        "category": "time_of_day",
                        "finding": f"Predictions at {best_hour:02d}:00 UTC are {best_acc*100:.0f}% accurate vs {worst_acc*100:.0f}% at {worst_hour:02d}:00",
                        "impact": "high" if best_acc - worst_acc > 0.25 else "medium",
                        "recommendation": f"Favor predictions around {best_hour:02d}:00 UTC, be cautious at {worst_hour:02d}:00"
                    })

        # 2. Day of week patterns
        if "dayofweek" in df.columns:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            day_perf = df.groupby("dayofweek")["correct"].agg(["mean", "count"])
            day_perf = day_perf[day_perf["count"] >= 5]
            if len(day_perf) > 1:
                best_day = int(day_perf["mean"].idxmax())
                worst_day = int(day_perf["mean"].idxmin())
                best_acc = day_perf.loc[best_day, "mean"]
                worst_acc = day_perf.loc[worst_day, "mean"]

                if best_acc - worst_acc > 0.15:
                    patterns.append({
                        "category": "day_of_week",
                        "finding": f"{day_names[best_day]} predictions are {best_acc*100:.0f}% accurate vs {day_names[worst_day]} at {worst_acc*100:.0f}%",
                        "impact": "high" if best_acc - worst_acc > 0.25 else "medium",
                        "recommendation": f"Increase position size on {day_names[best_day]}, reduce on {day_names[worst_day]}"
                    })

        # 3. Volatility patterns
        if "volatility" in df.columns:
            df["vol_quartile"] = pd.qcut(df["volatility"].dropna(), q=4, labels=["low", "med-low", "med-high", "high"], duplicates="drop")
            vol_perf = df.groupby("vol_quartile", observed=False)["correct"].agg(["mean", "count"])
            vol_perf = vol_perf[vol_perf["count"] >= 5]
            if len(vol_perf) >= 2:
                low_vol = vol_perf.loc["low", "mean"] if "low" in vol_perf.index else None
                high_vol = vol_perf.loc["high", "mean"] if "high" in vol_perf.index else None
                if low_vol and high_vol and low_vol - high_vol > 0.1:
                    patterns.append({
                        "category": "volatility",
                        "finding": f"Low volatility predictions: {low_vol*100:.0f}% accurate. High volatility: {high_vol*100:.0f}%",
                        "impact": "high" if low_vol - high_vol > 0.2 else "medium",
                        "recommendation": "Reduce position size or skip trades during high volatility periods"
                    })

        # 4. RSI extremes
        if "rsi" in df.columns:
            df["rsi_zone"] = pd.cut(df["rsi"], bins=[0, 30, 45, 55, 70, 100], 
                                     labels=["oversold", "low", "mid", "high", "overbought"])
            rsi_perf = df.groupby("rsi_zone", observed=False)["correct"].agg(["mean", "count"])
            rsi_perf = rsi_perf[rsi_perf["count"] >= 5]
            if len(rsi_perf) >= 2:
                best_zone = rsi_perf["mean"].idxmax()
                worst_zone = rsi_perf["mean"].idxmin()
                best_acc = rsi_perf.loc[best_zone, "mean"]
                worst_acc = rsi_perf.loc[worst_zone, "mean"]
                if best_acc - worst_acc > 0.15:
                    patterns.append({
                        "category": "rsi_zone",
                        "finding": f"RSI {best_zone} zone: {best_acc*100:.0f}% accurate vs {worst_zone}: {worst_acc*100:.0f}%",
                        "impact": "medium",
                        "recommendation": f"Favor setups when RSI is {best_zone}, avoid {worst_zone}"
                    })

        # 5. Confidence vs accuracy pattern
        if "confidence" in df.columns:
            high_conf = df[df["confidence"] >= 80]
            low_conf = df[df["confidence"] <= 60]
            if len(high_conf) >= 10 and len(low_conf) >= 10:
                hc_acc = high_conf["correct"].mean()
                lc_acc = low_conf["correct"].mean()
                if hc_acc - lc_acc < 0.1:
                    patterns.append({
                        "category": "confidence_discrimination",
                        "finding": f"High confidence (≥80%) is only {hc_acc*100:.0f}% accurate vs low confidence (≤60%) at {lc_acc*100:.0f}%",
                        "impact": "high",
                        "recommendation": "Confidence scores are not discriminating well — recalibrate or retrain model"
                    })

        # 6. XGBoost probability effectiveness
        if "xgb_probability" in df.columns:
            df["xgb_bin"] = pd.cut(df["xgb_probability"], bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                   labels=["low", "uncertain", "moderate", "high"])
            xgb_perf = df.groupby("xgb_bin", observed=False)["correct"].agg(["mean", "count"])
            xgb_perf = xgb_perf[xgb_perf["count"] >= 5]
            if len(xgb_perf) >= 2:
                best_bin = xgb_perf["mean"].idxmax()
                best_acc = xgb_perf.loc[best_bin, "mean"]
                patterns.append({
                    "category": "xgb_confidence",
                    "finding": f"XGBoost {best_bin} probability predictions are {best_acc*100:.0f}% accurate",
                    "impact": "medium",
                    "recommendation": f"Use XGBoost probability as primary filter — focus on {best_bin} confidence region"
                })

        return sorted(patterns, key=lambda x: 0 if x.get("impact") == "high" else 1)

    # ─────────────────────────────────────────────────────────
    # RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────

    def generate_recommendations(self) -> List[Dict]:
        """
        Generate actionable recommendations based on all analyses.
        """
        recommendations = []

        # 1. Drift-based
        drift = self.detect_drift() if self.has_data(200) else {}
        if drift.get("status") in ["declining", "warning"]:
            recommendations.append({
                "priority": "critical" if drift["status"] == "declining" else "high",
                "category": "model_drift",
                "title": "Retrain Model",
                "description": drift.get("recommendation", ""),
                "action": "Retrain XGBoost with latest 30 days of data"
            })

        # 2. Calibration-based
        cal = self.calibrate_confidence() if self.has_data(50) else {}
        if "bins" in cal:
            overconfident = [b for b in cal["bins"] if b.get("reliability") == "Overconfident"]
            if len(overconfident) >= 2:
                recommendations.append({
                    "priority": "high",
                    "category": "confidence_calibration",
                    "title": "Reduce Displayed Confidence",
                    "description": f"Model is overconfident in {len(overconfident)} confidence bins. Reduce displayed confidence by 10-15%.",
                    "action": "Apply calibration multiplier: displayed_conf = raw_conf * 0.85"
                })

        # 3. Regime-based
        regimes = self.accuracy_by_regime()
        if regimes and "error" not in regimes:
            best_regime = max(regimes.items(), key=lambda x: x[1]["accuracy"])
            worst_regime = min(regimes.items(), key=lambda x: x[1]["accuracy"])
            if best_regime[1]["accuracy"] - worst_regime[1]["accuracy"] > 0.2:
                recommendations.append({
                    "priority": "medium",
                    "category": "regime_optimization",
                    "title": f"Optimize for {best_regime[0]}",
                    "description": f"{best_regime[0]}: {best_regime[1]['accuracy_pct']}% accuracy vs {worst_regime[0]}: {worst_regime[1]['accuracy_pct']}%",
                    "action": f"Increase position size in {best_regime[0]}, reduce in {worst_regime[0]}"
                })

        # 4. Pattern-based
        patterns = self.discover_patterns()
        for p in patterns:
            if p.get("impact") == "high":
                recommendations.append({
                    "priority": "high",
                    "category": f"pattern_{p['category']}",
                    "title": p["finding"][:60],
                    "description": p["finding"],
                    "action": p.get("recommendation", "Review and adjust strategy")
                })

        # 5. Feature-based
        features = self.feature_effectiveness()
        if features and "error" not in features:
            top_feat = list(features.items())[0]
            if top_feat[1].get("significant") and top_feat[1]["importance"] > 0.1:
                recommendations.append({
                    "priority": "low",
                    "category": "feature_engineering",
                    "title": f"Leverage {top_feat[0]}",
                    "description": f"{top_feat[0]} is the most discriminative feature (importance: {top_feat[1]['importance']})",
                    "action": f"Consider adding more {top_feat[0]}-derived features"
                })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

        return recommendations

    # ─────────────────────────────────────────────────────────
    # REPORTS
    # ─────────────────────────────────────────────────────────

    def daily_summary(self) -> Dict[str, Any]:
        """Quick daily summary for dashboard display."""
        if not self.has_data():
            return {"error": "No data available"}

        df = self.df.copy()
        if "timestamp" in df.columns:
            today = pd.Timestamp.utcnow().normalize()
            today_data = df[df["timestamp"] >= today]
        else:
            today_data = df.tail(50)

        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "overall": self.overall_accuracy(),
            "by_window": self.accuracy_by_window(),
            "by_regime": self.accuracy_by_regime(),
            "calibration": self.calibrate_confidence(),
            "drift": self.detect_drift() if self.has_data(200) else {"status": "insufficient_data"},
            "patterns": self.discover_patterns()[:3],
            "recommendations": self.generate_recommendations()[:5],
            "today": {
                "predictions": len(today_data),
                "correct": int(today_data["correct"].sum()) if "correct" in today_data.columns else 0,
                "accuracy": round(today_data["correct"].mean() * 100, 1) if len(today_data) > 0 and "correct" in today_data.columns else 0
            }
        }
        return summary

    def weekly_report(self) -> Dict[str, Any]:
        """Comprehensive weekly report."""
        if not self.has_data(50):
            return {"error": "Need at least 50 predictions for weekly report"}

        df = self.df.dropna(subset=["correct"]).copy()
        total = len(df)
        correct = int(df["correct"].sum())
        accuracy = correct / total if total > 0 else 0

        # Best/worst windows
        by_window = self.accuracy_by_window()
        best_window = max(by_window.items(), key=lambda x: x[1]["accuracy"]) if by_window else ("N/A", {})
        worst_window = min(by_window.items(), key=lambda x: x[1]["accuracy"]) if by_window else ("N/A", {})

        # Best/worst regimes
        by_regime = self.accuracy_by_regime()
        best_regime = max(by_regime.items(), key=lambda x: x[1]["accuracy"]) if by_regime else ("N/A", {})
        worst_regime = min(by_regime.items(), key=lambda x: x[1]["accuracy"]) if by_regime else ("N/A", {})

        # Confidence reliability
        cal = self.calibrate_confidence()
        reliable_range = "N/A"
        if "bins" in cal:
            best_bin = max(cal["bins"], key=lambda x: x["actual_accuracy"])
            reliable_range = best_bin["bin_label"]

        report = {
            "report_type": "weekly",
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "from": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
                "to": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None,
            },
            "summary": {
                "total_predictions": total,
                "correct": correct,
                "accuracy_pct": round(accuracy * 100, 1),
                "incorrect": total - correct
            },
            "by_window": by_window,
            "by_regime": by_regime,
            "best_window": {"name": best_window[0], **best_window[1]},
            "worst_window": {"name": worst_window[0], **worst_window[1]},
            "best_regime": {"name": best_regime[0], **best_regime[1]},
            "worst_regime": {"name": worst_regime[0], **worst_regime[1]},
            "most_reliable_confidence": reliable_range,
            "calibration": cal,
            "drift": self.detect_drift() if self.has_data(200) else {"status": "insufficient_data"},
            "patterns": self.discover_patterns(),
            "recommendations": self.generate_recommendations(),
            "feature_effectiveness": self.feature_effectiveness()
        }

        return report

    def save_report(self, report: Dict, filename: Optional[str] = None):
        """Save report to artifacts directory."""
        if filename is None:
            filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.artifacts_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Report saved to: {path}")
        return path


if __name__ == "__main__":
    # Demo with synthetic data
    engine = LearningEngine(log_dir="logs")

    # Create sample data for testing
    np.random.seed(42)
    n = 500
    sample_data = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.utcnow(), periods=n, freq="H"),
        "current_price": np.random.uniform(60000, 70000, n),
        "prediction": np.random.choice(["ABOVE", "BELOW"], n),
        "confidence": np.random.uniform(50, 95, n),
        "xgb_probability": np.random.uniform(0.1, 0.9, n),
        "trust_score": np.random.uniform(0.3, 0.95, n),
        "rsi": np.random.uniform(20, 85, n),
        "atr": np.random.uniform(0.1, 1.5, n),
        "ema_distance": np.random.uniform(-0.5, 0.5, n),
        "volatility": np.random.uniform(0.1, 0.8, n),
        "volume": np.random.uniform(100000, 500000, n),
        "market_regime": np.random.choice(
            ["uptrend_low_vol", "uptrend_high_vol", "ranging", "downtrend"], n,
            p=[0.3, 0.2, 0.3, 0.2]
        ),
        "prediction_window": np.random.choice(["15m", "1h", "4h"], n, p=[0.5, 0.3, 0.2]),
        "correct": np.random.choice([True, False], n, p=[0.65, 0.35]),
        "hour": np.random.randint(0, 24, n),
        "dayofweek": np.random.randint(0, 7, n)
    })

    # Make regime affect correctness for realistic patterns
    for idx, row in sample_data.iterrows():
        if row["market_regime"] == "uptrend_low_vol":
            sample_data.at[idx, "correct"] = np.random.choice([True, False], p=[0.82, 0.18])
        elif row["market_regime"] == "downtrend":
            sample_data.at[idx, "correct"] = np.random.choice([True, False], p=[0.55, 0.45])

        if row["hour"] in [2, 3, 4]:
            sample_data.at[idx, "correct"] = np.random.choice([True, False], p=[0.45, 0.55])

    engine.df = sample_data

    print("=" * 60)
    print("PHASE 4 — LEARNING ENGINE DEMO")
    print("=" * 60)

    print("\n📊 OVERALL ACCURACY")
    print(json.dumps(engine.overall_accuracy(), indent=2))

    print("\n⏱️  ACCURACY BY WINDOW")
    print(json.dumps(engine.accuracy_by_window(), indent=2))

    print("\n🌊 ACCURACY BY REGIME")
    print(json.dumps(engine.accuracy_by_regime(), indent=2))

    print("\n🎯 CONFIDENCE CALIBRATION")
    print(json.dumps(engine.calibrate_confidence(), indent=2))

    print("\n📉 DRIFT DETECTION")
    print(json.dumps(engine.detect_drift(), indent=2))

    print("\n🔍 PATTERNS DISCOVERED")
    for p in engine.discover_patterns():
        print(f"  • [{p['category']}] {p['finding']}")

    print("\n💡 TOP RECOMMENDATIONS")
    for i, rec in enumerate(engine.generate_recommendations()[:5], 1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
        print(f"     → {rec['action']}")

    print("\n📄 GENERATING WEEKLY REPORT...")
    report = engine.weekly_report()
    engine.save_report(report, "weekly_report_demo.json")