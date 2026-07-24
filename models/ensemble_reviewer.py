"""
Ensemble + AI Reviewer layer.

Combines predictions from:
  1. Your existing bot (technical analysis model)
  2. XGBoost model (trained on historical labels)

When both agree → high confidence signal
When they disagree → "AI Reviewer" flags as uncertain/skip

Usage:
    from ensemble_reviewer import EnsembleReviewer

    reviewer = EnsembleReviewer()
    result = reviewer.predict(features, existing_bot_prediction)

    # result = {
    #     "final_verdict": "UP" | "DOWN" | "UNCERTAIN",
    #     "confidence": 0.85,
    #     "existing_bot": {"prediction": "UP", "confidence": 0.81},
    #     "xgboost": {"prediction": "UP", "confidence": 0.76},
    #     "agreement": True,
    #     "reason": "Both models agree with strong confidence"
    # }
"""

import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from indicators import FEATURE_COLUMNS


class EnsembleReviewer:
    """Combines existing bot + XGBoost predictions with conflict resolution."""

    # Thresholds for agreement/disagreement
    AGREEMENT_THRESHOLD = 0.05  # prob difference < 5% = agreement
    STRONG_CONFIDENCE = 0.60    # both models > 60% = strong signal
    MIN_CONFIDENCE = 0.53     # below this = uncertain regardless

    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")

        # Load XGBoost model
        xgb_path = os.path.join(artifacts_dir, "xgboost_model.pkl")
        cal_path = os.path.join(artifacts_dir, "xgboost_calibrator.pkl")

        if not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"XGBoost model not found at {xgb_path}. "
                "Run phase2_xgb.py first to train the model."
            )

        with open(xgb_path, "rb") as f:
            self.xgb_model = pickle.load(f)
        with open(cal_path, "rb") as f:
            self.calibrator = pickle.load(f)

    def _get_xgb_prediction(self, features_dict):
        """Get XGBoost prediction from feature dict."""
        import pandas as pd

        # Convert dict to array in correct order
        X = np.array([[features_dict.get(col, 0.0) for col in FEATURE_COLUMNS]])

        raw_prob = self.xgb_model.predict_proba(X)[0, 1]
        prob_up = float(self.calibrator.transform([raw_prob])[0])
        prob_up = min(max(prob_up, 0.01), 0.99)

        prediction = "UP" if prob_up >= 0.5 else "DOWN"

        return {
            "prediction": prediction,
            "confidence": prob_up if prediction == "UP" else 1 - prob_up,
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
        }

    def predict(self, features_dict, existing_bot_prediction, existing_bot_confidence=None):
        """
        Combine existing bot + XGBoost predictions.

        Args:
            features_dict: dict of current feature values (RSI, EMA, etc.)
            existing_bot_prediction: "UP" or "DOWN"
            existing_bot_confidence: float 0-1 (optional)

        Returns:
            dict with final_verdict, confidence, and reasoning
        """
        # Get XGBoost prediction
        xgb = self._get_xgb_prediction(features_dict)

        # Normalize existing bot
        bot = {
            "prediction": existing_bot_prediction.upper(),
            "confidence": existing_bot_confidence or 0.55,
            "prob_up": existing_bot_confidence if existing_bot_prediction.upper() == "UP" else 1 - (existing_bot_confidence or 0.55),
        }

        # Check agreement
        agree = bot["prediction"] == xgb["prediction"]
        prob_diff = abs(bot["prob_up"] - xgb["prob_up"])
        near_agree = prob_diff < self.AGREEMENT_THRESHOLD

        # Determine final verdict
        if agree and bot["confidence"] >= self.STRONG_CONFIDENCE and xgb["confidence"] >= self.STRONG_CONFIDENCE:
            final_verdict = bot["prediction"]
            confidence = (bot["confidence"] + xgb["confidence"]) / 2
            reason = f"Both models strongly agree: {bot['prediction']} (bot: {bot['confidence']:.1%}, xgb: {xgb['confidence']:.1%})"

        elif agree and bot["confidence"] >= self.MIN_CONFIDENCE and xgb["confidence"] >= self.MIN_CONFIDENCE:
            final_verdict = bot["prediction"]
            confidence = (bot["confidence"] + xgb["confidence"]) / 2
            reason = f"Both models agree: {bot['prediction']} (bot: {bot['confidence']:.1%}, xgb: {xgb['confidence']:.1%})"

        elif agree:
            final_verdict = bot["prediction"]
            confidence = max(bot["confidence"], xgb["confidence"])
            reason = f"Weak agreement: {bot['prediction']} (low confidence from one model)"

        elif near_agree:
            final_verdict = "UNCERTAIN"
            confidence = 0.5
            reason = f"Models disagree slightly — too close to call (bot: {bot['prediction']} {bot['confidence']:.1%}, xgb: {xgb['prediction']} {xgb['confidence']:.1%})"

        else:
            final_verdict = "SKIP"
            confidence = 0.5
            reason = f"Models DISAGREE — skip trade (bot: {bot['prediction']} {bot['confidence']:.1%}, xgb: {xgb['prediction']} {xgb['confidence']:.1%})"

        return {
            "final_verdict": final_verdict,
            "confidence": confidence,
            "existing_bot": {
                "prediction": bot["prediction"],
                "confidence": bot["confidence"],
            },
            "xgboost": {
                "prediction": xgb["prediction"],
                "confidence": xgb["confidence"],
                "prob_up": xgb["prob_up"],
            },
            "agreement": agree,
            "prob_diff": prob_diff,
            "reason": reason,
        }

    def predict_from_live(self, get_existing_bot_fn):
        """
        Convenience method: pass a function that returns (prediction, confidence, features_dict).

        Example:
            def get_bot_prediction():
                # your existing bot logic
                return "UP", 0.81, {"rsi_14": 62.4, "ema_dist": 0.001, ...}

            result = reviewer.predict_from_live(get_bot_prediction)
        """
        bot_pred, bot_conf, features = get_existing_bot_fn()
        return self.predict(features, bot_pred, bot_conf)


# Simple test
if __name__ == "__main__":
    print("EnsembleReviewer loaded.")
    print("Run phase2_xgb.py first to train the XGBoost model.")
    print("Then use:")
    print("  reviewer = EnsembleReviewer()")
    print("  result = reviewer.predict(features_dict, 'UP', 0.81)")