"""
Inference-time wrapper around the trained fusion classifier.
Loads the model saved by train_fusion.py and produces a verdict, confidence,
calibrated probabilities, optional abstention (selective classification),
and a structured evidence dict ready for the explanation module.
"""
import joblib
import numpy as np
from app.fusion.calibration import apply_abstention


class FusionClassifier:
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler", None)
        self.threshold = bundle.get("threshold", 0.50)
        self.feature_columns = bundle["feature_columns"]
        self.explainer = bundle["explainer"]
        self.calibrator = bundle.get("calibrator", None)

    def predict(self, features: dict, abstain_margin: float = 0.0) -> dict:
        """
        features: dict containing at least all keys in self.feature_columns.
        abstain_margin: margin around threshold for selective classification (abstention).
                        If > 0, returns verdict='uncertain' when |prob - threshold| <= abstain_margin.
        """
        x = np.array([[features[col] for col in self.feature_columns]])
        x_scaled = self.scaler.transform(x) if self.scaler is not None else x

        raw_prob_ai = float(self.model.predict_proba(x_scaled)[0][1])

        # Apply probability calibrator if bundled, else raw_prob_ai
        if self.calibrator is not None:
            calibrated_prob = float(self.calibrator.predict(raw_prob_ai))
        else:
            calibrated_prob = raw_prob_ai

        # Selective classification (Abstention)
        decision = apply_abstention(calibrated_prob, threshold=self.threshold, abstain_margin=abstain_margin)

        shap_values = self.explainer.shap_values(x_scaled)[0]

        evidence = {
            "verdict": decision["verdict"],
            "abstained": decision["abstained"],
            "confidence": decision["confidence"],
            "calibrated_prob_ai": decision["calibrated_prob_ai"],
            "raw_prob_ai": round(raw_prob_ai, 4),
            "abstain_margin": abstain_margin,
            "abstention_reason": decision["abstention_reason"],
        }

        for i, col in enumerate(self.feature_columns):
            evidence[col] = {
                "value": round(float(features[col]), 4),
                "contribution": round(float(shap_values[i]), 4),
            }

        # Carry through any extra evidence fields not directly in feature_columns
        for key, value in features.items():
            if key not in self.feature_columns:
                evidence[key] = value

        ranked = sorted(
            self.feature_columns, key=lambda c: abs(evidence[c]["contribution"]), reverse=True
        )
        evidence["top_evidence"] = ranked[:2]

        return evidence


if __name__ == "__main__":
    clf = FusionClassifier("models/genimage_fusion_model.joblib")
    example_features = {
        "clip_semantic_score": 0.82,
        "clip_contrast_margin": 0.045,
        "clip_max_generator_similarity": 0.74,
        "fft_anomaly_score": 0.41,
        "dct_anomaly_score": 0.07,
    }
    print(clf.predict(example_features, abstain_margin=0.08))
