"""
Inference-time wrapper around the trained fusion classifier.
Loads the model saved by train_fusion.py and produces a verdict, confidence,
and a structured evidence dict ready for the explanation module.
"""
import joblib
import numpy as np


class FusionClassifier:
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler", None)
        self.threshold = bundle.get("threshold", 0.56)
        self.feature_columns = bundle["feature_columns"]
        self.explainer = bundle["explainer"]

    def predict(self, features: dict) -> dict:
        """
        features: dict containing at least all keys in self.feature_columns
        (extra keys, e.g. gradcam_region or temporal_score, are carried
        through into the evidence dict but not used by the classifier itself
        unless you add them to FEATURE_COLUMNS in train_fusion.py).
        """
        x = np.array([[features[col] for col in self.feature_columns]])
        x_scaled = self.scaler.transform(x) if self.scaler is not None else x

        prob_ai = float(self.model.predict_proba(x_scaled)[0][1])
        verdict_label = 1 if prob_ai >= self.threshold else 0
        confidence = prob_ai if verdict_label == 1 else (1.0 - prob_ai)
        shap_values = self.explainer.shap_values(x_scaled)[0]

        evidence = {
            "verdict": "AI-generated" if verdict_label == 1 else "real",
            "confidence": round(confidence, 3),
        }
        for i, col in enumerate(self.feature_columns):
            evidence[col] = {
                "value": round(float(features[col]), 4),
                "contribution": round(float(shap_values[i]), 4),
            }

        # carry through any extra evidence fields not used by the classifier
        # (e.g. gradcam_region, temporal_score) so the explanation module can use them
        for key, value in features.items():
            if key not in self.feature_columns:
                evidence[key] = value

        ranked = sorted(
            self.feature_columns, key=lambda c: abs(evidence[c]["contribution"]), reverse=True
        )
        evidence["top_evidence"] = ranked[:2]

        return evidence


if __name__ == "__main__":
    clf = FusionClassifier("models/fusion_model_test.joblib")
    example_features = {
        "fft_anomaly_score": 0.79,
        "dct_anomaly_score": 0.31,
        "clip_max_generator_similarity": 0.81,
    }
    print(clf.predict(example_features))
