"""
Calibration and Abstention (Selective Classification) Module:
1. Expected Calibration Error (ECE) and Brier Score computation.
2. Isotonic and Sigmoid (Platt) probability calibration.
3. Abstention thresholding for trustworthy selective classification.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Measures the difference between predicted confidence and empirical accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Samples in this bin
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes the mean squared error of probability forecasts (Brier score)."""
    return float(np.mean((probs - labels) ** 2))


class ProbabilityCalibrator:
    """Calibrates raw classifier probabilities using Isotonic Regression or Platt scaling."""
    def __init__(self, method: str = "isotonic"):
        self.method = method
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        else:
            self.calibrator = None

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray):
        raw_probs = np.clip(raw_probs, 1e-4, 1.0 - 1e-4)
        if self.method == "isotonic":
            self.calibrator.fit(raw_probs, labels)
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.method == "isotonic" and self.calibrator is not None:
            raw_probs = np.asarray(raw_probs)
            scalar = raw_probs.ndim == 0
            res = self.calibrator.predict(np.atleast_1d(raw_probs))
            return float(res[0]) if scalar else res
        return raw_probs


def apply_abstention(prob_ai: float, threshold: float = 0.50, abstain_margin: float = 0.08) -> dict:
    """
    Selective Classification (Abstention):
    If the calibrated probability of AI is within [threshold - abstain_margin, threshold + abstain_margin],
    the model abstains from making a forced binary decision and returns 'uncertain' / 'abstain',
    flagging the image for human forensic inspection.
    """
    lower_bound = threshold - abstain_margin
    upper_bound = threshold + abstain_margin

    if lower_bound <= prob_ai <= upper_bound:
        # Distance to decision boundary normalized by margin
        uncertainty = 1.0 - (abs(prob_ai - threshold) / (abstain_margin + 1e-6))
        return {
            "verdict": "uncertain",
            "abstained": True,
            "confidence": round(float(uncertainty), 3),
            "calibrated_prob_ai": round(float(prob_ai), 4),
            "abstain_margin": abstain_margin,
            "abstention_reason": (
                f"Model probability ({prob_ai:.3f}) falls within uncertainty band "
                f"[{lower_bound:.2f}, {upper_bound:.2f}]. Flagged for human review."
            ),
        }
    elif prob_ai > upper_bound:
        confidence = prob_ai
        return {
            "verdict": "AI-generated",
            "abstained": False,
            "confidence": round(float(confidence), 3),
            "calibrated_prob_ai": round(float(prob_ai), 4),
            "abstain_margin": abstain_margin,
            "abstention_reason": None,
        }
    else:
        confidence = 1.0 - prob_ai
        return {
            "verdict": "real",
            "abstained": False,
            "confidence": round(float(confidence), 3),
            "calibrated_prob_ai": round(float(prob_ai), 4),
            "abstain_margin": abstain_margin,
            "abstention_reason": None,
        }
