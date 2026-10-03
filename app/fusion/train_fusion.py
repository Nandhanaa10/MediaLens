"""
Trains the fusion classifier on a feature table (one row per image, one column
per feature, plus a label column). This is a small, interpretable model on
purpose - see the project's research gap for why.

Expected input CSV columns (adjust FEATURE_COLUMNS below if you add/remove features):
    fft_anomaly_score, dct_anomaly_score, clip_max_generator_similarity, label
where label is 1 for AI-generated, 0 for real.

Usage:
    python app/fusion/train_fusion.py --csv data/cifake_features.csv --out models/fusion_model.joblib
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURE_COLUMNS = [
    "fft_anomaly_score",
    "dct_anomaly_score",
    "clip_max_generator_similarity",
]


def train(csv_path: str, model_out_path: str, threshold: float = 0.56):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "calibrated_threshold": threshold,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc_roc": roc_auc_score(y_test, probs),
    }
    print("Held-out test metrics:")
    print(json.dumps(metrics, indent=2))

    # SHAP explainer on scaled features
    explainer = shap.LinearExplainer(model, X_train_scaled)
    sample_shap_values = explainer.shap_values(X_test_scaled[:3])
    print("\nSample SHAP values for first 3 test rows (one row per prediction):")
    print(pd.DataFrame(sample_shap_values, columns=FEATURE_COLUMNS))

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
            "explainer": explainer,
            "threshold": threshold,
        },
        model_out_path,
    )
    print(f"\nSaved model + scaler + explainer to {model_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to feature CSV")
    parser.add_argument("--out", required=True, help="Where to save the trained model")
    args = parser.parse_args()
    train(args.csv, args.out)
