"""
Batch Robustness Benchmark Evaluation Script:
Evaluates detector performance across 100 held-out test images (50 Real, 50 AI) under:
- JPEG Compression (Q=90, 75, 50, 30)
- Resizing / Downscaling (75%, 50%, 35%)
- Combined Social Media Re-compression (50% + Q=50)

Outputs summary metrics and saves data/robustness_benchmark_results.json.
"""
import io
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from app.fusion.classifier import FusionClassifier
from app.evaluation.robustness import (
    apply_jpeg_compression,
    apply_resize_degradation,
    apply_social_media_pipeline,
    extract_features_for_image,
)

PARQUET_PATH = "data/genimage_cache/data/train-00000-of-00014.parquet"
MODEL_PATH = "models/genimage_fusion_model.joblib"
CENTROIDS_PATH = "models/genimage_centroids.joblib"
OUT_JSON = "data/robustness_benchmark_results.json"


def main(num_samples: int = 100):
    print("Loading test dataset and models for Robustness Benchmark...")
    table = pq.read_table(PARQUET_PATH)
    df_meta = table.select(["label", "generator"]).to_pandas()

    classifier = FusionClassifier(MODEL_PATH)
    centroids = joblib.load(CENTROIDS_PATH)

    # Sample 50 Real and 50 AI
    np.random.seed(99)
    real_indices = df_meta[df_meta["label"] == 0].sample(n=num_samples // 2, random_state=99).index.tolist()
    fake_indices = df_meta[df_meta["label"] == 1].sample(n=num_samples // 2, random_state=99).index.tolist()
    all_indices = real_indices + fake_indices

    print(f"Loaded {len(all_indices)} images ({len(real_indices)} Real, {len(fake_indices)} AI).")

    perturbation_conditions = [
        {"key": "clean", "name": "Clean (Unperturbed)", "fn": lambda img: img},
        {"key": "jpeg_90", "name": "JPEG (Quality 90)", "fn": lambda img: apply_jpeg_compression(img, 90)},
        {"key": "jpeg_75", "name": "JPEG (Quality 75)", "fn": lambda img: apply_jpeg_compression(img, 75)},
        {"key": "jpeg_50", "name": "JPEG (Quality 50)", "fn": lambda img: apply_jpeg_compression(img, 50)},
        {"key": "jpeg_30", "name": "JPEG (Quality 30)", "fn": lambda img: apply_jpeg_compression(img, 30)},
        {"key": "resize_75", "name": "Resize (75% Scale)", "fn": lambda img: apply_resize_degradation(img, 0.75)},
        {"key": "resize_50", "name": "Resize (50% Scale)", "fn": lambda img: apply_resize_degradation(img, 0.50)},
        {"key": "resize_35", "name": "Resize (35% Scale)", "fn": lambda img: apply_resize_degradation(img, 0.35)},
        {"key": "social_recompress", "name": "Social Media (50% + Q50)", "fn": lambda img: apply_social_media_pipeline(img, 0.50, 50)},
    ]

    labels = [int(df_meta.loc[i, "label"]) for i in all_indices]

    results_by_condition = {}

    for cond in perturbation_conditions:
        c_key = cond["key"]
        c_name = cond["name"]
        fn = cond["fn"]
        print(f"\nEvaluating Condition: {c_name}...")

        preds = []
        probs = []

        for idx in tqdm(all_indices, desc=c_name):
            cell = table["image"][int(idx)].as_py()
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            
            # Apply perturbation
            img_pert = fn(img)

            feats = extract_features_for_image(img_pert, centroids)
            pred = classifier.predict(feats, abstain_margin=0.0)

            v_label = 1 if pred["verdict"] == "AI-generated" else 0
            preds.append(v_label)
            probs.append(pred.get("calibrated_prob_ai", 0.5))

        acc = float(accuracy_score(labels, preds))
        auc = float(roc_auc_score(labels, probs))

        results_by_condition[c_key] = {
            "condition_name": c_name,
            "accuracy": round(acc, 4),
            "auc_roc": round(auc, 4),
        }

    clean_acc = results_by_condition["clean"]["accuracy"]
    for k, v in results_by_condition.items():
        v["retention_rate"] = round(v["accuracy"] / (clean_acc + 1e-8), 3)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results_by_condition, f, indent=2)

    print("\n================== ROBUSTNESS BENCHMARK RESULTS ==================")
    print(f"{'Condition':32s} | {'Accuracy':8s} | {'AUC-ROC':8s} | {'Retention':8s}")
    print("-" * 65)
    for k, v in results_by_condition.items():
        print(f"{v['condition_name']:32s} | {v['accuracy']*100:6.1f}%  | {v['auc_roc']:6.3f}   | {v['retention_rate']*100:6.1f}%")

    print(f"\nSaved benchmark results to {OUT_JSON}")


if __name__ == "__main__":
    main(num_samples=100)
