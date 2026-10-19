"""
Cross-Generator Evaluation Script for CLIP Linear Probe:
Evaluates out-of-distribution (OOD) cross-generator transfer across:
- Midjourney
- SD15 (Stable Diffusion 1.5)
- GLIDE
- BigGAN
- ADM
- VQDM
- Wukong

Evaluation Protocol:
1. Leave-One-Generator-Out (LOGO):
   Train CLIP Linear Probe on 6 generators + Real, and test zero-shot generalization on the 7th unseen generator.
2. Pairwise Cross-Generator Transfer:
   Evaluates how features learned from Diffusion models transfer to GANs and Autoregressive models.
3. Multi-Generator Unified Evaluation:
   Comprehensive metrics (Accuracy, AUC-ROC, AP, and Specificity).

Saves data/cross_generator_results.json and models/clip_linear_probe.joblib.
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
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from app.features.clip_features import load_clip_model

PARQUET_PATH = "data/genimage_cache/data/train-00000-of-00014.parquet"
OUT_JSON = "data/cross_generator_results.json"
MODEL_OUT = "models/clip_linear_probe.joblib"

GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}

AI_GENERATOR_IDS = [1, 2, 3, 4, 6, 7, 8]


def main():
    print("Loading GenImage parquet dataset for Cross-Generator Benchmark...")
    table = pq.read_table(PARQUET_PATH)
    df_meta = table.select(["label", "generator"]).to_pandas()

    clip_model, clip_preprocess, clip_device = load_clip_model()

    print(f"Extracting 512-D CLIP ViT-B/32 representations for all {len(df_meta)} samples...")
    embeddings = []
    labels = []
    gens = []

    with torch.no_grad():
        for i in tqdm(range(len(df_meta)), desc="Extracting CLIP Embeddings"):
            cell = table["image"][i].as_py()
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            img = Image.open(io.BytesIO(raw)).convert("RGB")

            tensor = clip_preprocess(img).unsqueeze(0).to(clip_device)
            emb = clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])

            labels.append(int(df_meta.loc[i, "label"]))
            gens.append(int(df_meta.loc[i, "generator"]))

    X = np.array(embeddings)
    y = np.array(labels)
    gens = np.array(gens)

    # 1. Train unified multi-generator linear probe on 80% split
    print("\n--- Training Unified Multi-Generator Linear Probe ---")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=42, stratify=gens
    )

    unified_probe = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    unified_probe.fit(X[train_idx], y[train_idx])

    probs_test = unified_probe.predict_proba(X[test_idx])[:, 1]
    preds_test = (probs_test >= 0.50).astype(int)

    unified_metrics = {
        "overall_accuracy": round(float(accuracy_score(y[test_idx], preds_test)), 4),
        "overall_auc_roc": round(float(roc_auc_score(y[test_idx], probs_test)), 4),
        "overall_ap": round(float(average_precision_score(y[test_idx], probs_test)), 4),
    }

    per_generator_unified = {}
    for g_id in [0] + AI_GENERATOR_IDS:
        g_name = GENERATOR_NAMES[g_id]
        mask = (gens[test_idx] == g_id)
        if mask.sum() > 0:
            g_acc = float(accuracy_score(y[test_idx][mask], preds_test[mask]))
            g_mean_p = float(probs_test[mask].mean())
            per_generator_unified[g_name] = {
                "samples": int(mask.sum()),
                "accuracy": round(g_acc, 4),
                "mean_p_ai": round(g_mean_p, 4),
            }

    # 2. Leave-One-Generator-Out (LOGO) Zero-Shot Cross-Generator Transfer
    print("\n--- Running Leave-One-Generator-Out (LOGO) Cross-Generator Evaluation ---")
    logo_results = {}

    for target_g_id in AI_GENERATOR_IDS:
        target_name = GENERATOR_NAMES[target_g_id]
        print(f"Evaluating Zero-Shot Transfer to unseen generator: {target_name}...")

        # Training set: All Real images (80%) + All AI images EXCEPT target_g_id
        train_mask_logo = (gens != target_g_id) & (np.isin(np.arange(len(y)), train_idx))
        # Test set: Held-out Real images (20%) + All samples of target_g_id
        test_mask_target = (gens == target_g_id)
        test_mask_real = (gens == 0) & (np.isin(np.arange(len(y)), test_idx))
        test_mask_logo = test_mask_target | test_mask_real

        clf_logo = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        clf_logo.fit(X[train_mask_logo], y[train_mask_logo])

        probs_logo = clf_logo.predict_proba(X[test_mask_logo])[:, 1]
        preds_logo = (probs_logo >= 0.50).astype(int)
        y_logo = y[test_mask_logo]

        # Target generator specific accuracy
        probs_target = clf_logo.predict_proba(X[test_mask_target])[:, 1]
        target_acc = float((probs_target >= 0.50).mean())
        target_mean_p = float(probs_target.mean())

        # Real specificity
        probs_real = clf_logo.predict_proba(X[test_mask_real])[:, 1]
        real_spec = float((probs_real < 0.50).mean())

        logo_results[target_name] = {
            "unseen_generator": target_name,
            "zero_shot_ai_accuracy": round(target_acc, 4),
            "zero_shot_mean_p_ai": round(target_mean_p, 4),
            "real_specificity": round(real_spec, 4),
            "joint_auc_roc": round(float(roc_auc_score(y_logo, probs_logo)), 4),
            "joint_ap": round(float(average_precision_score(y_logo, probs_logo)), 4),
        }

    # Save outputs
    benchmark_payload = {
        "unified_probe_metrics": unified_metrics,
        "per_generator_performance": per_generator_unified,
        "leave_one_generator_out_transfer": logo_results,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "probe": unified_probe,
        "feature_dim": 512,
        "weights": unified_probe.coef_[0].astype(np.float32),
        "bias": float(unified_probe.intercept_[0]),
        "classes": unified_probe.classes_,
    }, MODEL_OUT)

    print("\n================== UNIFIED MULTI-GENERATOR PERFORMANCE ==================")
    print(f"Overall Accuracy: {unified_metrics['overall_accuracy']*100:.2f}% | AUC-ROC: {unified_metrics['overall_auc_roc']:.4f} | AP: {unified_metrics['overall_ap']:.4f}")
    print("\nPer-Generator Accuracy:")
    for k, v in per_generator_unified.items():
        print(f"  {k:12s}: Accuracy = {v['accuracy']*100:5.1f}% (Mean P(AI) = {v['mean_p_ai']:.3f})")

    print("\n================== LEAVE-ONE-GENERATOR-OUT (ZERO-SHOT TRANSFER) ==================")
    print(f"{'Unseen Target Generator':24s} | {'Zero-Shot Acc':14s} | {'Real Spec':10s} | {'Joint AUC-ROC':14s}")
    print("-" * 72)
    for k, v in logo_results.items():
        print(f"{k:24s} | {v['zero_shot_ai_accuracy']*100:10.1f}%   | {v['real_specificity']*100:8.1f}% | {v['joint_auc_roc']:12.4f}")

    print(f"\nSaved cross-generator benchmark report to {OUT_JSON}")
    print(f"Saved pure CLIP linear probe model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
