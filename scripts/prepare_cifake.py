"""
Starter script for Step 4 in the README: walks the downloaded CIFAKE train
folder, extracts frequency + CLIP features for each image, and writes them
to a CSV ready for train_fusion.py.

Adjust N_PER_CLASS while developing (e.g. 300) so this runs in a minute or
two; increase it once you're confident the pipeline works, before your final
training run.

Usage:
    PYTHONPATH=. python3 scripts/prepare_cifake.py
"""
import glob
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from app.features.frequency import extract_frequency_features
from app.features.clip_features import extract_clip_features, build_generator_centroids

DATA_DIR = "data/cifake/train"
N_PER_CLASS = 100  # lower this while developing, raise it for the final run
OUT_CSV = "data/cifake_features.csv"


def build_centroids_from_fake_samples(fake_dir: str, n_samples: int = 20) -> dict:
    sample_paths = sorted(glob.glob(os.path.join(fake_dir, "*")))[:n_samples]
    sample_images = [Image.open(p).convert("RGB") for p in sample_paths]
    return build_generator_centroids({"stable_diffusion_v1_4": sample_images})


def extract_row(image_path: str, label: int, centroids: dict) -> dict:
    pil_image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) if False else None

    row = {}
    # frequency features need a BGR numpy array
    import numpy as np
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    row.update(extract_frequency_features(image_bgr))

    clip_feats = extract_clip_features(pil_image, centroids)
    row["clip_max_generator_similarity"] = clip_feats["clip_max_generator_similarity"]

    row["label"] = label
    return row


def main():
    real_dir = os.path.join(DATA_DIR, "REAL")
    fake_dir = os.path.join(DATA_DIR, "FAKE")

    print("Building generator centroids from a handful of FAKE samples...")
    centroids = build_centroids_from_fake_samples(fake_dir)
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(centroids, "models/generator_centroids.joblib")
    print("Saved generator centroids to models/generator_centroids.joblib")

    real_paths = sorted(glob.glob(os.path.join(real_dir, "*")))[:N_PER_CLASS]
    fake_paths = sorted(glob.glob(os.path.join(fake_dir, "*")))[:N_PER_CLASS]

    rows = []
    for path in tqdm(real_paths, desc="real images"):
        rows.append(extract_row(path, label=0, centroids=centroids))
    for path in tqdm(fake_paths, desc="fake images"):
        rows.append(extract_row(path, label=1, centroids=centroids))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
