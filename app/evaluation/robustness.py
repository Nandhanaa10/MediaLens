"""
Robustness Testing Module:
Simulates real-world media post-processing degradations:
1. JPEG Compression (varying quality factors Q=90, 75, 50, 30, 15)
2. Spatial Downscaling & Upscaling (resizing by scale factors 0.75, 0.50, 0.35, 0.25)
3. Social Media Simulator (Combined Downscale 50% + JPEG Q=50)

Evaluates prediction stability, confidence drift, and accuracy retention under distribution shift.
"""
import io
import cv2
import numpy as np
from PIL import Image

from app.features.frequency import extract_frequency_features
from app.features.clip_features import extract_clip_features


def apply_jpeg_compression(pil_img: Image.Image, quality: int = 75) -> Image.Image:
    """Simulates lossy JPEG compression at specified quality level (1-100)."""
    buffer = io.BytesIO()
    # Save as JPEG into buffer
    pil_img.save(buffer, format="JPEG", quality=int(np.clip(quality, 1, 100)))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_resize_degradation(pil_img: Image.Image, scale_factor: float = 0.50) -> Image.Image:
    """
    Simulates spatial downsampling (e.g. platform thumbnail or preview generation),
    followed by upsampling back to original resolution.
    """
    orig_w, orig_h = pil_img.size
    new_w = max(16, int(orig_w * scale_factor))
    new_h = max(16, int(orig_h * scale_factor))

    # Downscale and upscale back
    downscaled = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    restored = downscaled.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    return restored


def apply_social_media_pipeline(pil_img: Image.Image, scale: float = 0.50, quality: int = 50) -> Image.Image:
    """Simulates social messaging platform compression (e.g. WhatsApp, Twitter, Discord)."""
    resized = apply_resize_degradation(pil_img, scale_factor=scale)
    return apply_jpeg_compression(resized, quality=quality)


def extract_features_for_image(pil_img: Image.Image, centroids: dict) -> dict:
    """Extracts standardized frequency and semantic features for a PIL image."""
    # Frequency domain standardization to 256x256
    freq_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
    bgr = cv2.cvtColor(np.array(freq_img), cv2.COLOR_RGB2BGR)
    
    feats = {}
    feats.update(extract_frequency_features(bgr))

    if centroids:
        clip_feats = extract_clip_features(pil_img, centroids)
        feats["clip_semantic_score"] = clip_feats["clip_semantic_score"]
        feats["clip_contrast_margin"] = clip_feats["clip_contrast_margin"]
        feats["clip_max_generator_similarity"] = clip_feats["clip_max_generator_similarity"]
        if clip_feats.get("clip_best_match_generator"):
            feats["clip_best_match_generator"] = clip_feats["clip_best_match_generator"]
    else:
        feats["clip_semantic_score"] = 0.50
        feats["clip_contrast_margin"] = 0.0
        feats["clip_max_generator_similarity"] = 0.68

    return feats


def stress_test_single_image(pil_img: Image.Image, classifier, centroids: dict, abstain_margin: float = 0.0) -> dict:
    """
    Runs a full degradation stress-test battery on a single image.
    Evaluates how the classifier's verdict and confidence shift under:
    - Clean (Original)
    - JPEG Quality: 90, 75, 50, 30
    - Resizing: 75%, 50%, 35%
    - Social Media Re-compression (50% scale + JPEG Q=50)
    """
    perturbation_specs = [
        {"name": "Original (Clean)", "type": "clean", "param": 100},
        {"name": "JPEG (Quality 90)", "type": "jpeg", "param": 90},
        {"name": "JPEG (Quality 75)", "type": "jpeg", "param": 75},
        {"name": "JPEG (Quality 50)", "type": "jpeg", "param": 50},
        {"name": "JPEG (Quality 30)", "type": "jpeg", "param": 30},
        {"name": "Resize (75% Scale)", "type": "resize", "param": 0.75},
        {"name": "Resize (50% Scale)", "type": "resize", "param": 0.50},
        {"name": "Resize (35% Scale)", "type": "resize", "param": 0.35},
        {"name": "Social Media (50% + Q50)", "type": "social", "param": 50},
    ]

    results = []
    baseline_verdict = None

    for spec in perturbation_specs:
        p_type = spec["type"]
        p_param = spec["param"]

        if p_type == "clean":
            img_variant = pil_img
        elif p_type == "jpeg":
            img_variant = apply_jpeg_compression(pil_img, quality=p_param)
        elif p_type == "resize":
            img_variant = apply_resize_degradation(pil_img, scale_factor=p_param)
        elif p_type == "social":
            img_variant = apply_social_media_pipeline(pil_img, scale=0.50, quality=p_param)
        else:
            img_variant = pil_img

        feats = extract_features_for_image(img_variant, centroids)
        pred = classifier.predict(feats, abstain_margin=abstain_margin)

        if baseline_verdict is None:
            baseline_verdict = pred["verdict"]

        results.append({
            "test_name": spec["name"],
            "type": p_type,
            "parameter": p_param,
            "verdict": pred["verdict"],
            "confidence": pred["confidence"],
            "calibrated_prob_ai": pred.get("calibrated_prob_ai"),
            "abstained": pred.get("abstained", False),
            "verdict_preserved": (pred["verdict"] == baseline_verdict),
            "clip_semantic_score": feats.get("clip_semantic_score"),
            "fft_anomaly_score": feats.get("fft_anomaly_score"),
        })

    preserved_count = sum(1 for r in results if r["verdict_preserved"])
    stability_score = round(preserved_count / len(results), 3)

    return {
        "baseline_verdict": baseline_verdict,
        "stability_score": stability_score,
        "total_stress_tests": len(results),
        "tests": results,
    }
