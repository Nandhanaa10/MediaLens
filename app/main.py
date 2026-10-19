"""
FastAPI backend for Explainable AI-Generated Media Detector.
Provides:
- Web Demo UI at GET / and GET /demo
- Single-image detection at POST /predict with abstention & robustness simulation
- Video temporal consistency detection at POST /predict-video
- Interactive robustness stress test at POST /evaluate/robustness
- Benchmark metrics endpoints at GET /metrics/cross-generator and GET /metrics/robustness
"""
import io
import os
import json
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

from app.features.frequency import extract_frequency_features
from app.fusion.classifier import FusionClassifier
from app.explain.llm_client import generate_explanation
from app.video.frame_extractor import extract_frames_from_video
from app.features.temporal import extract_video_features
from app.evaluation.robustness import (
    apply_jpeg_compression,
    apply_resize_degradation,
    stress_test_single_image,
)

try:
    from app.features.clip_features import extract_clip_features
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

app = FastAPI(title="Explainable AI-Generated Media Detector")

# Model paths: Prefer Version 3 (GenImage) if available, fallback to Version 1 (CIFAKE)
MODEL_PATH = (
    "models/genimage_fusion_model.joblib"
    if os.path.exists("models/genimage_fusion_model.joblib")
    else "models/fusion_model.joblib"
)
CENTROIDS_PATH = (
    "models/genimage_centroids.joblib"
    if os.path.exists("models/genimage_centroids.joblib")
    else "models/generator_centroids.joblib"
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

_classifier = None
_generator_centroids = None


def get_classifier() -> FusionClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FusionClassifier(MODEL_PATH)
    return _classifier


def get_generator_centroids() -> dict:
    global _generator_centroids
    if _generator_centroids is None:
        if os.path.exists(CENTROIDS_PATH):
            _generator_centroids = joblib.load(CENTROIDS_PATH)
        else:
            _generator_centroids = {}
    return _generator_centroids


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    """Serves the interactive forensic web demo interface."""
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>VeritasAI Web Demo</h1><p>Template file not found.</p>")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "v3_genimage" if "genimage" in MODEL_PATH else "v1_cifake",
        "active_centroids": [
            k for k in get_generator_centroids().keys() if not k.startswith("_")
        ],
        "clip_available": CLIP_AVAILABLE,
    }


@app.get("/metrics/cross-generator")
def get_cross_generator_metrics():
    """Returns the zero-shot cross-generator benchmark evaluation report."""
    results_path = Path("data/cross_generator_results.json")
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Cross-generator metrics not generated yet. Run scripts/evaluate_cross_generator.py")


@app.get("/metrics/robustness")
def get_robustness_metrics():
    """Returns the compression and resizing robustness benchmark evaluation report."""
    results_path = Path("data/robustness_benchmark_results.json")
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Robustness benchmark not generated yet. Run scripts/evaluate_robustness.py")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    abstain_margin: float = Query(0.08, ge=0.0, le=0.50, description="Selective classification margin around 0.50"),
    simulate_jpeg: Optional[int] = Query(None, ge=10, le=100, description="Simulate JPEG compression (quality 10-100)"),
    simulate_resize: Optional[float] = Query(None, ge=0.1, le=1.0, description="Simulate downscaling factor"),
):
    """
    Evaluates an uploaded image for AI generation artifacts using:
    - 2D Spatial Frequency analysis (FFT & DCT)
    - Universal CLIP ViT-B/32 semantic probe & contrastive margins
    - Exact SHAP feature contributions
    - Calibrated probability with selective classification (abstention)
    - Faithfully grounded natural-language explanation
    """
    contents = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    # Optional robustness degradation simulation
    if simulate_resize is not None and simulate_resize < 1.0:
        pil_image = apply_resize_degradation(pil_image, scale_factor=simulate_resize)
    if simulate_jpeg is not None:
        pil_image = apply_jpeg_compression(pil_image, quality=simulate_jpeg)

    # Standardize image to (256, 256) for frequency analysis
    freq_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
    image_bgr = cv2.cvtColor(np.array(freq_image), cv2.COLOR_RGB2BGR)

    features = {}
    features.update(extract_frequency_features(image_bgr))

    centroids = get_generator_centroids()
    if CLIP_AVAILABLE and centroids:
        clip_features = extract_clip_features(pil_image, centroids)
        features["clip_semantic_score"] = clip_features["clip_semantic_score"]
        features["clip_contrast_margin"] = clip_features["clip_contrast_margin"]
        features["clip_max_generator_similarity"] = clip_features["clip_max_generator_similarity"]
        if clip_features.get("clip_best_match_generator"):
            features["clip_best_match_generator"] = clip_features["clip_best_match_generator"]
    else:
        features["clip_semantic_score"] = 0.50
        features["clip_contrast_margin"] = 0.0
        features["clip_max_generator_similarity"] = 0.68

    classifier = get_classifier()
    evidence = classifier.predict(features, abstain_margin=abstain_margin)

    explanation_result = generate_explanation(evidence)

    response = {
        "verdict": evidence["verdict"],
        "confidence": evidence["confidence"],
        "abstained": evidence.get("abstained", False),
        "abstention_reason": evidence.get("abstention_reason"),
        "model_version": "v3_genimage" if "genimage" in MODEL_PATH else "v1_cifake",
        "evidence": evidence,
        "explanation": explanation_result["explanation"],
        "explanation_source": explanation_result["source"],
        "faithfulness_audit": explanation_result.get("faithfulness_audit", {}),
    }
    if evidence.get("clip_best_match_generator"):
        response["closest_ai_generator"] = evidence["clip_best_match_generator"]

    return response


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    abstain_margin: float = Query(0.08, ge=0.0, le=0.50, description="Selective classification margin around 0.50"),
):
    """
    Evaluates an uploaded video for AI generation artifacts using:
    - Spatial frequency analysis (FFT & DCT across frames)
    - Semantic generator similarity (CLIP across frames)
    - Farneback optical flow motion warping discrepancy
    - Inter-frame frequency variance (temporal jitter)
    - CLIP semantic temporal drift
    - Calibrated probability with selective classification (abstention)
    """
    import tempfile

    filename = file.filename or "video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: .mp4, .mov, .avi, .webm, .mkv",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        frame_data = extract_frames_from_video(tmp_path, max_frames=16, target_size=(256, 256))
        centroids = get_generator_centroids()
        features = extract_video_features(
            frame_data["frames_bgr"],
            frame_data["frames_pil"],
            generator_centroids=centroids,
        )

        classifier = get_classifier()
        evidence = classifier.predict(features, abstain_margin=abstain_margin)

        # Include video metadata and temporal signals in evidence for the LLM
        evidence["media_type"] = "video"
        evidence["sampled_frames"] = frame_data["metadata"]["sampled_frames"]
        evidence["duration_seconds"] = frame_data["metadata"]["duration_seconds"]
        evidence["optical_flow_warping_error"] = round(features["optical_flow_warping_error"], 4)
        evidence["interframe_fft_variance"] = round(features["interframe_fft_variance"], 6)
        evidence["clip_temporal_drift"] = round(features["clip_temporal_drift"], 4)

        explanation_result = generate_explanation(evidence)

        return {
            "media_type": "video",
            "verdict": evidence["verdict"],
            "confidence": evidence["confidence"],
            "abstained": evidence.get("abstained", False),
            "abstention_reason": evidence.get("abstention_reason"),
            "metadata": frame_data["metadata"],
            "temporal_evidence": {
                "optical_flow_warping_error": features["optical_flow_warping_error"],
                "interframe_fft_variance": features["interframe_fft_variance"],
                "clip_temporal_drift": features["clip_temporal_drift"],
            },
            "evidence": evidence,
            "explanation": explanation_result["explanation"],
            "explanation_source": explanation_result["source"],
            "faithfulness_audit": explanation_result.get("faithfulness_audit", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/evaluate/robustness")
async def evaluate_robustness(
    file: UploadFile = File(...),
    abstain_margin: float = Query(0.08, ge=0.0, le=0.50),
):
    """
    Runs a multi-level degradation stress test battery on an uploaded image:
    Evaluates verdict stability across JPEG quality levels and spatial resizing.
    """
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    classifier = get_classifier()
    centroids = get_generator_centroids()

    report = stress_test_single_image(pil_image, classifier, centroids, abstain_margin=abstain_margin)
    return report
