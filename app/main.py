"""
FastAPI backend. Run with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs to test the /predict endpoint by
uploading an image, without needing a frontend yet.
"""
import io

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from PIL import Image

from app.features.frequency import extract_frequency_features
from app.fusion.classifier import FusionClassifier
from app.explain.llm_client import generate_explanation
from app.video.frame_extractor import extract_frames_from_video
from app.features.temporal import extract_video_features

try:
    from app.features.clip_features import extract_clip_features
    CLIP_AVAILABLE = True
except ImportError:
    # torch/open_clip not installed yet - the rest of the pipeline still works
    # without CLIP features, just with a weaker evidence set (see README Step 4)
    CLIP_AVAILABLE = False

app = FastAPI(title="Explainable AI-Generated Media Detector")

import os
import joblib

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


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "v3_genimage" if "genimage" in MODEL_PATH else "v1_cifake",
        "active_centroids": list(get_generator_centroids().keys()),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    # Standardize image to (256, 256) for high-resolution GenImage frequency analysis
    if pil_image.size != (256, 256):
        freq_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
    else:
        freq_image = pil_image

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
    evidence = classifier.predict(features)

    explanation_result = generate_explanation(evidence)

    response = {
        "verdict": evidence["verdict"],
        "confidence": evidence["confidence"],
        "model_version": "v3_genimage" if "genimage" in MODEL_PATH else "v1_cifake",
        "evidence": evidence,
        "explanation": explanation_result["explanation"],
        "explanation_source": explanation_result["source"],
    }
    if evidence.get("clip_best_match_generator"):
        response["closest_ai_generator"] = evidence["clip_best_match_generator"]

    return response


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """
    Evaluates an uploaded video for AI generation artifacts using:
    - Spatial frequency analysis (FFT & DCT across frames)
    - Semantic generator similarity (CLIP across frames)
    - Temporal optical flow warping error
    - Inter-frame frequency variance (temporal jitter)
    - CLIP semantic temporal drift
    """
    import tempfile

    filename = file.filename or "video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: .mp4, .mov, .avi, .webm, .mkv",
        )

    # Save uploaded video to temporary file
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
        evidence = classifier.predict(features)

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
            "metadata": frame_data["metadata"],
            "temporal_evidence": {
                "optical_flow_warping_error": features["optical_flow_warping_error"],
                "interframe_fft_variance": features["interframe_fft_variance"],
                "clip_temporal_drift": features["clip_temporal_drift"],
            },
            "evidence": evidence,
            "explanation": explanation_result["explanation"],
            "explanation_source": explanation_result["source"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

