"""
Temporal forensic feature extractor for video analysis.
Computes:
1. Optical flow motion warping error across consecutive frames.
2. Inter-frame frequency (FFT) variance (detecting temporal flicker/shimmer).
3. CLIP semantic temporal drift across frames.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.features.frequency import fft_anomaly_score, dct_anomaly_score

try:
    from app.features.clip_features import embed_image, extract_clip_features
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def compute_optical_flow_warping_error(frames_bgr: list) -> float:
    """
    Computes dense optical flow (Farneback) between consecutive frames,
    warps frame t into frame t+1, and measures the average photometric discrepancy.

    Real physical camera motion produces smooth warping with low photometric error.
    Generative video models (Runway, Pika, Sora, etc.) exhibit pixel-level morphing
    and regeneration artifacts that produce significantly higher warping error.
    """
    if len(frames_bgr) < 2:
        return 0.0

    errors = []

    for i in range(len(frames_bgr) - 1):
        prev = frames_bgr[i]
        curr = frames_bgr[i + 1]

        # Convert to grayscale and resize for fast, stable optical flow
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Scale down to 256x256 for uniform calculation and speed
        prev_gray = cv2.resize(prev_gray, (256, 256), interpolation=cv2.INTER_AREA)
        curr_gray = cv2.resize(curr_gray, (256, 256), interpolation=cv2.INTER_AREA)

        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Build coordinate flow grid to warp prev_gray forward to curr_gray
        h, w = prev_gray.shape
        flow_map = np.empty((h, w, 2), dtype=np.float32)
        flow_map[..., 0] = np.tile(np.arange(w), (h, 1)) + flow[..., 0]
        flow_map[..., 1] = np.tile(np.arange(h)[:, None], (1, w)) + flow[..., 1]

        warped_prev = cv2.remap(
            prev_gray,
            flow_map,
            None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Photometric absolute discrepancy normalized to [0, 1]
        diff = np.abs(curr_gray.astype(np.float32) - warped_prev.astype(np.float32)) / 255.0
        # Discard outer 5% border where camera flow boundaries naturally occur
        pad = int(0.05 * min(h, w))
        inner_diff = diff[pad : h - pad, pad : w - pad]

        errors.append(float(np.mean(inner_diff)))

    return float(np.mean(errors)) if errors else 0.0


def compute_interframe_fft_variance(frames_bgr: list) -> tuple:
    """
    Computes the FFT anomaly score for each frame, and returns:
    (mean_fft_score, fft_variance across time).
    A high variance indicates high-frequency temporal flickering/jitter typical of diffusion video.
    """
    if not frames_bgr:
        return 0.0, 0.0

    scores = [fft_anomaly_score(f) for f in frames_bgr]
    mean_val = float(np.mean(scores))
    var_val = float(np.var(scores))
    return mean_val, var_val


def compute_clip_temporal_drift(pil_frames: list, generator_centroids: dict = None) -> tuple:
    """
    Computes semantic drift across consecutive frames using CLIP embeddings,
    and average generator similarity against known centroids.

    Returns:
        (clip_temporal_drift, mean_clip_generator_similarity)
    """
    if not CLIP_AVAILABLE or len(pil_frames) < 1:
        return 0.0, 0.795

    embeddings = [embed_image(f) for f in pil_frames]

    # Temporal drift: 1.0 - cosine_similarity(e_t, e_{t+1})
    if len(embeddings) > 1:
        drifts = []
        for i in range(len(embeddings) - 1):
            e1 = embeddings[i]
            e2 = embeddings[i + 1]
            cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
            drifts.append(1.0 - cos_sim)
        mean_drift = float(np.mean(drifts))
    else:
        mean_drift = 0.0

    # Generator similarity across frames
    if generator_centroids:
        sims = []
        for emb in embeddings:
            similarities = {
                name: float(np.dot(emb, centroid))
                for name, centroid in generator_centroids.items()
            }
            if similarities:
                sims.append(max(similarities.values()))
        mean_gen_sim = float(np.mean(sims)) if sims else 0.795
    else:
        mean_gen_sim = 0.795

    return mean_drift, mean_gen_sim


def extract_video_features(
    frames_bgr: list,
    pil_frames: list,
    generator_centroids: dict = None,
) -> dict:
    """
    Main entry point for video forensic extraction.
    Combines spatial forensics (FFT, DCT, CLIP) with temporal motion signals
    (Optical Flow Warping Error, Inter-frame FFT Variance, and Semantic Drift).
    """
    # 1. Spatial Frequency signals (averaged over frames)
    mean_fft, fft_variance = compute_interframe_fft_variance(frames_bgr)
    dct_scores = [dct_anomaly_score(f) for f in frames_bgr]
    mean_dct = float(np.mean(dct_scores)) if dct_scores else 0.0

    # 2. Temporal Optical Flow Warping Error
    warping_error = compute_optical_flow_warping_error(frames_bgr)

    # 3. CLIP Semantic Temporal Drift and Generator Similarity
    temporal_drift, mean_gen_sim = compute_clip_temporal_drift(
        pil_frames, generator_centroids=generator_centroids
    )

    return {
        # Spatial foundation
        "fft_anomaly_score": mean_fft,
        "dct_anomaly_score": mean_dct,
        "clip_max_generator_similarity": mean_gen_sim,
        # Temporal extensions
        "optical_flow_warping_error": warping_error,
        "interframe_fft_variance": fft_variance,
        "clip_temporal_drift": temporal_drift,
    }


if __name__ == "__main__":
    # Quick self-test with synthetic test frames
    rng = np.random.default_rng(42)
    test_bgr = [
        np.uint8(rng.normal(128, 30 + i * 2, (128, 128, 3)).clip(0, 255))
        for i in range(4)
    ]
    test_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in test_bgr]

    features = extract_video_features(test_bgr, test_pil)
    print("Video Features Extraction Test:")
    for k, v in features.items():
        print(f"  {k}: {v:.6f}")
