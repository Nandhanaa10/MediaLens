"""
Video frame extractor.
Uniformly extracts frames and metadata from video files for temporal forensic analysis.
"""
import os
import cv2
import numpy as np
from PIL import Image


def extract_frames_from_video(
    video_path: str,
    max_frames: int = 16,
    max_duration_seconds: float = 15.0,
    target_size: tuple = None,
) -> dict:
    """
    Extracts uniformly spaced frames and metadata from a video file.

    Parameters:
        video_path: Path to the video file.
        max_frames: Number of frames to sample across the video.
        max_duration_seconds: Maximum duration to inspect from the start of the video.
        target_size: Optional (width, height) to resize frames (e.g. (256, 256) for fast CPU processing).

    Returns:
        dict containing:
            - 'frames_bgr': list of np.ndarray in BGR format
            - 'frames_pil': list of PIL.Image in RGB format
            - 'metadata': dict of video properties
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # reasonable fallback if metadata is missing

        duration_seconds = total_frames / fps if total_frames > 0 else 0.0

        # Limit to max_duration_seconds if needed
        effective_duration = min(duration_seconds, max_duration_seconds) if duration_seconds > 0 else max_duration_seconds
        effective_frames = min(total_frames, int(effective_duration * fps)) if total_frames > 0 else 100

        if effective_frames <= 0:
            effective_frames = 1

        # Select uniformly spaced frame indices
        num_samples = min(max_frames, effective_frames)
        if num_samples <= 1:
            frame_indices = [0]
        else:
            frame_indices = np.linspace(0, effective_frames - 1, num=num_samples, dtype=int)

        frame_indices_set = set(frame_indices)
        frames_bgr = []
        frames_pil = []

        current_idx = 0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened() and len(frames_bgr) < len(frame_indices_set):
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in frame_indices_set:
                if target_size is not None:
                    processed_bgr = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                else:
                    processed_bgr = frame

                rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                frames_bgr.append(processed_bgr)
                frames_pil.append(Image.fromarray(rgb))

            current_idx += 1
            if current_idx > effective_frames:
                break

        if not frames_bgr:
            raise ValueError("No frames could be extracted from the video.")

        metadata = {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "sampled_frames": len(frames_bgr),
            "duration_seconds": round(duration_seconds, 2),
            "effective_duration_seconds": round(effective_duration, 2),
            "original_resolution": [orig_width, orig_height],
            "extracted_resolution": [frames_bgr[0].shape[1], frames_bgr[0].shape[0]],
        }

        return {
            "frames_bgr": frames_bgr,
            "frames_pil": frames_pil,
            "metadata": metadata,
        }

    finally:
        cap.release()


if __name__ == "__main__":
    # Test generator: synthesize a small test video in memory and extract frames
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        test_video_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (128, 128))
        for i in range(20):
            # create synthetic moving gradient
            frame = np.full((128, 128, 3), (i * 10) % 255, dtype=np.uint8)
            out.write(frame)
        out.release()

        result = extract_frames_from_video(test_video_path, max_frames=5)
        print("Frame extraction test passed!")
        print("Extracted frames count:", len(result["frames_bgr"]))
        print("Metadata:", result["metadata"])
    finally:
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
