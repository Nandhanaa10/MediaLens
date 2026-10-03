"""
CLIP-based semantic evidence extractor.

Uses a frozen, pretrained CLIP model (no training here) to embed an image,
then compares it against a small bank of "generator centroid" embeddings
built once from a handful of known AI-generated sample images per generator.

On an M2 Mac, this runs fine on CPU. If you want to use the GPU (Metal),
set device="mps" - PyTorch's Metal backend works out of the box on Apple
Silicon with a standard `pip install torch` (no special index-url needed).
"""
import numpy as np
import torch
import open_clip
from PIL import Image


_MODEL = None
_PREPROCESS = None
_DEVICE = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_model():
    """Loads the CLIP model once and reuses it across calls (it's not cheap to reload)."""
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is None:
        _DEVICE = get_device()
        _MODEL, _, _PREPROCESS = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _MODEL = _MODEL.to(_DEVICE).eval()
    return _MODEL, _PREPROCESS, _DEVICE


@torch.no_grad()
def embed_image(pil_image: Image.Image) -> np.ndarray:
    model, preprocess, device = load_clip_model()
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    embedding = model.encode_image(image_tensor)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]


def build_generator_centroids(sample_images_by_generator: dict) -> dict:
    """
    sample_images_by_generator: {"midjourney": [PIL.Image, ...], "sdxl": [...], ...}
    Run this ONCE offline on a handful of known-generator sample images
    (10-20 per generator is enough) and save the result - don't recompute
    this on every request.
    """
    centroids = {}
    for generator_name, images in sample_images_by_generator.items():
        embeddings = np.stack([embed_image(img) for img in images])
        centroids[generator_name] = embeddings.mean(axis=0)
    return centroids


def extract_clip_features(pil_image: Image.Image, generator_centroids: dict) -> dict:
    """Main entry point used by the rest of the pipeline."""
    embedding = embed_image(pil_image)

    # Extract probe weights if present
    probe_w = generator_centroids.get("_probe_weights")
    probe_b = generator_centroids.get("_probe_bias", 0.0)

    # Filter out metadata keys and real centroid for generator matching
    ai_centroids = {
        name: centroid
        for name, centroid in generator_centroids.items()
        if not name.startswith("_") and name.lower() != "real"
    }

    similarities = {
        name: float(np.dot(embedding, centroid))
        for name, centroid in ai_centroids.items()
    }
    best_match = max(similarities, key=similarities.get) if similarities else None
    max_ai_sim = max(similarities.values()) if similarities else 0.0

    real_centroid = generator_centroids.get("real")
    if real_centroid is not None:
        real_sim = float(np.dot(embedding, real_centroid))
        contrast_margin = float(max_ai_sim - real_sim)
    else:
        real_sim = 0.65
        contrast_margin = float(max_ai_sim - 0.65)

    if probe_w is not None:
        logit = float(np.dot(embedding, probe_w) + probe_b)
        semantic_score = float(1.0 / (1.0 + np.exp(-logit)))
    else:
        semantic_score = max_ai_sim

    return {
        "clip_semantic_score": semantic_score,
        "clip_contrast_margin": contrast_margin,
        "clip_max_generator_similarity": max_ai_sim,
        "clip_best_match_generator": best_match,
        "clip_similarities_by_generator": similarities,
        "clip_real_similarity": real_sim,
        "clip_embedding": embedding,
    }


if __name__ == "__main__":
    # quick self-test: embed two random synthetic images and compare them.
    # replace this with real generator sample images once you have CIFAKE downloaded.
    rng = np.random.default_rng(0)
    img_a = Image.fromarray(np.uint8(rng.normal(128, 40, (224, 224, 3)).clip(0, 255)))
    img_b = Image.fromarray(np.uint8(rng.normal(128, 40, (224, 224, 3)).clip(0, 255)))

    centroids = build_generator_centroids({"synthetic_test_generator": [img_a]})
    print(extract_clip_features(img_b, centroids))
