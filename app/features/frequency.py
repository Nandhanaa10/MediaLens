"""
Frequency-domain evidence extractor.

Computes an FFT-based and a DCT-based anomaly score for a single image.
Both scores are cheap to compute (no training, no GPU) and are known in the
literature to separate real camera images from diffusion/GAN-generated images,
which tend to leave periodic artifacts in the frequency spectrum.
"""
import numpy as np
import cv2
from scipy.fft import dctn


def _to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr


def fft_anomaly_score(image_bgr: np.ndarray) -> float:
    """
    Ratio of high-frequency energy to total energy in the FFT magnitude spectrum.
    Higher values tend to indicate synthetic high-frequency artifacts.
    """
    gray = _to_grayscale(image_bgr).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # radius defining the "low frequency" center region (25% of the smaller dimension)
    radius = int(0.25 * min(h, w))

    yy, xx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_freq_mask = dist_from_center <= radius

    total_energy = magnitude.sum() + 1e-8
    high_freq_energy = magnitude[~low_freq_mask].sum()

    return float(high_freq_energy / total_energy)


def dct_anomaly_score(image_bgr: np.ndarray) -> float:
    """
    Energy concentrated in high-frequency DCT coefficients (bottom-right of the
    DCT block), normalized by total energy. Diffusion models can leave block-level
    artifacts that show up here even when the FFT signal is weak.
    """
    gray = _to_grayscale(image_bgr).astype(np.float32)
    # resize to a fixed size so the score is comparable across images
    gray = cv2.resize(gray, (256, 256))
    coeffs = dctn(gray, norm="ortho")

    h, w = coeffs.shape
    high_freq_block = coeffs[h // 2:, w // 2:]

    total_energy = np.sum(np.abs(coeffs)) + 1e-8
    high_freq_energy = np.sum(np.abs(high_freq_block))

    return float(high_freq_energy / total_energy)


def extract_frequency_features(image_bgr: np.ndarray) -> dict:
    """Main entry point used by the rest of the pipeline."""
    return {
        "fft_anomaly_score": fft_anomaly_score(image_bgr),
        "dct_anomaly_score": dct_anomaly_score(image_bgr),
    }


if __name__ == "__main__":
    # quick self-test with a synthetic image so you can confirm the module runs
    # before wiring it into anything else
    rng = np.random.default_rng(0)
    fake_smooth_image = np.uint8(rng.normal(128, 5, (256, 256, 3)).clip(0, 255))
    fake_noisy_image = np.uint8(rng.normal(128, 60, (256, 256, 3)).clip(0, 255))

    print("Smooth synthetic image:", extract_frequency_features(fake_smooth_image))
    print("Noisy synthetic image:", extract_frequency_features(fake_noisy_image))
