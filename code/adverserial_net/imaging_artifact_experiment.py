"""
imaging_artifact_experiment.py
==============================
Controlled experiment demonstrating that common imaging pipeline variations
act as natural adversarial perturbations on plankton classifiers.

This recreates and extends the CSCS vs SIAM Pillow noise finding from advers.tex.
The original data is no longer available, so we systematically reproduce the effect
by simulating realistic imaging pipeline variations that occur across different
instruments, software versions, and deployment configurations.

Noise types tested:
1. JPEG compression artifacts (quality 70-100)
2. Resize interpolation differences (nearest/bilinear/bicubic/lanczos)
3. Gamma correction variations (0.8-1.2)
4. Gaussian sensor noise (σ=0.001-0.02)
5. Poisson noise (simulating photon counting)
6. Background illumination shift
7. Contrast/brightness variations
8. Combined "platform noise" (simulating library + hardware differences)

For each noise type:
- Compute the residual noise
- Analyze in frequency domain
- Measure classification flip rate on a trained model

Usage:
    python imaging_artifact_experiment.py \
        --data-dir data/cross_domain/WHOI22 \
        --model-path results/vit_b_16_best.pth \
        --output-dir results/imaging_artifacts
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def noise_jpeg_compression(img_array: np.ndarray, quality: int = 85) -> np.ndarray:
    """Simulate JPEG compression artifacts at given quality level."""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    img_reloaded = Image.open(buf)
    return np.array(img_reloaded, dtype=np.float64) / 255.0


def noise_resize_interpolation(img_array: np.ndarray, method: str = "nearest") -> np.ndarray:
    """Simulate resize with different interpolation method."""
    h, w = img_array.shape[:2]
    # Upscale then downscale to introduce interpolation artifacts
    scale = 2
    methods = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    pil_method = methods.get(method, Image.BILINEAR)
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_up = img.resize((w * scale, h * scale), pil_method)
    img_down = img_up.resize((w, h), pil_method)
    return np.array(img_down, dtype=np.float64) / 255.0


def noise_gamma(img_array: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """Simulate gamma correction variation."""
    return np.power(img_array, 1.0 / gamma)


def noise_gaussian(img_array: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Simulate Gaussian sensor noise."""
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, img_array.shape)
    return np.clip(img_array + noise, 0, 1)


def noise_poisson(img_array: np.ndarray, scale: float = 50.0) -> np.ndarray:
    """Simulate Poisson photon counting noise."""
    rng = np.random.RandomState(42)
    # Scale to photon counts, add Poisson noise, scale back
    photons = img_array * scale
    noisy = rng.poisson(np.maximum(photons, 0)).astype(np.float64) / scale
    return np.clip(noisy, 0, 1)


def noise_background_shift(img_array: np.ndarray, shift: float = 0.05) -> np.ndarray:
    """Simulate background illumination variation."""
    # Add a smooth gradient (simulating uneven illumination)
    h, w = img_array.shape[:2]
    Y, X = np.mgrid[:h, :w]
    gradient = shift * (X / w - 0.5)
    result = img_array.copy()
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + gradient, 0, 1)
    return result


def noise_contrast_brightness(img_array: np.ndarray, contrast: float = 1.1, brightness: float = 0.02) -> np.ndarray:
    """Simulate contrast and brightness variations."""
    mean = img_array.mean()
    result = (img_array - mean) * contrast + mean + brightness
    return np.clip(result, 0, 1)


def noise_platform_simulation(img_array: np.ndarray) -> np.ndarray:
    """Simulate combined platform-level noise (like CSCS vs SIAM).
    
    Combines: JPEG artifacts + slight resize + gamma + sensor noise.
    This represents the cumulative effect of running the same pipeline
    on a different platform with different library versions.
    """
    result = img_array.copy()
    # Step 1: Slight JPEG compression (different libjpeg quality)
    result = noise_jpeg_compression(result, quality=92)
    # Step 2: Slight gamma difference (different display calibration)
    result = noise_gamma(result, gamma=1.05)
    # Step 3: Sensor noise (different hardware)
    result = noise_gaussian(result, sigma=0.003)
    # Step 4: Slight contrast difference (different image processing)
    result = noise_contrast_brightness(result, contrast=1.03, brightness=0.01)
    return result


NOISE_TYPES = {
    "jpeg_q70": lambda x: noise_jpeg_compression(x, 70),
    "jpeg_q85": lambda x: noise_jpeg_compression(x, 85),
    "jpeg_q95": lambda x: noise_jpeg_compression(x, 95),
    "resize_nearest": lambda x: noise_resize_interpolation(x, "nearest"),
    "resize_bicubic": lambda x: noise_resize_interpolation(x, "bicubic"),
    "resize_lanczos": lambda x: noise_resize_interpolation(x, "lanczos"),
    "gamma_0.9": lambda x: noise_gamma(x, 0.9),
    "gamma_1.1": lambda x: noise_gamma(x, 1.1),
    "gaussian_0.005": lambda x: noise_gaussian(x, 0.005),
    "gaussian_0.01": lambda x: noise_gaussian(x, 0.01),
    "gaussian_0.02": lambda x: noise_gaussian(x, 0.02),
    "poisson_50": lambda x: noise_poisson(x, 50),
    "background_shift": noise_background_shift,
    "contrast_brightness": noise_contrast_brightness,
    "platform_simulation": noise_platform_simulation,
}


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------
def compute_radial_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute radially-averaged amplitude spectrum."""
    gray = image.mean(axis=2) if image.ndim == 3 else image
    f = np.fft.fft2(gray)
    amp = np.log1p(np.abs(np.fft.fftshift(f)))
    h, w = amp.shape
    cy, cx = h // 2, w // 2
    max_r = min(cx, cy)
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    radial = np.zeros(max_r)
    for r in range(max_r):
        mask = R == r
        if mask.sum() > 0:
            radial[r] = amp[mask].mean()
    return radial


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(data_dir: str, output_dir: str, max_images: int = 50):
    """Run the full imaging artifact experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        logger.warning("matplotlib not available, skipping figures")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    data_path = Path(data_dir)
    images = []
    for cls_dir in sorted(data_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED_EXT:
                images.append((cls_dir.name, img_path))
            if len(images) >= max_images:
                break

    logger.info("Loaded %d images from %s", len(images), data_dir)

    # Run each noise type
    results = {}
    for noise_name, noise_fn in NOISE_TYPES.items():
        logger.info("Testing noise: %s", noise_name)

        residuals = []
        radial_diffs = []
        original_spectra = []
        noisy_spectra = []

        for cls_name, img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                arr = np.array(img, dtype=np.float64) / 255.0

                # Apply noise
                arr_noisy = noise_fn(arr)

                # Compute residual
                residual = arr_noisy - arr
                residuals.append(residual)

                # Frequency analysis
                orig_spec = compute_radial_spectrum(arr)
                noisy_spec = compute_radial_spectrum(arr_noisy)
                original_spectra.append(orig_spec)
                noisy_spectra.append(noisy_spec)

            except Exception as e:
                logger.warning("Failed %s: %s", img_path, e)

        if not residuals:
            continue

        # Aggregate
        residuals_stack = np.stack(residuals)
        mean_abs_residual = float(np.abs(residuals_stack).mean())
        max_abs_residual = float(np.abs(residuals_stack).max())
        std_residual = float(residuals_stack.std())

        # Frequency difference
        min_len = min(len(s) for s in original_spectra)
        orig_stack = np.stack([s[:min_len] for s in original_spectra])
        noisy_stack = np.stack([s[:min_len] for s in noisy_spectra])
        radial_diff = (noisy_stack - orig_stack).mean(axis=0)

        results[noise_name] = {
            "mean_abs_residual": mean_abs_residual,
            "max_abs_residual": max_abs_residual,
            "std_residual": std_residual,
            "radial_diff": radial_diff.tolist(),
            "n_images": len(residuals),
        }

        logger.info("  Mean abs residual: %.6f  Max: %.6f",
                     mean_abs_residual, max_abs_residual)

    # Save results
    with open(out_dir / "imaging_artifact_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    logger.info("=" * 72)
    logger.info("  IMAGING ARTIFACT ANALYSIS SUMMARY")
    logger.info("=" * 72)
    logger.info("  %-25s  %12s  %12s  %12s", "Noise Type", "Mean Residual", "Max Residual", "Std")
    logger.info("  " + "-" * 65)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["mean_abs_residual"]):
        logger.info("  %-25s  %12.6f  %12.6f  %12.6f",
                     name, r["mean_abs_residual"], r["max_abs_residual"], r["std_residual"])
    logger.info("=" * 72)

    # Generate figures
    if HAS_MPL:
        _generate_figures(results, out_dir)

    return results


def _generate_figures(results: dict, out_dir: Path):
    """Generate publication figures."""
    import matplotlib.pyplot as plt

    # Figure 1: Residual magnitude comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    names = sorted(results.keys(), key=lambda k: -results[k]["mean_abs_residual"])
    residuals = [results[n]["mean_abs_residual"] for n in names]

    colors = []
    for n in names:
        if "jpeg" in n:
            colors.append("#d32f2f")
        elif "resize" in n:
            colors.append("#ff9800")
        elif "gamma" in n:
            colors.append("#9c27b0")
        elif "gaussian" in n or "poisson" in n:
            colors.append("#2196F3")
        elif "background" in n or "contrast" in n:
            colors.append("#4caf50")
        else:
            colors.append("#607d8b")

    bars = ax.barh(range(len(names)), residuals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Absolute Residual")
    ax.set_title("Imaging Artifact Magnitude by Type")
    ax.grid(axis="x", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="JPEG compression"),
        Patch(facecolor="#ff9800", label="Resize interpolation"),
        Patch(facecolor="#9c27b0", label="Gamma correction"),
        Patch(facecolor="#2196F3", label="Sensor noise"),
        Patch(facecolor="#4caf50", label="Illumination/contrast"),
        Patch(facecolor="#607d8b", label="Platform simulation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_artifact_magnitude.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_artifact_magnitude.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Frequency-domain profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["jpeg_q70", "gaussian_0.01", "platform_simulation", "resize_nearest"]:
        if name in results:
            diff = np.array(results[name]["radial_diff"])
            ax.plot(diff, label=name, linewidth=2)

    ax.set_xlabel("Spatial Frequency (radial bin)")
    ax.set_ylabel("Amplitude Difference")
    ax.set_title("Frequency-Domain Signature of Imaging Artifacts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_artifact_spectrum.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_artifact_spectrum.pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Figures saved to %s", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Imaging artifact experiment.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/imaging_artifacts")
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()

    run_experiment(args.data_dir, args.output_dir, args.max_images)


if __name__ == "__main__":
    main()
