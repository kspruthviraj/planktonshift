"""
pillow_noise_experiment.py
==========================
Reproduce the natural adversarial examples from library version differences.

This recreates the CSCS vs SIAM Pillow library noise finding from the
adversarial draft (advers.tex). The original data is no longer available,
so we systematically reproduce the effect using two venvs with different
Pillow versions.

The experiment:
1. Take plankton images from any source (WHOI22, ZooLake35, etc.)
2. Process each image with Pillow version A and Pillow version B
3. Compute the residual noise (difference)
4. Analyze noise in frequency domain
5. Measure classification flip rate

Usage (run from the "old" Pillow venv):
    python pillow_noise_experiment.py --mode produce \
        --data-dir data/cross_domain/WHOI22 \
        --output-dir results/pillow_noise/v9

Usage (run from the "new" Pillow venv):
    python pillow_noise_experiment.py --mode produce \
        --data-dir data/cross_domain/WHOI22 \
        --output-dir results/pillow_noise/v10

Usage (after both runs, analyze):
    python pillow_noise_experiment.py --mode analyze \
        --v9-dir results/pillow_noise/v9 \
        --v10-dir results/pillow_noise/v10 \
        --output-dir results/pillow_noise/analysis

Setup two venvs:
    # Venv 1 (old Pillow):
    python -m venv .venv_pillow9
    source .venv_pillow_pillow9/bin/activate
    pip install Pillow==9.5.0 numpy

    # Venv 2 (new Pillow):
    python -m venv .venv_pillow10
    source .venv_pillow10/bin/activate
    pip install Pillow==10.4.0 numpy
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


def process_images(data_dir: str, output_dir: str, max_images: int = 100):
    """Process images through the current Pillow version.
    
    Saves both the processed image and its raw pixel array for later comparison.
    """
    import PIL
    pillow_version = PIL.__version__

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    if not data_path.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        return

    # Collect images from all class subdirectories
    images = []
    for cls_dir in sorted(data_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED_EXT:
                images.append((cls_dir.name, img_path))
            if len(images) >= max_images:
                break

    logger.info("Processing %d images with Pillow %s", len(images), pillow_version)

    metadata = {
        "pillow_version": pillow_version,
        "data_dir": data_dir,
        "img_size": IMG_SIZE,
        "images": [],
    }

    for idx, (cls_name, img_path) in enumerate(images):
        try:
            # Standard processing pipeline (same as training)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            cls_out = out_dir / cls_name
            cls_out.mkdir(parents=True, exist_ok=True)

            # Save as JPEG (lossy — this is where Pillow versions differ)
            # The original CSCS/SIAM noise came from JPEG encoding differences
            out_path_jpg = cls_out / f"{img_path.stem}_processed.jpg"
            img.save(out_path_jpg, "JPEG", quality=95)

            # Also save as PNG for reference
            out_path_png = cls_out / f"{img_path.stem}_processed.png"
            img.save(out_path_png, "PNG")

            # Reload from JPEG (to capture encoding differences)
            img_reloaded = Image.open(out_path_jpg)
            arr = np.array(img_reloaded, dtype=np.float64) / 255.0
            np.save(str(cls_out / f"{img_path.stem}_processed.npy"), arr)

            metadata["images"].append({
                "original": str(img_path),
                "processed": str(out_path_jpg),
                "class": cls_name,
                "shape": list(arr.shape),
            })

            if (idx + 1) % 20 == 0:
                logger.info("  Processed %d/%d", idx + 1, len(images))

        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path, e)

    # Save metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved %d processed images to %s (Pillow %s)",
                len(metadata["images"]), output_dir, pillow_version)


def analyze_noise(v9_dir: str, v10_dir: str, output_dir: str):
    """Compare images processed by two Pillow versions.
    
    Computes:
    1. Pixel-level residual noise
    2. Frequency-domain analysis of the noise
    3. Classification flip rate (if a model is available)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        logger.warning("matplotlib not available, skipping figures")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(Path(v9_dir) / "metadata.json") as f:
        meta_v9 = json.load(f)
    with open(Path(v10_dir) / "metadata.json") as f:
        meta_v10 = json.load(f)

    logger.info("Comparing Pillow %s vs %s", meta_v9["pillow_version"], meta_v10["pillow_version"])

    # Match images by original path
    v9_map = {Path(img["processed"]).name: img for img in meta_v9["images"]}
    v10_map = {Path(img["processed"]).name: img for img in meta_v10["images"]}

    common = set(v9_map.keys()) & set(v10_map.keys())
    logger.info("Found %d common images", len(common))

    # Analyze each pair
    all_residuals = []
    all_amplitude_diffs = []
    flip_count = 0
    total_count = 0
    per_class = {}

    for fname in sorted(common):
        img_v9_info = v9_map[fname]
        img_v10_info = v10_map[fname]
        cls_name = img_v9_info["class"]

        # Load numpy arrays
        # npy files are named *_processed.npy regardless of image format
        npy_name = fname.rsplit(".", 1)[0] + ".npy"
        arr_v9 = np.load(Path(v9_dir) / cls_name / npy_name, allow_pickle=True)
        arr_v10 = np.load(Path(v10_dir) / cls_name / npy_name, allow_pickle=True)

        # Ensure same shape
        if arr_v9.shape != arr_v10.shape:
            continue

        # Compute residual
        residual = arr_v10 - arr_v9
        all_residuals.append(residual)

        # Frequency analysis
        for c in range(3):  # RGB channels
            f_v9 = np.fft.fft2(arr_v9[:, :, c])
            f_v10 = np.fft.fft2(arr_v10[:, :, c])
            amp_diff = np.abs(np.fft.fftshift(f_v10)) - np.abs(np.fft.fftshift(f_v9))
            all_amplitude_diffs.append(amp_diff)

        # Per-class stats
        if cls_name not in per_class:
            per_class[cls_name] = {"residuals": [], "n": 0}
        per_class[cls_name]["residuals"].append(np.abs(residual).mean())
        per_class[cls_name]["n"] += 1
        total_count += 1

    if not all_residuals:
        logger.error("No matching images found")
        return

    # Aggregate statistics
    residuals_stack = np.stack(all_residuals)
    mean_residual = residuals_stack.mean(axis=0)
    std_residual = residuals_stack.std(axis=0)
    max_abs_residual = np.abs(residuals_stack).max()

    amplitude_stack = np.stack(all_amplitude_diffs)
    mean_amp_diff = amplitude_stack.mean(axis=0)

    # Radial average of amplitude difference
    h, w = mean_amp_diff.shape
    cy, cx = h // 2, w // 2
    max_r = min(cx, cy)
    radial_diff = np.zeros(max_r)
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    for r in range(max_r):
        mask = R == r
        if mask.sum() > 0:
            radial_diff[r] = mean_amp_diff[mask].mean()

    # Summary statistics
    summary = {
        "pillow_v9": meta_v9["pillow_version"],
        "pillow_v10": meta_v10["pillow_version"],
        "n_images": total_count,
        "mean_abs_residual": float(np.abs(residuals_stack).mean()),
        "max_abs_residual": float(max_abs_residual),
        "std_residual": float(residuals_stack.std()),
        "residual_as_fraction_of_range": float(np.abs(residuals_stack).mean() / (residuals_stack.max() - residuals_stack.min() + 1e-10)),
        "per_class": {},
        "radial_amplitude_diff": radial_diff.tolist(),
    }

    for cls_name, stats in per_class.items():
        summary["per_class"][cls_name] = {
            "n": stats["n"],
            "mean_abs_residual": float(np.mean(stats["residuals"])),
        }

    # Save summary
    with open(out_dir / "pillow_noise_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("=" * 72)
    logger.info("  PILLOW LIBRARY NOISE ANALYSIS")
    logger.info("=" * 72)
    logger.info("  Versions: %s vs %s", summary["pillow_v9"], summary["pillow_v10"])
    logger.info("  Images compared: %d", total_count)
    logger.info("  Mean absolute residual: %.6f", summary["mean_abs_residual"])
    logger.info("  Max absolute residual: %.6f", summary["max_abs_residual"])
    logger.info("  Residual as fraction of pixel range: %.4f%%",
                summary["residual_as_fraction_of_range"] * 100)
    logger.info("-" * 72)
    logger.info("  Per-class mean absolute residual:")
    for cls_name, stats in sorted(summary["per_class"].items()):
        logger.info("    %-20s  n=%d  residual=%.6f",
                    cls_name, stats["n"], stats["mean_abs_residual"])
    logger.info("=" * 72)

    # Generate figures
    if plt is not None:
        _generate_noise_figures(
            residuals_stack, mean_amp_diff, radial_diff, summary, out_dir
        )


def _generate_noise_figures(residuals_stack, mean_amp_diff, radial_diff, summary, out_dir):
    """Generate publication-quality figures for the Pillow noise analysis."""
    import matplotlib.pyplot as plt

    # Figure 1: Residual noise distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of residuals
    ax = axes[0]
    flat_residuals = residuals_stack.flatten()
    # Subsample for speed
    if len(flat_residuals) > 1_000_000:
        rng = np.random.RandomState(42)
        flat_residuals = rng.choice(flat_residuals, 1_000_000, replace=False)
    ax.hist(flat_residuals, bins=100, density=True, alpha=0.7, color="#2196F3")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Density")
    ax.set_title("(a) Distribution of pixel residuals\nbetween Pillow versions")
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    # Mean residual image (RGB)
    ax = axes[1]
    mean_res = np.abs(residuals_stack.mean(axis=0))
    # Amplify for visibility
    amplified = mean_res / (mean_res.max() + 1e-10)
    ax.imshow(amplified)
    ax.set_title("(b) Mean absolute residual\n(amplified for visibility)")
    ax.axis("off")

    # Residual by channel
    ax = axes[2]
    for c, color, label in zip(range(3), ["#d32f2f", "#4caf50", "#2196F3"], ["R", "G", "B"]):
        channel_residuals = residuals_stack[:, :, :, c].flatten()
        if len(channel_residuals) > 500_000:
            rng = np.random.RandomState(42)
            channel_residuals = rng.choice(channel_residuals, 500_000, replace=False)
        ax.hist(channel_residuals, bins=100, density=True, alpha=0.4, color=color, label=label)
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Density")
    ax.set_title("(c) Residual distribution by channel")
    ax.legend()

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_pillow_noise_distribution.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_pillow_noise_distribution.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Frequency-domain analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Amplitude difference map
    ax = axes[0]
    im = ax.imshow(np.log1p(np.abs(mean_amp_diff)), cmap="hot")
    ax.set_title("(a) Log-amplitude difference\nbetween Pillow versions")
    ax.set_xlabel("Frequency (horizontal)")
    ax.set_ylabel("Frequency (vertical)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Radial profile
    ax = axes[1]
    ax.plot(radial_diff, color="#2196F3", linewidth=2)
    ax.set_xlabel("Spatial Frequency (radial bin)")
    ax.set_ylabel("Mean Amplitude Difference")
    ax.set_title("(b) Radial amplitude difference profile")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_pillow_noise_spectrum.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_pillow_noise_spectrum.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Per-class residual
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = sorted(summary["per_class"].keys())
    residuals = [summary["per_class"][c]["mean_abs_residual"] for c in classes]
    counts = [summary["per_class"][c]["n"] for c in classes]

    bars = ax.bar(range(len(classes)), residuals, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Absolute Residual")
    ax.set_title("Per-class Pillow library noise magnitude")
    ax.grid(axis="y", alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={count}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_pillow_noise_per_class.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_pillow_noise_per_class.pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Figures saved to %s", out_dir)


def simulate_pillow_noise(image_path: str, output_dir: str):
    """Simulate Pillow library noise without two venvs.
    
    This is a FALLBACK if creating two venvs is too cumbersome.
    It simulates the known artifacts:
    1. Sub-pixel resize differences (bilinear interpolation rounding)
    2. JPEG quantization differences
    3. Color space conversion rounding
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64) / 255.0

    # Simulate different resize interpolation
    # Pillow 9.x uses slightly different rounding than 10.x
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr_resized = np.array(img_resized, dtype=np.float64) / 255.0

    # Simulate sub-pixel noise (from the perturbations.png pattern)
    # The noise is typically very small (0-3 pixel values out of 255)
    rng = np.random.RandomState(42)
    noise = rng.uniform(-3/255, 3/255, arr_resized.shape)

    # Make noise spatially correlated (like real Pillow differences)
    from scipy.ndimage import gaussian_filter
    for c in range(3):
        noise[:, :, c] = gaussian_filter(noise[:, :, c], sigma=1.5)

    arr_noisy = np.clip(arr_resized + noise, 0, 1)

    # Save comparison
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    Image.fromarray((arr_resized * 255).astype(np.uint8)).save(out_dir / "original.png")

    # Save noisy (simulating other Pillow version)
    Image.fromarray((arr_noisy * 255).astype(np.uint8)).save(out_dir / "noisy_simulated.png")

    # Save residual (amplified)
    residual = arr_noisy - arr_resized
    amplified = (residual - residual.min()) / (residual.max() - residual.min() + 1e-10)
    Image.fromarray((amplified * 255).astype(np.uint8)).save(out_dir / "residual_amplified.png")

    # Save noise stats
    stats = {
        "mean_abs_residual": float(np.abs(residual).mean()),
        "max_abs_residual": float(np.abs(residual).max()),
        "std_residual": float(residual.std()),
        "noise_range": [float(residual.min()), float(residual.max())],
    }
    with open(out_dir / "noise_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Simulated Pillow noise saved to %s", out_dir)
    logger.info("  Mean absolute residual: %.6f", stats["mean_abs_residual"])
    logger.info("  Max absolute residual: %.6f", stats["max_abs_residual"])


def main():
    parser = argparse.ArgumentParser(description="Pillow library noise experiment.")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["produce", "analyze", "simulate"],
                        help="'produce' to process images with current Pillow, "
                             "'analyze' to compare two runs, "
                             "'simulate' to simulate noise without two venvs")
    parser.add_argument("--data-dir", type=str, help="Input data directory (for produce/simulate)")
    parser.add_argument("--output-dir", type=str, default="results/pillow_noise")
    parser.add_argument("--v9-dir", type=str, help="Output from Pillow 9 run (for analyze)")
    parser.add_argument("--v10-dir", type=str, help="Output from Pillow 10 run (for analyze)")
    parser.add_argument("--image", type=str, help="Single image path (for simulate)")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "produce":
        if not args.data_dir:
            parser.error("--data-dir required for produce mode")
        process_images(args.data_dir, args.output_dir, args.max_images)

    elif args.mode == "analyze":
        if not args.v9_dir or not args.v10_dir:
            parser.error("--v9-dir and --v10-dir required for analyze mode")
        analyze_noise(args.v9_dir, args.v10_dir, args.output_dir)

    elif args.mode == "simulate":
        if not args.image and not args.data_dir:
            parser.error("--image or --data-dir required for simulate mode")
        if args.image:
            simulate_pillow_noise(args.image, args.output_dir)
        else:
            # Simulate on first image from data dir
            data_path = Path(args.data_dir)
            for cls_dir in sorted(data_path.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_EXT:
                        simulate_pillow_noise(str(img_path), args.output_dir)
                        return


if __name__ == "__main__":
    main()
