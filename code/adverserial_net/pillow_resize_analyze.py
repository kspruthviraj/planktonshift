"""
pillow_resize_analyze.py
========================
Analyze the pixel-level differences between images processed by Pillow 6.x
(NEAREST resize default) and Pillow 7.x (BICUBIC resize default).

This reproduces the CSCS vs SIAM cluster noise finding.

Usage:
    python pillow_resize_analyze.py \
        --v6-dir results/pillow_noise/v6.2.2 \
        --v7-dir results/pillow_noise/v7.0.0 \
        --output-dir results/pillow_noise/analysis
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze Pillow 6 vs 7 resize noise.")
    parser.add_argument("--v6-dir", type=str, required=True)
    parser.add_argument("--v7-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/pillow_noise/analysis")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(Path(args.v6_dir) / "metadata.json") as f:
        meta_v6 = json.load(f)
    with open(Path(args.v7_dir) / "metadata.json") as f:
        meta_v7 = json.load(f)

    logger.info("Comparing Pillow %s (NEAREST) vs %s (BICUBIC)",
                meta_v6["pillow_version"], meta_v7["pillow_version"])

    # Match files
    v6_files = {f.name: f for f in Path(args.v6_dir).glob("*.npy")}
    v7_files = {f.name: f for f in Path(args.v7_dir).glob("*.npy")}
    common = sorted(set(v6_files.keys()) & set(v7_files.keys()))
    logger.info("Found %d common images", len(common))

    # Analyze
    residuals = []
    per_class = {}

    for fname in common:
        arr6 = np.load(v6_files[fname])
        arr7 = np.load(v7_files[fname])
        residual = arr7 - arr6
        residuals.append(residual)

        cls = fname.rsplit("_img_", 1)[0] if "_img_" in fname else fname.split("_")[0]
        if cls not in per_class:
            per_class[cls] = []
        per_class[cls].append(float(np.abs(residual).mean()))

    residuals_stack = np.stack(residuals)

    # Summary
    summary = {
        "pillow_v6": meta_v6["pillow_version"],
        "pillow_v7": meta_v7["pillow_version"],
        "v6_filter": "NEAREST",
        "v7_filter": "BICUBIC",
        "n_images": len(common),
        "mean_abs_residual": float(np.abs(residuals_stack).mean()),
        "max_abs_residual": float(np.abs(residuals_stack).max()),
        "std_residual": float(residuals_stack.std()),
        "fraction_nonzero": float((residuals_stack != 0).mean()),
        "residual_range": [float(residuals_stack.min()), float(residuals_stack.max())],
        "per_class": {cls: {"n": len(v), "mean_abs_residual": float(np.mean(v))}
                      for cls, v in per_class.items()},
    }

    with open(out_dir / "pillow_resize_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    logger.info("=" * 72)
    logger.info("  PILLOW RESIZE DEFAULT CHANGE: NEAREST → BICUBIC")
    logger.info("  This is the root cause of the CSCS vs SIAM noise")
    logger.info("=" * 72)
    logger.info("  Pillow %s (NEAREST) vs %s (BICUBIC)", summary["pillow_v6"], summary["pillow_v7"])
    logger.info("  Images compared:      %d", summary["n_images"])
    logger.info("  Mean abs residual:    %.6f  (~%.1f pixel values)", summary["mean_abs_residual"], summary["mean_abs_residual"] * 255)
    logger.info("  Max abs residual:     %.6f  (~%.1f pixel values)", summary["max_abs_residual"], summary["max_abs_residual"] * 255)
    logger.info("  Std residual:         %.6f", summary["std_residual"])
    logger.info("  Fraction nonzero:     %.2f%%", summary["fraction_nonzero"] * 100)
    logger.info("-" * 72)
    logger.info("  Per-class mean absolute residual:")
    for cls, stats in sorted(summary["per_class"].items()):
        logger.info("    %-25s  n=%3d  residual=%.6f", cls, stats["n"], stats["mean_abs_residual"])
    logger.info("=" * 72)

    # Figures
    if HAS_MPL:
        _make_figures(residuals_stack, per_class, summary, out_dir)


def _make_figures(residuals_stack, per_class, summary, out_dir):
    import matplotlib.pyplot as plt

    # Figure 1: Residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    flat = residuals_stack.flatten()
    rng = np.random.RandomState(42)
    sample = rng.choice(flat, min(1_000_000, len(flat)), replace=False)
    ax.hist(sample, bins=100, density=True, alpha=0.7, color="#d32f2f")
    ax.set_xlabel("Residual (Pillow 7 BICUBIC - Pillow 6 NEAREST)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Pixel residual distribution\nPillow 6.2.2 vs 7.0.0 resize() default")
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    ax = axes[1]
    # Mean absolute residual image
    mean_res = np.abs(residuals_stack.mean(axis=0))
    amplified = mean_res / (mean_res.max() + 1e-10)
    ax.imshow(amplified)
    ax.set_title("(b) Mean absolute residual\n(amplified for visibility)")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_pillow_resize_distribution.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_pillow_resize_distribution.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Per-class residual
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = sorted(per_class.keys())
    resids = [np.mean(per_class[c]) for c in classes]
    counts = [len(per_class[c]) for c in classes]

    bars = ax.bar(range(len(classes)), resids, color="#d32f2f", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Absolute Residual")
    ax.set_title("Per-class noise from Pillow resize default change")
    ax.grid(axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={count}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(str(out_dir / "fig_pillow_resize_per_class.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_dir / "fig_pillow_resize_per_class.pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
