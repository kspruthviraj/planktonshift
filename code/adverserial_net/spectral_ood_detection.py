"""
spectral_ood_detection.py
==========================
Out-of-distribution detection using spectral (Fourier) features.

Train an Isolation Forest on source-domain amplitude spectra.
At test time, check if a test image's spectrum is OOD.

If OOD → route to VLM (more robust)
If ID → use conventional model (faster, cheaper)

Usage:
    python spectral_ood_detection.py \
        --train-dir data/cross_domain/cross_instrument/train/DataShift_IFCB \
        --test-dirs data/cross_domain/cross_instrument/test/DataShift_ZooScan \
                    data/cross_domain/whoi22_full/test/ZooLake35 \
        --output-dir results/ood_detection
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


def compute_radial_spectrum(image_path: str) -> np.ndarray:
    """Compute radially-averaged amplitude spectrum."""
    img = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float64) / 255.0
    f = np.fft.fft2(arr)
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


def collect_features(data_dir: str, max_images: int = 200):
    """Collect spectral features from a directory."""
    features = []
    paths = []
    data_path = Path(data_dir)
    count = 0
    for cls_dir in sorted(data_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            try:
                feat = compute_radial_spectrum(str(img_path))
                features.append(feat)
                paths.append(str(img_path))
                count += 1
            except Exception:
                continue
            if count >= max_images:
                break
    return np.array(features), paths


def main():
    parser = argparse.ArgumentParser(description="Spectral OOD detection.")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="results/ood_detection")
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=200)
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

    # Collect source features
    logger.info("Collecting source features from %s...", args.train_dir)
    src_features, src_paths = collect_features(args.train_dir, args.max_train)
    logger.info("  %d source features", len(src_features))

    # Train Isolation Forest
    logger.info("Training Isolation Forest...")
    clf = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    clf.fit(src_features)

    # Source scores (should be high/in-distribution)
    src_scores = clf.decision_function(src_features)

    # Evaluate on each test directory
    results = {"source": {
        "n": len(src_features),
        "mean_score": float(src_scores.mean()),
        "std_score": float(src_scores.std()),
    }}

    for test_dir in args.test_dirs:
        logger.info("Evaluating %s...", test_dir)
        test_features, test_paths = collect_features(test_dir, args.max_test)
        if len(test_features) == 0:
            continue

        test_scores = clf.decision_function(test_features)

        # Binary labels: source=1 (ID), test=0 (OOD)
        all_scores = np.concatenate([src_scores, test_scores])
        all_labels = np.concatenate([np.ones(len(src_scores)), np.zeros(len(test_scores))])

        # ROC AUC
        auc_score = roc_auc_score(all_labels, all_scores)

        # Precision-recall
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall, precision)

        domain_name = Path(test_dir).name
        results[domain_name] = {
            "n_test": len(test_features),
            "mean_score": float(test_scores.mean()),
            "std_score": float(test_scores.std()),
            "roc_auc": float(auc_score),
            "pr_auc": float(pr_auc),
            "separation": float(src_scores.mean() - test_scores.mean()),
        }

        logger.info("  %s: ROC-AUC=%.3f  PR-AUC=%.3f  Separation=%.4f",
                     domain_name, auc_score, pr_auc,
                     src_scores.mean() - test_scores.mean())

        # Figure
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(src_scores, bins=30, density=True, alpha=0.6, color="#2196F3", label="Source (ID)")
            ax.hist(test_scores, bins=30, density=True, alpha=0.6, color="#d32f2f", label=f"{domain_name} (OOD)")
            ax.set_xlabel("Isolation Forest Score")
            ax.set_ylabel("Density")
            ax.set_title(f"OOD Detection: {domain_name} vs Source (AUC={auc_score:.3f})")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(str(out_dir / f"fig_ood_{domain_name}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Save results
    with open(out_dir / "ood_detection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("  SPECTRAL OOD DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Source: mean_score=%.4f (std=%.4f)",
                results["source"]["mean_score"], results["source"]["std_score"])
    for domain, data in results.items():
        if domain == "source":
            continue
        logger.info("  %-20s  ROC-AUC=%.3f  PR-AUC=%.3f  Sep=%.4f",
                     domain, data["roc_auc"], data["pr_auc"], data["separation"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
