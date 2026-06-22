"""
fourier_shift_analysis.py
=========================
Fourier-domain analysis of dataset shift across plankton imaging modalities.

Computes 2D FFT of images from each domain (IFCB, ZooScan, ZooLake),
analyzes amplitude spectra differences, and identifies which frequency
bands carry domain-specific information vs. morphological information.

This is the foundational analysis for the paper:
"From Adversarial Noise to Domain Shift: Fourier-Grounded Robust Plankton
Classification Across Imaging Modalities"

Usage:
    python fourier_shift_analysis.py \
        --data-root data/eval \
        --domains IFCB ZooScan \
        --output-dir results/fourier_analysis
"""

import argparse
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = [
    "Amphipoda", "Annelida", "Ceratium", "Chaetognatha",
    "Coscinodiscus", "Noctiluca",
]
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_images(data_root: str, domain: str, cls: str, max_images: int = 40):
    """Load images as grayscale numpy arrays."""
    cls_dir = Path(data_root) / domain / cls
    if not cls_dir.is_dir():
        return []
    images = []
    for img_path in sorted(cls_dir.iterdir()):
        if img_path.suffix.lower() not in SUPPORTED_EXT:
            continue
        try:
            img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float64) / 255.0)
        except Exception:
            continue
        if len(images) >= max_images:
            break
    return images


# ---------------------------------------------------------------------------
# Fourier analysis
# ---------------------------------------------------------------------------
def compute_amplitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute log-amplitude spectrum of a 2D image."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    amplitude = np.abs(fshift)
    return np.log1p(amplitude)


def radial_average(amplitude: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Compute radially-averaged amplitude spectrum."""
    h, w = amplitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for r in range(max_r):
        mask = R == r
        if mask.sum() > 0:
            radial_profile[r] = amplitude[mask].mean()
    return radial_profile


def compute_domain_spectra(
    data_root: str, domain: str, classes: list, max_per_class: int = 40
) -> dict:
    """Compute average amplitude spectra for a domain."""
    all_radial = []
    all_amplitudes = []

    for cls in classes:
        images = load_images(data_root, domain, cls, max_per_class)
        if not images:
            logger.warning("No images for %s/%s", domain, cls)
            continue
        for img in images:
            amp = compute_amplitude_spectrum(img)
            radial = radial_average(amp)
            all_radial.append(radial)
            all_amplitudes.append(amp)

    if not all_radial:
        return {}

    # Pad to same length
    max_len = max(len(r) for r in all_radial)
    radial_matrix = np.zeros((len(all_radial), max_len))
    for i, r in enumerate(all_radial):
        radial_matrix[i, :len(r)] = r

    return {
        "radial_mean": radial_matrix.mean(axis=0).tolist(),
        "radial_std": radial_matrix.std(axis=0).tolist(),
        "n_samples": len(all_radial),
        "amplitude_mean": np.mean(all_amplitudes, axis=0).tolist(),
    }


def compute_shift_spectrum(spec_a: dict, spec_b: dict) -> dict:
    """Compute the frequency-domain shift between two domains."""
    r_a = np.array(spec_a["radial_mean"])
    r_b = np.array(spec_b["radial_mean"])
    min_len = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:min_len], r_b[:min_len]

    diff = r_a - r_b
    relative_diff = diff / (np.maximum(r_a, r_b) + 1e-8)

    # Identify frequency bands with largest differences
    n_bands = 5
    band_size = min_len // n_bands
    band_diffs = []
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else min_len
        band_diffs.append({
            "band": i,
            "freq_range": [start, end],
            "mean_abs_diff": float(np.abs(diff[start:end]).mean()),
            "mean_rel_diff": float(np.abs(relative_diff[start:end]).mean()),
        })

    return {
        "diff": diff.tolist(),
        "relative_diff": relative_diff.tolist(),
        "band_analysis": band_diffs,
        "total_shift_energy": float(np.sum(diff ** 2)),
    }


# ---------------------------------------------------------------------------
# Domain classification from amplitude spectra
# ---------------------------------------------------------------------------
def train_domain_classifier(data_root: str, domains: list, classes: list):
    """Train a linear probe on amplitude spectra to classify domain.
    
    High accuracy = the shift is concentrated in identifiable frequency bands.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = [], []
    for d_idx, domain in enumerate(domains):
        for cls in classes:
            images = load_images(data_root, domain, cls, max_images=40)
            for img in images:
                amp = compute_amplitude_spectrum(img)
                radial = radial_average(amp)
                X.append(radial)
                y.append(d_idx)

    if len(X) < 10:
        logger.warning("Not enough samples for domain classification")
        return {}

    max_len = max(len(x) for x in X)
    X_matrix = np.zeros((len(X), max_len))
    for i, x in enumerate(X):
        X_matrix[i, :len(x)] = x

    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, X_matrix, y, cv=5, scoring="accuracy")

    # Also get feature importances (which frequencies are most discriminative)
    clf.fit(X_matrix, y)
    importances = np.abs(clf.coef_).mean(axis=0)

    return {
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_samples": len(X),
        "freq_importance": importances.tolist(),
        "most_discriminative_freqs": np.argsort(importances)[-10:][::-1].tolist(),
    }


# ---------------------------------------------------------------------------
# Morphology preservation analysis
# ---------------------------------------------------------------------------
def compute_class_separability(
    data_root: str, domain: str, classes: list, max_per_class: int = 40
):
    """Measure how well classes are separated in frequency space.
    
    If classes remain separable in high-frequency bands, morphology is preserved.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score

    X, y = [], []
    for cls_idx, cls in enumerate(classes):
        images = load_images(data_root, domain, cls, max_per_class)
        for img in images:
            amp = compute_amplitude_spectrum(img)
            radial = radial_average(amp)
            X.append(radial)
            y.append(cls_idx)

    if len(X) < 10:
        return {}

    max_len = max(len(x) for x in X)
    X_matrix = np.zeros((len(X), max_len))
    for i, x in enumerate(X):
        X_matrix[i, :len(x)] = x

    # Test separability in different frequency bands
    n_bands = 5
    band_size = max_len // n_bands
    band_results = []

    for band in range(n_bands):
        start = band * band_size
        end = (band + 1) * band_size if band < n_bands - 1 else max_len
        X_band = X_matrix[:, start:end]

        clf = LinearDiscriminantAnalysis()
        scores = cross_val_score(clf, X_band, y, cv=5, scoring="accuracy")
        band_results.append({
            "band": band,
            "freq_range": [int(start), int(end)],
            "class_accuracy": float(scores.mean()),
            "class_accuracy_std": float(scores.std()),
        })

    return {"band_separability": band_results, "n_classes": len(classes)}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_figures(results: dict, output_dir: str):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Radial amplitude spectra per domain
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for domain, spec in results.get("domain_spectra", {}).items():
        radial = np.array(spec["radial_mean"])
        ax.plot(radial, label=domain)
    ax.set_xlabel("Spatial Frequency (radial bin)")
    ax.set_ylabel("Log Amplitude")
    ax.set_title("Radial Amplitude Spectra Across Domains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_radial_spectra.png"), dpi=150)
    plt.close(fig)

    # Figure 2: Shift spectrum (difference between domains)
    if "shift_spectra" in results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for pair, shift in results["shift_spectra"].items():
            diff = np.array(shift["diff"])
            ax.plot(diff, label=pair)
        ax.set_xlabel("Spatial Frequency (radial bin)")
        ax.set_ylabel("Amplitude Difference")
        ax.set_title("Cross-Domain Shift Spectrum")
        ax.legend()
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig2_shift_spectrum.png"), dpi=150)
        plt.close(fig)

    # Figure 3: Frequency band importance for domain classification
    if "domain_classifier" in results and results["domain_classifier"]:
        dc = results["domain_classifier"]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        importances = np.array(dc["freq_importance"])
        ax.bar(range(len(importances)), importances, alpha=0.7)
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Classification Importance")
        ax.set_title(
            f"Frequency Bands for Domain Classification "
            f"(Acc: {dc['accuracy']:.1%})"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig3_domain_freq_importance.png"), dpi=150)
        plt.close(fig)

    # Figure 4: Class separability per frequency band
    if "class_separability" in results:
        for domain, sep in results["class_separability"].items():
            if not sep:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            bands = sep["band_separability"]
            band_labels = [f"{b['freq_range'][0]}-{b['freq_range'][1]}" for b in bands]
            accuracies = [b["class_accuracy"] for b in bands]
            errors = [b["class_accuracy_std"] for b in bands]
            ax.bar(band_labels, accuracies, yerr=errors, alpha=0.7)
            ax.set_xlabel("Frequency Band Range")
            ax.set_ylabel("Class Separability (LDA Accuracy)")
            ax.set_title(f"Class Separability by Frequency Band ({domain})")
            ax.set_ylim([0, 1])
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, f"fig4_class_separability_{domain}.png"),
                dpi=150,
            )
            plt.close(fig)

    logger.info("Figures saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fourier-domain shift analysis.")
    parser.add_argument("--data-root", type=str, default="data/eval")
    parser.add_argument("--domains", nargs="+", default=["IFCB", "ZooScan"])
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Classes to analyze. If not specified, auto-discovers from directory.")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default="results/fourier_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-discover classes if not specified
    if args.classes is None:
        all_classes = set()
        for domain in args.domains:
            domain_dir = Path(args.data_root) / domain
            if domain_dir.is_dir():
                for cls_dir in sorted(domain_dir.iterdir()):
                    if cls_dir.is_dir() and not cls_dir.name.startswith("."):
                        all_classes.add(cls_dir.name)
        args.classes = sorted(all_classes)
        logger.info("Auto-discovered %d classes: %s", len(args.classes), args.classes[:10])

    # 1. Compute per-domain spectra
    logger.info("Computing per-domain amplitude spectra...")
    domain_spectra = {}
    for domain in args.domains:
        logger.info("  Processing %s...", domain)
        domain_spectra[domain] = compute_domain_spectra(
            args.data_root, domain, args.classes, args.max_per_class
        )

    # 2. Compute cross-domain shift spectra
    logger.info("Computing shift spectra...")
    shift_spectra = {}
    for i, d1 in enumerate(args.domains):
        for d2 in args.domains[i + 1:]:
            if not domain_spectra.get(d1) or not domain_spectra.get(d2):
                logger.warning("  Skipping %s→%s (empty domain spectra)", d1, d2)
                continue
            pair = f"{d1}→{d2}"
            logger.info("  %s", pair)
            shift_spectra[pair] = compute_shift_spectrum(
                domain_spectra[d1], domain_spectra[d2]
            )

    # 3. Domain classification from spectra
    logger.info("Training domain classifier from amplitude spectra...")
    domain_classifier = train_domain_classifier(
        args.data_root, args.domains, args.classes
    )

    # 4. Class separability per frequency band
    logger.info("Computing class separability per frequency band...")
    class_separability = {}
    for domain in args.domains:
        logger.info("  %s", domain)
        class_separability[domain] = compute_class_separability(
            args.data_root, domain, args.classes, args.max_per_class
        )

    # 5. Compile results
    results = {
        "domain_spectra": domain_spectra,
        "shift_spectra": shift_spectra,
        "domain_classifier": domain_classifier,
        "class_separability": class_separability,
    }

    # Save results
    output_path = os.path.join(args.output_dir, "fourier_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # 6. Generate figures
    generate_figures(results, args.output_dir)

    # 7. Print summary
    logger.info("=" * 72)
    logger.info("  FOURIER SHIFT ANALYSIS SUMMARY")
    logger.info("=" * 72)
    for pair, shift in shift_spectra.items():
        bands = shift["band_analysis"]
        most_shifted = max(bands, key=lambda b: b["mean_abs_diff"])
        logger.info("  %s: Total shift energy = %.4f", pair, shift["total_shift_energy"])
        logger.info("    Most shifted band: %d (freq %d-%d, diff=%.4f)",
                     most_shifted["band"], most_shifted["freq_range"][0],
                     most_shifted["freq_range"][1], most_shifted["mean_abs_diff"])

    if domain_classifier:
        logger.info("  Domain classifier accuracy: %.1f%% (±%.1f%%)",
                     domain_classifier["accuracy"] * 100,
                     domain_classifier["accuracy_std"] * 100)
        logger.info("  Top discriminative frequencies: %s",
                     domain_classifier["most_discriminative_freqs"][:5])

    for domain, sep in class_separability.items():
        if sep:
            for band in sep["band_separability"]:
                logger.info("  %s band %d: class separability = %.1f%%",
                            domain, band["band"], band["class_accuracy"] * 100)

    logger.info("=" * 72)


if __name__ == "__main__":
    main()
