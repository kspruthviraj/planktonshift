"""
01_fourier_analysis.py — How do different cameras "see" plankton differently?

STEP 1: FOURIER-DOMAIN CHARACTERISATION OF CROSS-INSTRUMENT SHIFT
==================================================================

When two different cameras photograph the same plankton species, the images
look different — not because the organism changed, but because the camera's
lighting, resolution, and optics are different. This script decomposes those
differences into frequency layers using a 2D Fourier Transform (the same
math behind audio equalisers, but for images).

Think of it like separating a song into bass, midrange, and treble. We do
the same for images: low frequencies = overall shape and brightness, mid
frequencies = fine texture, high frequencies = pixel-level noise.

MATHEMATICAL PIPELINE:
======================

Input: RGB image I(x, y) of shape (224, 224, 3), pixel values in [0, 1].

Step 1 — Convert to grayscale:
    I_gray(x, y) = mean(I_R, I_G, I_B)

Step 2 — 2D Discrete Fourier Transform:
    F(u, v) = SUM_x SUM_y I_gray(x, y) * exp(-2*pi*i * (ux/W + vy/H))

    This converts the image from SPATIAL domain (pixel positions) to
    FREQUENCY domain (how fast pixels change in each direction).
    F(u, v) is a complex number at each frequency coordinate (u, v).

Step 3 — Amplitude spectrum (log-scaled):
    A(u, v) = log(1 + |F(u, v)|)

    |F(u, v)| = sqrt(Re(F)^2 + Im(F)^2) gives the "strength" of each
    frequency component. We take log(1 + .) to compress the dynamic range
    (amplitude spans several orders of magnitude).

Step 4 — Shift to center zero-frequency:
    A_shifted = fftshift(A)

    The FFT puts zero frequency at corner (0,0). We shift it to the center
    so that low frequencies are in the middle and high frequencies at edges.

Step 5 — Radial average (collapse 2D to 1D profile):
    A_bar(r) = mean{ A_shifted(u, v) : sqrt((u-cx)^2 + (v-cy)^2) = r }

    where (cx, cy) is the center of the spectrum. This averages all
    frequency components at the same distance r from the center, giving
    a 1D profile of "how much energy at each spatial frequency scale."

Step 6 — Shift spectrum (difference between cameras):
    Delta_A(r) = A_bar_source(r) - A_bar_target(r)

    This shows which frequency scales differ most between cameras.

Step 7 — Shift energy (single number summary):
    E = SUM_r [Delta_A(r)]^2

    Total squared difference across all frequency scales.

ANALYSIS:
=========
  (a) Domain classifier: logistic regression on A_bar(r) features to
      predict which camera took the image. If accuracy >> chance, the
      amplitude spectrum encodes camera identity.
  (b) Class separability: LDA on A_bar(r) features within each of 5
      frequency bands to measure how much species info each band carries.
  (c) Shift spectrum: Delta_A(r) between each pair of cameras, identifying
      which frequency scales carry the most camera-specific information.

WHY IT MATTERS:
  If cameras differ mostly in mid frequencies but species info is in low
  frequencies, we can augment only the camera-specific bands during training
  — making the model robust to camera differences without losing the ability
  to identify species.

Output: results/tier1_corrected/fourier_analysis.json
"""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RESULTS, SEEDS
from utils_pipeline import preprocess_image, compute_amplitude_spectrum, radial_average, bootstrap_ci

OUT = RESULTS / "tier1_corrected" / "fourier_analysis.json"
IMG_SIZE = 224
MAX_PER_CLASS = 40


def load_domain_images(root, max_per_class=MAX_PER_CLASS):
    """Load images using Pipeline A (proportional padding), return (images, class_labels, class_names)."""
    root = Path(root)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    images, labels = [], []
    for ci, cn in enumerate(class_names):
        n = 0
        for p in sorted((root / cn).iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                continue
            if n >= max_per_class:
                break
            try:
                im = Image.open(p).convert("RGB")
                arr = preprocess_image(im)
                images.append(arr)
                labels.append(ci)
                n += 1
            except Exception:
                continue
    return images, np.array(labels), class_names


def extract_radial_features(images, channel="gray"):
    """Extract radial amplitude features from a list of images.
    channel: 'gray' (mean), 'r', 'g', 'b', or 'all' (concatenated RGB).
    """
    feats = []
    for arr in images:
        if channel == "gray":
            gray = np.mean(arr, axis=2)
            amp = compute_amplitude_spectrum(gray)
            feats.append(radial_average(amp))
        elif channel in ("r", "g", "b"):
            ci = {"r": 0, "g": 1, "b": 2}[channel]
            amp = compute_amplitude_spectrum(arr[:, :, ci])
            feats.append(radial_average(amp))
        elif channel == "all":
            profs = []
            for ci in range(3):
                amp = compute_amplitude_spectrum(arr[:, :, ci])
                profs.append(radial_average(amp))
            feats.append(np.concatenate(profs))
    maxlen = max(len(f) for f in feats)
    X = np.zeros((len(feats), maxlen))
    for i, f in enumerate(feats):
        X[i, :len(f)] = f
    return X


def run_domain_classifier(domains_data, channel="gray"):
    """Train domain classifier from radial amplitude features, 5-fold CV."""
    X_all, y_all = [], []
    for di, (imgs, _, _) in enumerate(domains_data):
        X = extract_radial_features(imgs, channel)
        X_all.append(X)
        y_all.append(np.full(len(X), di))
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEEDS["numpy_default"])
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)
    importances = np.abs(clf.coef_).mean(axis=0)
    # Bootstrap CI over CV folds
    mean, lo, hi = bootstrap_ci(scores, seed=42)
    return {
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "ci_95": [lo, hi],
        "n_samples": len(X),
        "freq_importance": importances.tolist(),
        "most_discriminative_freqs": np.argsort(importances)[-10:][::-1].tolist(),
    }


def run_class_separability(imgs, labels, channel="gray"):
    """LDA class separability per frequency band (5 equal bands)."""
    X = extract_radial_features(imgs, channel)
    maxlen = X.shape[1]
    n_bands = 5
    band_size = maxlen // n_bands
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for bi in range(n_bands):
        s = bi * band_size
        e = (bi + 1) * band_size if bi < n_bands - 1 else maxlen
        X_band = X[:, s:e]
        if len(np.unique(labels)) < 2 or X_band.shape[1] < 1:
            results.append({"band": bi, "freq_range": [int(s), int(e)], "class_accuracy": 0.0})
            continue
        clf = LinearDiscriminantAnalysis()
        scores = cross_val_score(clf, X_band, labels, cv=cv, scoring="accuracy")
        results.append({
            "band": bi,
            "freq_range": [int(s), int(e)],
            "class_accuracy": float(scores.mean()),
            "class_accuracy_std": float(scores.std()),
        })
    return {"band_separability": results, "n_classes": len(np.unique(labels))}


def compute_shift_spectrum(spec_a, spec_b):
    r_a = np.array(spec_a["radial_mean"])
    r_b = np.array(spec_b["radial_mean"])
    min_len = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:min_len], r_b[:min_len]
    diff = r_a - r_b
    n_bands = 5
    band_size = min_len // n_bands
    bands = []
    for i in range(n_bands):
        s = i * band_size
        e = (i + 1) * band_size if i < n_bands - 1 else min_len
        bands.append({
            "band": i,
            "freq_range": [int(s), int(e)],
            "mean_abs_diff": float(np.abs(diff[s:e]).mean()),
        })
    return {
        "diff": diff.tolist(),
        "band_analysis": bands,
        "total_shift_energy": float(np.sum(diff ** 2)),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    domains = ["WHOI22", "ZooScan20", "ZooLake2"]
    paths = [DATA["whoi22"], DATA["zooscan20"], DATA["zoolake2"]]

    print(f"Fourier analysis — Pipeline A (Proportional Padding), {MAX_PER_CLASS} imgs/class")
    domain_data = {}
    for name, path in zip(domains, paths):
        print(f"  Loading {name} from {path}...")
        imgs, labels, cls_names = load_domain_images(path)
        domain_data[name] = (imgs, labels, cls_names)
        print(f"    {len(imgs)} images, {len(cls_names)} classes")

    # 1. Domain spectra (radial means)
    domain_spectra = {}
    for name, (imgs, _, _) in domain_data.items():
        X = extract_radial_features(imgs, "gray")
        domain_spectra[name] = {
            "radial_mean": X.mean(axis=0).tolist(),
            "radial_std": X.std(axis=0).tolist(),
            "n_samples": len(X),
        }

    # 2. Shift spectra
    shift_spectra = {}
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            pair = f"{domains[i]}→{domains[j]}"
            shift_spectra[pair] = compute_shift_spectrum(
                domain_spectra[domains[i]], domain_spectra[domains[j]])

    # 3. Domain classifier — grayscale
    print("\nDomain classifier (grayscale)...")
    dc_gray = run_domain_classifier(list(domain_data.values()), "gray")
    print(f"  Accuracy: {dc_gray['accuracy']:.4f} CI[{dc_gray['ci_95'][0]:.4f},{dc_gray['ci_95'][1]:.4f}]")

    # 4. Domain classifier — per-channel RGB (captures colour domain cues)
    print("Domain classifier (per-channel RGB)...")
    dc_rgb = run_domain_classifier(list(domain_data.values()), "all")
    print(f"  Accuracy: {dc_rgb['accuracy']:.4f} CI[{dc_rgb['ci_95'][0]:.4f},{dc_rgb['ci_95'][1]:.4f}]")

    # 5. Class separability per band
    class_sep = {}
    for name, (imgs, labels, _) in domain_data.items():
        print(f"  Class separability: {name}...")
        class_sep[name] = run_class_separability(imgs, labels, "gray")

    results = {
        "preprocessing": "ProportionalPadding(128)->Resize(224,BILINEAR) [Pipeline A]",
        "domain_spectra": domain_spectra,
        "shift_spectra": shift_spectra,
        "domain_classifier_gray": dc_gray,
        "domain_classifier_rgb": dc_rgb,
        "class_separability": class_sep,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}\nFOURIER ANALYSIS SUMMARY (Pipeline A)\n{'='*60}")
    for pair, sh in shift_spectra.items():
        bands = sh["band_analysis"]
        ms = max(bands, key=lambda b: b["mean_abs_diff"])
        print(f"  {pair}: energy={sh['total_shift_energy']:.3f}, most shifted band {ms['band']} (freq {ms['freq_range'][0]}-{ms['freq_range'][1]})")
    print(f"  Domain classifier (gray): {dc_gray['accuracy']:.1%} CI[{dc_gray['ci_95'][0]:.1%},{dc_gray['ci_95'][1]:.1%}]")
    print(f"  Domain classifier (RGB):  {dc_rgb['accuracy']:.1%} CI[{dc_rgb['ci_95'][0]:.1%},{dc_rgb['ci_95'][1]:.1%}]")
    for dom, sep in class_sep.items():
        b0 = sep["band_separability"][0]
        print(f"  {dom} band0 separability: {b0['class_accuracy']:.1%}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
