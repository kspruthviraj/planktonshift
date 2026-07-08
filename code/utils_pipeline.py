"""
utils_pipeline.py — Shared preprocessing, FFT, and statistics utilities.

This module centralises the CORRECTED pipeline (Pipeline A: Chen's Proportional
Padding) used by ALL experiments, ensuring consistency between the Fourier
analysis, frequency masking, SBA training, and temporal OOD evaluation.

EQUATIONS FOR KEY OPERATIONS:
==============================

1. PREPROCESSING (Pipeline A):
   Input:  PIL Image of arbitrary size (H, W)
   Step 1: Shrink keeping aspect ratio so max(H, W) = 128
           ratio = 128 / max(H, W)
           (H_new, W_new) = (int(H * ratio), int(W * ratio))
           I_shrunk = resize(I, (H_new, W_new), LANCZOS)
   Step 2: Black-pad to 128x128 square
           I_pad = zeros(128, 128, 3)
           offset = ((128 - W_new) // 2, (128 - H_new) // 2)
           I_pad[offset:offset+H_new, offset:offset+W_new] = I_shrunk
   Step 3: Resize to final size (224x224)
           I_out = resize(I_pad, (224, 224), BILINEAR)
   Step 4: Normalise to [0, 1]
           I_out = I_out / 255.0

2. AMPLITUDE SPECTRUM:
   Input:  grayscale image I_gray(x, y)
   F(u, v) = FFT2{ I_gray }                      (2D Fourier Transform)
   F_shift = fftshift(F)                          (center zero-frequency)
   A(u, v) = log(1 + |F_shift(u, v)|)            (log-amplitude)

3. RADIAL AVERAGE:
   For each radial bin r = 0, 1, ..., r_max:
       A_bar(r) = mean{ A(u, v) : sqrt((u-cx)^2 + (v-cy)^2) == r }
   where (cx, cy) = center of the spectrum, r_max = min(cx, cy).

4. BANDPASS FILTER:
   Given band edges (r_frac_inner, r_frac_outer) as fractions of r_max:
       M(u, v) = 1  if  r_frac_inner * r_max <= r(u,v) <= r_frac_outer * r_max
                 0  otherwise
   F_filtered = fftshift(F) * M
   I_filtered = real( IFFT2{ ifftshift(F_filtered) } )
   Output: I_filtered clipped to [0, 1], replicated to 3 channels.

5. BOOTSTRAP CI (for accuracy):
   Given N binary values c_1, ..., c_N (1=correct, 0=wrong):
   Repeat B=2000 times:
       sample = resample N values with replacement
       stat_b = mean(sample)
   CI_95 = [percentile(stat, 2.5), percentile(stat, 97.5)]

6. McNEMAR'S TEST:
   Given predictions from classifiers A and B on the same N test images:
       n01 = count(A correct AND B wrong)
       n10 = count(A wrong AND B correct)
   Under H0 (A and B equally good):  n01 ~ Binomial(n01+n10, 0.5)
   p-value = 2 * P(Binomial <= min(n01, n10) | n01+n10, 0.5)
   If p < 0.05, the difference is statistically significant.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.stats import binomtest
from config import FREQ_BAND_FRACTIONS

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Preprocessing: Chen's Proportional-Padding pipeline (Pipeline A)
# ---------------------------------------------------------------------------
def resize_with_proportions(im, desired_size=128):
    """Chen et al. ResizeWithProportions: shrink keeping aspect ratio, black-pad to square."""
    old_size = im.size
    if max(old_size) > desired_size:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple(int(x * ratio) for x in old_size)
        im = im.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (desired_size, desired_size), color=0)
    new_im.paste(im, ((desired_size - im.size[0]) // 2,
                      (desired_size - im.size[1]) // 2))
    return new_im


def preprocess_image(im_pil, pad_size=128, final_size=224):
    """Full Pipeline A: proportional padding -> resize to final_size."""
    im = resize_with_proportions(im_pil, desired_size=pad_size)
    im = im.resize((final_size, final_size), Image.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# FFT utilities
# ---------------------------------------------------------------------------
def compute_amplitude_spectrum(image_gray):
    """Compute log-amplitude spectrum (shifted)."""
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    return np.log1p(np.abs(fshift))


def radial_average(amplitude, n_bins=None):
    """Compute radially-averaged amplitude profile."""
    h, w = amplitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    rmax = min(cy, cx)
    if n_bins is None:
        n_bins = rmax
    profile = np.zeros(n_bins)
    for r in range(min(n_bins, rmax)):
        mask = R == r
        if mask.any():
            profile[r] = amplitude[mask].mean()
    return profile


def bandpass_filter(arr01, band, band_edges=None):
    """Apply annular bandpass on GRAYSCALE, keep phase, reconstruct to 3ch.

    Uses config.FREQ_BAND_FRACTIONS by default (matching the paper's bins).
    No per-image min-max rescale (that was a contrast confound in the original).
    """
    if band_edges is None:
        band_edges = FREQ_BAND_FRACTIONS
    gray = np.mean(arr01, axis=2) if arr01.ndim == 3 else arr01
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    r_max = min(cy, cx)
    Y, X = np.ogrid[:rows, :cols]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_inner, r_outer = (r_max * e for e in band_edges[band])
    mask = ((R >= r_inner) & (R <= r_outer)).astype(np.float32)
    Fshift_filtered = Fshift * mask
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(Fshift_filtered)))
    img_back = np.clip(img_back, 0.0, 1.0)
    return np.stack([img_back] * 3, axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistics: REAL bootstrap CIs + McNemar test
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap CI for a mean of 0/1 values (per-image correctness)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    stats = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return float(values.mean()), lo, hi


def bootstrap_auc_ci(scores, labels, n_boot=2000, seed=42):
    """Bootstrap CI for ROC-AUC using real resampling (not binomial approximation)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    aucs = np.array(aucs)
    point = roc_auc_score(labels, scores)
    return float(point), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def mcnemar_test(preds_a, preds_b, labels):
    """McNemar's test for paired comparison of two classifiers.

    Returns (statistic, p_value, discordant_counts).
    """
    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)
    # b: both correct or both wrong; c: a correct b wrong; d: a wrong b correct
    n01 = int((a_correct & ~b_correct).sum())  # A right, B wrong
    n10 = int((~a_correct & b_correct).sum())  # A wrong, B right
    n_discordant = n01 + n10
    if n_discordant == 0:
        return 0.0, 1.0, (n01, n10)
    # Use exact binomial for small samples
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 25 else None
    # binomtest.pvalue is already two-sided; do NOT multiply by 2
    test = binomtest(min(n01, n10), n01 + n10, 0.5)
    p_val = float(test.pvalue)
    return float(stat) if stat else float(n01 + n10), float(p_val), (n01, n10)


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def discover_classes(*roots):
    """Find class directories present in ALL given roots."""
    sets = []
    for root in roots:
        s = {d.name for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith(".")}
        sets.append(s)
    common = set.intersection(*sets) if sets else set()
    names = sorted(common)
    return {name: i for i, name in enumerate(names)}, names


def load_image_paths(root, classes_dict, max_per_class=0):
    """Load (path, label) pairs from a directory."""
    samples = []
    counts = {}
    for cls_name, idx in classes_dict.items():
        cls_dir = Path(root) / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if max_per_class and counts.get(cls_name, 0) >= max_per_class:
                break
            samples.append((str(p), idx))
            counts[cls_name] = counts.get(cls_name, 0) + 1
    return samples
