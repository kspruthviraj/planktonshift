"""
09_generality_test.py — Do these frequency-domain findings work beyond plankton?

STEP 9: GENERALITY TEST (NON-PLANKTON DATASETS)
================================================

A common critique: "your findings only work because plankton are transparent."
This script tests the frequency-domain framework on ANY imaging dataset, not
just plankton.

MATHEMATICAL PIPELINE:
======================

Same as Steps 01 + 03 combined, but generalised to any dataset:

Input: Two directories (source, target), each containing class subdirectories.
       Or: one directory + --simulate-domains flag.

Step 1 — Preprocess all images (Pipeline A: proportional padding + resize).

Step 2 — For each image, compute radial amplitude profile:
    A_bar(r) = radial_average( log(1 + |fftshift(FFT2{ I_gray })|) )

Step 3 — For each frequency band (low, mid, high):
    Build annular bandpass mask M_band(u, v) (see Step 03 for equations).
    Apply mask, reconstruct filtered image.
    Train ViT-B/16 on filtered source, test on filtered target.

Step 4 — Domain accuracy (de-confounded):
    For each band, extract radial amplitude features from BOTH domains.
    Train logistic regression to predict domain.
    Report delta vs all-frequencies baseline.

SIMULATED DOMAINS (--simulate-domains):
========================================
When only one dataset is available, we simulate a second "instrument" by
applying a deterministic acquisition transform:

    Step A — Split images 80/20 into train/test (no data leakage).
    Step B — For test images, apply simulate_acquisition():
        1. Resize to 96x96 with BICUBIC, then back to 224x224 with NEAREST
           (interpolation method signature — alters high-frequency content)
        2. Gamma correction: I_new = I^0.9
           (illumination signature — alters mid-frequency contrast)
        3. JPEG compression at quality=70
           (compression signature — adds high-frequency ringing artifacts)

    The simulated domain has a known instrument signature embedded in its
    frequency spectrum, providing a controlled test of the framework.

WHY IT MATTERS:
  If the frequency-domain separation of species info (low freq) from instrument
  artifacts (mid freq) holds across datasets, it's a general principle — not a
  plankton-specific coincidence.

Usage:
  # Plankton cross-instrument:
  python code/09_generality_test.py \
      --source data/cross_instrument/train/DataShift_IFCB \
      --target data/cross_instrument/test/DataShift_ZooScan \
      --tag plankton_cleaned --epochs 15

  # Non-plankton with simulated second domain:
  python code/09_generality_test.py \
      --source /path/to/any/dataset --simulate-domains \
      --tag my_dataset_generality --epochs 15

Output: results/generality/<tag>/frequency_decomposition.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RESULTS, FREQ_BAND_FRACTIONS as CONFIG_BANDS


# ---------------------------------------------------------------------------
# Preprocessing: Chen's Proportional-Padding pipeline (Pipeline A), exactly.
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


def preprocess(im_pil, pad_size=128, final_size=224):
    im = resize_with_proportions(im_pil, desired_size=pad_size)
    im = im.resize((final_size, final_size), Image.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Optional second "instrument" via deterministic acquisition transform.
# This is an on-thesis manipulation: it injects an instrument signature
# (resize-method + gamma + JPEG) into the frequency domain.
# ---------------------------------------------------------------------------
def simulate_acquisition(arr01):
    pil = Image.fromarray((arr01 * 255).clip(0, 255).astype(np.uint8))
    # resize-method signature (bicubic vs nearest differs in high-freq content)
    pil = pil.resize((96, 96), Image.BICUBIC).resize((224, 224), Image.NEAREST)
    arr = np.array(pil, dtype=np.float32) / 255.0
    # gamma signature (mid-frequency illumination/contrast change)
    arr = np.clip(arr, 0, 1) ** 0.9
    # JPEG compression signature (high-frequency ringing)
    import io
    buf = io.BytesIO()
    Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=70)
    arr = np.array(Image.open(buf), dtype=np.float32) / 255.0
    return arr


# ---------------------------------------------------------------------------
# Bandpass using the SAME band definition as the Fourier analysis.
# Edges are radial-bin fractions of r_max; default edges reproduce the paper's
# "0-22 / 22-44 / 44+" split (5 equal bands -> low=0-0.2, mid=0.2-0.4, high=0.4-1.0).
# ---------------------------------------------------------------------------
BAND_EDGES = CONFIG_BANDS  # from config.py, matching the paper


def bandpass_filter(arr01, band):
    """Apply annular bandpass on GRAYSCALE amplitude, keep phase, reconstruct to 3ch."""
    gray = np.mean(arr01, axis=2) if arr01.ndim == 3 else arr01
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    r_max = min(cy, cx)
    Y, X = np.ogrid[:rows, :cols]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_inner, r_outer = (r_max * e for e in BAND_EDGES[band])
    mask = ((R >= r_inner) & (R <= r_outer)).astype(np.float32)
    Fshift_filtered = Fshift * mask
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(Fshift_filtered)))
    # NOTE: no per-image min-max rescale (the original did this; it is a
    # contrast confound). We only clip to [0,1]; the DC band is in 'low' anyway.
    img_back = np.clip(img_back, 0.0, 1.0)
    return np.stack([img_back] * 3, axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def split_train_test_paths(root, classes, train_frac=0.8, seed=42):
    """Split image paths into train/test sets by class (80/20).
    Returns (train_paths, test_paths) as dicts {cls_name: [path, ...]}."""
    rng = np.random.RandomState(seed)
    train_paths, test_paths = {}, {}
    for cls_name in classes:
        cls_dir = Path(root) / cls_name
        if not cls_dir.is_dir():
            train_paths[cls_name] = []
            test_paths[cls_name] = []
            continue
        all_p = sorted([str(p) for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT])
        rng.shuffle(all_p)
        n_train = max(1, int(len(all_p) * train_frac))
        train_paths[cls_name] = all_p[:n_train]
        test_paths[cls_name] = all_p[n_train:]
    return train_paths, test_paths


class FreqDataset(Dataset):
    def __init__(self, root, classes, band="all", augment=False,
                 simulate_domain=False, allowed_paths=None):
        self.band = band
        self.augment = augment
        self.simulate_domain = simulate_domain
        self.samples = []
        for cls_name, idx in classes.items():
            if allowed_paths is not None:
                for p in allowed_paths.get(cls_name, []):
                    self.samples.append((p, idx))
            else:
                cls_dir = Path(root) / cls_name
                if not cls_dir.is_dir():
                    continue
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in SUPPORTED_EXT:
                        self.samples.append((str(p), idx))
        from torchvision import transforms
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        im = Image.open(path).convert("RGB")
        arr = preprocess(im)
        if self.simulate_domain:
            arr = simulate_acquisition(arr)
        if self.band != "all":
            arr = bandpass_filter(arr, self.band)
        if self.augment:
            pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            arr = np.array(self.aug_tf(pil), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------
def train_model(model, dataset, device, epochs, lr, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in tqdm(loader, desc=f"      {dataset.band} ep{ep+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def predict_species(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        preds.append(p)
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ---------------------------------------------------------------------------
# CLEAN domain-accuracy metric:
# Amplitude-feature classifier on HELD-OUT images from BOTH domains (not CLS of
# a source-trained model). Reports accuracy AND the delta vs the 'all' baseline.
# ---------------------------------------------------------------------------
def amplitude_features(root, classes, max_per_class=40, simulate_domain=False):
    feats, y = [], []
    for cls_name, idx in classes.items():
        cls_dir = Path(root) / cls_name
        if not cls_dir.is_dir():
            continue
        n = 0
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if n >= max_per_class:
                break
            im = Image.open(p).convert("RGB")
            arr = preprocess(im)
            if simulate_domain:
                arr = simulate_acquisition(arr)
            gray = np.mean(arr, axis=2)
            f = np.fft.fftshift(np.fft.fft2(gray))
            amp = np.log1p(np.abs(f))
            # radial average
            h, w = amp.shape
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
            rmax = min(cy, cx)
            prof = np.zeros(rmax)
            for r in range(rmax):
                m = R == r
                if m.any():
                    prof[r] = amp[m].mean()
            feats.append(prof)
            y.append(idx)
            n += 1
    return np.array(feats), np.array(y)


def domain_accuracy_amplitude(source_root, target_root, classes,
                              source_sim=False, target_sim=False, max_per_class=40):
    """Cross-validated accuracy of telling source vs target from amplitude spectra."""
    Xs, _ = amplitude_features(source_root, classes, max_per_class, source_sim)
    Xt, _ = amplitude_features(target_root, classes, max_per_class, target_sim)
    if len(Xs) == 0 or len(Xt) == 0:
        return None, 0
    maxlen = max(Xs.shape[1], Xt.shape[1])
    Xs2 = np.zeros((len(Xs), maxlen)); Xs2[:, :Xs.shape[1]] = Xs
    Xt2 = np.zeros((len(Xt), maxlen)); Xt2[:, :Xt.shape[1]] = Xt
    X = np.vstack([Xs2, Xt2])
    y = np.array([0] * len(Xs2) + [1] * len(Xt2))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()), len(y)


def bootstrap_accuracy_ci(preds, labels, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    correct = (preds == labels).astype(float)
    n = len(correct)
    stats = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_classes(source_root, target_root, simulate_domains):
    if simulate_domains:
        s = {d.name for d in Path(source_root).iterdir() if d.is_dir()}
        # target == source (synthesised); use all classes
        common = sorted(s)
    else:
        s = {d.name for d in Path(source_root).iterdir() if d.is_dir()}
        t = {d.name for d in Path(target_root).iterdir() if d.is_dir()}
        common = sorted(s & t)
    return {name: i for i, name in enumerate(common)}, common


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DATA["cross_ifcb"]), help="source domain root (domain/class/imgs)")
    ap.add_argument("--target", default=str(DATA["cross_zooscan"]), help="target domain root (ignored if --simulate-domains)")
    ap.add_argument("--tag", default="run", help="output subfolder name")
    ap.add_argument("--simulate-domains", action="store_true",
                    help="synthesize a second 'instrument' from --source (for single-acquisition non-plankton data)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-images", type=int, default=0, help="0=all")
    ap.add_argument("--bands", nargs="+", default=["low", "mid", "high", "all"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pipeline: Proportional-Padding(128)->Resize(224,BILINEAR)  [Chen-correct]")
    print(f"Band edges (frac of r_max): {BAND_EDGES}")
    print(f"Domain metric: amplitude-feature 5-fold CV on BOTH domains (de-confounded)")

    classes, names = discover_classes(args.source, args.target, args.simulate_domains)
    print(f"Common classes ({len(names)}): {names[:8]}{' ...' if len(names) > 8 else ''}")
    if len(names) < 2:
        print("ERROR: need >=2 classes shared by both domains.")
        return

    source_root = args.source
    target_root = args.source if args.simulate_domains else args.target

    # When simulating domains from a single dataset, split into train/test
    # to avoid data leakage (same images in both sets).
    train_paths_dict, test_paths_dict = None, None
    if args.simulate_domains:
        train_paths_dict, test_paths_dict = split_train_test_paths(
            source_root, classes, train_frac=0.8, seed=42)
        n_train = sum(len(v) for v in train_paths_dict.values())
        n_test = sum(len(v) for v in test_paths_dict.values())
        print(f"Simulate-domains: train/test split = {n_train}/{n_test} images")

    out_dir = RESULTS / "generality" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    baseline_domain = None
    # Precompute the no-masking ('all') amplitude domain accuracy as the reference.
    base_dom, base_n = domain_accuracy_amplitude(
        source_root, target_root, classes,
        source_sim=False,
        target_sim=args.simulate_domains)
    baseline_domain = base_dom
    print(f"\nBaseline amplitude domain accuracy (no masking): {base_dom:.3f} (n={base_n})")

    for band in args.bands:
        print(f"\n{'='*60}\nBand: {band.upper()}  (edges {BAND_EDGES[band]})\n{'='*60}")
        # Train on SOURCE with this band; species model.
        train_ds = FreqDataset(source_root, classes, band=band, augment=True,
                               simulate_domain=False,
                               allowed_paths=train_paths_dict)
        # Test on TARGET with the SAME band (target domain = real or simulated).
        test_ds = FreqDataset(target_root, classes, band=band, augment=False,
                              simulate_domain=args.simulate_domains,
                              allowed_paths=test_paths_dict)
        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"  skip {band}: empty")
            continue

        model = timm.create_model("vit_base_patch16_224", pretrained=True,
                                  num_classes=len(classes)).to(device)
        model = train_model(model, train_ds, device, args.epochs, args.lr)

        preds, labels = predict_species(model, test_ds, device)
        species_acc = float((preds == labels).mean())
        mean, lo, hi = bootstrap_accuracy_ci(preds, labels)
        print(f"  Species acc: {species_acc:.3f}  CI[{lo:.3f},{hi:.3f}]  (n_test={len(labels)})")

        # Clean domain accuracy: amplitude features of BOTH domains, band-restricted.
        dom_acc, dom_n = domain_accuracy_amplitude_band(
            source_root, target_root, classes, band,
            source_sim=False, target_sim=args.simulate_domains)
        delta_dom = (dom_acc - baseline_domain) if (dom_acc is not None and baseline_domain is not None) else None
        print(f"  Domain acc (amplitude, both domains): {dom_acc}  delta_vs_all={delta_dom}")

        results[band] = {
            "species_accuracy": species_acc,
            "species_ci95": [lo, hi],
            "n_test": int(len(labels)),
            "domain_accuracy_amplitude": dom_acc,
            "domain_delta_vs_all": delta_dom,
            "domain_n": dom_n,
            "band_edges_frac": BAND_EDGES[band],
        }
        del model
        torch.cuda.empty_cache()

    results["_meta"] = {
        "source": str(source_root),
        "target": str(target_root),
        "simulate_domains": args.simulate_domains,
        "preprocessing": "ProportionalPadding(128)->Resize(224,BILINEAR)",
        "domain_metric": "amplitude-feature 5-fold CV on both domains (de-confounded)",
        "baseline_domain_accuracy_all": baseline_domain,
        "n_classes": len(names),
        "classes": names,
        "epochs": args.epochs,
        "bands_tested": args.bands,
    }

    out_path = out_dir / "frequency_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nFREQUENCY DECOMPOSITION  (tag={args.tag})\n{'='*60}")
    print(f"{'Band':<6} {'Species':>8} {'95% CI':>16} {'Domain(amp)':>12} {'Δvs all':>9}")
    for b in args.bands:
        if b in results:
            r = results[b]
            ci = f"[{r['species_ci95'][0]:.3f},{r['species_ci95'][1]:.3f}]"
            dd = f"{r['domain_delta_vs_all']:+.3f}" if r['domain_delta_vs_all'] is not None else "NA"
            da = f"{r['domain_accuracy_amplitude']:.3f}" if r['domain_accuracy_amplitude'] else "NA"
            print(f"{b:<6} {r['species_accuracy']:>8.3f} {ci:>16} {da:>12} {dd:>9}")
    print(f"\nSaved: {out_path}")


def domain_accuracy_amplitude_band(source_root, target_root, classes, band,
                                   source_sim=False, target_sim=False, max_per_class=40):
    """Amplitude domain accuracy RESTRICTED to a frequency band (clean causal probe)."""
    feats_s, _ = _amplitude_features_band(source_root, classes, band, max_per_class, source_sim)
    feats_t, _ = _amplitude_features_band(target_root, classes, band, max_per_class, target_sim)
    if len(feats_s) == 0 or len(feats_t) == 0:
        return None, 0
    maxlen = max(feats_s.shape[1], feats_t.shape[1])
    Xs = np.zeros((len(feats_s), maxlen)); Xs[:, :feats_s.shape[1]] = feats_s
    Xt = np.zeros((len(feats_t), maxlen)); Xt[:, :feats_t.shape[1]] = feats_t
    X = np.vstack([Xs, Xt])
    y = np.array([0] * len(Xs) + [1] * len(Xt))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()), len(y)


def _amplitude_features_band(root, classes, band, max_per_class, simulate_domain):
    r_inner_frac, r_outer_frac = BAND_EDGES[band]
    feats, y = [], []
    for cls_name, idx in classes.items():
        cls_dir = Path(root) / cls_name
        if not cls_dir.is_dir():
            continue
        n = 0
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if n >= max_per_class:
                break
            im = Image.open(p).convert("RGB")
            arr = preprocess(im)
            if simulate_domain:
                arr = simulate_acquisition(arr)
            gray = np.mean(arr, axis=2)
            f = np.fft.fftshift(np.fft.fft2(gray))
            amp = np.log1p(np.abs(f))
            h, w = amp.shape
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            rmax = min(cy, cx)
            band_mask = ((R >= rmax * r_inner_frac) & (R <= rmax * r_outer_frac))
            rmax_int = int(rmax)
            prof = np.zeros(rmax_int)
            for r in range(rmax_int):
                m = (R == r) & band_mask
                if m.any():
                    prof[r] = amp[m].mean()
            feats.append(prof)
            y.append(idx)
            n += 1
    return np.array(feats), np.array(y)


if __name__ == "__main__":
    main()
