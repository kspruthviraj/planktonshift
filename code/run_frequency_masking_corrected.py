"""
run_frequency_masking_corrected.py — Causal frequency-masking experiment.

FIXES (P0-2, P0-3, P0-4):
  - Pipeline A (Proportional Padding) for all images.
  - Band definitions from config.FREQ_BAND_FRACTIONS matching the paper
    (low=0-22, mid=22-44, high=44+), NOT the fraction-of-Nyquist annuli.
  - De-confounded domain accuracy: amplitude-feature classifier on held-out
    images from BOTH domains (not CLS of a source-trained model). Reports
    the DELTA over the no-masking baseline.
  - Real bootstrap CIs on species accuracy.
  - Uses vendored data paths (no external dependencies).

Output: results/tier1_corrected/frequency_masking.json
"""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RESULTS, SEEDS, FREQ_BAND_FRACTIONS
from utils_pipeline import preprocess_image, bandpass_filter, bootstrap_ci, discover_classes, SUPPORTED_EXT

OUT = RESULTS / "tier1_corrected" / "frequency_masking.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FreqMaskedDataset(Dataset):
    def __init__(self, root, classes_dict, band="all", augment=False):
        self.band = band
        self.augment = augment
        self.samples = []
        for cls_name, idx in classes_dict.items():
            cls_dir = Path(root) / cls_name
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(p), idx))
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
        arr = preprocess_image(im)  # Pipeline A
        if self.band != "all":
            arr = bandpass_filter(arr, self.band)
        if self.augment:
            pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            arr = np.array(self.aug_tf(pil), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


def train_model(model, dataset, epochs=15, lr=1e-4, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in tqdm(loader, desc=f"      ep{ep+1}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def predict_species(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
        preds.append(p)
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# De-confounded domain accuracy: amplitude features from BOTH domains
def amplitude_domain_accuracy(source_root, target_root, classes_dict, band="all", max_per_class=40):
    """Domain classifier on radial amplitude features restricted to a band.
    Uses held-out images from BOTH domains — not CLS of a source-trained model.
    """
    Xs, _ = _amp_features_band(source_root, classes_dict, band, max_per_class)
    Xt, _ = _amp_features_band(target_root, classes_dict, band, max_per_class)
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


def _amp_features_band(root, classes_dict, band, max_per_class):
    r_in, r_out = FREQ_BAND_FRACTIONS[band]
    feats, y = [], []
    for cn, idx in classes_dict.items():
        cd = Path(root) / cn
        if not cd.is_dir():
            continue
        n = 0
        for p in sorted(cd.iterdir()):
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if n >= max_per_class:
                break
            try:
                im = Image.open(p).convert("RGB")
                arr = preprocess_image(im)
                gray = np.mean(arr, axis=2)
                f = np.fft.fftshift(np.fft.fft2(gray))
                amp = np.log1p(np.abs(f))
                h, w = amp.shape
                cy, cx = h // 2, w // 2
                Y, X = np.ogrid[:h, :w]
                R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                rmax = min(cy, cx)
                bmask = ((R >= rmax * r_in) & (R <= rmax * r_out))
                rmax_i = int(rmax)
                prof = np.zeros(rmax_i)
                for r in range(rmax_i):
                    m = (R == r) & bmask
                    if m.any():
                        prof[r] = amp[m].mean()
                feats.append(prof)
                y.append(idx)
                n += 1
            except Exception:
                continue
    return np.array(feats), np.array(y)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = DATA["cross_ifcb"]
    target = DATA["cross_zooscan"]
    print(f"Device: {DEVICE}")
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Pipeline: Proportional Padding (A)")
    print(f"Bands: {FREQ_BAND_FRACTIONS}")

    classes_dict, names = discover_classes(source, target)
    print(f"Common classes ({len(names)}): {names}")
    if len(names) < 2:
        print("ERROR: need >=2 common classes")
        return

    # Baseline domain accuracy (no masking = 'all')
    base_dom, base_n = amplitude_domain_accuracy(source, target, classes_dict, "all")
    print(f"\nBaseline domain accuracy (amplitude, both domains): {base_dom:.4f} (n={base_n})")

    results = {"_meta": {
        "source": str(source), "target": str(target),
        "preprocessing": "ProportionalPadding(128)->Resize(224) [Pipeline A]",
        "domain_metric": "amplitude 5-fold CV on BOTH domains (de-confounded)",
        "baseline_domain_all": base_dom, "n_classes": len(names),
        "bands": list(FREQ_BAND_FRACTIONS.keys()),
    }}

    for band in ["low", "mid", "high", "all"]:
        print(f"\n{'='*60}\nBand: {band.upper()} (edges {FREQ_BAND_FRACTIONS[band]})\n{'='*60}")
        train_ds = FreqMaskedDataset(source, classes_dict, band=band, augment=True)
        test_ds = FreqMaskedDataset(target, classes_dict, band=band, augment=False)
        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"  skip {band}: empty")
            continue
        print(f"  train={len(train_ds)} test={len(test_ds)}")

        torch.manual_seed(SEEDS["ablation"])
        np.random.seed(SEEDS["ablation"])
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=len(names)).to(DEVICE)
        model = train_model(model, train_ds, epochs=15)

        preds, labels = predict_species(model, test_ds)
        acc = float((preds == labels).mean())
        mean, lo, hi = bootstrap_ci((preds == labels).astype(float))
        print(f"  Species acc: {acc:.4f} CI[{lo:.4f},{hi:.4f}] n={len(labels)}")

        dom_acc, dom_n = amplitude_domain_accuracy(source, target, classes_dict, band)
        delta = (dom_acc - base_dom) if (dom_acc and base_dom) else None
        print(f"  Domain acc (amp): {dom_acc:.4f} delta_vs_all={delta:+.4f}" if dom_acc else "  Domain acc: N/A")

        results[band] = {
            "species_accuracy": acc,
            "species_ci95": [lo, hi],
            "n_test": int(len(labels)),
            "domain_accuracy_amplitude": dom_acc,
            "domain_delta_vs_all": delta,
            "domain_n": dom_n,
        }
        del model
        torch.cuda.empty_cache()

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nFREQUENCY MASKING (corrected)\n{'='*60}")
    print(f"{'Band':<6} {'Species':>8} {'95% CI':>18} {'Domain(amp)':>12} {'Δvs all':>9}")
    for b in ["low", "mid", "high", "all"]:
        if b in results:
            r = results[b]
            ci = f"[{r['species_ci95'][0]:.3f},{r['species_ci95'][1]:.3f}]"
            dd = f"{r['domain_delta_vs_all']:+.3f}" if r['domain_delta_vs_all'] is not None else "NA"
            da = f"{r['domain_accuracy_amplitude']:.3f}" if r['domain_accuracy_amplitude'] else "NA"
            print(f"{b:<6} {r['species_accuracy']:>8.3f} {ci:>18} {da:>12} {dd:>9}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
