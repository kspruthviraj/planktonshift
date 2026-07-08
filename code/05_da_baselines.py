"""
05_da_baselines.py -- How does SBA compare to generic domain adaptation methods?

STEP 5: DOMAIN ADAPTATION BASELINES COMPARISON
===============================================

SBA is a domain adaptation (DA) method specifically designed for frequency-
domain shift. But there are generic DA methods that don't use frequency
analysis at all. This script compares SBA against:

  - Standard augmentation (random flips, rotations)
  - RandAugment (strong generic augmentation)
  - Heavy augmentation (color jitter, blur, rotations)
  - FDA (Fourier Domain Adaptation -- swaps low-frequency amplitude)
  - CORAL (Correlation Alignment -- matches feature statistics)

All methods are evaluated on the same IFCB->ZooScan benchmark with the same
preprocessing pipeline and random seed, so the comparison is fair.

WHY IT MATTERS:
  If generic methods perform as well as SBA, then the frequency-domain
  analysis (Steps 01-03) was unnecessary. If SBA outperforms, it validates
  the frequency-domain approach.

Output: results/tier1_corrected/da_baselines.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "adverserial_net"))
from config import DATA, RESULTS, SEEDS
from utils_pipeline import preprocess_image, bootstrap_ci, discover_classes, SUPPORTED_EXT
from spectral_augmentation import SpectralAugmentation

OUT = RESULTS / "tier1_corrected" / "da_baselines.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DADataset(Dataset):
    """Dataset with configurable augmentation strategy. Pipeline A preprocessing."""
    def __init__(self, root, classes_dict, augment_strategy="standard", target_root=None,
                 augment=True, sba=None):
        self.augment = augment
        self.strategy = augment_strategy
        self.sba = sba
        self.samples = []
        for cn, idx in classes_dict.items():
            cd = Path(root) / cn
            if not cd.is_dir():
                continue
            for p in sorted(cd.iterdir()):
                if p.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(p), idx))
        # Preload target images for FDA
        self.target_images = []
        if augment_strategy == "fda" and target_root:
            self._load_targets(target_root, classes_dict, max_total=50)
        self._setup_transforms()

    def _load_targets(self, root, classes_dict, max_total):
        for cn in classes_dict:
            cd = Path(root) / cn
            if not cd.is_dir():
                continue
            for p in sorted(cd.iterdir()):
                if p.suffix.lower() not in SUPPORTED_EXT:
                    continue
                if len(self.target_images) >= max_total:
                    return
                try:
                    im = Image.open(p).convert("L")
                    im = im.resize((224, 224))
                    self.target_images.append(np.array(im, dtype=np.float64) / 255.0)
                except Exception:
                    continue

    def _setup_transforms(self):
        if self.strategy == "standard":
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
            ])
        elif self.strategy == "randaugment":
            self.tf = transforms.Compose([
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
            ])
        elif self.strategy == "heavy":
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
            ])

    def __len__(self):
        return len(self.samples)

    def _apply_fda(self, arr01):
        """FDA: swap low-freq amplitude with random target image."""
        if not self.target_images:
            return arr01
        gray = np.mean(arr01, axis=2)
        target = self.target_images[np.random.randint(len(self.target_images))]
        f_src = np.fft.fft2(gray)
        f_tgt = np.fft.fft2(target)
        amp_src = np.abs(f_src)
        amp_tgt = np.abs(f_tgt)
        pha_src = np.angle(f_src)
        h, w = gray.shape
        beta = np.random.uniform(0.01, 0.05)
        hs = max(1, int(h * beta / 2))
        ws = max(1, int(w * beta / 2))
        amp_new = amp_src.copy()
        amp_new[:hs, :ws] = amp_tgt[:hs, :ws]
        amp_new[:hs, -ws:] = amp_tgt[:hs, -ws:]
        amp_new[-hs:, :ws] = amp_tgt[-hs:, :ws]
        amp_new[-hs:, -ws:] = amp_tgt[-hs:, -ws:]
        f_new = amp_new * np.exp(1j * pha_src)
        img_back = np.real(np.fft.ifft2(f_new)).clip(0, 1)
        return np.stack([img_back] * 3, axis=2).astype(np.float32)

    def __getitem__(self, i):
        path, label = self.samples[i]
        im = Image.open(path).convert("RGB")
        arr = preprocess_image(im)
        if self.augment:
            if self.strategy == "sba" and self.sba:
                gray = np.mean(arr, axis=2)
                gray_aug = self.sba(gray)
                arr = np.stack([gray_aug] * 3, axis=2)
            elif self.strategy == "fda":
                arr = self._apply_fda(arr)
            pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            arr = np.array(self.tf(pil), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


class CORALLoss(nn.Module):
    """CORAL: Correlation Alignment loss (Sun & Saenko 2016).
    Minimizes the difference between source and target feature covariances."""
    def forward(self, source_feats, target_feats):
        d = source_feats.size(1)
        n_s = source_feats.size(0)
        n_t = target_feats.size(0)
        if n_s < 2 or n_t < 2:
            return torch.tensor(0.0, device=source_feats.device)
        source = source_feats - source_feats.mean(0)
        target = target_feats - target_feats.mean(0)
        cov_s = (source.T @ source) / (n_s - 1)
        cov_t = (target.T @ target) / (n_t - 1)
        diff = cov_s - cov_t
        return (diff * diff).sum() / (4 * d * d)


def train_with_coral(train_ds, target_ds, test_ds, num_classes, seed, epochs=30, lr=1e-4, coral_weight=0.5):
    """Train with CORAL feature alignment (uses unlabeled target features)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    target_loader = DataLoader(target_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    coral = CORALLoss().to(DEVICE)
    model.train()
    for ep in range(epochs):
        target_iter = iter(target_loader)
        for x_s, y_s in tqdm(train_loader, desc=f"    CORAL ep{ep+1}/{epochs}", leave=False):
            x_s, y_s = x_s.to(DEVICE), y_s.to(DEVICE)
            try:
                x_t, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_t, _ = next(target_iter)
            x_t = x_t.to(DEVICE)
            opt.zero_grad()
            feats_s = model.forward_features(x_s)[:, 0]
            feats_t = model.forward_features(x_t)[:, 0]
            logits = model.head(feats_s) if hasattr(model, 'head') else model(x_s)
            cls_loss = crit(logits, y_s)
            coral_loss = coral(feats_s, feats_t)
            loss = cls_loss + coral_weight * coral_loss
            loss.backward()
            opt.step()
        sch.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            preds.append(p)
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    acc = float((preds == labels).mean())
    mean, lo, hi = bootstrap_ci((preds == labels).astype(float))
    del model
    torch.cuda.empty_cache()
    return acc, lo, hi


def train_standard(train_ds, test_ds, num_classes, seed, epochs=30, lr=1e-4):
    """Standard training (no DA)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in tqdm(train_loader, desc=f"    ep{ep+1}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            preds.append(p)
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    acc = float((preds == labels).mean())
    mean, lo, hi = bootstrap_ci((preds == labels).astype(float))
    del model
    torch.cuda.empty_cache()
    return acc, lo, hi


def load_shift_spectrum():
    path = RESULTS / "tier1_corrected" / "fourier_analysis.json"
    if path.exists():
        with open(path) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "WHOI" in key and "ZooScan" in key:
                return np.array(val.get("diff", []))
    path2 = RESULTS / "adverserial_net" / "fourier_analysis_zoolake2" / "fourier_analysis.json"
    if path2.exists():
        with open(path2) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "WHOI" in key and "ZooScan" in key:
                return np.array(val.get("diff", []))
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = DATA["cross_ifcb"]
    target = DATA["cross_zooscan"]
    print(f"Device: {DEVICE}")
    print(f"DA Baselines comparison (Pipeline A)")
    classes_dict, names = discover_classes(source, target)
    print(f"Common classes ({len(names)}): {names}")
    seed = SEEDS["ablation"]

    shift_spectrum = load_shift_spectrum()
    sba = SpectralAugmentation(
        shift_spectrum=shift_spectrum, strength=0.5,
        strategies=["spectral_noise", "band_adversarial"], p=0.8,
    ) if shift_spectrum is not None else None

    strategies = {
        "standard": {"strategy": "standard"},
        "randaugment": {"strategy": "randaugment"},
        "heavy": {"strategy": "heavy"},
        "fda": {"strategy": "fda"},
        "sba_band": {"strategy": "sba", "sba": sba},
        "coral": {"strategy": "standard", "coral": True},
    }

    results = {"_meta": {
        "source": str(source), "target": str(target),
        "preprocessing": "ProportionalPadding(128)->Resize(224) [Pipeline A]",
        "seed": seed, "n_classes": len(names),
    }}

    for name, cfg in strategies.items():
        print(f"\n{'='*60}\nStrategy: {name}\n{'='*60}")
        train_ds = DADataset(source, classes_dict, augment_strategy=cfg["strategy"],
                             target_root=target if cfg["strategy"] == "fda" else None,
                             augment=True, sba=cfg.get("sba"))
        test_ds = DADataset(target, classes_dict, augment=False)
        if cfg.get("coral"):
            target_ds = DADataset(target, classes_dict, augment_strategy="standard", augment=True)
            acc, lo, hi = train_with_coral(train_ds, target_ds, test_ds, len(names), seed)
        else:
            acc, lo, hi = train_standard(train_ds, test_ds, len(names), seed)
        results[name] = {"accuracy": acc, "ci_95": [lo, hi]}
        print(f"  {name}: acc={acc:.4f} CI[{lo:.4f},{hi:.4f}]")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nDA BASELINES COMPARISON\n{'='*60}")
    print(f"{'Strategy':<15} {'Accuracy':>10} {'95% CI':>18}")
    for name in strategies:
        if name in results:
            r = results[name]
            ci = f"[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]"
            print(f"{name:<15} {r['accuracy']:>10.3f} {ci:>18}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
