"""
tier1_reverse_transfer.py — Test SBA on reverse cross-instrument transfers.

Tests:
  1. ZooScan → IFCB (reverse of main benchmark)
  2. WHOI22 → ZooLake2 (cross-ecosystem)

For each: baseline vs SBA augmentation.
"""

import json, sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_augmentation import SpectralAugmentation

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "tier1"

# Data paths
WHOI_DIR = Path("/home/sreenath/research-space/TraitMind/data/whoi22-preprocessed")
ZOOSCAN_DIR = Path("/home/sreenath/research-space/TraitMind/data/zooscan20-preprocessed")
ZOOlAKE_DIR = ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0"

# DataShift cross-instrument data (has overlapping classes)
DS_IFCB = Path("/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument/train/DataShift_IFCB")
DS_ZOOSCAN = Path("/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument/test/DataShift_ZooScan")


class PlanktonDataset(Dataset):
    def __init__(self, data_dir, classes, augment=False, sba=None):
        self.images, self.labels = [], []
        self.augment = augment
        self.sba = sba

        for cls_dir in sorted(Path(data_dir).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)

        self.basic_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ])
        print(f"  Loaded {len(self.images)} images from {data_dir.name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.augment:
            if self.sba:
                gray = np.mean(arr, axis=2)
                gray_aug = self.sba(gray)
                gray_uint8 = (gray_aug * 255).clip(0, 255).astype(np.uint8)
                img_aug = Image.fromarray(gray_uint8, mode="L").convert("RGB")
                arr = np.array(img_aug, dtype=np.float32) / 255.0
            pil_img = Image.fromarray((arr * 255).astype(np.uint8))
            pil_img = self.basic_aug(pil_img)
            arr = np.array(pil_img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor, self.labels[idx]


def train_and_eval(train_dataset, test_dataset, num_classes, device, epochs=20, lr=1e-4):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/total:.4f} acc={correct/total:.4f}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load shift spectrum
    shift_path = ROOT / "results" / "fourier_analysis.json"
    shift_spectrum = None
    if shift_path.exists():
        with open(shift_path) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "ZooScan" in key and "WHOI" in key:
                shift_spectrum = np.array(val.get("diff", []))
                break

    sba = SpectralAugmentation(
        shift_spectrum=shift_spectrum,
        strength=0.5,
        strategies=["spectral_noise", "band_adversarial"],
        p=0.8,
    ) if shift_spectrum is not None else None

    results = {}

    # ── Experiment 1: ZooScan → IFCB (reverse of main benchmark) ──
    if DS_ZOOSCAN.exists() and DS_IFCB.exists():
        print(f"\n{'='*60}")
        print("EXPERIMENT 1: ZooScan → IFCB (reverse transfer)")
        print(f"{'='*60}")

        zooscan_classes = np.array(sorted([d.name for d in DS_ZOOSCAN.iterdir() if d.is_dir()]))
        ifcb_classes = np.array(sorted([d.name for d in DS_IFCB.iterdir() if d.is_dir()]))
        # Find common classes
        common = np.intersect1d(zooscan_classes, ifcb_classes)
        print(f"  Common classes: {len(common)} — {list(common)}")

        if len(common) >= 2:
            # Baseline
            print("\n  Baseline:")
            train_ds = PlanktonDataset(DS_ZOOSCAN, common, augment=True)
            test_ds = PlanktonDataset(DS_IFCB, common, augment=False)
            baseline_acc = train_and_eval(train_ds, test_ds, len(common), device)
            print(f"    ZooScan→IFCB baseline: {baseline_acc:.4f}")

            # SBA
            print("\n  With SBA:")
            train_ds_sba = PlanktonDataset(DS_ZOOSCAN, common, augment=True, sba=sba)
            sba_acc = train_and_eval(train_ds_sba, test_ds, len(common), device)
            print(f"    ZooScan→IFCB SBA: {sba_acc:.4f}")

            results["zooscan_to_ifcb"] = {
                "baseline": float(baseline_acc),
                "sba": float(sba_acc),
                "lift": float((sba_acc - baseline_acc) * 100),
                "n_classes": len(common),
            }

    # ── Experiment 2: WHOI → ZooScan (alternative cross-instrument) ──
    if WHOI_DIR.exists() and ZOOSCAN_DIR.exists():
        print(f"\n{'='*60}")
        print("EXPERIMENT 2: WHOI22 → ZooScan20 (cross-instrument)")
        print(f"{'='*60}")

        whoi_classes = np.array(sorted([d.name for d in WHOI_DIR.iterdir() if d.is_dir()]))
        zooscan_classes = np.array(sorted([d.name for d in ZOOSCAN_DIR.iterdir() if d.is_dir()]))
        common = np.intersect1d(whoi_classes, zooscan_classes)
        print(f"  Common classes: {len(common)} — {list(common)}")

        if len(common) >= 2:
            # Baseline
            print("\n  Baseline:")
            train_ds = PlanktonDataset(WHOI_DIR, common, augment=True)
            test_ds = PlanktonDataset(ZOOSCAN_DIR, common, augment=False)
            baseline_acc = train_and_eval(train_ds, test_ds, len(common), device)
            print(f"    WHOI→ZooScan baseline: {baseline_acc:.4f}")

            # SBA
            print("\n  With SBA:")
            train_ds_sba = PlanktonDataset(WHOI_DIR, common, augment=True, sba=sba)
            sba_acc = train_and_eval(train_ds_sba, test_ds, len(common), device)
            print(f"    WHOI→ZooScan SBA: {sba_acc:.4f}")

            results["whoi_to_zooscan"] = {
                "baseline": float(baseline_acc),
                "sba": float(sba_acc),
                "lift": float((sba_acc - baseline_acc) * 100),
                "n_classes": len(common),
                "common_classes": list(common),
            }

    # Summary
    print(f"\n{'='*60}")
    print("REVERSE TRANSFER RESULTS")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name}: baseline={res['baseline']:.4f}, SBA={res['sba']:.4f}, lift={res['lift']:+.2f}%")

    out_path = RESULTS_DIR / "reverse_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
