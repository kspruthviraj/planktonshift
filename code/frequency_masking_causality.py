"""
tier1_frequency_masking.py — Causality experiment: train with frequency band subsets.

Trains ViT-B/16 on IFCB data with only specific frequency bands preserved:
  - Low only (bins 0-22): should preserve species, not domain
  - Mid only (bins 22-44): should preserve domain, not species
  - High only (bins 44+): should preserve neither

Then evaluates on ZooScan for both species accuracy and domain classification.

This is the key experiment that turns the Fourier correlation into causation.
"""

import json, sys, os, argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import cv2

ROOT = Path(__file__).resolve().parent.parent
IFCB_DIR = Path("/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument/train/DataShift_IFCB")
ZOO_DIR = Path("/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument/test/DataShift_ZooScan")
RESULTS_DIR = ROOT / "results" / "tier1"

# WHOI22 and ZooScan paths
WHOI_DIR = Path("/home/sreenath/research-space/TraitMind/data/whoi22-preprocessed")
ZOOSCAN_DIR = Path("/home/sreenath/research-space/TraitMind/data/zooscan20-preprocessed")


def bandpass_filter(image_np, band):
    """Apply frequency band mask to image. band: 'low', 'mid', 'high', 'all'."""
    gray = np.mean(image_np, axis=2) if len(image_np.shape) == 3 else image_np
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    r_max = min(crow, ccol)

    mask = np.zeros((rows, cols), dtype=np.float32)
    if band == 'low':
        r_inner, r_outer = 0, int(r_max * 0.25)
    elif band == 'mid':
        r_inner, r_outer = int(r_max * 0.25), int(r_max * 0.75)
    elif band == 'high':
        r_inner, r_outer = int(r_max * 0.75), r_max
    else:
        r_inner, r_outer = 0, r_max

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if r_inner <= dist <= r_outer:
                mask[i, j] = 1.0

    Fshift_filtered = Fshift * mask
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    img_back = np.fft.ifft2(F_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
    return img_back


class FrequencyMaskedDataset(Dataset):
    def __init__(self, data_dir, classes, band='all', augment=False):
        self.images, self.labels = [], []
        self.band = band
        self.augment = augment

        data_path = Path(data_dir)
        for cls_dir in sorted(data_path.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ])
        print(f"  [{band}] Loaded {len(self.images)} images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.band != 'all':
            filtered = bandpass_filter(arr, self.band)
            arr = np.stack([filtered] * 3, axis=2)

        if self.augment:
            pil_img = Image.fromarray((arr * 255).astype(np.uint8))
            pil_img = self.transform(pil_img)
            arr = np.array(pil_img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor, self.labels[idx]


def evaluate_accuracy(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, dataset, device, epochs=15, lr=1e-4, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(loader, desc=f"      Epoch {epoch+1}/{epochs}", leave=False):
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
        acc = correct / total
        print(f"      Epoch {epoch+1}: loss={total_loss/total:.4f} acc={acc:.4f}")
    return model


def extract_features_for_domain(model, dataset, device, batch_size=64):
    """Extract CLS features for domain classification."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            # Get CLS token
            x = model.patch_embed(images) if hasattr(model, 'patch_embed') else None
            if x is not None:
                cls_token = model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + model.pos_embed if hasattr(model, 'pos_embed') else x
                for blk in model.blocks:
                    x = blk(x)
                x = model.norm(x)
                cls_feat = x[:, 0].cpu().numpy()
            else:
                cls_feat = model.forward_features(images)[:, 0].cpu().numpy()
            features.append(cls_feat)
    return np.concatenate(features, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-domain", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check if data exists
    if not IFCB_DIR.exists():
        print(f"ERROR: Data not found at {IFCB_DIR}")
        return
    if not ZOO_DIR.exists():
        print(f"ERROR: Data not found at {ZOO_DIR}")
        return

    train_dir = IFCB_DIR
    test_dir = ZOO_DIR

    # Load classes from directory
    classes = np.array(sorted([d.name for d in train_dir.iterdir() if d.is_dir()]))
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {list(classes[:5])}...")

    bands = ['low', 'mid', 'high', 'all']
    results = {}

    for band in bands:
        print(f"\n{'='*60}")
        print(f"Training with {band.upper()} frequencies only")
        print(f"{'='*60}")

        # Load data
        train_dataset = FrequencyMaskedDataset(train_dir, classes, band=band, augment=True)
        test_dataset = FrequencyMaskedDataset(test_dir, classes, band=band, augment=False)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"  Skipping {band}: no data")
            continue

        # Train model
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        model = model.to(device)
        model = train_model(model, train_dataset, device, epochs=args.epochs, lr=args.lr)

        # Evaluate species accuracy
        species_acc, preds, labels = evaluate_accuracy(model, test_dataset, device)
        print(f"  Species accuracy ({band}): {species_acc:.4f}")

        # Evaluate domain classification (if not skipped)
        domain_acc = None
        if not args.skip_domain:
            try:
                # Extract features from source and target
                source_feats = extract_features_for_domain(model, train_dataset, device)
                target_feats = extract_features_for_domain(model, test_dataset, device)

                # Domain labels: 0=source, 1=target
                X = np.vstack([source_feats, target_feats])
                y = np.array([0]*len(source_feats) + [1]*len(target_feats))

                clf = LogisticRegression(max_iter=1000, random_state=42)
                domain_acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
                print(f"  Domain accuracy ({band}): {domain_acc:.4f}")
            except Exception as e:
                print(f"  Domain classification failed: {e}")

        results[band] = {
            'species_accuracy': float(species_acc),
            'domain_accuracy': float(domain_acc) if domain_acc else None,
            'n_train': len(train_dataset),
            'n_test': len(test_dataset),
        }

        # Save model checkpoint
        torch.save(model.state_dict(), str(RESULTS_DIR / f"model_{band}.pth"))

    # Summary table
    print(f"\n{'='*60}")
    print("FREQUENCY MASKING CAUSALITY RESULTS")
    print(f"{'='*60}")
    print(f"{'Band':<8} {'Species Acc':>12} {'Domain Acc':>12}")
    print("-"*32)
    for band in bands:
        if band in results:
            sp = results[band]['species_accuracy']
            dm = results[band]['domain_accuracy']
            dm_str = f"{dm:.4f}" if dm else "N/A"
            print(f"{band:<8} {sp:>12.4f} {dm_str:>12}")

    # Save results
    out_path = RESULTS_DIR / "frequency_masking.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
