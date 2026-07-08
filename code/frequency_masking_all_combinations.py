"""
frequency_masking_all_combinations.py — Run ALL frequency band combinations.
Adds rows for low+mid, low+high to Table 1.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA
IFCB_DIR = Path(DATA["cross_ifcb"])
ZOO_DIR = Path(DATA["cross_zooscan"])
RESULTS_DIR = ROOT / "results" / "tier1"


def bandpass_filter(image_np, band):
    gray = np.mean(image_np, axis=2) if len(image_np.shape) == 3 else image_np
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    r_max = min(crow, ccol)

    mask = np.zeros((rows, cols), dtype=np.float32)
    bands = {
        'low': (0, 0.25),
        'mid': (0.25, 0.75),
        'high': (0.75, 1.0),
        'low_mid': (0, 0.75),
        'low_high': (0, 1.0),  # This is basically all
        'mid_high': (0.25, 1.0),
        'all': (0, 1.0),
    }
    r_inner_frac, r_outer_frac = bands.get(band, (0, 1.0))
    r_inner = int(r_max * r_inner_frac)
    r_outer = int(r_max * r_outer_frac)

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
        for cls_dir in sorted(Path(data_dir).iterdir()):
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
        return torch.from_numpy(arr).permute(2, 0, 1).float(), self.labels[idx]


def train_and_eval(band, classes, device, epochs=15, lr=1e-4):
    train_ds = FrequencyMaskedDataset(IFCB_DIR, classes, band=band, augment=True)
    test_ds = FrequencyMaskedDataset(ZOO_DIR, classes, band=band, augment=False)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    num_classes = len(classes)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Species accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    species_acc = correct / total

    # Domain accuracy
    train_feats, test_feats = [], []
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            feats = model.forward_features(images)[:, 0].cpu().numpy()
            train_feats.append(feats)
        for images, _ in test_loader:
            images = images.to(device)
            feats = model.forward_features(images)[:, 0].cpu().numpy()
            test_feats.append(feats)

    train_feats = np.concatenate(train_feats)
    test_feats = np.concatenate(test_feats)
    X = np.vstack([train_feats, test_feats])
    y = np.array([0]*len(train_feats) + [1]*len(test_feats))
    clf = LogisticRegression(max_iter=1000, random_state=42)
    domain_acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

    del model
    torch.cuda.empty_cache()

    return {'species_accuracy': float(species_acc), 'domain_accuracy': float(domain_acc)}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.array(sorted([d.name for d in IFCB_DIR.iterdir() if d.is_dir()]))
    print(f"Classes ({len(classes)})")

    # Load existing results
    existing_path = RESULTS_DIR / "frequency_masking.json"
    if existing_path.exists():
        with open(existing_path) as f:
            results = json.load(f)
    else:
        results = {}

    # New combinations to test
    new_bands = ['low_mid', 'mid_high']

    for band in new_bands:
        print(f"\n{'='*60}")
        print(f"Training with {band.upper()} frequencies")
        print(f"{'='*60}")
        res = train_and_eval(band, classes, device)
        if res:
            results[band] = res
            print(f"  Species: {res['species_accuracy']:.4f}, Domain: {res['domain_accuracy']:.4f}")

    # Save updated results
    out_path = RESULTS_DIR / "frequency_masking_all.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("COMPLETE FREQUENCY MASKING RESULTS")
    print(f"{'='*60}")
    print(f"{'Band':<15} {'Bins':<15} {'Species':>10} {'Domain':>10}")
    print("-"*50)
    band_info = {
        'low': '0-28',
        'mid': '28-84',
        'high': '84-112',
        'low_mid': '0-84',
        'mid_high': '28-112',
        'all': '0-112',
    }
    for band in ['low', 'mid', 'high', 'low_mid', 'mid_high', 'all']:
        if band in results:
            r = results[band]
            bins = band_info.get(band, '?')
            print(f"{band:<15} {bins:<15} {r['species_accuracy']:>10.1%} {r['domain_accuracy']:>10.1%}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
