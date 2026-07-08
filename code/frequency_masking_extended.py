"""
frequency_masking_extended.py — Extended frequency masking with mid+high combination.
Adds a row to Table 1: mid+high frequencies combined.
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
    if band == 'low':
        r_inner, r_outer = 0, int(r_max * 0.25)
    elif band == 'mid':
        r_inner, r_outer = int(r_max * 0.25), int(r_max * 0.75)
    elif band == 'high':
        r_inner, r_outer = int(r_max * 0.75), r_max
    elif band == 'mid_high':
        r_inner, r_outer = int(r_max * 0.25), r_max
    elif band == 'all':
        r_inner, r_outer = 0, r_max
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
        print(f"  [{band}] Loaded {len(self.images)} images")

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
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


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
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/total:.4f} acc={correct/total:.4f}")
    return model


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not IFCB_DIR.exists() or not ZOO_DIR.exists():
        print("Data not found")
        return

    classes = np.array(sorted([d.name for d in IFCB_DIR.iterdir() if d.is_dir()]))
    num_classes = len(classes)
    print(f"Classes ({num_classes})")

    # Only run the new mid+high band
    band = 'mid_high'
    print(f"\n{'='*60}")
    print(f"Training with {band.upper()} frequencies (mid + high combined)")
    print(f"{'='*60}")

    train_dataset = FrequencyMaskedDataset(IFCB_DIR, classes, band=band, augment=True)
    test_dataset = FrequencyMaskedDataset(ZOO_DIR, classes, band=band, augment=False)

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    model = train_model(model, train_dataset, device, epochs=15, lr=1e-4)

    species_acc = evaluate_accuracy(model, test_dataset, device)
    print(f"\n  Species accuracy ({band}): {species_acc:.4f}")

    # Also evaluate camera accuracy
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Extract features for domain classification
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model.eval()
    train_feats, test_feats = [], []
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
    print(f"  Domain accuracy ({band}): {domain_acc:.4f}")

    # Save
    result = {
        band: {
            'species_accuracy': float(species_acc),
            'domain_accuracy': float(domain_acc),
            'n_train': len(train_dataset),
            'n_test': len(test_dataset),
        }
    }
    out_path = RESULTS_DIR / "frequency_masking_mid_high.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nSummary:")
    print(f"  Mid+High frequencies: Species={species_acc:.1%}, Domain={domain_acc:.1%}")


if __name__ == "__main__":
    main()
