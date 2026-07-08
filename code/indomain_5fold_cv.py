"""
indomain_5fold_cv.py — 5-fold cross-validation for in-domain IFCB accuracy.
Train on 4 folds, test on 1 fold, rotate 5 times, average.
"""

import json, sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import timm
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from config import DATA
IFCB_DIR = Path(DATA["cross_ifcb"])
RESULTS_DIR = ROOT / "results" / "tier1"


class IFCBDataset(Dataset):
    def __init__(self, data_dir, classes, augment=False):
        self.images, self.labels = [], []
        self.augment = augment
        for cls_dir in sorted(Path(data_dir).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)
        self.labels = np.array(self.labels)
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        if self.augment:
            img = self.aug(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), self.labels[idx]


def train_and_eval_fold(model, train_loader, test_loader, device, epochs=15, lr=1e-4):
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

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.array(sorted([d.name for d in IFCB_DIR.iterdir() if d.is_dir()]))
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {list(classes)}")

    dataset = IFCBDataset(IFCB_DIR, classes, augment=True)
    print(f"Total images: {len(dataset)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.images, dataset.labels)):
        print(f"\n--- Fold {fold+1}/5 ---")
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Test subset should NOT have augmentation
        test_dataset_noaug = IFCBDataset(IFCB_DIR, classes, augment=False)
        test_subset = Subset(test_dataset_noaug, test_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)

        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        model = model.to(device)

        acc = train_and_eval_fold(model, train_loader, test_loader, device, epochs=15, lr=1e-4)
        fold_accs.append(acc)
        print(f"  Fold {fold+1} accuracy: {acc:.4f}")

        del model
        torch.cuda.empty_cache()

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"\n{'='*50}")
    print(f"In-domain IFCB 5-fold CV accuracy")
    print(f"{'='*50}")
    print(f"Per-fold: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*50}")

    output = {
        "in_domain_accuracy_mean": float(mean_acc),
        "in_domain_accuracy_std": float(std_acc),
        "per_fold": [float(a) for a in fold_accs],
        "n_folds": 5,
        "n_images": len(dataset),
        "n_classes": num_classes,
    }
    out_path = RESULTS_DIR / "indomain_5fold_cv.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
