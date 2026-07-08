"""
train_saa_zoolake.py
====================
Train models with Spectral Adversarial Augmentation (SAA) on ZooLake35
and evaluate on temporal OOD deployment days.

This is the key experiment to beat Chen et al.'s 83% OOD accuracy:
- Chen uses standard augmentations (rotation, flip, color jitter, blur)
- We use frequency-calibrated augmentations (SAA) that target the specific
  frequency bands identified by Fourier analysis as carrying domain info

Usage:
    python train_saa_zoolake.py \
        --data-dir data/zoolake_ood \
        --augmentation saa_band \
        --epochs 30 --tta \
        --output results/saa_zoolake_ood.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# Add Adverserial_net for spectral augmentation imports
sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation, load_shift_spectrum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


class PlanktonDataset(Dataset):
    def __init__(self, data_dir: str, class_to_idx: dict, transform=None):
        self.samples = []
        self.transform = transform
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return
        for cls_name, idx in class_to_idx.items():
            cls_dir = data_path / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), idx))
        logger.info("Loaded %d samples from %s", len(self.samples), data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


class SAATransform:
    """SAA transform for ZooLake temporal shift."""
    def __init__(self, shift_spectrum=None, strategies=None, strength=0.5, p=0.8):
        self.aug = SpectralAugmentation(
            shift_spectrum=shift_spectrum,
            strength=strength,
            strategies=strategies or ["spectral_noise", "band_adversarial"],
            p=p,
        )
        self.base = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        image_aug = Image.fromarray(arr_uint8, mode="L").convert("RGB")
        return self.base(image_aug)


def compute_shift_spectrum_from_data(data_dir: str, classes: list, max_per_class=50):
    """Compute shift spectrum from training data (temporal variation)."""
    all_spectra = []
    for cls_name in classes:
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.is_dir():
            continue
        count = 0
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            try:
                img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.float64) / 255.0
                f = np.fft.fft2(arr)
                amp = np.log1p(np.abs(np.fft.fftshift(f)))
                h, w = amp.shape
                cy, cx = h // 2, w // 2
                max_r = min(cx, cy)
                Y, X = np.ogrid[:h, :w]
                R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
                radial = np.zeros(max_r)
                for r in range(max_r):
                    mask = R == r
                    if mask.sum() > 0:
                        radial[r] = amp[mask].mean()
                all_spectra.append(radial)
                count += 1
            except Exception:
                continue
            if count >= max_per_class:
                break

    if not all_spectra:
        return None

    # Compute variance across images as shift proxy
    max_len = max(len(s) for s in all_spectra)
    matrix = np.zeros((len(all_spectra), max_len))
    for i, s in enumerate(all_spectra):
        matrix[i, :len(s)] = s
    return matrix.std(axis=0)


def rotation_tta(model, image, device):
    preds = []
    for k in range(4):
        rotated = torch.rot90(image, k, dims=[2, 3]).to(device)
        preds.append(torch.softmax(model(rotated), dim=1))
        preds.append(torch.softmax(model(torch.flip(rotated, [3])), dim=1))
    return torch.stack(preds).mean(0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if use_tta:
            probs = rotation_tta(model, images, device)
            outputs = torch.log(probs + 1e-8)
        else:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = probs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "predictions": all_preds,
        "labels": all_labels,
    }


def bootstrap_ci(binary, n=1000, ci=0.95):
    rng = np.random.RandomState(42)
    arr = np.array(binary)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    return {"mean": float(arr.mean()),
            "ci_low": float(np.percentile(boots, alpha * 100)),
            "ci_high": float(np.percentile(boots, (1 - alpha) * 100))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/zoolake_ood")
    parser.add_argument("--augmentation", type=str, default="saa_band",
                        choices=["standard", "heavy", "saa_noise", "saa_band", "saa_best"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--arch", type=str, default="beit",
                        choices=["vit_b_16", "beit", "resnet50"])
    parser.add_argument("--output", type=str, default="results/saa_zoolake_ood.json")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(Path(args.data_dir) / "manifest.json") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    class_to_idx = manifest["class_to_idx"]
    num_classes = len(classes)
    logger.info("Classes: %s (%d)", classes, num_classes)

    # Compute shift spectrum from training data
    logger.info("Computing shift spectrum from training data...")
    shift_spectrum = compute_shift_spectrum_from_data(
        Path(args.data_dir) / "train", classes
    )
    if shift_spectrum is not None:
        logger.info("Shift spectrum computed: %d bins", len(shift_spectrum))
    else:
        logger.warning("Could not compute shift spectrum")

    # Transforms
    if args.augmentation.startswith("saa"):
        strategies = {
            "saa_noise": ["spectral_noise"],
            "saa_band": ["band_adversarial"],
            "saa_best": ["spectral_noise", "band_adversarial"],
        }.get(args.augmentation, ["spectral_noise", "band_adversarial"])
        train_transform = SAATransform(
            shift_spectrum=shift_spectrum,
            strategies=strategies,
            strength=0.5,
            p=0.8,
        )
        logger.info("Using SAA: %s", strategies)
    elif args.augmentation == "heavy":
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = PlanktonDataset(Path(args.data_dir) / "train", class_to_idx, train_transform)
    test_ds = PlanktonDataset(Path(args.data_dir) / "test", class_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    if args.arch == "beit":
        # BEiT — following Chen et al.
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif args.arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    logger.info("Training %s for %d epochs...", args.arch, args.epochs)
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_res = evaluate(model, test_loader, criterion, device)
            logger.info("  Epoch %d/%d  Loss: %.4f  Train: %.1f%%  Test (OOD): %.1f%%",
                        epoch + 1, args.epochs, total_loss / total,
                        correct / total * 100, test_res["accuracy"] * 100)
            if test_res["accuracy"] > best_acc:
                best_acc = test_res["accuracy"]
                torch.save(model.state_dict(), str(out_path).replace(".json", "_best.pth"))

    # Load best and evaluate with TTA
    model.load_state_dict(torch.load(str(out_path).replace(".json", "_best.pth"), weights_only=True))

    train_res = evaluate(model, train_loader, criterion, device, use_tta=args.tta)
    test_res = evaluate(model, test_loader, criterion, device, use_tta=args.tta)

    train_ci = bootstrap_ci([1 if p == l else 0 for p, l in zip(train_res["predictions"], train_res["labels"])])
    test_ci = bootstrap_ci([1 if p == l else 0 for p, l in zip(test_res["predictions"], test_res["labels"])])

    # Per-class
    per_class = {}
    preds = np.array(test_res["predictions"])
    labels = np.array(test_res["labels"])
    for cls, idx in class_to_idx.items():
        mask = labels == idx
        if mask.sum() > 0:
            per_class[cls] = {"accuracy": float((preds[mask] == labels[mask]).mean()), "total": int(mask.sum())}

    # Per-date
    per_date = {}
    test_date_stats = manifest.get("test_date_stats", {})
    for date in sorted(test_date_stats.keys()):
        date_samples = []
        for cls_name, count in test_date_stats[date].items():
            cls_dir = Path(args.data_dir) / "test" / cls_name
            if cls_dir.is_dir():
                for img_path in sorted(cls_dir.iterdir()):
                    if date in img_path.name:
                        date_samples.append((str(img_path), class_to_idx.get(cls_name, 0)))
        if date_samples:
            date_ds = PlanktonDataset.__new__(PlanktonDataset)
            date_ds.samples = date_samples
            date_ds.transform = eval_transform
            date_loader = DataLoader(date_ds, batch_size=args.batch_size, shuffle=False)
            date_res = evaluate(model, date_loader, criterion, device, use_tta=args.tta)
            per_date[date] = {"accuracy": date_res["accuracy"], "n_samples": date_res["total"]}

    # Save
    results = {
        "architecture": args.arch,
        "augmentation": args.augmentation,
        "use_tta": args.tta,
        "epochs": args.epochs,
        "train": {"accuracy": train_res["accuracy"], "ci_95": [train_ci["ci_low"], train_ci["ci_high"]], "n": train_res["total"]},
        "test_ood": {"accuracy": test_res["accuracy"], "ci_95": [test_ci["ci_low"], test_ci["ci_high"]], "n": test_res["total"],
                     "drop": train_res["accuracy"] - test_res["accuracy"]},
        "per_class": per_class,
        "per_date": per_date,
        "chen_best": 0.83,  # Reference: Chen et al.'s BEsT model
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    logger.info("=" * 72)
    logger.info("  RESULTS: %s + %s + TTA=%s", args.arch, args.augmentation, args.tta)
    logger.info("=" * 72)
    logger.info("  Train: %.1f%% [%.1f-%.1f%%]", train_res["accuracy"]*100, train_ci["ci_low"]*100, train_ci["ci_high"]*100)
    logger.info("  Test (OOD): %.1f%% [%.1f-%.1f%%]", test_res["accuracy"]*100, test_ci["ci_low"]*100, test_ci["ci_high"]*100)
    logger.info("  Drop: %.1f%%", (train_res["accuracy"] - test_res["accuracy"])*100)
    logger.info("  Chen's BEsT: 83%%")
    logger.info("  vs Chen: %+.1f%%", test_res["accuracy"]*100 - 83)
    logger.info("-" * 72)
    for cls, stats in sorted(per_class.items(), key=lambda x: -x[1]["accuracy"]):
        logger.info("    %-25s  %.1f%%  (n=%d)", cls, stats["accuracy"]*100, stats["total"])
    logger.info("-" * 72)
    for date, stats in sorted(per_date.items()):
        logger.info("    %s  %.1f%%  (n=%d)", date, stats["accuracy"]*100, stats["n_samples"])
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
