"""
temporal_ood_evaluation.py
==========================
Temporal OOD evaluation on ZooLake35 dataset.

Uses deployment dates embedded in image filenames to create temporal splits:
- Train on early dates (2018)
- Test on later dates (2019-2020) — temporal shift

This replicates and extends Chen et al.'s (2025) OOD evaluation approach
using the ZooLake35 dataset that's available on disk.

Usage:
    python temporal_ood_evaluation.py \
        --data-dir /home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed \
        --output-dir results/temporal_ood \
        --classes ceratium daphnia bosmina conochilus keratella-quadrata fragilaria
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224

# Timestamp regex: SPC-EAWAG-0P5X-{microseconds}-...
TS_PATTERN = re.compile(r"SPC-EAWAG-0P5X-(\d+)-")


def extract_date(filename: str) -> str:
    """Extract date string (YYYY-MM-DD) from ZooLake35 filename."""
    m = TS_PATTERN.search(filename)
    if m:
        ts_us = int(m.group(1))
        ts_sec = ts_us / 1_000_000
        try:
            return datetime.fromtimestamp(ts_sec).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return "unknown"


def collect_images_by_date(data_dir: str, classes: list) -> dict:
    """Collect images grouped by deployment date."""
    date_images = defaultdict(list)
    data_path = Path(data_dir)

    for cls_name in classes:
        cls_dir = data_path / cls_name
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            date = extract_date(img_path.name)
            if date != "unknown":
                date_images[date].append((str(img_path), cls_name))

    return dict(date_images)


class PlanktonDataset(Dataset):
    def __init__(self, samples: list, class_to_idx: dict, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx.get(cls_name, 0)
        return image, label, path


def get_transforms():
    train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, eval_t


def rotation_tta(model, image, device):
    """Rotation TTA: average over 4 rotations × 2 flips = 8 predictions."""
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
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boots, alpha * 100)),
        "ci_high": float(np.percentile(boots, (1 - alpha) * 100)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed")
    parser.add_argument("--output-dir", type=str, default="results/temporal_ood")
    parser.add_argument("--classes", nargs="+",
                        default=["ceratium", "daphnia", "bosmina", "conochilus",
                                 "keratella-quadrata", "fragilaria"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augmentation", type=str, default="standard",
                        choices=["standard", "saa_band"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect images by date
    logger.info("Collecting images by deployment date...")
    date_images = collect_images_by_date(args.data_dir, args.classes)

    # Get all dates sorted
    all_dates = sorted(date_images.keys())
    logger.info("Found %d deployment dates across %d classes", len(all_dates), len(args.classes))

    # Date statistics
    date_stats = {}
    for date in all_dates:
        cls_counts = defaultdict(int)
        for _, cls in date_images[date]:
            cls_counts[cls] += 1
        date_stats[date] = {"total": len(date_images[date]), "classes": dict(cls_counts)}

    # Select top dates with enough samples per class
    # Relax: require at least 5 samples for at least 3 classes
    min_per_class = 5
    min_classes = 3
    valid_dates = []
    for date in all_dates:
        cls_counts = date_stats[date]["classes"]
        n_valid = sum(1 for c in args.classes if cls_counts.get(c, 0) >= min_per_class)
        if n_valid >= min_classes:
            valid_dates.append(date)

    logger.info("Dates with >=%d samples in >=%d classes: %d dates",
                min_per_class, min_classes, len(valid_dates))
    if valid_dates:
        logger.info("First 10: %s", valid_dates[:10])

    if len(valid_dates) < 2:
        # Fallback: use ALL dates, split by year
        logger.warning("Not enough valid dates for threshold-based split.")
        logger.info("Using year-based split: train on 2018, test on 2019-2020")
        train_dates = [d for d in all_dates if d.startswith("2018")]
        test_dates = [d for d in all_dates if d.startswith("2019") or d.startswith("2020")]
        # Ensure each split has enough data
        if not train_dates or not test_dates:
            # Just split 60/40
            split_idx = int(len(all_dates) * 0.6)
            train_dates = all_dates[:split_idx]
            test_dates = all_dates[split_idx:]
    else:
        # Temporal split: use first 60% as train, rest as OOD test
        split_idx = int(len(valid_dates) * 0.6)
        train_dates = valid_dates[:split_idx]
        test_dates = valid_dates[split_idx:]

    logger.info("Train dates (%d): %s", len(train_dates), train_dates[:5])
    logger.info("Test dates (%d): %s", len(test_dates), test_dates[:5])

    # Build datasets
    class_to_idx = {c: i for i, c in enumerate(args.classes)}

    train_samples = []
    for date in train_dates:
        for path, cls in date_images[date]:
            if cls in class_to_idx:
                train_samples.append((path, cls))

    test_samples = []
    for date in test_dates:
        for path, cls in date_images[date]:
            if cls in class_to_idx:
                test_samples.append((path, cls))

    logger.info("Train: %d samples, Test: %d samples", len(train_samples), len(test_samples))

    train_t, eval_t = get_transforms()
    train_ds = PlanktonDataset(train_samples, class_to_idx, train_t)
    test_ds = PlanktonDataset(test_samples, class_to_idx, eval_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Train model
    num_classes = len(args.classes)
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger.info("Training ViT-B/16 for %d epochs...", args.epochs)
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
            train_acc = correct / total * 100
            test_res = evaluate(model, test_loader, criterion, device)
            logger.info("  Epoch %d/%d  Loss: %.4f  Train: %.1f%%  Test (OOD): %.1f%%",
                        epoch + 1, args.epochs, total_loss / total,
                        train_acc, test_res["accuracy"] * 100)

    # Final evaluation
    train_res = evaluate(model, train_loader, criterion, device, use_tta=args.tta)
    test_res = evaluate(model, test_loader, criterion, device, use_tta=args.tta)

    train_ci = bootstrap_ci([1 if p == l else 0 for p, l in
                              zip(train_res["predictions"], train_res["labels"])])
    test_ci = bootstrap_ci([1 if p == l else 0 for p, l in
                             zip(test_res["predictions"], test_res["labels"])])

    # Per-class results
    per_class = {}
    preds = np.array(test_res["predictions"])
    labels = np.array(test_res["labels"])
    for cls, idx in class_to_idx.items():
        mask = labels == idx
        if mask.sum() > 0:
            per_class[cls] = {
                "accuracy": float((preds[mask] == labels[mask]).mean()),
                "total": int(mask.sum()),
            }

    # Per-date results
    per_date = {}
    for date in test_dates:
        date_samples = [(p, c) for p, c in date_images[date] if c in class_to_idx]
        if not date_samples:
            continue
        date_ds = PlanktonDataset(date_samples, class_to_idx, eval_t)
        date_loader = DataLoader(date_ds, batch_size=args.batch_size, shuffle=False)
        date_res = evaluate(model, date_loader, criterion, device, use_tta=args.tta)
        per_date[date] = {
            "accuracy": date_res["accuracy"],
            "n_samples": date_res["total"],
        }

    # Save results
    results = {
        "train_dates": train_dates,
        "test_dates": test_dates,
        "classes": args.classes,
        "augmentation": args.augmentation,
        "use_tta": args.tta,
        "train": {
            "accuracy": train_res["accuracy"],
            "ci_95": [train_ci["ci_low"], train_ci["ci_high"]],
            "n_samples": train_res["total"],
        },
        "test_ood": {
            "accuracy": test_res["accuracy"],
            "ci_95": [test_ci["ci_low"], test_ci["ci_high"]],
            "n_samples": test_res["total"],
            "accuracy_drop": train_res["accuracy"] - test_res["accuracy"],
        },
        "per_class": per_class,
        "per_date": per_date,
        "date_stats": {d: date_stats[d] for d in valid_dates},
    }

    with open(out / "temporal_ood_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 72)
    logger.info("  TEMPORAL OOD EVALUATION RESULTS")
    logger.info("=" * 72)
    logger.info("  Train dates: %s ... (%d total)", train_dates[0], len(train_dates))
    logger.info("  Test dates:  %s ... (%d total)", test_dates[0], len(test_dates))
    logger.info("  Train accuracy: %.1f%% [%.1f-%.1f%%]",
                train_res["accuracy"] * 100, train_ci["ci_low"] * 100, train_ci["ci_high"] * 100)
    logger.info("  Test (OOD) accuracy: %.1f%% [%.1f-%.1f%%]",
                test_res["accuracy"] * 100, test_ci["ci_low"] * 100, test_ci["ci_high"] * 100)
    logger.info("  Accuracy drop: %.1f%%", (train_res["accuracy"] - test_res["accuracy"]) * 100)
    logger.info("-" * 72)
    logger.info("  Per-class OOD accuracy:")
    for cls, stats in sorted(per_class.items(), key=lambda x: -x[1]["accuracy"]):
        logger.info("    %-25s  %.1f%%  (n=%d)", cls, stats["accuracy"] * 100, stats["total"])
    logger.info("-" * 72)
    logger.info("  Per-date OOD accuracy:")
    for date, stats in sorted(per_date.items()):
        logger.info("    %s  %.1f%%  (n=%d)", date, stats["accuracy"] * 100, stats["n_samples"])
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
