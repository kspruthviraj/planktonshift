"""
train_classifier_on_crops.py
=============================
Train ViT/BEiT classifier on v2a-segmented plankton crops (background removed).

This tests the hypothesis: does removing background artifacts via segmentation
improve cross-domain robustness?

Compare:
1. Full images (with background) → baseline accuracy
2. v2a crops (background removed) → improved accuracy?

If crops improve OOD accuracy, it proves that background artifacts are the
primary source of domain shift — validating our Fourier analysis.

Usage:
    python train_classifier_on_crops.py \
        --train-dir data_segmentation/zoolake2 \
        --test-dir data_segmentation/ood \
        --epochs 30 --tta --output results/classifier_on_crops.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}
IMG_SIZE = 224


class CropDataset(Dataset):
    """Load cropped plankton images (organism on white background)."""
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return
        for cls_name, idx in class_to_idx.items():
            cls_dir = data_path / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*_crop.png")):
                self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


def rotation_tta(model, images, device):
    preds = []
    for k in range(4):
        rotated = torch.rot90(images, k, dims=[2, 3]).to(device)
        preds.append(torch.softmax(model(rotated), dim=1))
        preds.append(torch.softmax(model(torch.flip(rotated, [3])), dim=1))
    return torch.stack(preds).mean(0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_paths = [], [], []
    for images, labels, paths in loader:
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
        all_paths.extend(paths)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "predictions": all_preds,
        "labels": all_labels,
        "paths": all_paths,
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
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dirs", nargs="+", required=True)
    parser.add_argument("--arch", type=str, default="vit_b_16",
                        choices=["vit_b_16", "resnet50", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output", type=str, default="results/classifier_on_crops.json")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Discover classes from train directory
    train_path = Path(args.train_dir)
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("Classes (%d): %s", num_classes, classes[:10])

    # Transforms
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
    train_ds = CropDataset(args.train_dir, class_to_idx, train_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    logger.info("Train: %d samples", len(train_ds))

    test_loaders = {}
    for test_dir in args.test_dirs:
        name = Path(test_dir).name
        ds = CropDataset(test_dir, class_to_idx, eval_transform)
        if len(ds) > 0:
            test_loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
            logger.info("Test %s: %d samples", name, len(ds))

    # Model
    if args.arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif args.arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    logger.info("Training %s for %d epochs...", args.arch, args.epochs)
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  Epoch %d/%d  Loss: %.4f  Train: %.1f%%",
                        epoch + 1, args.epochs, total_loss / total, correct / total * 100)

    # Save model
    model_path = args.output.replace(".json", "_best.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # Evaluate
    results = {"architecture": args.arch, "augmentation": "none (clean crops)", "tta": args.tta}

    train_res = evaluate(model, train_loader, criterion, device, use_tta=args.tta)
    results["train"] = {"accuracy": train_res["accuracy"], "n": train_res["total"]}

    for name, loader in test_loaders.items():
        res = evaluate(model, loader, criterion, device, use_tta=args.tta)
        binary = [1 if p == l else 0 for p, l in zip(res["predictions"], res["labels"])]
        ci = bootstrap_ci(binary)
        results[name] = {
            "accuracy": res["accuracy"],
            "ci_95": [ci["ci_low"], ci["ci_high"]],
            "n": res["total"],
            "drop": train_res["accuracy"] - res["accuracy"],
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    logger.info("=" * 60)
    logger.info("  CLASSIFIER ON CROPS RESULTS")
    logger.info("=" * 60)
    logger.info("  Train: %.1f%%", train_res["accuracy"] * 100)
    for name, data in results.items():
        if isinstance(data, dict) and "accuracy" in data and name != "train":
            logger.info("  %s: %.1f%% (drop: %.1f%%)", name, data["accuracy"] * 100, data["drop"] * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
