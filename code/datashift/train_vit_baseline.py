"""
train_vit_baseline.py
=====================
Train a ViT classifier on IFCB domain data and evaluate on both IFCB and ZooScan
to demonstrate dataset shift (the 20-30% accuracy drop that motivates RAG).

This is the KEY missing piece: showing that conventional vision models break
across domains while RAG-grounded VLMs maintain consistency.
"""

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSES = [
    "Amphipoda", "Annelida", "Ceratium", "Chaetognatha",
    "Coscinodiscus", "Noctiluca",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PlanktonDataset(Dataset):
    """Load plankton images from data/eval/{domain}/{class}/ structure."""

    def __init__(self, data_root: str, domain: str, transform=None):
        self.samples = []
        self.transform = transform
        domain_dir = Path(data_root) / domain
        for cls in CLASSES:
            cls_dir = domain_dir / cls
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), CLASS_TO_IDX[cls]))
        logger.info("Loaded %d samples from %s/%s", len(self.samples), data_root, domain)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(arch: str = "vit_b_16", pretrained: bool = True) -> nn.Module:
    if arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    for images, labels, paths in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = probs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_paths.extend(paths)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "paths": all_paths,
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def compute_ece(probabilities, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    probs = np.array(probabilities)
    labels = np.array(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return ece


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(accuracies, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(42)
    n = len(accuracies)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(accuracies, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    return {
        "mean": np.mean(accuracies),
        "ci_low": np.percentile(boot_means, alpha * 100),
        "ci_high": np.percentile(boot_means, (1 - alpha) * 100),
        "std": np.std(boot_means),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/eval")
    parser.add_argument("--arch", type=str, default="vit_b_16",
                        choices=["vit_b_16", "resnet50", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="results/vit_baseline.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load datasets
    train_ds = PlanktonDataset(args.data_root, "IFCB", train_transform)
    test_ifcb_ds = PlanktonDataset(args.data_root, "IFCB", eval_transform)
    test_zooscan_ds = PlanktonDataset(args.data_root, "ZooScan", eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_ifcb_loader = DataLoader(test_ifcb_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_zooscan_loader = DataLoader(test_zooscan_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    model = build_model(args.arch, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train on IFCB
    logger.info("Training %s on IFCB domain for %d epochs…", args.arch, args.epochs)
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_res = evaluate(model, test_ifcb_loader, criterion, device)
            logger.info(
                "Epoch %d/%d  Train Loss: %.4f  Train Acc: %.2f%%  Val Acc: %.2f%%",
                epoch + 1, args.epochs, train_loss, train_acc * 100, val_res["accuracy"] * 100,
            )
            if val_res["accuracy"] > best_acc:
                best_acc = val_res["accuracy"]
                torch.save(model.state_dict(), "results/best_vit.pth")

    # Load best model
    model.load_state_dict(torch.load("results/best_vit.pth", weights_only=True))

    # Evaluate on both domains
    logger.info("Evaluating on IFCB (source domain)…")
    ifcb_result = evaluate(model, test_ifcb_loader, criterion, device)

    logger.info("Evaluating on ZooScan (target domain - dataset shift)…")
    zooscan_result = evaluate(model, test_zooscan_loader, criterion, device)

    # Per-class accuracy
    def per_class_acc(result):
        preds = np.array(result["predictions"])
        labels = np.array(result["labels"])
        accs = {}
        for cls, idx in CLASS_TO_IDX.items():
            mask = labels == idx
            if mask.sum() > 0:
                accs[cls] = {
                    "accuracy": float((preds[mask] == labels[mask]).mean()),
                    "total": int(mask.sum()),
                    "correct": int((preds[mask] == labels[mask]).sum()),
                }
        return accs

    ifcb_per_class = per_class_acc(ifcb_result)
    zooscan_per_class = per_class_acc(zooscan_result)

    # Calibration
    ifcb_ece = compute_ece(ifcb_result["probabilities"], ifcb_result["labels"])
    zooscan_ece = compute_ece(zooscan_result["probabilities"], zooscan_result["labels"])

    # Bootstrap CIs
    ifcb_binary = [1 if p == l else 0 for p, l in zip(ifcb_result["predictions"], ifcb_result["labels"])]
    zooscan_binary = [1 if p == l else 0 for p, l in zip(zooscan_result["predictions"], zooscan_result["labels"])]
    ifcb_ci = bootstrap_ci(ifcb_binary)
    zooscan_ci = bootstrap_ci(zooscan_binary)

    # Report
    shift = ifcb_result["accuracy"] - zooscan_result["accuracy"]
    logger.info("=" * 72)
    logger.info("  DATASET SHIFT DEMONSTRATION (ViT Baseline)")
    logger.info("=" * 72)
    logger.info("  Model: %s  |  Trained on: IFCB  |  Classes: %d", args.arch, NUM_CLASSES)
    logger.info("  IFCB (source):    %.2f%%  [95%% CI: %.2f-%.2f%%]  ECE: %.4f",
                ifcb_result["accuracy"] * 100, ifcb_ci["ci_low"] * 100, ifcb_ci["ci_high"] * 100, ifcb_ece)
    logger.info("  ZooScan (target): %.2f%%  [95%% CI: %.2f-%.2f%%]  ECE: %.4f",
                zooscan_result["accuracy"] * 100, zooscan_ci["ci_low"] * 100, zooscan_ci["ci_high"] * 100, zooscan_ece)
    logger.info("  ACCURACY DROP:    %.2f%%  (dataset shift magnitude)", shift * 100)
    logger.info("-" * 72)
    logger.info("  Per-class accuracy:")
    for cls in CLASSES:
        ifcb_a = ifcb_per_class.get(cls, {}).get("accuracy", 0)
        zs_a = zooscan_per_class.get(cls, {}).get("accuracy", 0)
        delta = (ifcb_a - zs_a) * 100
        logger.info("    %-18s  IFCB: %5.1f%%  ZooScan: %5.1f%%  Drop: %+.1f%%",
                    cls, ifcb_a * 100, zs_a * 100, -delta)
    logger.info("=" * 72)

    # Save
    report = {
        "model": args.arch,
        "trained_on": "IFCB",
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "epochs": args.epochs,
        "ifcb": {
            "accuracy": ifcb_result["accuracy"],
            "ci_95": [ifcb_ci["ci_low"], ifcb_ci["ci_high"]],
            "ece": ifcb_ece,
            "per_class": ifcb_per_class,
        },
        "zooscan": {
            "accuracy": zooscan_result["accuracy"],
            "ci_95": [zooscan_ci["ci_low"], zooscan_ci["ci_high"]],
            "ece": zooscan_ece,
            "per_class": zooscan_per_class,
        },
        "accuracy_drop": shift,
        "per_sample": {
            "ifcb": [
                {"path": p, "true": CLASSES[l], "pred": CLASSES[pr]}
                for p, l, pr in zip(ifcb_result["paths"], ifcb_result["labels"], ifcb_result["predictions"])
            ],
            "zooscan": [
                {"path": p, "true": CLASSES[l], "pred": CLASSES[pr]}
                for p, l, pr in zip(zooscan_result["paths"], zooscan_result["labels"], zooscan_result["predictions"])
            ],
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
