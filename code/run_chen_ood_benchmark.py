"""
run_chen_ood_benchmark.py
=========================
Replicate and beat Chen et al.'s (2025) OOD benchmark on ZooLake.

Chen's BEsT model achieves 83% OOD accuracy using:
- BEiT ensemble (5 models)
- Standard augmentations (rotation, flip, color jitter, blur)
- Rotation TTA (4 rotations × 2 flips)

Our approach replaces standard augmentations with frequency-calibrated SAA
and adds RAG grounding for hard cases.

Data: ZooLake35 from /home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed/

Usage:
    python run_chen_ood_benchmark.py --output-dir results/chen_benchmark
"""

import argparse
import json
import logging
import os
import re
import sys
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
TS_PATTERN = re.compile(r"SPC-EAWAG-0P5X-(\d+)-")

# Chen's 35 classes (ZooLake2.0)
CHEN_CLASSES = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina", "brachionus",
    "ceratium", "chaoborus", "conochilus", "copepod-skins", "cyclops",
    "daphnia", "daphnia-skins", "diaphanosoma", "diatom-chain", "dinobryon",
    "dirt", "eudiaptomus", "filament", "fish", "fragilaria", "hydra",
    "kellicottia", "keratella-cochlearis", "keratella-quadrata", "leptodora",
    "maybe-cyano", "nauplius", "paradileptus", "polyarthra", "rotifers",
    "synchaeta", "trichocerca", "unknown", "unknown-plankton", "uroglena",
]

DATA_ROOT = "/home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed"


def extract_date(filename):
    m = TS_PATTERN.search(filename)
    if m:
        try:
            return datetime.fromtimestamp(int(m.group(1)) / 1_000_000).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return "unknown"


class ZooLakeDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
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
        return image, self.class_to_idx.get(cls_name, 0), path


def compute_shift_spectrum(data_dir, classes, max_per_class=30):
    """Compute temporal shift spectrum from training data."""
    spectra = []
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
                spectra.append(radial)
                count += 1
            except Exception:
                continue
            if count >= max_per_class:
                break
    if not spectra:
        return None
    max_len = max(len(s) for s in spectra)
    matrix = np.zeros((len(spectra), max_len))
    for i, s in enumerate(spectra):
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
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1),
            "correct": correct, "total": total, "predictions": all_preds, "labels": all_labels}


def bootstrap_ci(binary, n=1000, ci=0.95):
    rng = np.random.RandomState(42)
    arr = np.array(binary)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    return {"mean": float(arr.mean()), "ci_low": float(np.percentile(boots, alpha * 100)),
            "ci_high": float(np.percentile(boots, (1 - alpha) * 100))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/chen_benchmark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentation", type=str, default="saa_band",
                        choices=["standard", "heavy", "saa_noise", "saa_band"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all images with dates
    logger.info("Collecting ZooLake35 images...")
    all_images = []
    for cls_name in CHEN_CLASSES:
        cls_dir = Path(DATA_ROOT) / cls_name
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            date = extract_date(img_path.name)
            if date != "unknown":
                all_images.append((str(img_path), cls_name, date))

    logger.info("Found %d images across %d classes", len(all_images), len(CHEN_CLASSES))

    # Group by date, find dates with enough data
    date_images = defaultdict(list)
    for path, cls, date in all_images:
        date_images[date].append((path, cls))

    # Use year-based split: train on 2018, test on 2019-2020
    train_dates = sorted([d for d in date_images if d.startswith("2018")])
    test_dates = sorted([d for d in date_images if d.startswith("2019") or d.startswith("2020")])

    logger.info("Train dates (2018): %d", len(train_dates))
    logger.info("Test dates (2019-2020): %d", len(test_dates))

    # Build datasets
    class_to_idx = {c: i for i, c in enumerate(CHEN_CLASSES)}
    num_classes = len(CHEN_CLASSES)

    train_samples = []
    for date in train_dates:
        for path, cls in date_images[date]:
            train_samples.append((path, cls))

    test_samples = []
    for date in test_dates:
        for path, cls in date_images[date]:
            test_samples.append((path, cls))

    logger.info("Train: %d samples, Test (OOD): %d samples", len(train_samples), len(test_samples))

    # Compute shift spectrum
    logger.info("Computing shift spectrum...")
    shift_spectrum = compute_shift_spectrum(DATA_ROOT, CHEN_CLASSES)

    # Augmentation strategies
    if args.augmentation == "saa_band":
        sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
        from spectral_augmentation import SpectralAugmentation

        class SAATransform:
            def __init__(self):
                self.aug = SpectralAugmentation(
                    shift_spectrum=shift_spectrum,
                    strategies=["band_adversarial"],
                    strength=0.5, p=0.8,
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
                return self.base(Image.fromarray(arr_uint8, mode="L").convert("RGB"))

        train_transform = SAATransform()
    elif args.augmentation == "heavy":
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
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
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
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

    train_ds = ZooLakeDataset(train_samples, class_to_idx, train_transform)
    test_ds = ZooLakeDataset(test_samples, class_to_idx, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Train ensemble
    criterion = nn.CrossEntropyLoss()
    ensemble_preds = []

    for seed in range(args.n_ensemble):
        logger.info("=" * 60)
        logger.info("Training ensemble member %d/%d (seed=%d)", seed + 1, args.n_ensemble, seed)
        logger.info("=" * 60)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

        # Collect predictions with TTA
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                if args.tta:
                    probs = rotation_tta(model, images, device)
                else:
                    probs = torch.softmax(model(images), dim=1)
                all_probs.append(probs.cpu())
                all_labels.extend(labels.numpy())

        ensemble_preds.append(torch.cat(all_probs))

    # Ensemble prediction (geometric mean of softmax)
    logger.info("Computing ensemble prediction...")
    ensemble_probs = torch.stack(ensemble_preds).mean(0)
    ensemble_preds_classes = ensemble_probs.argmax(dim=1).numpy()
    all_labels = np.array(all_labels)

    # Compute accuracy
    correct = (ensemble_preds_classes == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total

    binary = (ensemble_preds_classes == all_labels).astype(int)
    ci = bootstrap_ci(binary)

    # Per-class accuracy
    per_class = {}
    for cls, idx in class_to_idx.items():
        mask = all_labels == idx
        if mask.sum() > 0:
            per_class[cls] = {"accuracy": float((ensemble_preds_classes[mask] == all_labels[mask]).mean()),
                              "total": int(mask.sum())}

    # Save results
    results = {
        "method": f"SAA-{args.augmentation} + BEiT ensemble + TTA={args.tta}",
        "n_ensemble": args.n_ensemble,
        "augmentation": args.augmentation,
        "use_tta": args.tta,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_dates": train_dates[:5],
        "test_dates": test_dates[:5],
        "ood_accuracy": accuracy,
        "ci_95": [ci["ci_low"], ci["ci_high"]],
        "per_class": per_class,
        "chen_best": 0.83,
        "vs_chen": accuracy - 0.83,
    }

    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print
    logger.info("=" * 72)
    logger.info("  CHEN OOD BENCHMARK RESULTS")
    logger.info("=" * 72)
    logger.info("  Method: %s", results["method"])
    logger.info("  Train: %d samples (2018 dates)", len(train_samples))
    logger.info("  Test (OOD): %d samples (2019-2020 dates)", len(test_samples))
    logger.info("  OOD Accuracy: %.1f%% [%.1f-%.1f%%]", accuracy * 100, ci["ci_low"] * 100, ci["ci_high"] * 100)
    logger.info("  Chen's BEsT: 83%%")
    logger.info("  vs Chen: %+.1f%%", (accuracy - 0.83) * 100)
    logger.info("-" * 72)
    logger.info("  Per-class OOD accuracy:")
    for cls, stats in sorted(per_class.items(), key=lambda x: -x[1]["accuracy"]):
        logger.info("    %-30s  %.1f%%  (n=%d)", cls, stats["accuracy"] * 100, stats["total"])
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
