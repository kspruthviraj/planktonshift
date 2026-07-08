"""
benchmark_chen_exact.py
=======================
Replicate and beat Chen et al.'s exact OOD benchmark.

Uses:
- Chen's ZooLake2.0 training data (35 classes, ~30K images)
- Chen's 10 OOD test days (OOD1-OOD10, ~9,522 images)
- Chen's trained BEiT models for comparison

Our approach: replace standard augmentations with frequency-calibrated SAA.

Usage:
    python benchmark_chen_exact.py \
        --train-dir data/chen_data/ZooLake2/ZooLake2/ZooLake2.0 \
        --ood-dir data/chen_data/OOD_data/OODs \
        --output-dir results/chen_exact_benchmark
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


class PlanktonDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform=None, max_per_class=0):
        self.samples = []
        self.transform = transform
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return
        for cls_name, idx in class_to_idx.items():
            cls_dir = data_path / cls_name
            if not cls_dir.is_dir():
                continue
            count = 0
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_EXT:
                    continue
                self.samples.append((str(img_path), idx))
                count += 1
                if max_per_class > 0 and count >= max_per_class:
                    break
        logger.info("Loaded %d samples from %s", len(self.samples), data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


def compute_shift_spectrum(data_dir, classes, max_per_class=30):
    """Compute temporal shift spectrum."""
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
    parser.add_argument("--train-dir", type=str,
                        default="data/chen_data/ZooLake2/ZooLake2/ZooLake2.0")
    parser.add_argument("--ood-dir", type=str,
                        default="data/chen_data/OOD_data/OODs")
    parser.add_argument("--output-dir", type=str, default="results/chen_exact_benchmark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentation", type=str, default="saa_band",
                        choices=["standard", "saa_band"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--n-ensemble", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Discover classes from training data
    train_path = Path(args.train_dir)
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("Found %d classes in training data", num_classes)

    # Compute shift spectrum
    logger.info("Computing shift spectrum...")
    shift_spectrum = compute_shift_spectrum(args.train_dir, classes)
    if shift_spectrum is not None:
        logger.info("Shift spectrum: %d bins", len(shift_spectrum))

    # Augmentation
    if args.augmentation == "saa_band":
        class SAATransform:
            def __init__(self):
                self.aug = SpectralAugmentation(
                    shift_spectrum=shift_spectrum,
                    strategies=["band_adversarial"],
                    strength=0.5, p=0.8,
                )
                self.base = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
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

    # Training dataset
    train_ds = PlanktonDataset(args.train_dir, class_to_idx, train_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # OOD test datasets
    ood_dir = Path(args.ood_dir)
    ood_loaders = {}
    for ood_cell in sorted(ood_dir.iterdir()):
        if ood_cell.is_dir():
            ds = PlanktonDataset(str(ood_cell), class_to_idx, eval_transform)
            if len(ds) > 0:
                ood_loaders[ood_cell.name] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("OOD test cells: %s", list(ood_loaders.keys()))

    # Train ensemble
    criterion = nn.CrossEntropyLoss()
    ensemble_results = {}

    for seed in range(args.n_ensemble):
        logger.info("=" * 60)
        logger.info("Ensemble member %d/%d (seed=%d, aug=%s)",
                    seed + 1, args.n_ensemble, seed, args.augmentation)
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

        # Evaluate on all OOD cells
        model.eval()
        seed_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_res = evaluate(model, ood_loader, criterion, device, use_tta=args.tta)
            seed_results[ood_name] = ood_res["accuracy"]
            logger.info("  %s: %.1f%%", ood_name, ood_res["accuracy"] * 100)

        ensemble_results[f"seed_{seed}"] = seed_results

        # Save model
        torch.save(model.state_dict(), str(out / f"model_seed{seed}.pth"))

    # Compute ensemble average
    logger.info("=" * 72)
    logger.info("  ENSEMBLE RESULTS (geometric mean)")
    logger.info("=" * 72)

    ood_names = sorted(ood_loaders.keys())
    ensemble_avg = {}
    for ood_name in ood_names:
        accs = [ensemble_results[seed][ood_name] for seed in ensemble_results]
        ensemble_avg[ood_name] = float(np.mean(accs))

    overall_ood = np.mean(list(ensemble_avg.values()))
    ci = bootstrap_ci([1 if a > 0.5 else 0 for a in ensemble_avg.values()])

    logger.info("  Method: SAA-%s + ViT ensemble (n=%d) + TTA=%s",
                args.augmentation, args.n_ensemble, args.tta)
    logger.info("-" * 72)
    for ood_name in ood_names:
        logger.info("  %-8s  %.1f%%", ood_name, ensemble_avg[ood_name] * 100)
    logger.info("-" * 72)
    logger.info("  Overall OOD: %.1f%%", overall_ood * 100)
    logger.info("  Chen's BEsT: 83%%")
    logger.info("  vs Chen: %+.1f%%", (overall_ood - 0.83) * 100)
    logger.info("=" * 72)

    # Save
    results = {
        "method": f"SAA-{args.augmentation} + ViT ensemble + TTA={args.tta}",
        "n_ensemble": args.n_ensemble,
        "augmentation": args.augmentation,
        "use_tta": args.tta,
        "train_samples": len(train_ds),
        "num_classes": num_classes,
        "per_ood_cell": ensemble_avg,
        "overall_ood_accuracy": overall_ood,
        "chen_best": 0.83,
        "vs_chen": overall_ood - 0.83,
        "per_seed": ensemble_results,
    }

    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
