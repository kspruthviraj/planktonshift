"""
train_with_saa.py
=================
Main training script integrating Spectral Band Adversarial augmentation (SBA) for
cross-domain plankton classification.

Trains multiple architectures with multiple augmentation strategies and evaluates
across all available domains.

Usage:
    python train_with_saa.py \
        --data-dir data/cross_domain/cross_domain_2plus \
        --source-domain WHOI22 \
        --target-domains ZooScan20 ZooLake35 \
        --architectures vit_b_16 deit_b \
        --augmentation saa_all \
        --epochs 30 \
        --output results/sba_experiment.json
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

from spectral_augmentation import SpectralAugmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CrossDomainDataset(Dataset):
    """Load images from cross-domain directory structure."""

    def __init__(self, data_dir: str, domain: str, class_to_idx: dict, transform=None):
        self.samples = []
        self.transform = transform
        domain_dir = Path(data_dir) / domain
        if not domain_dir.is_dir():
            logger.warning("Domain directory not found: %s", domain_dir)
            return
        for cls_name, idx in class_to_idx.items():
            cls_dir = domain_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), idx))
        logger.info("Loaded %d samples from %s/%s", len(self.samples), data_dir, domain)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


# ---------------------------------------------------------------------------
# Augmentation strategies
# ---------------------------------------------------------------------------
def load_shift_spectrum(path: str) -> np.ndarray:
    """Load shift spectrum from fourier_shift_analysis.py output."""
    if not Path(path).exists():
        logger.warning("Shift spectrum not found: %s", path)
        return None
    with open(path) as f:
        data = json.load(f)
    for pair, shift in data.get("shift_spectra", {}).items():
        return np.array(shift["diff"])
    # Try alternative format
    if "diff" in data:
        return np.array(data["diff"])
    return None


class SBATransform:
    """Applies Spectral Adversarial Augmentation to a PIL Image, then standard transforms."""

    def __init__(self, sba_strategies: list, shift_spectrum=None,
                 target_image_paths=None, strength=0.5, p=0.5):
        # Pre-load target-domain images for FDA
        target_images = []
        if target_image_paths:
            for path in target_image_paths[:50]:  # Limit to 50 for memory
                try:
                    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                    target_images.append(np.array(img, dtype=np.float64) / 255.0)
                except Exception:
                    continue
            logger.info("FDA: pre-loaded %d target-domain images", len(target_images))

        self.aug = SpectralAugmentation(
            shift_spectrum=shift_spectrum,
            target_images=target_images if target_images else None,
            strength=strength,
            strategies=sba_strategies,
            p=p,
        )
        self.base_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        image_aug = Image.fromarray(arr_uint8, mode="L").convert("RGB")
        return self.base_transform(image_aug)


SBA_STRATEGIES_MAP = {
    "saa_amplitude": ["amplitude_mix"],
    "saa_noise": ["spectral_noise"],
    "saa_band": ["band_adversarial"],
    "saa_fda": ["fda_swap"],
    "saa_best": ["spectral_noise", "band_adversarial", "fda_swap"],
    "saa_all": ["amplitude_mix", "spectral_noise", "band_adversarial", "fda_swap"],
}


def get_train_transform(augmentation: str, shift_spectrum=None, target_image_paths=None):
    """Get training transform based on augmentation strategy."""

    if augmentation == "standard":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    elif augmentation == "heavy":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    elif augmentation in SBA_STRATEGIES_MAP:
        strategies = SBA_STRATEGIES_MAP[augmentation]
        has_fda = "fda_swap" in strategies
        logger.info("SBA strategies: %s  shift_spectrum: %s  FDA target images: %s",
                     strategies,
                     "loaded" if shift_spectrum is not None else "None",
                     len(target_image_paths) if target_image_paths else 0)
        return SBATransform(
            sba_strategies=strategies,
            shift_spectrum=shift_spectrum,
            target_image_paths=target_image_paths if has_fda else None,
            strength=0.5,
            p=0.8,
        )

    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch == "deit_b":
        # DeiT-B has same architecture as ViT-B but different training
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, sba_augmenter=None):
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
def evaluate(model, loader, criterion, device, use_tta=False):
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

        if use_tta:
            # Rotation TTA: average over 4 rotations + horizontal flip
            tta_preds = []
            for k in range(4):
                rotated = torch.rot90(images, k, dims=[2, 3])
                tta_preds.append(torch.softmax(model(rotated), dim=1))
                tta_preds.append(torch.softmax(model(torch.flip(rotated, [3])), dim=1))
            probs = torch.stack(tta_preds).mean(0)
            outputs = torch.log(probs + 1e-8)  # For loss computation
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
# ECE and Bootstrap CI
# ---------------------------------------------------------------------------
def compute_ece(probabilities, labels, n_bins=15):
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


def bootstrap_ci(accuracies, n_bootstrap=1000, ci=0.95):
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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train with SBA for cross-domain plankton classification.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--source-domain", type=str, default="WHOI22")
    parser.add_argument("--target-domains", nargs="+", default=["ZooScan20", "ZooLake35"])
    parser.add_argument("--architectures", nargs="+", default=["vit_b_16", "resnet50"],
                        choices=["vit_b_16", "deit_b", "resnet50", "convnext_tiny"])
    parser.add_argument("--augmentation", type=str, default="standard",
                        choices=["standard", "heavy", "saa_amplitude", "saa_noise",
                                 "saa_band", "saa_fda", "saa_best", "saa_all"])
    parser.add_argument("--tta", action="store_true",
                        help="Use rotation TTA at evaluation time")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds for ensemble training")
    parser.add_argument("--output", type=str, default="results/sba_experiment.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Discover classes from directory structure
    source_dir = Path(args.data_dir) / args.source_domain
    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("Found %d classes: %s", num_classes, classes)

    # Transforms
    shift_spectrum = None
    target_image_paths = []
    if args.augmentation.startswith("saa"):
        # Load shift spectrum
        spectrum_paths = [
            "results/fourier_analysis/cross_domain/fourier_analysis.json",
            "results/shift_spectrum_ifcb_zooscan.json",
        ]
        for sp in spectrum_paths:
            shift_spectrum = load_shift_spectrum(sp)
            if shift_spectrum is not None:
                logger.info("Loaded shift spectrum from %s", sp)
                break
        if shift_spectrum is None:
            logger.warning("No shift spectrum found — SBA will use default parameters")

        # Load target-domain images for FDA-style augmentation
        for target_domain in args.target_domains:
            target_dir = Path(args.data_dir) / target_domain
            if target_dir.is_dir():
                for cls_dir in sorted(target_dir.iterdir()):
                    if cls_dir.is_dir():
                        for img_path in sorted(cls_dir.iterdir()):
                            if img_path.suffix.lower() in SUPPORTED_EXT:
                                target_image_paths.append(str(img_path))
        logger.info("Found %d target-domain images for FDA", len(target_image_paths))

    train_transform = get_train_transform(
        args.augmentation, shift_spectrum=shift_spectrum,
        target_image_paths=target_image_paths
    )
    eval_transform = get_eval_transform()

    # Datasets
    train_ds = CrossDomainDataset(args.data_dir, args.source_domain, class_to_idx, train_transform)
    test_loaders = {}
    for domain in [args.source_domain] + args.target_domains:
        ds = CrossDomainDataset(args.data_dir, domain, class_to_idx, eval_transform)
        if len(ds) > 0:
            test_loaders[domain] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    all_results = {}

    for arch in args.architectures:
        for seed in args.seeds:
            logger.info("=" * 60)
            logger.info("Training %s (seed=%d, aug=%s)", arch, seed, args.augmentation)
            logger.info("=" * 60)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            model = build_model(arch, num_classes, pretrained=True).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            best_acc = 0
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                scheduler.step()
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    val_res = evaluate(model, test_loaders[args.source_domain], criterion, device)
                    logger.info(
                        "Epoch %d/%d  Loss: %.4f  Train Acc: %.2f%%  Val Acc: %.2f%%",
                        epoch + 1, args.epochs, train_loss, train_acc * 100,
                        val_res["accuracy"] * 100,
                    )
                    if val_res["accuracy"] > best_acc:
                        best_acc = val_res["accuracy"]
                        torch.save(model.state_dict(), f"results/{arch}_seed{seed}_best.pth")

            # Load best and evaluate on all domains
            model.load_state_dict(torch.load(f"results/{arch}_seed{seed}_best.pth", weights_only=True))

            result_key = f"{arch}_seed{seed}_{args.augmentation}"
            all_results[result_key] = {
                "architecture": arch,
                "seed": seed,
                "augmentation": args.augmentation,
                "epochs": args.epochs,
                "source_domain": args.source_domain,
                "domains": {},
            }

            for domain, loader in test_loaders.items():
                res = evaluate(model, loader, criterion, device, use_tta=args.tta)
                binary = [1 if p == l else 0 for p, l in zip(res["predictions"], res["labels"])]
                ci = bootstrap_ci(binary)
                ece = compute_ece(res["probabilities"], res["labels"])

                per_class = {}
                preds = np.array(res["predictions"])
                labels = np.array(res["labels"])
                for cls, idx in class_to_idx.items():
                    mask = labels == idx
                    if mask.sum() > 0:
                        per_class[cls] = {
                            "accuracy": float((preds[mask] == labels[mask]).mean()),
                            "total": int(mask.sum()),
                        }

                all_results[result_key]["domains"][domain] = {
                    "accuracy": res["accuracy"],
                    "ci_95": [ci["ci_low"], ci["ci_high"]],
                    "ece": ece,
                    "per_class": per_class,
                    "n_samples": res["total"],
                }

                logger.info("  %s: %.2f%% [95%% CI: %.2f-%.2f%%] ECE: %.4f",
                            domain, res["accuracy"] * 100,
                            ci["ci_low"] * 100, ci["ci_high"] * 100, ece)

            # Compute cross-domain drops
            source_acc = all_results[result_key]["domains"][args.source_domain]["accuracy"]
            for domain in args.target_domains:
                if domain in all_results[result_key]["domains"]:
                    target_acc = all_results[result_key]["domains"][domain]["accuracy"]
                    drop = source_acc - target_acc
                    all_results[result_key]["domains"][domain]["accuracy_drop"] = drop
                    logger.info("  Drop %s→%s: %.2f%%", args.source_domain, domain, drop * 100)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Print summary
    logger.info("=" * 72)
    logger.info("  SUMMARY")
    logger.info("=" * 72)
    for key, result in all_results.items():
        source_acc = result["domains"].get(args.source_domain, {}).get("accuracy", 0)
        logger.info("  %s: Source=%.2f%%", key, source_acc * 100)
        for domain in args.target_domains:
            if domain in result["domains"]:
                acc = result["domains"][domain]["accuracy"]
                drop = result["domains"][domain].get("accuracy_drop", 0)
                logger.info("    %s: %.2f%% (drop: %.2f%%)", domain, acc * 100, drop * 100)


if __name__ == "__main__":
    main()
