"""
evaluate_ood_from_crops.py
==========================
Evaluate a trained ViT on v2a-segmented OOD crops.

The model was trained on ZooLake2 crops (organism on white background).
Now evaluate on each OOD day separately.

Usage:
    python evaluate_ood_from_crops.py \
        --model-path results/crop_classifier_best.pth \
        --ood-dir data_segmentation/ood \
        --output results/ood_from_crops.json
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}
IMG_SIZE = 224

# Classes from ZooLake2 (35 classes)
CLASSES = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina",
    "ceratium", "chaoborus", "collotheca", "conochilus", "copepod_skins",
    "cyclops", "daphnia", "daphnia_skins", "diaphanosoma", "diatom_chain",
    "dinobryon", "dirt", "eudiaptomus", "filament", "fish", "fragilaria",
    "hydra", "kellicottia", "keratella_cochlearis", "keratella_quadrata",
    "leptodora", "maybe_cyano", "nauplius", "paradileptus", "polyarthra",
    "rotifers", "synchaeta", "trichocerca", "unknown", "unknown_plankton", "uroglena",
]


class CropDataset(Dataset):
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
                self.samples.append((str(img_path), idx, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cls_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, cls_name


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
    per_class = {}
    for images, labels, cls_names in loader:
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

        for pred, true, cls in zip(predicted.cpu().numpy(), labels.cpu().numpy(), cls_names):
            if cls not in per_class:
                per_class[cls] = {"correct": 0, "total": 0}
            per_class[cls]["total"] += 1
            if pred == true:
                per_class[cls]["correct"] += 1

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "per_class": per_class,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--ood-dir", type=str, required=True)
    parser.add_argument("--arch", type=str, default="vit_b_16")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output", type=str, default="results/ood_from_crops.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    num_classes = len(CLASSES)

    # Load model
    if args.arch == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif args.arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model = model.to(device)
    logger.info("Loaded model from %s", args.model_path)

    criterion = nn.CrossEntropyLoss()
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Evaluate on each OOD day
    ood_dir = Path(args.ood_dir)
    results = {"model": args.arch, "tta": args.tta, "per_day": {}}

    for ood_day in sorted(ood_dir.iterdir()):
        if not ood_day.is_dir():
            continue
        ds = CropDataset(str(ood_day), class_to_idx, eval_transform)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        res = evaluate(model, loader, criterion, device, use_tta=args.tta)
        binary = [1 if p == l else 0 for p, l in zip(res["predictions"], res["labels"])]
        ci = bootstrap_ci(binary)

        results["per_day"][ood_day.name] = {
            "accuracy": res["accuracy"],
            "ci_95": [ci["ci_low"], ci["ci_high"]],
            "n": res["total"],
            "per_class": res["per_class"],
        }

        logger.info("  %s: %.1f%% [%.1f-%.1f%%] (n=%d)",
                     ood_day.name, res["accuracy"] * 100,
                     ci["ci_low"] * 100, ci["ci_high"] * 100, res["total"])

    # Overall
    all_acc = [v["accuracy"] for v in results["per_day"].values()]
    results["overall"] = float(np.mean(all_acc))
    results["overall_std"] = float(np.std(all_acc))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  OOD EVALUATION ON SEGMENTED CROPS")
    logger.info("=" * 60)
    logger.info("  Model: %s", args.arch)
    logger.info("  Overall OOD: %.1f%% (±%.1f%%)", results["overall"] * 100, results["overall_std"] * 100)
    logger.info("  Chen's BEsT: 83.0%%")
    logger.info("  vs Chen: %+.1f%%", results["overall"] * 100 - 83)
    for day, data in sorted(results["per_day"].items()):
        logger.info("    %s: %.1f%%", day, data["accuracy"] * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
