"""
ensemble_all_methods.py
=======================
Smart ensembling: loads checkpoints, runs inference ONCE per model,
saves per-sample probabilities, then computes ALL ensemble methods.

Ensemble methods:
1. Arithmetic mean of softmax (current approach)
2. Geometric mean of softmax (Chen's approach)
3. Majority vote
4. Confidence-weighted average
5. Temperature-scaled ensemble

Usage:
    python ensemble_all_methods.py \
        --ood-dir data/chen_data/OOD_data/OODs \
        --output-dir results/ensemble_analysis
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224
RESULTS_DIR = Path("/home/sreenath/research-space/PlanktonShift/results")
OOD_DIR = "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs"

CHEN_CLASSES = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina", "brachionus",
    "ceratium", "chaoborus", "collotheca", "conochilus", "copepod_skins",
    "cyclops", "daphnia", "daphnia_skins", "diaphanosoma", "diatom_chain",
    "dinobryon", "dirt", "eudiaptomus", "filament", "fish", "fragilaria",
    "hydra", "kellicottia", "keratella_cochlearis", "keratella_quadrata",
    "leptodora", "maybe_cyano", "nauplius", "paradileptus", "polyarthra",
    "rotifers", "synchaeta", "trichocerca", "unknown", "unknown_plankton", "uroglena",
]


class OODDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform):
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
                    self.samples.append((str(img_path), idx, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cls_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


def rotation_tta_probs(model, images, device, model_type="beit"):
    """Get TTA-averaged softmax probabilities."""
    all_probs = []
    with torch.no_grad():
        for k in range(4):
            rotated = torch.rot90(images, k, dims=[2, 3]).to(device)
            out = model(rotated)
            logits = out.logits if hasattr(out, 'logits') else out
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

            flipped = torch.flip(rotated, [3])
            out2 = model(flipped)
            logits2 = out2.logits if hasattr(out2, 'logits') else out2
            all_probs.append(torch.softmax(logits2, dim=1).cpu().numpy())

    return np.mean(all_probs, axis=0)


@torch.no_grad()
def extract_probs(model, loader, device, model_type="beit"):
    """Extract per-sample TTA probabilities from a model."""
    model.eval()
    all_probs = []
    all_labels = []
    all_paths = []
    all_cls_names = []

    for images, labels, paths in loader:
        images = images.to(device)
        probs = rotation_tta_probs(model, images, device, model_type)
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
        # Extract class names from paths
        for p in paths:
            cls_name = Path(p).parent.name
            all_cls_names.append(cls_name)

    return np.vstack(all_probs), np.array(all_labels), all_paths, all_cls_names


def ensemble_arithmetic_mean(probs_list):
    """Arithmetic mean of softmax probabilities."""
    return np.mean(probs_list, axis=0)


def ensemble_geometric_mean(probs_list):
    """Geometric mean of softmax probabilities (Chen's approach)."""
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    log_probs = [np.log(p + eps) for p in probs_list]
    mean_log = np.mean(log_probs, axis=0)
    # Normalize to sum to 1
    result = np.exp(mean_log)
    return result / result.sum(axis=1, keepdims=True)


def ensemble_majority_vote(probs_list):
    """Majority vote of argmax predictions."""
    preds = [np.argmax(p, axis=1) for p in probs_list]
    preds_stack = np.stack(preds)  # (n_models, n_samples)
    n_classes = probs_list[0].shape[1]

    # For each sample, count votes for each class
    result = np.zeros_like(probs_list[0])
    for i in range(preds_stack.shape[1]):
        votes = preds_stack[:, i]
        counts = np.bincount(votes, minlength=n_classes)
        result[i] = counts / len(probs_list)

    return result


def ensemble_confidence_weighted(probs_list):
    """Weight each model by its confidence (max probability)."""
    weights = []
    for p in probs_list:
        max_conf = np.max(p, axis=1).mean()
        weights.append(max_conf)
    weights = np.array(weights)
    weights = weights / weights.sum()

    result = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        result += w * p
    return result


def ensemble_temperature_scaled(probs_list, temperature=2.0):
    """Temperature-scaled ensemble (sharpen predictions)."""
    mean_probs = np.mean(probs_list, axis=0)
    # Apply temperature scaling
    log_probs = np.log(mean_probs + 1e-8) / temperature
    result = np.exp(log_probs)
    return result / result.sum(axis=1, keepdims=True)


def compute_metrics(probs, labels, cls_names, ood_days):
    """Compute accuracy, per-day accuracy, per-class accuracy."""
    preds = np.argmax(probs, axis=1)
    overall_acc = float((preds == labels).mean())

    # Per-day
    per_day = {}
    for day in ood_days:
        day_mask = np.array([day in p for p in cls_names])  # This won't work, need to match by path
        # Actually, OOD days are directories, so we need to match by path structure
        pass

    # Simpler: match by the OOD directory structure
    # OOD paths look like: .../OOD1/ClassName/img.png
    per_day = {}
    for day in sorted(set([Path(p).parent.parent.name for p in cls_names if 'OOD' in str(p)])):
        day_mask = np.array([Path(p).parent.parent.name == day for p in cls_names])
        if day_mask.sum() > 0:
            per_day[day] = float((preds[day_mask] == labels[day_mask]).mean())

    # Per-class
    per_class = {}
    unique_cls = sorted(set(cls_names))
    for cls in unique_cls:
        cls_mask = np.array([c == cls for c in cls_names])
        if cls_mask.sum() > 0:
            per_class[cls] = float((preds[cls_mask] == labels[cls_mask]).mean())

    return overall_acc, per_day, per_class


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood-dir", type=str, default=OOD_DIR)
    parser.add_argument("--output-dir", type=str, default="results/ensemble_analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    class_to_idx = {c: i for i, c in enumerate(CHEN_CLASSES)}
    num_classes = len(CHEN_CLASSES)

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Load OOD data
    ood_dir = Path(args.ood_dir)
    all_samples = []
    for ood_cell in sorted(ood_dir.iterdir()):
        if ood_cell.is_dir():
            ds = OODDataset(str(ood_cell), class_to_idx, eval_transform)
            all_samples.extend(ds.samples)

    combined_ds = OODDataset.__new__(OODDataset)
    combined_ds.samples = all_samples
    combined_ds.transform = eval_transform
    loader = DataLoader(combined_ds, batch_size=16, shuffle=False, num_workers=4)

    logger.info("Loaded %d OOD samples", len(all_samples))

    # Load BEiT checkpoints and extract probabilities
    all_probs = {}
    model_type = "beit"

    for seed in range(3):
        ckpt_path = RESULTS_DIR / f"beit_chen_ood_seed{seed}_checkpoint.pth"
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s", ckpt_path)
            continue

        logger.info("Loading BEiT seed=%d...", seed)
        from transformers import BeitForImageClassification
        model = BeitForImageClassification.from_pretrained(
            "microsoft/beit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        logger.info("  Extracting probabilities with TTA...")
        probs, labels, paths, cls_names = extract_probs(model, loader, device, model_type)
        all_probs[f"seed_{seed}"] = probs

        # Save per-sample probabilities
        np.save(str(out / f"beit_seed{seed}_probs.npy"), probs)
        logger.info("  Saved per-sample probabilities: %s", probs.shape)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Also load ViT checkpoints
    for seed in range(3):
        ckpt_path = RESULTS_DIR / f"vit_chen_ood_seed{seed}_checkpoint.pth"
        if not ckpt_path.exists():
            # Try the Adverserial_net results
            alt_path = Path("/home/sreenath/research-space/Adverserial_net/results") / f"vit_b_16_seed{seed}_best.pth"
            if alt_path.exists():
                ckpt_path = alt_path
            else:
                continue

        logger.info("Loading ViT seed=%d...", seed)
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)

        logger.info("  Extracting probabilities with TTA...")
        probs, labels, paths, cls_names = extract_probs(model, loader, device, "vit")
        all_probs[f"vit_seed_{seed}"] = probs

        np.save(str(out / f"vit_seed{seed}_probs.npy"), probs)
        logger.info("  Saved per-sample probabilities: %s", probs.shape)

        del model
        torch.cuda.empty_cache()

    # Save labels and paths
    np.save(str(out / "labels.npy"), labels)
    with open(str(out / "paths.json"), "w") as f:
        json.dump(paths, f)

    logger.info("Loaded %d models: %s", len(all_probs), list(all_probs.keys()))

    # ===================================================================
    # Compute all ensemble methods
    # ===================================================================
    ood_days = sorted(set([Path(p).parent.parent.name for p in paths if 'OOD' in str(p)]))

    # BEiT-only ensembles
    beit_probs = [all_probs[k] for k in sorted(all_probs.keys()) if k.startswith("seed_")]

    ensemble_methods = {}
    if len(beit_probs) >= 2:
        ensemble_methods["BEiT arithmetic_mean"] = ensemble_arithmetic_mean(beit_probs)
        ensemble_methods["BEiT geometric_mean"] = ensemble_geometric_mean(beit_probs)
        ensemble_methods["BEiT majority_vote"] = ensemble_majority_vote(beit_probs)
        ensemble_methods["BEiT confidence_weighted"] = ensemble_confidence_weighted(beit_probs)
        ensemble_methods["BEiT temperature_0.5"] = ensemble_temperature_scaled(beit_probs, 0.5)
        ensemble_methods["BEiT temperature_2.0"] = ensemble_temperature_scaled(beit_probs, 2.0)

    # ViT-only ensembles
    vit_probs = [all_probs[k] for k in sorted(all_probs.keys()) if k.startswith("vit_seed_")]
    if len(vit_probs) >= 2:
        ensemble_methods["ViT arithmetic_mean"] = ensemble_arithmetic_mean(vit_probs)
        ensemble_methods["ViT geometric_mean"] = ensemble_geometric_mean(vit_probs)

    # Mixed BEiT + ViT ensemble
    if len(beit_probs) >= 1 and len(vit_probs) >= 1:
        mixed = beit_probs + vit_probs
        ensemble_methods["Mixed arithmetic_mean"] = ensemble_arithmetic_mean(mixed)
        ensemble_methods["Mixed geometric_mean"] = ensemble_geometric_mean(mixed)

    # Individual models
    for name, probs in all_probs.items():
        ensemble_methods[name] = probs

    # ===================================================================
    # Compute and compare all methods
    # ===================================================================
    results = {}
    logger.info("")
    logger.info("=" * 80)
    logger.info("  ENSEMBLE COMPARISON")
    logger.info("=" * 80)
    logger.info(f"  {'Method':35s} {'Overall':>8s} ", end="")
    for day in ood_days[:3]:
        logger.info(f" {day:>6s}", end="")
    logger.info(" ...")
    logger.info("  " + "-" * 75)

    best_acc = 0
    best_method = ""

    for method_name, probs in sorted(ensemble_methods.items()):
        overall_acc, per_day, per_class = compute_metrics(probs, labels, cls_names, ood_days)

        results[method_name] = {
            "overall": overall_acc,
            "per_day": per_day,
            "per_class": per_class,
            "n_models": 1 if "seed" in method_name and "mean" not in method_name else len(beit_probs) + len(vit_probs),
        }

        logger.info(f"  {method_name:35s} {overall_acc*100:7.1f}% ", end="")
        for day in ood_days[:3]:
            acc = per_day.get(day, 0)
            logger.info(f" {acc*100:5.1f}%", end="")
        logger.info("")

        if overall_acc > best_acc:
            best_acc = overall_acc
            best_method = method_name

    logger.info("  " + "-" * 75)
    logger.info(f"  BEST: {best_method} = {best_acc*100:.1f}%")
    logger.info(f"  Chen BEsT: 83.0%")
    logger.info(f"  vs Chen: {best_acc*100 - 83:+.1f}%")
    logger.info("=" * 80)

    # Save results
    with open(out / "ensemble_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", out / "ensemble_comparison.json")

    # Print best per-day breakdown
    if best_method in results:
        logger.info("")
        logger.info(f"  Best method: {best_method}")
        logger.info(f"  Per-day breakdown:")
        for day in ood_days:
            acc = results[best_method]["per_day"].get(day, 0)
            logger.info(f"    {day}: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
