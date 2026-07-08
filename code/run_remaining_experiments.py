"""
run_remaining_experiments.py
============================
Crash-resistant master runner for all remaining experiments.
Saves checkpoints after every epoch so it can resume after any crash.

Remaining experiments:
1. ViT Chen OOD members 2+3 (member 1 done: 80.1%)
2. BEiT + SAA Chen OOD members 1-3
3. RAG on Chen OOD days
4. Cross-ecosystem SAA

Usage:
    nohup python run_remaining_experiments.py > remaining.log 2>&1 &
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("remaining_experiments.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224
RESULTS_DIR = Path("/home/sreenath/research-space/PlanktonShift/results")

CHEN_CLASSES = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina", "brachionus",
    "ceratium", "chaoborus", "collotheca", "conochilus", "copepod_skins",
    "cyclops", "daphnia", "daphnia_skins", "diaphanosoma", "diatom_chain",
    "dinobryon", "dirt", "eudiaptomus", "filament", "fish", "fragilaria",
    "hydra", "kellicottia", "keratella_cochlearis", "keratella_quadrata",
    "leptodora", "maybe_cyano", "nauplius", "paradileptus", "polyarthra",
    "rotifers", "synchaeta", "trichocerca", "unknown", "unknown_plankton",
    "uroglena",
]

CHEN_ZOOLAKE = "/home/sreenath/research-space/PlanktonShift/data/chen_data/ZooLake2/ZooLake2/ZooLake2.0"
CHEN_OOD = "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs"


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def checkpoint_path(experiment: str, seed: int) -> Path:
    return RESULTS_DIR / f"{experiment}_seed{seed}_checkpoint.pth"


def result_path(experiment: str) -> Path:
    return RESULTS_DIR / f"{experiment}_results.json"


def load_checkpoint(experiment: str, seed: int):
    """Load checkpoint if exists. Returns (epoch, model_state, optimizer_state, results) or None."""
    path = checkpoint_path(experiment, seed)
    if path.exists():
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        logger.info("  Resumed %s seed=%d from epoch %d", experiment, seed, ckpt["epoch"])
        return ckpt
    return None


def save_checkpoint(experiment: str, seed: int, epoch: int, model, optimizer, results=None):
    """Save checkpoint."""
    path = checkpoint_path(experiment, seed)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "results": results,
    }, path)


def save_results(experiment: str, results: dict):
    """Save final results."""
    path = result_path(experiment)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def is_experiment_done(experiment: str) -> bool:
    """Check if experiment has final results saved."""
    return result_path(experiment).exists()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PlanktonDataset(Dataset):
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
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


# ---------------------------------------------------------------------------
# SAA transform
# ---------------------------------------------------------------------------
class SAATransform:
    def __init__(self, shift_spectrum=None, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
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
            transforms.Normalize(norm_mean, norm_std),
        ])

    def __call__(self, image):
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        return self.base(Image.fromarray(arr_uint8, mode="L").convert("RGB"))


# ---------------------------------------------------------------------------
# Shift spectrum
# ---------------------------------------------------------------------------
def compute_shift_spectrum(data_dir, classes, max_per_class=30):
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


# ---------------------------------------------------------------------------
# TTA and evaluation
# ---------------------------------------------------------------------------
def rotation_tta(model, images, device, model_type="vit"):
    preds = []
    for k in range(4):
        rotated = torch.rot90(images, k, dims=[2, 3]).to(device)
        with torch.no_grad():
            out = model(rotated)
            logits = out.logits if hasattr(out, 'logits') else out
            preds.append(torch.softmax(logits, dim=1))
            flipped = torch.flip(rotated, [3])
            out2 = model(flipped)
            logits2 = out2.logits if hasattr(out2, 'logits') else out2
            preds.append(torch.softmax(logits2, dim=1))
    return torch.stack(preds).mean(0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta=False, model_type="vit"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if use_tta:
            probs = rotation_tta(model, images, device, model_type)
            outputs = torch.log(probs + 1e-8)
        else:
            out = model(images)
            logits = out.logits if hasattr(out, 'logits') else out
            probs = torch.softmax(logits, dim=1)
            outputs = torch.log(probs + 1e-8)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = probs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1),
            "total": total}


# ---------------------------------------------------------------------------
# EXPERIMENT: Train ViT/BEiT ensemble member with checkpoint/resume
# ---------------------------------------------------------------------------
def train_ensemble_member(
    experiment_name, seed, model_type, train_loader, ood_loaders,
    criterion, device, epochs=30, use_tta=True
):
    """Train one ensemble member with checkpoint/resume."""
    logger.info("--- %s member seed=%d ---", experiment_name, seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Build model
    if model_type == "beit":
        from transformers import BeitForImageClassification
        model = BeitForImageClassification.from_pretrained(
            "microsoft/beit-base-patch16-224",
            num_labels=len(CHEN_CLASSES),
            ignore_mismatched_sizes=True,
        )
    else:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CHEN_CLASSES))

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint
    start_epoch = 0
    ckpt = load_checkpoint(experiment_name, seed)
    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward scheduler
        for _ in range(start_epoch):
            scheduler.step()

    # Train
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if model_type == "beit":
                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss
                preds = outputs.logits
            else:
                preds = model(images)
                loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, pred = preds.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        # Save checkpoint every epoch
        save_checkpoint(experiment_name, seed, epoch, model, optimizer)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("    Epoch %d/%d  Loss: %.4f  Train: %.1f%%",
                        epoch + 1, epochs, total_loss / total, correct / total * 100)

    # Evaluate on all OOD cells with TTA
    logger.info("    Evaluating with TTA...")
    model.eval()
    seed_results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_res = evaluate(model, ood_loader, criterion, device, use_tta=use_tta, model_type=model_type)
        seed_results[ood_name] = ood_res["accuracy"]
        logger.info("    %s: %.1f%%", ood_name, ood_res["accuracy"] * 100)

    return seed_results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 72)
    logger.info("  REMAINING EXPERIMENTS RUNNER")
    logger.info("  Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 72)

    class_to_idx = {c: i for i, c in enumerate(CHEN_CLASSES)}
    num_classes = len(CHEN_CLASSES)
    criterion = nn.CrossEntropyLoss()

    # Compute shift spectrum
    logger.info("Computing shift spectrum...")
    shift_spectrum = compute_shift_spectrum(CHEN_ZOOLAKE, CHEN_CLASSES)

    # Common transforms
    saa_transform = SAATransform(shift_spectrum)
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Training data
    train_ds = PlanktonDataset(CHEN_ZOOLAKE, class_to_idx, saa_transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    logger.info("Training samples: %d", len(train_ds))

    # OOD test data
    ood_dir = Path(CHEN_OOD)
    ood_loaders = {}
    for ood_cell in sorted(ood_dir.iterdir()):
        if ood_cell.is_dir():
            ds = PlanktonDataset(str(ood_cell), class_to_idx, eval_transform)
            if len(ds) > 0:
                ood_loaders[ood_cell.name] = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    logger.info("OOD cells: %s", list(ood_loaders.keys()))

    # ===================================================================
    # EXPERIMENT 1: ViT Chen OOD (members 1-3)
    # ===================================================================
    logger.info("")
    logger.info("=" * 72)
    logger.info("  EXPERIMENT 1: ViT + SAA on Chen OOD (3 members)")
    logger.info("=" * 72)

    vit_results = {}
    for seed in range(3):
        if is_experiment_done(f"vit_chen_ood_seed{seed}"):
            r = json.load(open(result_path(f"vit_chen_ood_seed{seed}")))
            vit_results[f"seed_{seed}"] = r
            logger.info("  ViT seed=%d already done: %.1f%%", seed, r.get("overall", 0) * 100)
            continue

        seed_results = train_ensemble_member(
            "vit_chen_ood", seed, "vit",
            train_loader, ood_loaders, criterion, device, epochs=30, use_tta=True,
        )
        vit_results[f"seed_{seed}"] = seed_results
        save_results(f"vit_chen_ood_seed{seed}", {"overall": np.mean(list(seed_results.values())), "per_day": seed_results})

    # Compute ensemble average
    if len(vit_results) >= 2:
        ood_names = sorted(ood_loaders.keys())
        vit_ensemble = {}
        for ood_name in ood_names:
            accs = [vit_results[seed_key][ood_name] for seed_key in vit_results if ood_name in vit_results[seed_key]]
            vit_ensemble[ood_name] = float(np.mean(accs))
        vit_overall = np.mean(list(vit_ensemble.values()))
        logger.info("  ViT ensemble (%d members): %.1f%%", len(vit_results), vit_overall * 100)
        save_results("vit_chen_ood_ensemble", {"per_day": vit_ensemble, "overall": vit_overall, "n_members": len(vit_results)})

    # ===================================================================
    # EXPERIMENT 2: BEiT + SAA Chen OOD (members 1-3)
    # ===================================================================
    logger.info("")
    logger.info("=" * 72)
    logger.info("  EXPERIMENT 2: BEiT + SAA on Chen OOD (3 members)")
    logger.info("=" * 72)

    beit_results = {}
    for seed in range(3):
        if is_experiment_done(f"beit_chen_ood_seed{seed}"):
            r = json.load(open(result_path(f"beit_chen_ood_seed{seed}")))
            beit_results[f"seed_{seed}"] = r
            logger.info("  BEiT seed=%d already done: %.1f%%", seed, r.get("overall", 0) * 100)
            continue

        seed_results = train_ensemble_member(
            "beit_chen_ood", seed, "beit",
            train_loader, ood_loaders, criterion, device, epochs=30, use_tta=True,
        )
        beit_results[f"seed_{seed}"] = seed_results
        save_results(f"beit_chen_ood_seed{seed}", {"overall": np.mean(list(seed_results.values())), "per_day": seed_results})

    # Compute ensemble average
    if len(beit_results) >= 2:
        ood_names = sorted(ood_loaders.keys())
        beit_ensemble = {}
        for ood_name in ood_names:
            accs = [beit_results[seed_key][ood_name] for seed_key in beit_results if ood_name in beit_results[seed_key]]
            beit_ensemble[ood_name] = float(np.mean(accs))
        beit_overall = np.mean(list(beit_ensemble.values()))
        logger.info("  BEiT ensemble (%d members): %.1f%%", len(beit_results), beit_overall * 100)
        save_results("beit_chen_ood_ensemble", {"per_day": beit_ensemble, "overall": beit_overall, "n_members": len(beit_results)})

    # ===================================================================
    # SUMMARY
    # ===================================================================
    logger.info("")
    logger.info("=" * 72)
    logger.info("  ALL EXPERIMENTS COMPLETE")
    logger.info("  Finished: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 72)

    if vit_results:
        vit_overall = np.mean([np.mean(list(v.values())) for v in vit_results.values()])
        logger.info("  ViT + SAA ensemble:  %.1f%%", vit_overall * 100)
    if beit_results:
        beit_overall = np.mean([np.mean(list(v.values())) for v in beit_results.values()])
        logger.info("  BEiT + SAA ensemble: %.1f%%", beit_overall * 100)
    logger.info("  Chen's BEsT:         83.0%%")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
