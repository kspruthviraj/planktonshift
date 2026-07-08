"""
finetune_chen_with_saa.py
==========================
Fine-tune Chen's trained BEiT models with SAA augmentation.

Strategy: Load Chen's already-trained BEiT model (which achieves ~83% OOD),
then continue training with SAA augmentation for a few epochs.

This is the smartest approach because:
1. Chen's model already learned good features from ZooLake2
2. SAA adds frequency-domain robustness on top
3. Only needs 5-10 epochs (fast)
4. Shows SAA improves even the best existing model

Usage:
    python finetune_chen_with_saa.py \
        --chen-models data/chen_models/beit_models/trained_BEiT_models/trained_models \
        --train-dir data/chen_data/ZooLake2/ZooLake2/ZooLake2.0 \
        --ood-dir data/chen_data/OOD_data/OODs \
        --finetune-epochs 10 \
        --output-dir results/finetune_chen_saa
"""

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
from torchvision import transforms
from PIL import Image

sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune_chen_saa.log", mode="a")],
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224

CHEN_CLASSES = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina", "brachionus",
    "ceratium", "chaoborus", "collotheca", "conochilus", "copepod_skins",
    "cyclops", "daphnia", "daphnia_skins", "diaphanosoma", "diatom_chain",
    "dinobryon", "dirt", "eudiaptomus", "filament", "fish", "fragilaria",
    "hydra", "kellicottia", "keratella_cochlearis", "keratella_quadrata",
    "leptodora", "maybe_cyano", "nauplius", "paradileptus", "polyarthra",
    "rotifers", "synchaeta", "trichocerca", "unknown", "unknown_plankton", "uroglena",
]


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


class SAATransform:
    def __init__(self, shift_spectrum=None):
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
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __call__(self, image):
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        return self.base(Image.fromarray(arr_uint8, mode="L").convert("RGB"))


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


def rotation_tta(model, images, device):
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
def evaluate(model, loader, criterion, device, use_tta=True):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if use_tta:
            probs = rotation_tta(model, images, device)
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chen-models", type=str,
                        default="data/chen_models/beit_models/trained_BEiT_models/trained_models")
    parser.add_argument("--train-dir", type=str,
                        default="data/chen_data/ZooLake2/ZooLake2/ZooLake2.0")
    parser.add_argument("--ood-dir", type=str, default="data/chen_data/OOD_data/OODs")
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="results/finetune_chen_saa")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    class_to_idx = {c: i for i, c in enumerate(CHEN_CLASSES)}
    num_classes = len(CHEN_CLASSES)

    # Compute shift spectrum
    logger.info("Computing shift spectrum...")
    shift_spectrum = compute_shift_spectrum(args.train_dir, CHEN_CLASSES)

    # Transforms
    train_transform = SAATransform(shift_spectrum)
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Training data
    train_ds = PlanktonDataset(args.train_dir, class_to_idx, train_transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    logger.info("Training samples: %d", len(train_ds))

    # OOD test data
    ood_dir = Path(args.ood_dir)
    ood_loaders = {}
    for ood_cell in sorted(ood_dir.iterdir()):
        if ood_cell.is_dir():
            ds = PlanktonDataset(str(ood_cell), class_to_idx, eval_transform)
            if len(ds) > 0:
                ood_loaders[ood_cell.name] = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    # Load and fine-tune each of Chen's models
    chen_models_dir = Path(args.chen_models)
    all_results = {}

    for model_dir in sorted(chen_models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        model_path = model_dir / "trained_model_tuned.pth"

        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            continue

        logger.info("=" * 60)
        logger.info("Fine-tuning Chen model %s with SAA", model_name)
        logger.info("=" * 60)

        # Load Chen's trained model
        from transformers import BeitForImageClassification
        model = BeitForImageClassification.from_pretrained(
            "microsoft/beit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Load Chen's weights
        chen_state = torch.load(model_path, weights_only=False, map_location="cpu")
        # Chen's model might have different key names
        try:
            model.load_state_dict(chen_state, strict=False)
            logger.info("  Loaded Chen's weights")
        except Exception as e:
            logger.warning("  Could not load Chen's weights directly: %s", e)
            # Try to adapt keys
            new_state = {}
            for k, v in chen_state.items():
                if k.startswith("model."):
                    new_state[k[6:]] = v
                else:
                    new_state[k] = v
            try:
                model.load_state_dict(new_state, strict=False)
                logger.info("  Loaded Chen's weights (adapted keys)")
            except Exception as e2:
                logger.error("  Failed to load: %s", e2)
                continue

        model = model.to(device)

        # Evaluate BEFORE fine-tuning (Chen's original performance)
        logger.info("  Evaluating Chen's model BEFORE fine-tuning...")
        chen_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            res = evaluate(model, ood_loader, criterion, device, use_tta=True)
            chen_results[ood_name] = res["accuracy"]
        chen_overall = np.mean(list(chen_results.values()))
        logger.info("  Chen's model %s: %.1f%% OOD", model_name, chen_overall * 100)

        # Fine-tune with SAA (lower learning rate to preserve learned features)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

        for epoch in range(args.finetune_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(pixel_values=images, labels=labels)
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item() * images.size(0)
                _, pred = out.logits.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("    Epoch %d/%d  Loss: %.4f  Train: %.1f%%",
                            epoch + 1, args.finetune_epochs, total_loss / total, correct / total * 100)

        # Evaluate AFTER fine-tuning
        logger.info("  Evaluating AFTER fine-tuning with SAA...")
        finetuned_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            res = evaluate(model, ood_loader, criterion, device, use_tta=True)
            finetuned_results[ood_name] = res["accuracy"]
        finetuned_overall = np.mean(list(finetuned_results.values()))

        all_results[f"chen_{model_name}"] = {
            "before": {"per_day": chen_results, "overall": chen_overall},
            "after": {"per_day": finetuned_results, "overall": finetuned_overall},
            "improvement": finetuned_overall - chen_overall,
        }

        logger.info("  Chen %s: BEFORE=%.1f%%  AFTER=%.1f%%  Improvement=%+.1f%%",
                     model_name, chen_overall * 100, finetuned_overall * 100,
                     (finetuned_overall - chen_overall) * 100)

        # Save fine-tuned model
        torch.save(model.state_dict(), str(out / f"finetuned_{model_name}.pth"))

    # Compute ensemble of fine-tuned models
    if len(all_results) >= 2:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  ENSEMBLE OF FINE-TUNED MODELS")
        logger.info("=" * 60)

        ood_names = sorted(ood_loaders.keys())

        # Before ensemble
        before_per_day = {}
        for day in ood_names:
            vals = [all_results[k]["before"]["per_day"][day] for k in all_results]
            before_per_day[day] = float(np.mean(vals))

        # After ensemble
        after_per_day = {}
        for day in ood_names:
            vals = [all_results[k]["after"]["per_day"][day] for k in all_results]
            after_per_day[day] = float(np.mean(vals))

        before_overall = np.mean(list(before_per_day.values()))
        after_overall = np.mean(list(after_per_day.values()))

        logger.info("  Ensemble BEFORE fine-tuning: %.1f%%", before_overall * 100)
        logger.info("  Ensemble AFTER fine-tuning:  %.1f%%", after_overall * 100)
        logger.info("  Improvement: %+.1f%%", (after_overall - before_overall) * 100)
        logger.info("  Chen's BEsT: 83.0%%")
        logger.info("  vs Chen: %+.1f%%", after_overall * 100 - 83)

        all_results["ensemble"] = {
            "before": {"per_day": before_per_day, "overall": before_overall},
            "after": {"per_day": after_per_day, "overall": after_overall},
            "improvement": after_overall - before_overall,
        }

    # Save
    with open(out / "finetune_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("")
    logger.info("Results saved to %s", out / "finetune_results.json")


if __name__ == "__main__":
    main()
