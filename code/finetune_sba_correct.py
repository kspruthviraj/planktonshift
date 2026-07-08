"""
finetune_sba_correct.py
=======================
Fine-tune Chen's BEiT models with SBA + targeted augmentations
using Chen's EXACT preprocessing pipeline.

Goal: Beat 83% OOD accuracy.

Strategy:
  - Chen's ResizeWithProportions(128) → Resize(224) preprocessing
  - SBA band adversarial augmentation (frequency-domain)
  - Chen's targeted augmentations (spatial/color)
  - Low LR fine-tuning (1e-5) for 10 epochs
  - Geometric ensemble + 4-rotation TTA at evaluation
"""

import sys, os, json, time, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from scipy.stats import gmean

sys.path.insert(0, str(Path(__file__).resolve().parent / "adverserial_net"))
from spectral_augmentation import SpectralAugmentation

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
ZOOlAKE_DIR = ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
RESULTS_DIR = ROOT / "results" / "finetune_sba_correct"
CLASSES_PATH = MODEL_DIR / "classes.npy"

CHEN_MODEL_FILES = [
    MODEL_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
]


# ══════════════════════════════════════════════
# Chen's EXACT preprocessing
# ══════════════════════════════════════════════
def resize_with_proportions(im, desired_size=128):
    """Chen's ResizeWithProportions: shrink to fit, then black-pad to square."""
    old_size = im.size
    if max(old_size) > desired_size:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (desired_size, desired_size), color=0)
    offset = ((desired_size - im.size[0]) // 2, (desired_size - im.size[1]) // 2)
    new_im.paste(im, offset)
    return new_im


# ══════════════════════════════════════════════
# Dataset with SBA + targeted augmentations
# ══════════════════════════════════════════════
class ZooLakeDataset(Dataset):
    """ZooLake dataset with Chen's preprocessing + SBA + targeted augmentations."""

    def __init__(self, data_dir, classes, sba=None, augment=False, max_per_class=None):
        self.classes = classes
        self.sba = sba
        self.augment = augment
        self.images = []
        self.labels = []

        data_path = Path(data_dir)
        for cls_dir in sorted(data_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            if cls_name not in classes:
                continue
            cls_idx = np.where(classes == cls_name)[0][0]
            count = 0
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)
                    count += 1
                    if max_per_class and count >= max_per_class:
                        break

        # Chen's targeted augmentations (spatial/color)
        self.targeted_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomPerspective(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        ])

        print(f"  Loaded {len(self.images)} images from {data_path.name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        im = Image.open(img_path).convert("RGB")

        # Chen's preprocessing: ResizeWithProportions(128)
        im = resize_with_proportions(im, desired_size=128)

        # Apply targeted augmentations (before SBA, on PIL image)
        if self.augment:
            im = self.targeted_aug(im)

        # Resize to 224 (Chen's pipeline)
        im = im.resize((224, 224), Image.BILINEAR)

        # Convert to numpy [0,1] float
        arr = np.array(im, dtype=np.float32) / 255.0

        # Apply SBA on grayscale (per SAATransform convention)
        if self.sba and self.augment:
            gray = np.array(im.convert("L"), dtype=np.float64) / 255.0
            gray_aug = self.sba(gray)
            gray_uint8 = (gray_aug * 255).clip(0, 255).astype(np.uint8)
            im_aug = Image.fromarray(gray_uint8, mode="L").convert("RGB")
            arr = np.array(im_aug, dtype=np.float32) / 255.0

        # To tensor CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


# ══════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════
def load_beit(num_classes, device):
    model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
    return model


def load_chen_model(ckpt_path, num_classes, device):
    model = load_beit(num_classes, device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device)


# ══════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════
def finetune_model(model, train_loader, device, epochs=10, lr=1e-5, wd=0.03):
    """Fine-tune with low LR, no class weights (matching Chen's eval)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"    Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        acc = correct / total
        avg_loss = total_loss / total
        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f} acc={acc:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    return model


# ══════════════════════════════════════════════
# Evaluation with TTA
# ══════════════════════════════════════════════
@torch.no_grad()
def predict_with_tta(model, im_pil, device, angles=[0, 90, 180, 270]):
    """Predict with 4-rotation TTA using Chen's preprocessing."""
    all_probs = []
    for angle in angles:
        im = im_pil.copy()
        if angle > 0:
            im = im.rotate(angle, expand=False)
        im = im.resize((224, 224), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs)


def evaluate_ood(ood_path, models, classes, device, use_tta=True):
    """Evaluate one OOD cell."""
    images, labels = [], []
    for cls_dir in sorted(Path(ood_path).iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in classes:
            continue
        cls_idx = np.where(classes == cls_name)[0][0]
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                images.append(str(img_path))
                labels.append(cls_idx)

    labels = np.array(labels)
    all_ensemble_probs = []

    for img_path in tqdm(images, desc=f"  {Path(ood_path).name}", leave=False):
        im = Image.open(img_path).convert("RGB")
        im = resize_with_proportions(im, desired_size=128)

        model_probs = []
        for model in models:
            model.eval()
            if use_tta:
                tta_probs = predict_with_tta(model, im, device)
                model_probs.append(np.mean(tta_probs, axis=0))
            else:
                im_resized = im.resize((224, 224), Image.BILINEAR)
                arr = np.array(im_resized, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                model_probs.append(probs)

        ensemble_probs = gmean(model_probs)
        all_ensemble_probs.append(ensemble_probs)

    all_ensemble_probs = np.array(all_ensemble_probs)
    preds = all_ensemble_probs.argmax(axis=1)
    accuracy = (preds == labels).mean()

    return {"accuracy": float(accuracy), "correct": int((preds == labels).sum()),
            "total": int(len(labels)), "n_images": len(images)}


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sba-strength", type=float, default=0.5)
    parser.add_argument("--sba-p", type=float, default=0.8)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    print(f"Classes: {num_classes} — {list(classes)}")

    # Load shift spectrum for SBA
    shift_path = ROOT / "results" / "adverserial_net" / "fourier_analysis" / "cross_domain" / "fourier_analysis.json"
    shift_spectrum = None
    if shift_path.exists():
        with open(shift_path) as f:
            fa = json.load(f)
        if "shift_spectra" in fa:
            for key, val in fa["shift_spectra"].items():
                if "ZooScan" in key and "WHOI" in key:
                    shift_spectrum = np.array(val.get("diff", []))
                    break
        print(f"  Shift spectrum loaded: {len(shift_spectrum)} bins" if shift_spectrum is not None else "  No shift spectrum found")

    # SBA augmentation
    sba = SpectralAugmentation(
        shift_spectrum=shift_spectrum,
        strength=args.sba_strength,
        strategies=["spectral_noise", "band_adversarial"],
        p=args.sba_p,
    )

    finetuned_paths = []

    if not args.eval_only:
        # ── Fine-tune each model ──
        print(f"\nFine-tuning {len(CHEN_MODEL_FILES)} models with SBA + targeted aug...")
        print(f"  SBA: strength={args.sba_strength}, p={args.sba_p}")
        print(f"  LR: {args.lr}, Epochs: {args.epochs}")

        train_dataset = ZooLakeDataset(
            ZOOlAKE_DIR, classes, sba=sba, augment=True
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                   num_workers=4, pin_memory=True)

        for i, ckpt_path in enumerate(CHEN_MODEL_FILES):
            print(f"\n  Model {i+1}/3: {ckpt_path.parent.name}")
            model = load_chen_model(ckpt_path, num_classes, device)
            model = finetune_model(model, train_loader, device,
                                    epochs=args.epochs, lr=args.lr)

            out_path = RESULTS_DIR / f"model_{i+1:02d}_sba_finetuned.pth"
            torch.save(model.state_dict(), str(out_path))
            finetuned_paths.append(out_path)
            print(f"    Saved: {out_path}")
    else:
        # Load existing finetuned models
        for i in range(len(CHEN_MODEL_FILES)):
            out_path = RESULTS_DIR / f"model_{i+1:02d}_sba_finetuned.pth"
            if out_path.exists():
                finetuned_paths.append(out_path)
            else:
                print(f"  [WARN] {out_path} not found, using original")
                finetuned_paths.append(None)

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print("Evaluating on all 10 OOD cells...")
    print(f"  TTA: {'Yes (4 rotations)' if not args.no_tta else 'No'}")
    print(f"  Ensemble: Geometric mean of 3 models")
    print(f"{'='*60}")

    # Load finetuned models
    models = []
    for i, path in enumerate(finetuned_paths):
        if path and path.exists():
            model = load_beit(num_classes, device)
            model.load_state_dict(torch.load(str(path), map_location=device, weights_only=False))
            model.to(device).eval()
            models.append(model)
            print(f"  Loaded finetuned model {i+1}")
        else:
            model = load_chen_model(CHEN_MODEL_FILES[i], num_classes, device)
            model.eval()
            models.append(model)
            print(f"  Loaded original Chen model {i+1}")

    # Evaluate each OOD cell
    ood_names = sorted([d.name for d in OOD_DIR.iterdir() if d.is_dir()])
    results = {}

    for ood_name in ood_names:
        ood_path = OOD_DIR / ood_name
        if not ood_path.exists():
            continue
        res = evaluate_ood(ood_path, models, classes, device, use_tta=not args.no_tta)
        results[ood_name] = res
        print(f"  {ood_name}: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")

    # Overall
    all_correct = sum(r["correct"] for r in results.values())
    all_total = sum(r["total"] for r in results.values())
    micro = all_correct / all_total
    macro = np.mean([r["accuracy"] for r in results.values()])

    print(f"\n{'='*60}")
    print(f"Micro-OOD: {micro:.4f} ({all_correct}/{all_total})")
    print(f"Macro-OOD: {macro:.4f}")
    print(f"Per-day: {[f'{results[k]['accuracy']:.3f}' for k in sorted(results.keys())]}")
    print(f"Chen's BEsT: 83.05%")
    print(f"Delta: {(macro - 0.8305)*100:+.2f}%")
    print(f"{'='*60}")

    # Save
    output = {
        "overall_micro": float(micro),
        "overall_macro": float(macro),
        "vs_chen": float(macro - 0.8305),
        "per_day": {k: v["accuracy"] for k, v in results.items()},
        "config": {
            "sba_strength": args.sba_strength,
            "sba_p": args.sba_p,
            "lr": args.lr,
            "epochs": args.epochs,
            "tta": not args.no_tta,
            "preprocessing": "Chen_ResizeWithProportions(128)->Resize(224)",
            "augmentation": "SBA + Chen_targeted",
            "ensemble": "geometric_mean_3_models",
        }
    }
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
