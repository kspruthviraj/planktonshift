"""
centrecrop_sba_finetune.py — Fine-tune Chen's BEiT with CenterCrop + per-channel SBA + TTA.
Tests whether matching the wrong preprocessing (CenterCrop) during fine-tuning helps.
"""

import sys, json, argparse
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

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
ZOOlAKE_DIR = ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
RESULTS_DIR = ROOT / "results" / "centrecrop_sba"
CLASSES_PATH = MODEL_DIR / "classes.npy"

CHEN_MODEL_FILES = [
    MODEL_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
]


def apply_sba_per_channel(arr_rgb, sba):
    result = np.zeros_like(arr_rgb)
    for c in range(3):
        channel = arr_rgb[:, :, c].astype(np.float64)
        channel_aug = sba(channel)
        result[:, :, c] = channel_aug
    return result.clip(0, 1).astype(np.float32)


class CenterCropSBADataset(Dataset):
    def __init__(self, data_dir, classes, sba=None, augment=False):
        self.classes = classes
        self.sba = sba
        self.augment = augment
        self.images, self.labels = [], []

        for cls_dir in sorted(Path(data_dir).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)

        # Chen's targeted augmentations
        self.chen_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomPerspective(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        ])
        print(f"  Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx]).convert("RGB")

        # CenterCrop preprocessing
        im = transforms.Resize(224)(im)
        im = transforms.CenterCrop(224)(im)

        if self.augment:
            im = self.chen_aug(im)

        arr = np.array(im, dtype=np.float32) / 255.0

        # Per-channel SBA
        if self.sba and self.augment:
            arr = apply_sba_per_channel(arr, self.sba)

        return torch.from_numpy(arr).permute(2, 0, 1), self.labels[idx]


def load_chen_model(ckpt_path, num_classes, device):
    model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device)


@torch.no_grad()
def predict_tta_centrecrop(models, im_pil, device, angles=[0, 90, 180, 270]):
    all_model_probs = []
    for model in models:
        tta_probs = []
        for angle in angles:
            im = im_pil.copy()
            if angle > 0:
                im = im.rotate(angle, expand=False)
            # CenterCrop preprocessing
            im = transforms.Resize(224)(im)
            im = transforms.CenterCrop(224)(im)
            arr = np.array(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            tta_probs.append(probs)
        all_model_probs.append(np.mean(tta_probs, axis=0))
    return gmean(all_model_probs)


def evaluate(models, classes, device, angles=[0, 90, 180, 270]):
    per_day = {}
    for ood_name in sorted([d.name for d in OOD_DIR.iterdir() if d.is_dir()]):
        ood_path = OOD_DIR / ood_name
        images, labels = [], []
        for cls_dir in sorted(ood_path.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    images.append(str(img_path))
                    labels.append(cls_idx)
        labels = np.array(labels)
        correct = 0
        for i, img_path in enumerate(tqdm(images, desc=f"  {ood_name}", leave=False)):
            im = Image.open(img_path).convert("RGB")
            probs = predict_tta_centrecrop(models, im, device, angles)
            if np.argmax(probs) == labels[i]:
                correct += 1
        acc = correct / len(labels)
        per_day[ood_name] = float(acc)
        print(f"  {ood_name}: {acc:.4f}")
    return np.mean(list(per_day.values())), per_day


def finetune(model, loader, device, epochs, lr):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(loader, desc=f"    Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        print(f"    Epoch {epoch+1}: loss={total_loss/total:.4f} acc={correct/total:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sba-strength", type=float, default=0.5)
    parser.add_argument("--sba-p", type=float, default=0.8)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)

    # Load shift spectrum
    shift_path = ROOT / "results" / "tier1_corrected" / "fourier_analysis.json"
    shift_spectrum = None
    if shift_path.exists():
        with open(shift_path) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "WHOI" in key and "ZooScan" in key:
                shift_spectrum = np.array(val.get("diff", []))
                break
    if shift_spectrum is None:
        shift_path = ROOT / "results" / "adverserial_net" / "fourier_analysis_zoolake2" / "fourier_analysis.json"
        if shift_path.exists():
            with open(shift_path) as f:
                fa = json.load(f)
            for key, val in fa.get("shift_spectra", {}).items():
                if "WHOI" in key and "ZooScan" in key:
                    shift_spectrum = np.array(val.get("diff", []))
                    break

    sba = SpectralAugmentation(
        shift_spectrum=shift_spectrum,
        strength=args.sba_strength,
        strategies=["spectral_noise", "band_adversarial"],
        p=args.sba_p,
    ) if shift_spectrum is not None else None

    finetuned_paths = []

    if not args.eval_only:
        print(f"\nFine-tuning with CenterCrop + per-channel SBA...")
        train_dataset = CenterCropSBADataset(ZOOlAKE_DIR, classes, sba=sba, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

        for i, ckpt_path in enumerate(CHEN_MODEL_FILES):
            print(f"\n  Model {i+1}/3")
            model = load_chen_model(ckpt_path, num_classes, device)
            model = finetune(model, train_loader, device, args.epochs, args.lr)
            out_path = RESULTS_DIR / f"model_{i+1:02d}_centrecrop_sba.pth"
            torch.save(model.state_dict(), str(out_path))
            finetuned_paths.append(out_path)
            print(f"    Saved: {out_path}")
    else:
        for i in range(3):
            p = RESULTS_DIR / f"model_{i+1:02d}_centrecrop_sba.pth"
            finetuned_paths.append(p if p.exists() else CHEN_MODEL_FILES[i])

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating with CenterCrop + TTA...")
    models = []
    for i, path in enumerate(finetuned_paths):
        if path.exists() and "centrecrop" in str(path):
            m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=False, num_classes=num_classes)
            m.load_state_dict(torch.load(str(path), map_location=device, weights_only=False), strict=False)
        else:
            m = load_chen_model(CHEN_MODEL_FILES[i], num_classes, device)
        m.to(device).eval()
        models.append(m)

    macro, per_day = evaluate(models, classes, device)
    delta = (macro - 0.8305) * 100

    print(f"\n{'='*60}")
    print(f"CenterCrop + Per-channel SBA + TTA")
    print(f"{'='*60}")
    print(f"Macro-OOD: {macro:.4f}")
    print(f"vs Chen 83.05%: {delta:+.2f}%")
    print(f"Per-day: { {k: round(v, 3) for k, v in sorted(per_day.items())} }")
    if macro > 0.8305:
        print(f"\n*** BEATS CHEN by {delta:.2f}%! ***")
    print(f"{'='*60}")

    output = {
        "macro": float(macro), "vs_chen": float(delta), "per_day": per_day,
        "config": {"preprocessing": "CenterCrop(224)", "sba": "per_channel_RGB",
                   "tta": "4 rotations", "ensemble": "geometric_3_models"}
    }
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
