"""
finetune_crops_chen.py — Fine-tune Chen's BEiT on segmented crops
with Chen's exact preprocessing + gentle SBA.

Hypothesis: Segmented crops remove background domain artifacts.
Combined with Chen's augmentation pipeline, this should beat 83%.
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
CROP_DIR = ROOT / "data_segmentation"
OOD_DIR = CROP_DIR / "ood"
RESULTS_DIR = ROOT / "results" / "finetune_crops_chen"
CLASSES_PATH = MODEL_DIR / "classes.npy"

CHEN_MODEL_FILES = [
    MODEL_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
]


def resize_with_proportions(im, desired_size=128):
    old_size = im.size
    if max(old_size) > desired_size:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (desired_size, desired_size), color=0)
    offset = ((desired_size - im.size[0]) // 2, (desired_size - im.size[1]) // 2)
    new_im.paste(im, offset)
    return new_im


class CropDataset(Dataset):
    """Load _crop.png images with Chen's preprocessing."""
    def __init__(self, data_dir, classes, augment=False, sba=None):
        self.classes = classes
        self.augment = augment
        self.sba = sba
        self.images, self.labels = [], []

        for cls_dir in sorted(Path(data_dir).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*_crop.png")):
                self.images.append(str(img_path))
                self.labels.append(cls_idx)

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
        print(f"  Loaded {len(self.images)} crop images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx]).convert("RGB")
        im = resize_with_proportions(im, desired_size=128)

        if self.augment:
            im = self.chen_aug(im)

        im = im.resize((224, 224), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0

        if self.sba and self.augment:
            gray = np.array(im.convert("L"), dtype=np.float64) / 255.0
            gray_aug = self.sba(gray)
            gray_uint8 = (gray_aug * 255).clip(0, 255).astype(np.uint8)
            im_aug = Image.fromarray(gray_uint8, mode="L").convert("RGB")
            arr = np.array(im_aug, dtype=np.float32) / 255.0

        return torch.from_numpy(arr).permute(2, 0, 1), self.labels[idx]


def load_model(ckpt_path, num_classes, device):
    model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device)


@torch.no_grad()
def predict_tta(model, im_pil, device, angles=[0, 90, 180, 270]):
    all_probs = []
    for angle in angles:
        im = im_pil.copy()
        if angle > 0:
            im = im.rotate(angle, expand=False)
        im = im.resize((224, 224), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs)


def evaluate(models, classes, device, angles=[0, 90, 180, 270]):
    per_day = {}
    for ood_name in sorted([d.name for d in OOD_DIR.iterdir() if d.is_dir()]):
        ood_path = OOD_DIR / ood_name
        images, labels = [], []
        for cls_dir in sorted(ood_path.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*_crop.png")):
                images.append(str(img_path))
                labels.append(cls_idx)

        labels = np.array(labels)
        preds = []
        for img_path in tqdm(images, desc=f"  {ood_name}", leave=False):
            im = Image.open(img_path).convert("RGB")
            im = resize_with_proportions(im, desired_size=128)
            model_probs = []
            for model in models:
                model.eval()
                tta_probs = predict_tta(model, im, device, angles)
                model_probs.append(np.mean(tta_probs, axis=0))
            preds.append(np.argmax(gmean(model_probs)))

        preds = np.array(preds)
        per_day[ood_name] = float((preds == labels).mean())

    macro = np.mean(list(per_day.values()))
    return macro, per_day


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--sba-strength", type=float, default=0.1)
    parser.add_argument("--sba-p", type=float, default=0.3)
    parser.add_argument("--no-sba", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)

    # Load shift spectrum
    shift_path = ROOT / "results" / "adverserial_net" / "fourier_analysis" / "cross_domain" / "fourier_analysis.json"
    shift_spectrum = None
    if shift_path.exists() and not args.no_sba:
        with open(shift_path) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "ZooScan" in key and "WHOI" in key:
                shift_spectrum = np.array(val.get("diff", []))
                break

    sba = None
    if not args.no_sba and shift_spectrum is not None:
        sba = SpectralAugmentation(shift_spectrum=shift_spectrum,
                                    strength=args.sba_strength,
                                    strategies=["spectral_noise", "band_adversarial"],
                                    p=args.sba_p)

    finetuned_paths = []

    if not args.eval_only:
        print(f"\nFine-tuning {len(CHEN_MODEL_FILES)} models on segmented crops...")
        print(f"  SBA: {'ON' if sba else 'OFF'}, epochs={args.epochs}, lr={args.lr}")

        train_dataset = CropDataset(CROP_DIR / "zoolake2", classes, augment=True, sba=sba)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                   num_workers=4, pin_memory=True)

        for i, ckpt_path in enumerate(CHEN_MODEL_FILES):
            print(f"\n  Model {i+1}/3")
            model = load_model(ckpt_path, num_classes, device)
            model = finetune(model, train_loader, device, args.epochs, args.lr)
            out_path = RESULTS_DIR / f"model_{i+1:02d}_crops.pth"
            torch.save(model.state_dict(), str(out_path))
            finetuned_paths.append(out_path)
            print(f"    Saved: {out_path}")
    else:
        for i in range(3):
            p = RESULTS_DIR / f"model_{i+1:02d}_crops.pth"
            if p.exists():
                finetuned_paths.append(p)
            else:
                finetuned_paths.append(CHEN_MODEL_FILES[i])

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating on OOD crops with 4-TTA + geometric ensemble...")

    models = []
    for i, path in enumerate(finetuned_paths):
        if path.exists() and "crops" in str(path):
            m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                                   pretrained=False, num_classes=num_classes)
            m.load_state_dict(torch.load(str(path), map_location=device, weights_only=False), strict=False)
        else:
            m = load_model(CHEN_MODEL_FILES[i], num_classes, device)
        m.to(device).eval()
        models.append(m)

    macro, per_day = evaluate(models, classes, device)
    delta = (macro - 0.8305) * 100

    print(f"\n{'='*60}")
    print(f"Macro-OOD: {macro:.4f}  vs Chen 83.05%: {delta:+.2f}%")
    print(f"Per-day: { {k: round(v, 3) for k, v in sorted(per_day.items())} }")
    print(f"{'='*60}")

    output = {"macro": float(macro), "vs_chen": float(delta), "per_day": per_day,
              "config": {"sba": not args.no_sba, "sba_strength": args.sba_strength,
                         "sba_p": args.sba_p, "epochs": args.epochs, "lr": args.lr,
                         "preprocessing": "Chen_ResizeWithProportions(128)",
                         "input": "segmented_crops", "tta": "4_rotation",
                         "ensemble": "geometric_3_models"}}
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
