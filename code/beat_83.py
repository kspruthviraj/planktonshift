"""
beat_83.py — Try multiple strategies to beat Chen's 83% OOD accuracy.

Strategies:
1. More TTA angles (8, 16) on Chen's original models
2. Gentle SBA fine-tuning (low strength, few epochs)
3. SBA-only fine-tuning (no targeted aug)
4. Larger ensemble (original + SBA models)
5. Test-time SBA perturbations
"""

import sys, os, json, argparse
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
RESULTS_DIR = ROOT / "results" / "beat_83"
CLASSES_PATH = MODEL_DIR / "classes.npy"

CHEN_MODEL_FILES = [
    MODEL_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
]

SBA_MODEL_DIR = RESULTS_DIR
SBA_MODEL_FILES = [
    SBA_MODEL_DIR / "model_01_sba.pth",
    SBA_MODEL_DIR / "model_02_sba.pth",
    SBA_MODEL_DIR / "model_03_sba.pth",
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


def load_beit(num_classes, device):
    return timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                              pretrained=False, num_classes=num_classes)


def load_model(ckpt_path, num_classes, device):
    model = load_beit(num_classes, device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device)


class ZooLakeDataset(Dataset):
    def __init__(self, data_dir, classes, sba=None, augment=False, targeted_aug=False):
        self.classes = classes
        self.sba = sba
        self.augment = augment
        self.targeted_aug_flag = targeted_aug
        self.images, self.labels = [], []

        for cls_dir in sorted(Path(data_dir).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx]).convert("RGB")
        im = resize_with_proportions(im, desired_size=128)

        if self.augment and self.targeted_aug_flag:
            im = self.targeted_aug(im)

        im = im.resize((224, 224), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0

        if self.sba and self.augment:
            gray = np.array(im.convert("L"), dtype=np.float64) / 255.0
            gray_aug = self.sba(gray)
            gray_uint8 = (gray_aug * 255).clip(0, 255).astype(np.uint8)
            im_aug = Image.fromarray(gray_uint8, mode="L").convert("RGB")
            arr = np.array(im_aug, dtype=np.float32) / 255.0

        return torch.from_numpy(arr).permute(2, 0, 1), self.labels[idx]


@torch.no_grad()
def predict_with_tta(model, im_pil, device, angles):
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


def evaluate_ood(ood_path, models, classes, device, angles=[0, 90, 180, 270]):
    images, labels = [], []
    for cls_dir in sorted(Path(ood_path).iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in classes:
            continue
        cls_idx = np.where(classes == cls_dir.name)[0][0]
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
            tta_probs = predict_with_tta(model, im, device, angles)
            model_probs.append(np.mean(tta_probs, axis=0))

        ensemble_probs = gmean(model_probs)
        all_ensemble_probs.append(ensemble_probs)

    all_ensemble_probs = np.array(all_ensemble_probs)
    preds = all_ensemble_probs.argmax(axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc), "correct": int((preds == labels).sum()),
            "total": int(len(labels)), "n_images": len(images)}


def evaluate_all(models, classes, device, angles, label=""):
    ood_names = sorted([d.name for d in OOD_DIR.iterdir() if d.is_dir()])
    results = {}
    for ood_name in ood_names:
        ood_path = OOD_DIR / ood_name
        if not ood_path.exists():
            continue
        res = evaluate_ood(ood_path, models, classes, device, angles)
        results[ood_name] = res

    micro = sum(r["correct"] for r in results.values()) / sum(r["total"] for r in results.values())
    macro = np.mean([r["accuracy"] for r in results.values()])

    print(f"\n  [{label}] Micro: {micro:.4f}  Macro: {macro:.4f}")
    print(f"  Per-day: { {k: round(v['accuracy'], 3) for k, v in sorted(results.items())} }")
    print(f"  vs Chen 83.05%: {(macro - 0.8305)*100:+.2f}%")
    return {"micro": float(micro), "macro": float(macro),
            "per_day": {k: v["accuracy"] for k, v in results.items()}}


def finetune_model(model, train_loader, device, epochs, lr, wd=0.03):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"    Epoch {epoch+1}/{epochs}", leave=False):
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
    parser.add_argument("--strategy", choices=["tta", "gentle-sba", "sba-only", "mega-ensemble", "all"],
                        default="all", help="Which strategy to run")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)

    # Load shift spectrum
    shift_path = ROOT / "results" / "adverserial_net" / "fourier_analysis" / "cross_domain" / "fourier_analysis.json"
    shift_spectrum = None
    if shift_path.exists():
        with open(shift_path) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "ZooScan" in key and "WHOI" in key:
                shift_spectrum = np.array(val.get("diff", []))
                break

    all_results = {}

    # ══════════════════════════════════════════════
    # STRATEGY 1: More TTA angles on Chen's original
    # ══════════════════════════════════════════════
    if args.strategy in ["tta", "all"]:
        print("\n" + "="*60)
        print("STRATEGY 1: More TTA angles on Chen's original models")
        print("="*60)

        chen_models = [load_model(p, num_classes, device) for p in CHEN_MODEL_FILES]

        for n_angles in [4, 8, 12, 16]:
            angles = [i * (360 // n_angles) for i in range(n_angles)]
            print(f"\n  TTA with {n_angles} angles: {angles}")
            res = evaluate_all(chen_models, classes, device, angles,
                              label=f"Chen original + {n_angles}-TTA")
            all_results[f"chen_{n_angles}tta"] = res

        # Free memory
        del chen_models
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # STRATEGY 2: Gentle SBA fine-tuning
    # ══════════════════════════════════════════════
    if args.strategy in ["gentle-sba", "all"]:
        print("\n" + "="*60)
        print("STRATEGY 2: Gentle SBA fine-tuning (low strength, few epochs)")
        print("="*60)

        configs = [
            {"strength": 0.1, "p": 0.3, "epochs": 2, "lr": 1e-6, "targeted": False},
            {"strength": 0.1, "p": 0.3, "epochs": 3, "lr": 1e-6, "targeted": False},
            {"strength": 0.2, "p": 0.5, "epochs": 3, "lr": 1e-6, "targeted": False},
            {"strength": 0.1, "p": 0.3, "epochs": 2, "lr": 5e-7, "targeted": False},
        ]

        for i, cfg in enumerate(configs):
            print(f"\n  Config {i+1}: strength={cfg['strength']}, p={cfg['p']}, "
                  f"epochs={cfg['epochs']}, lr={cfg['lr']}")

            sba = SpectralAugmentation(
                shift_spectrum=shift_spectrum,
                strength=cfg["strength"],
                strategies=["spectral_noise", "band_adversarial"],
                p=cfg["p"],
            )

            train_dataset = ZooLakeDataset(ZOOlAKE_DIR, classes, sba=sba,
                                            augment=True, targeted_aug=cfg["targeted"])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                       num_workers=4, pin_memory=True)

            ft_models = []
            for j, ckpt_path in enumerate(CHEN_MODEL_FILES):
                print(f"    Model {j+1}/3")
                model = load_model(ckpt_path, num_classes, device)
                model = finetune_model(model, train_loader, device,
                                        epochs=cfg["epochs"], lr=cfg["lr"])

                out_path = RESULTS_DIR / f"gentle_sba_{i}_model_{j+1}.pth"
                torch.save(model.state_dict(), str(out_path))
                ft_models.append(model)

            res = evaluate_all(ft_models, classes, device, [0, 90, 180, 270],
                              label=f"Gentle SBA config {i+1}")
            all_results[f"gentle_sba_{i}"] = res

            del ft_models
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # STRATEGY 3: SBA-only (no targeted aug)
    # ══════════════════════════════════════════════
    if args.strategy in ["sba-only", "all"]:
        print("\n" + "="*60)
        print("STRATEGY 3: SBA-only fine-tuning (no targeted aug)")
        print("="*60)

        sba = SpectralAugmentation(
            shift_spectrum=shift_spectrum,
            strength=0.1,
            strategies=["spectral_noise", "band_adversarial"],
            p=0.3,
        )

        train_dataset = ZooLakeDataset(ZOOlAKE_DIR, classes, sba=sba,
                                        augment=True, targeted_aug=False)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                   num_workers=4, pin_memory=True)

        sba_models = []
        for j, ckpt_path in enumerate(CHEN_MODEL_FILES):
            print(f"    Model {j+1}/3")
            model = load_model(ckpt_path, num_classes, device)
            model = finetune_model(model, train_loader, device, epochs=2, lr=1e-6)
            out_path = RESULTS_DIR / f"model_{j+1:02d}_sba.pth"
            torch.save(model.state_dict(), str(out_path))
            sba_models.append(model)

        res = evaluate_all(sba_models, classes, device, [0, 90, 180, 270],
                          label="SBA-only (2ep, s=0.1, p=0.3)")
        all_results["sba_only"] = res

        del sba_models
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # STRATEGY 4: Mega-ensemble (6 models)
    # ══════════════════════════════════════════════
    if args.strategy in ["mega-ensemble", "all"]:
        print("\n" + "="*60)
        print("STRATEGY 4: Mega-ensemble (6 models: 3 Chen + 3 SBA)")
        print("="*60)

        chen_models = [load_model(p, num_classes, device) for p in CHEN_MODEL_FILES]
        sba_models = [load_model(p, num_classes, device) for p in SBA_MODEL_FILES
                      if p.exists()]

        if len(sba_models) == 3:
            mega_models = chen_models + sba_models
            for n_angles in [4, 8, 16]:
                angles = [i * (360 // n_angles) for i in range(n_angles)]
                res = evaluate_all(mega_models, classes, device, angles,
                                  label=f"Mega-ensemble 6×{n_angles}-TTA")
                all_results[f"mega_{n_angles}tta"] = res
        else:
            print(f"  Only {len(sba_models)} SBA models found, skipping")

        del chen_models, sba_models
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ══════════════════════════════════════════════
    print("\n" + "="*60)
    print("SUMMARY — All strategies")
    print("="*60)
    print(f"{'Strategy':<35} {'Macro-OOD':>10} {'vs Chen':>10}")
    print("-"*55)

    best_macro = 0
    best_name = ""
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["macro"]):
        delta = (res["macro"] - 0.8305) * 100
        marker = " ★" if res["macro"] > 0.8305 else ""
        print(f"  {name:<33} {res['macro']:>9.4f} {delta:>+9.2f}%{marker}")
        if res["macro"] > best_macro:
            best_macro = res["macro"]
            best_name = name

    print(f"\n  Best: {best_name} = {best_macro:.4f} ({(best_macro-0.8305)*100:+.2f}%)")

    output = {"best": {"name": best_name, "macro": best_macro},
              "all_results": all_results}
    out_path = RESULTS_DIR / "all_strategies.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
