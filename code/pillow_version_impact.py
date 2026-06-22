"""
tier1_pillow_impact.py — Does the Pillow version change affect classification accuracy?

Evaluates Chen's trained BEiT models on OOD images processed with:
  1. Pillow 6.x (nearest-neighbor resize)
  2. Pillow 7.0 (bicubic resize)

Measures accuracy drop from the software update alone.
"""

import json, sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import timm
from scipy.stats import gmean

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
CLASSES_PATH = MODEL_DIR / "classes.npy"
RESULTS_DIR = ROOT / "results" / "tier1"

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


def load_models(device, num_classes):
    models = []
    for p in CHEN_MODEL_FILES:
        m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=False)
        m.to(device).eval()
        models.append(m)
    return models


@torch.no_grad()
def predict(models, im_pil, device):
    """Predict with 4-rotation TTA + geometric ensemble."""
    all_model_probs = []
    for model in models:
        tta_probs = []
        for angle in [0, 90, 180, 270]:
            im = im_pil.copy()
            if angle > 0:
                im = im.rotate(angle, expand=False)
            im = im.resize((224, 224), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            tta_probs.append(probs)
        all_model_probs.append(np.mean(tta_probs, axis=0))
    return gmean(all_model_probs)


def evaluate(models, classes, device, resize_method='bicubic'):
    """Evaluate on OOD with specific resize method."""
    if resize_method == 'nearest':
        resample = Image.NEAREST
    else:
        resample = Image.BICUBIC

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
        for i, img_path in enumerate(tqdm(images, desc=f"  {ood_name} ({resize_method})", leave=False)):
            im = Image.open(img_path).convert("RGB")
            # Use Chen's preprocessing but with specified resize method
            im = resize_with_proportions(im, desired_size=128)
            # Override the final resize method
            im = im.resize((224, 224), resample)
            # But we need to undo the previous resize and redo with correct method
            # Actually, the resize_with_proportions already uses LANCZOS for the initial shrink
            # The final resize to 224 is what we're testing
            # Let's redo properly:
            im_orig = Image.open(img_path).convert("RGB")
            # Step 1: Proportional resize (same for both)
            old_size = im_orig.size
            if max(old_size) > 128:
                ratio = float(128) / max(old_size)
                new_size = tuple([int(x * ratio) for x in old_size])
                im_orig = im_orig.resize(new_size, Image.LANCZOS)
            im_square = Image.new("RGB", (128, 128), color=0)
            offset = ((128 - im_orig.size[0]) // 2, (128 - im_orig.size[1]) // 2)
            im_square.paste(im_orig, offset)
            # Step 2: Final resize with tested method
            im_final = im_square.resize((224, 224), resample)

            probs = predict(models, im_final, device)
            pred = np.argmax(probs)
            if pred == labels[i]:
                correct += 1

        acc = correct / len(labels)
        per_day[ood_name] = float(acc)

    micro = sum(v * 1000 for v in per_day.values()) / sum(1000 for _ in per_day)
    macro = np.mean(list(per_day.values()))
    return macro, per_day


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    models = load_models(device, num_classes)
    print(f"Loaded {len(models)} models, {num_classes} classes")

    # Evaluate with bicubic (Pillow 7.0 default)
    print(f"\n{'='*60}")
    print("Evaluating with BICUBIC (Pillow 7.0 default)")
    print(f"{'='*60}")
    macro_bicubic, per_day_bicubic = evaluate(models, classes, device, 'bicubic')
    print(f"  Macro-OOD (bicubic): {macro_bicubic:.4f}")

    # Evaluate with nearest (Pillow 6.x default)
    print(f"\n{'='*60}")
    print("Evaluating with NEAREST (Pillow 6.x default)")
    print(f"{'='*60}")
    macro_nearest, per_day_nearest = evaluate(models, classes, device, 'nearest')
    print(f"  Macro-OOD (nearest): {macro_nearest:.4f}")

    # Compute difference
    diff = (macro_bicubic - macro_nearest) * 100

    print(f"\n{'='*60}")
    print("PILLOW VERSION IMPACT ON CLASSIFICATION")
    print(f"{'='*60}")
    print(f"  Pillow 6.x (nearest):  {macro_nearest:.4f}")
    print(f"  Pillow 7.0 (bicubic):  {macro_bicubic:.4f}")
    print(f"  Difference:            {diff:+.2f}%")
    print(f"{'='*60}")

    # Per-day breakdown
    print("\nPer-day breakdown:")
    print(f"  {'OOD':<6} {'Nearest':>10} {'Bicubic':>10} {'Diff':>10}")
    for day in sorted(per_day_nearest.keys()):
        n = per_day_nearest[day]
        b = per_day_bicubic[day]
        d = (b - n) * 100
        print(f"  {day:<6} {n:>10.4f} {b:>10.4f} {d:>+10.2f}%")

    output = {
        "pillow_6_nearest_macro": float(macro_nearest),
        "pillow_7_bicubic_macro": float(macro_bicubic),
        "accuracy_drop_percent": float(diff),
        "per_day_nearest": per_day_nearest,
        "per_day_bicubic": per_day_bicubic,
    }
    out_path = RESULTS_DIR / "pillow_accuracy_impact.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
