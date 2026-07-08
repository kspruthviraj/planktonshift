"""
run_pillow_impact_corrected.py — Pillow version impact with FIXED accuracy weighting.

FIX (P0-5):
  - The original computed micro accuracy as sum(v*1000)/sum(1000), assuming
    every OOD day has exactly 1000 images. OOD3 has only 522. This script uses
    the ACTUAL per-day image counts for proper micro-averaging.
  - Reports macro AND correctly-weighted micro accuracy.
  - Adds a per-image paired sign test (McNemar) between bicubic and nearest.
  - Uses vendored data paths (no external dependencies).

Output: results/tier1_corrected/pillow_impact.json
"""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import timm
from scipy.stats import gmean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RESULTS, MODELS, CHEN_BEST_ACCURACY
from utils_pipeline import resize_with_proportions, mcnemar_test

OUT = RESULTS / "tier1_corrected" / "pillow_impact.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OOD_DIR = DATA["ood"]


def load_models(device, num_classes):
    models = []
    for key in ["chen_beit_01", "chen_beit_02", "chen_beit_03"]:
        p = MODELS[key]
        m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=False)
        m.to(device).eval()
        models.append(m)
    return models


@torch.no_grad()
def predict(models, im_pil, device, angles=[0, 90, 180, 270]):
    all_model_probs = []
    for model in models:
        tta_probs = []
        for angle in angles:
            im = im_pil.copy()
            if angle > 0:
                im = im.rotate(angle, expand=False)
            im = resize_with_proportions(im, desired_size=128)
            im = im.resize((224, 224), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            tta_probs.append(probs)
        all_model_probs.append(np.mean(tta_probs, axis=0))
    return gmean(all_model_probs)


def evaluate(models, classes, device, resample):
    """Evaluate on OOD with specific final resize method.
    Returns per_day dict with ACTUAL n_images, and flat prediction arrays."""
    per_day = {}
    all_preds = []
    all_labels = []
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
        day_preds = []
        for i, img_path in enumerate(tqdm(images, desc=f"  {ood_name} ({resample})", leave=False)):
            im_orig = Image.open(img_path).convert("RGB")
            # Step 1: Proportional resize (same for both)
            old_size = im_orig.size
            if max(old_size) > 128:
                ratio = float(128) / max(old_size)
                new_size = tuple(int(x * ratio) for x in old_size)
                im_orig = im_orig.resize(new_size, Image.LANCZOS)
            im_square = Image.new("RGB", (128, 128), color=0)
            offset = ((128 - im_orig.size[0]) // 2, (128 - im_orig.size[1]) // 2)
            im_square.paste(im_orig, offset)
            # Step 2: Final resize with tested method
            im_final = im_square.resize((224, 224), resample)
            probs = predict(models, im_final, device)
            pred = np.argmax(probs)
            day_preds.append(pred)
            if pred == labels[i]:
                correct += 1
        acc = correct / len(labels)
        per_day[ood_name] = {"accuracy": float(acc), "n_images": len(labels), "correct": correct}
        all_preds.extend(day_preds)
        all_labels.extend(labels.tolist())
        print(f"  {ood_name}: {acc:.4f} (n={len(labels)})")
    return per_day, np.array(all_preds), np.array(all_labels)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    print(f"Device: {device}")
    classes = np.load(str(MODELS["chen_classes"]), allow_pickle=True)
    num_classes = len(classes)
    models = load_models(device, num_classes)
    print(f"Loaded {len(models)} models, {num_classes} classes")

    # Bicubic (Pillow 7.0 default)
    print(f"\n{'='*60}\nBICUBIC (Pillow 7.0 default)\n{'='*60}")
    bicubic_per_day, bicubic_preds, bicubic_labels = evaluate(models, classes, device, Image.BICUBIC)
    bicubic_correct = (bicubic_preds == bicubic_labels).astype(float)
    bicubic_macro = np.mean([v["accuracy"] for v in bicubic_per_day.values()])
    # CORRECTED micro: use actual n_images
    bicubic_micro = sum(v["correct"] for v in bicubic_per_day.values()) / sum(v["n_images"] for v in bicubic_per_day.values())

    # Nearest (Pillow 6.x default)
    print(f"\n{'='*60}\nNEAREST (Pillow 6.x default)\n{'='*60}")
    nearest_per_day, nearest_preds, nearest_labels = evaluate(models, classes, device, Image.NEAREST)
    nearest_correct = (nearest_preds == nearest_labels).astype(float)
    nearest_macro = np.mean([v["accuracy"] for v in nearest_per_day.values()])
    nearest_micro = sum(v["correct"] for v in nearest_per_day.values()) / sum(v["n_images"] for v in nearest_per_day.values())

    diff_macro = (bicubic_macro - nearest_macro) * 100
    diff_micro = (bicubic_micro - nearest_micro) * 100

    # McNemar paired test
    stat, pval, (n01, n10) = mcnemar_test(bicubic_preds, nearest_preds, bicubic_labels)

    print(f"\n{'='*60}\nPILLOW VERSION IMPACT (CORRECTED)\n{'='*60}")
    print(f"  Pillow 6.x (nearest):  macro={nearest_macro:.4f} micro={nearest_micro:.4f}")
    print(f"  Pillow 7.0 (bicubic):  macro={bicubic_macro:.4f} micro={bicubic_micro:.4f}")
    print(f"  Diff:                  macro={diff_macro:+.2f}% micro={diff_micro:+.2f}%")
    print(f"  McNemar: stat={stat:.3f} p={pval:.4f} discordant bicubic_right/nearest_right=({n01},{n10})")
    print(f"\n  Per-day (n_images):")
    for day in sorted(bicubic_per_day.keys()):
        n = bicubic_per_day[day]["n_images"]
        b = bicubic_per_day[day]["accuracy"]
        nn = nearest_per_day[day]["accuracy"]
        print(f"    {day}: n={n} nearest={nn:.4f} bicubic={b:.4f} diff={(b-nn)*100:+.2f}%")

    output = {
        "pillow_6_nearest_macro": float(nearest_macro),
        "pillow_6_nearest_micro": float(nearest_micro),
        "pillow_7_bicubic_macro": float(bicubic_macro),
        "pillow_7_bicubic_micro": float(bicubic_micro),
        "macro_diff_percent": float(diff_macro),
        "micro_diff_percent": float(diff_micro),
        "mcnemar": {"statistic": stat, "p_value": pval, "discordant": {"bicubic_right_nearest_wrong": n01, "bicubic_wrong_nearest_right": n10}},
        "per_day_bicubic": {k: v for k, v in bicubic_per_day.items()},
        "per_day_nearest": {k: v for k, v in nearest_per_day.items()},
    }
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
