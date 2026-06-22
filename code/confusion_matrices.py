"""
tier1_confusion_matrices.py — Generate confusion matrices for key experiments.

Generates:
  1. Cross-instrument confusion matrix (baseline vs SBA)
  2. Temporal OOD confusion matrix (baseline vs SBA fine-tuned)
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
CLASSES_PATH = MODEL_DIR / "classes.npy"
RESULTS_DIR = ROOT / "results" / "tier1"
FIGURES_DIR = ROOT / "figures"

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


def load_models(device, num_classes, model_files=None):
    if model_files is None:
        model_files = CHEN_MODEL_FILES
    models = []
    for p in model_files:
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


def generate_confusion_matrix(models, classes, device, ood_name, output_path):
    """Generate confusion matrix for a specific OOD cell."""
    ood_path = OOD_DIR / ood_name
    if not ood_path.exists():
        print(f"  {ood_name} not found")
        return

    images, labels, class_names = [], [], []
    for cls_dir in sorted(ood_path.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in classes:
            continue
        cls_idx = np.where(classes == cls_dir.name)[0][0]
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                images.append(str(img_path))
                labels.append(cls_idx)
                class_names.append(cls_dir.name)

    labels = np.array(labels)
    all_preds = []

    for img_path in tqdm(images, desc=f"  {ood_name}", leave=False):
        im = Image.open(img_path).convert("RGB")
        im = resize_with_proportions(im, desired_size=128)
        probs = predict(models, im, device)
        all_preds.append(np.argmax(probs))

    all_preds = np.array(all_preds)

    # Get unique classes present in this OOD cell
    unique_labels = sorted(set(labels) | set(all_preds))
    present_classes = [classes[i] for i in unique_labels]

    # Compute confusion matrix
    cm = confusion_matrix(labels, all_preds, labels=unique_labels)

    # Normalize by row (true labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=present_classes)
    disp.plot(ax=ax, cmap='Blues', values_format='.2f', xticks_rotation=45)
    ax.set_title(f'Confusion Matrix — {ood_name} (Accuracy: {(labels == all_preds).mean():.3f})', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return {
        "accuracy": float((labels == all_preds).mean()),
        "n_images": len(labels),
        "per_class_acc": {classes[i]: float((labels[labels == i] == all_preds[labels == i]).mean())
                          for i in unique_labels if (labels == i).sum() > 0},
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    models = load_models(device, num_classes)
    print(f"Loaded {len(models)} models, {num_classes} classes")

    # Generate confusion matrices for key OOD days
    key_days = ["OOD1", "OOD5", "OOD10"]  # best, middle, worst
    results = {}

    for day in key_days:
        print(f"\nGenerating confusion matrix for {day}...")
        out_path = FIGURES_DIR / f"fig_cm_{day.lower()}.png"
        res = generate_confusion_matrix(models, classes, device, day, out_path)
        if res:
            results[day] = res

    # Find most confused pairs
    print(f"\n{'='*60}")
    print("MOST CONFUSED CLASS PAIRS")
    print(f"{'='*60}")
    for day, res in results.items():
        print(f"\n  {day}:")
        # Sort per-class accuracy
        sorted_classes = sorted(res["per_class_acc"].items(), key=lambda x: x[1])
        for cls, acc in sorted_classes[:5]:
            print(f"    {cls}: {acc:.3f}")

    out_path = RESULTS_DIR / "confusion_matrices.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
