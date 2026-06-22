"""
replicate_chen_83.py — Reproduce Chen et al.'s 83% OOD accuracy.

Uses Chen's EXACT preprocessing pipeline:
  ResizeWithProportions(128) → Resize(224) → ToTensor

With 3-model BEiT ensemble + 4-rotation TTA + geometric mean.
"""

import sys, os, json, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from scipy.stats import gmean

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
RESULTS_DIR = ROOT / "results"

MODEL_DIRS = [
    MODEL_DIR / "trained_models" / "01",
    MODEL_DIR / "trained_models" / "02",
    MODEL_DIR / "trained_models" / "03",
]
CLASSES_PATH = MODEL_DIR / "classes.npy"


# ── Chen's exact preprocessing ──
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


def chen_preprocess(image_path):
    """Load and preprocess image using Chen's exact pipeline."""
    im = Image.open(image_path).convert("RGB")
    im = resize_with_proportions(im, desired_size=128)
    im = im.resize((224, 224), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    return tensor


def chen_preprocess_rotated(image_path, angle):
    """Load and preprocess with rotation (0, 90, 180, 270 degrees)."""
    im = Image.open(image_path).convert("RGB")
    im = resize_with_proportions(im, desired_size=128)
    if angle > 0:
        im = im.rotate(angle, expand=False)
    im = im.resize((224, 224), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


# ── Load OOD data ──
def load_ood_cell(ood_path, classes):
    """Load all images and labels from an OOD test cell."""
    images = []
    labels = []
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
    return images, labels


SBA_MODEL_DIR = RESULTS_DIR / "finetune_chen_saa"
SBA_MODEL_FILES = [
    SBA_MODEL_DIR / "model_01_finetuned_v4.pth",
    SBA_MODEL_DIR / "model_02_finetuned_v4.pth",
    SBA_MODEL_DIR / "model_03_finetuned_v4.pth",
]


# ── Model loading ──
def load_models(device, variant="chen"):
    """Load all 3 BEiT models. variant: 'chen' or 'sba'."""
    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    models = []

    if variant == "sba":
        model_files = SBA_MODEL_FILES
        print(f"  Loading SBA-finetuned models from {SBA_MODEL_DIR}")
    else:
        model_files = [mdir / "trained_model_tuned.pth" for mdir in MODEL_DIRS]
        print(f"  Loading Chen's original models")

    for ckpt_path in model_files:
        model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                                   pretrained=False, num_classes=num_classes)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            print(f"    [WARN] {ckpt_path.name}: {len(result.unexpected_keys)} unexpected keys")
        if result.missing_keys:
            print(f"    [WARN] {ckpt_path.name}: {len(result.missing_keys)} missing keys")
        model.to(device)
        model.eval()
        models.append(model)
    return models, classes


# ── Prediction ──
@torch.no_grad()
def predict_single(model, tensor, device):
    """Single forward pass, returns softmax probabilities."""
    x = tensor.unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]


def predict_with_tta(model, image_path, device, angles=[0, 90, 180, 270]):
    """Predict with test-time augmentation (rotation)."""
    all_probs = []
    for angle in angles:
        tensor = chen_preprocess_rotated(image_path, angle)
        probs = predict_single(model, tensor, device)
        all_probs.append(probs)
    return np.array(all_probs)


def evaluate_ood_cell(ood_path, models, classes, device, use_tta=True):
    """Evaluate one OOD cell with ensemble + TTA."""
    images, labels = load_ood_cell(ood_path, classes)
    if len(images) == 0:
        return None
    labels = np.array(labels)
    
    all_ensemble_probs = []
    
    for img_path in tqdm(images, desc=f"  {Path(ood_path).name}", leave=False):
        model_probs = []
        for model in models:
            if use_tta:
                tta_probs = predict_with_tta(model, img_path, device)
                # Average TTA predictions per model
                model_probs.append(np.mean(tta_probs, axis=0))
            else:
                tensor = chen_preprocess(img_path)
                probs = predict_single(model, tensor, device)
                model_probs.append(probs)
        
        # Geometric mean across models
        ensemble_probs = gmean(model_probs)
        all_ensemble_probs.append(ensemble_probs)
    
    all_ensemble_probs = np.array(all_ensemble_probs)
    preds = all_ensemble_probs.argmax(axis=1)
    accuracy = (preds == labels).mean()
    
    return {
        "n_images": len(images),
        "accuracy": float(accuracy),
        "correct": int((preds == labels).sum()),
        "total": int(len(labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood-cells", nargs="*", default=None,
                        help="OOD cell names to evaluate (default: all 10)")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--variant", choices=["chen", "sba"], default="chen",
                        help="Model variant: 'chen' (original) or 'sba' (SBA-finetuned)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    print(f"Loading 3 BEiT models ({args.variant})...")
    models, classes = load_models(device, variant=args.variant)
    print(f"  Classes: {len(classes)} — {list(classes)}")
    
    # Determine OOD cells
    if args.ood_cells:
        ood_names = args.ood_cells
    else:
        ood_names = sorted([d.name for d in OOD_DIR.iterdir() if d.is_dir()])
    
    print(f"\nEvaluating {len(ood_names)} OOD cells with "
          f"{'TTA + ' if not args.no_tta else ''}geometric ensemble...")
    
    results = {}
    for ood_name in ood_names:
        ood_path = OOD_DIR / ood_name
        if not ood_path.exists():
            print(f"  [SKIP] {ood_name} — not found")
            continue
        
        print(f"\n  {ood_name}:")
        res = evaluate_ood_cell(ood_path, models, classes, device, use_tta=not args.no_tta)
        if res:
            results[ood_name] = res
            print(f"    Accuracy: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")
    
    # Overall
    if results:
        all_correct = sum(r["correct"] for r in results.values())
        all_total = sum(r["total"] for r in results.values())
        overall = all_correct / all_total
        
        per_day_accs = [r["accuracy"] for r in results.values()]
        macro_avg = np.mean(per_day_accs)
        
        print(f"\n{'='*50}")
        print(f"Micro-OOD accuracy: {overall:.4f} ({all_correct}/{all_total})")
        print(f"Macro-OOD accuracy: {macro_avg:.4f}")
        print(f"Per-day: {[f'{a:.3f}' for a in per_day_accs]}")
        print(f"{'='*50}")
        
        output = {
            "overall_micro": float(overall),
            "overall_macro": float(macro_avg),
            "per_day": {k: v["accuracy"] for k, v in results.items()},
            "details": results,
            "config": {
                "variant": args.variant,
                "models": [str(d) for d in MODEL_DIRS] if args.variant == "chen" else [str(d) for d in SBA_MODEL_FILES],
                "tta": not args.no_tta,
                "tta_angles": [0, 90, 180, 270] if not args.no_tta else None,
                "ensemble": "geometric_mean",
                "preprocessing": "Chen_ResizeWithProportions(128)->Resize(224)",
            }
        }
        
        out_path = args.output or str(RESULTS_DIR / f"chen_replication_{args.variant}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
