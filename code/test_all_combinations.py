"""
test_all_combinations.py — Test every combination of models, preprocessing, and TTA.
"""

import sys, json
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

# All model sets
MODEL_SETS = {
    "chen_original": [
        MODEL_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
        MODEL_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
        MODEL_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
    ],
    "sba_old_v4": [
        ROOT / "results" / "finetune_chen_saa" / "model_01_finetuned_v4.pth",
        ROOT / "results" / "finetune_chen_saa" / "model_02_finetuned_v4.pth",
        ROOT / "results" / "finetune_chen_saa" / "model_03_finetuned_v4.pth",
    ],
    "sba_new_correct": [
        ROOT / "results" / "finetune_sba_correct" / "model_01_sba_finetuned.pth",
        ROOT / "results" / "finetune_sba_correct" / "model_02_sba_finetuned.pth",
        ROOT / "results" / "finetune_sba_correct" / "model_03_sba_finetuned.pth",
    ],
}


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


def preprocess_chen(img_path):
    """Chen's exact preprocessing: ResizeWithProportions(128) -> Resize(224)"""
    im = Image.open(img_path).convert("RGB")
    im = resize_with_proportions(im, desired_size=128)
    im = im.resize((224, 224), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def preprocess_centercrop(img_path):
    """Our original preprocessing: Resize(224) -> CenterCrop(224)"""
    from torchvision import transforms
    im = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform(im)


def load_models(model_files, num_classes, device):
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
def predict(models, im_pil, device, use_tta, preprocess_fn):
    """Predict with optional TTA."""
    all_model_probs = []
    for model in models:
        if use_tta:
            tta_probs = []
            for angle in [0, 90, 180, 270]:
                im = im_pil.copy()
                if angle > 0:
                    im = im.rotate(angle, expand=False)
                tensor = preprocess_fn_from_pil(im, preprocess_fn)
                probs = F.softmax(model(tensor.unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
                tta_probs.append(probs)
            all_model_probs.append(np.mean(tta_probs, axis=0))
        else:
            tensor = preprocess_fn_from_pil(im_pil, preprocess_fn)
            probs = F.softmax(model(tensor.unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
            all_model_probs.append(probs)
    return gmean(all_model_probs)


def preprocess_fn_from_pil(im_pil, preprocess_type):
    """Apply preprocessing to a PIL image."""
    if preprocess_type == "chen":
        im = resize_with_proportions(im_pil, desired_size=128)
        im = im.resize((224, 224), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    else:  # centercrop
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        return transform(im_pil)


def evaluate(models, classes, device, use_tta, preprocess_type):
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
            probs = predict(models, im, device, use_tta, preprocess_type)
            if np.argmax(probs) == labels[i]:
                correct += 1

        per_day[ood_name] = float(correct / len(labels))

    macro = np.mean(list(per_day.values()))
    return macro, per_day


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    # All combinations
    configs = [
        # (model_set, preprocessing, tta, description)
        ("chen_original", "chen", True, "Chen original + TTA (should be ~83%)"),
        ("chen_original", "chen", False, "Chen original, no TTA"),
        ("chen_original", "centercrop", False, "Chen original + CenterCrop (our old baseline)"),
        ("sba_old_v4", "chen", True, "OLD SBA (v4) + Chen prep + TTA"),
        ("sba_old_v4", "chen", False, "OLD SBA (v4) + Chen prep, no TTA"),
        ("sba_old_v4", "centercrop", True, "OLD SBA (v4) + CenterCrop + TTA"),
        ("sba_old_v4", "centercrop", False, "OLD SBA (v4) + CenterCrop, no TTA"),
        ("sba_new_correct", "chen", True, "NEW SBA (correct prep) + TTA"),
        ("sba_new_correct", "chen", False, "NEW SBA (correct prep), no TTA"),
    ]

    results = {}
    for model_set, prep, tta, desc in configs:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"  Models: {model_set}, Prep: {prep}, TTA: {tta}")
        print(f"{'='*60}")

        models = load_models(MODEL_SETS[model_set], num_classes, device)
        macro, per_day = evaluate(models, classes, device, tta, prep)

        key = f"{model_set}__{prep}__tta{tta}"
        results[key] = {
            "description": desc,
            "macro": float(macro),
            "per_day": {k: round(v, 4) for k, v in sorted(per_day.items())},
            "model_set": model_set,
            "preprocessing": prep,
            "tta": tta,
        }
        print(f"  RESULT: {macro:.4f}")

        del models
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — ALL COMBINATIONS")
    print(f"{'='*70}")
    print(f"{'Description':<50} {'Macro':>8}")
    print("-" * 58)
    for key, res in sorted(results.items(), key=lambda x: -x[1]["macro"]):
        print(f"  {res['description']:<48} {res['macro']:>7.4f}")

    # Save
    out_path = ROOT / "results" / "all_combinations.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
