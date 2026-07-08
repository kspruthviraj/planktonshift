"""
test_sba_centrecrop_tta.py — Test OLD SBA (v4) + CenterCrop + TTA.
The OLD SBA models were trained with CenterCrop preprocessing.
Testing with matching preprocessing + TTA may beat 83%.
"""

import json, sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from scipy.stats import gmean

ROOT = Path(__file__).resolve().parent.parent
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
CLASSES_PATH = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models" / "classes.npy"

SBA_MODELS = [
    ROOT / "results" / "finetune_chen_saa" / "model_01_finetuned_v4.pth",
    ROOT / "results" / "finetune_chen_saa" / "model_02_finetuned_v4.pth",
    ROOT / "results" / "finetune_chen_saa" / "model_03_finetuned_v4.pth",
]

centercrop_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_models(device, num_classes):
    models = []
    for p in SBA_MODELS:
        m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state, strict=False)
        m.to(device).eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_tta(models, im_pil, device):
    all_model_probs = []
    for model in models:
        tta_probs = []
        for angle in [0, 90, 180, 270]:
            im = im_pil.copy()
            if angle > 0:
                im = im.rotate(angle, expand=False)
            tensor = centercrop_transform(im)
            probs = F.softmax(model(tensor.unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
            tta_probs.append(probs)
        all_model_probs.append(np.mean(tta_probs, axis=0))
    return gmean(all_model_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    print(f"\nLoading OLD SBA (v4) models...")
    models = load_models(device, num_classes)

    print(f"\nEvaluating with CenterCrop + TTA...")
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
            probs = predict_tta(models, im, device)
            if np.argmax(probs) == labels[i]:
                correct += 1

        acc = correct / len(labels)
        per_day[ood_name] = float(acc)
        print(f"  {ood_name}: {acc:.4f} ({correct}/{len(labels)})")

    macro = np.mean(list(per_day.values()))
    delta = (macro - 0.8305) * 100

    print(f"\n{'='*60}")
    print(f"OLD SBA (v4) + CenterCrop + TTA")
    print(f"{'='*60}")
    print(f"Macro-OOD: {macro:.4f}")
    print(f"vs Chen 83.05%: {delta:+.2f}%")
    print(f"Per-day: { {k: round(v, 3) for k, v in sorted(per_day.items())} }")
    if macro > 0.8305:
        print(f"\n*** BEATS CHEN by {delta:.2f}%! ***")
    print(f"{'='*60}")

    output = {
        "macro": float(macro),
        "vs_chen": float(delta),
        "per_day": per_day,
        "config": {
            "models": "OLD SBA (v4)",
            "preprocessing": "CenterCrop(224) + ToTensor (matching training preprocessing)",
            "tta": "4 rotations (0/90/180/270)",
            "ensemble": "geometric mean of 3 models",
        }
    }
    out_path = ROOT / "results" / "sba_centrecrop_tta.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
