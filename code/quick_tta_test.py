"""
quick_tta_test.py — Test more TTA angles on Chen's original BEiT models.
Fastest way to potentially beat 83%.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, torch.nn.functional as F, timm, json
from scipy.stats import gmean

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
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

def load_models(device):
    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    models = []
    for p in CHEN_MODEL_FILES:
        m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=False, num_classes=len(classes))
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
        m.to(device).eval()
        models.append(m)
    return models, classes

@torch.no_grad()
def predict_tta(models, im_pil, device, angles):
    model_probs = []
    for model in models:
        tta_probs = []
        for angle in angles:
            im = im_pil.copy()
            if angle > 0:
                im = im.rotate(angle, expand=False)
            im = im.resize((224, 224), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            tta_probs.append(probs)
        model_probs.append(np.mean(tta_probs, axis=0))
    return gmean(model_probs)

def evaluate(models, classes, device, angles):
    all_preds, all_labels = [], []
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
        preds = []
        for img_path in tqdm(images, desc=f"  {ood_name}", leave=False):
            im = Image.open(img_path).convert("RGB")
            im = resize_with_proportions(im, desired_size=128)
            probs = predict_tta(models, im, device, angles)
            preds.append(np.argmax(probs))
        preds = np.array(preds)
        acc = (preds == labels).mean()
        per_day[ood_name] = float(acc)
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    micro = (all_preds == all_labels).mean()
    macro = np.mean(list(per_day.values()))
    return micro, macro, per_day

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models, classes = load_models(device)
    print(f"Loaded {len(models)} models, {len(classes)} classes")

    configs = {
        "4_TTA":  [0, 90, 180, 270],
        "8_TTA":  [0, 45, 90, 135, 180, 225, 270, 315],
        "12_TTA": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        "16_TTA": [i * 22 for i in range(16)],
    }

    results = {}
    for name, angles in configs.items():
        print(f"\n{'='*50}")
        print(f"Testing {name}: {len(angles)} angles")
        print(f"{'='*50}")
        micro, macro, per_day = evaluate(models, classes, device, angles)
        results[name] = {"micro": float(micro), "macro": float(macro), "per_day": per_day}
        delta = (macro - 0.8305) * 100
        print(f"  Micro: {micro:.4f}  Macro: {macro:.4f}  vs Chen: {delta:+.2f}%")
        print(f"  Per-day: { {k: round(v, 3) for k, v in sorted(per_day.items())} }")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["macro"]):
        delta = (res["macro"] - 0.8305) * 100
        marker = " ★ BEATS CHEN!" if res["macro"] > 0.8305 else ""
        print(f"  {name:<10} Macro: {res['macro']:.4f}  {delta:+.2f}%{marker}")

    out_path = ROOT / "results" / "beat_83" / "tta_angles.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
