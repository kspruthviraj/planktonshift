"""
test_time_sba.py — Apply SBA perturbations at test time as additional TTA views.

Instead of modifying the model, we add frequency-domain diversity to TTA:
- Original image
- 4 rotation angles
- SBA spectral noise variants
- SBA band adversarial variants

This is like "frequency-domain TTA" — no retraining needed.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, torch.nn.functional as F, timm, json
from scipy.stats import gmean

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "adverserial_net"))
from spectral_augmentation import SpectralAugmentation

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
def predict_tensor(model, tensor, device):
    return F.softmax(model(tensor.unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]


def im_to_tensor(im_pil):
    arr = np.array(im_pil.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def evaluate(models, classes, device, sba, n_sba_variants=3):
    """Evaluate with rotation TTA + SBA test-time perturbations."""
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

            model_probs = []
            for model in models:
                tta_probs = []

                # Standard rotation TTA
                for angle in [0, 90, 180, 270]:
                    im_rot = im.copy()
                    if angle > 0:
                        im_rot = im.rotate(angle, expand=False)
                    tta_probs.append(predict_tensor(model, im_to_tensor(im_rot), device))

                # SBA test-time perturbations
                gray = np.array(im.convert("L"), dtype=np.float64) / 255.0
                for _ in range(n_sba_variants):
                    gray_aug = sba(gray)
                    gray_uint8 = (gray_aug * 255).clip(0, 255).astype(np.uint8)
                    im_aug = Image.fromarray(gray_uint8, mode="L").convert("RGB")
                    # Apply on original + 90° rotation
                    tta_probs.append(predict_tensor(model, im_to_tensor(im_aug), device))
                    im_aug_rot = im_aug.rotate(90, expand=False)
                    tta_probs.append(predict_tensor(model, im_to_tensor(im_aug_rot), device))

                model_probs.append(np.mean(tta_probs, axis=0))

            preds.append(np.argmax(gmean(model_probs)))

        preds = np.array(preds)
        per_day[ood_name] = float((preds == labels).mean())

    macro = np.mean(list(per_day.values()))
    return macro, per_day


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models, classes = load_models(device)
    print(f"Loaded {len(models)} models, {len(classes)} classes")

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
    print(f"Shift spectrum: {len(shift_spectrum) if shift_spectrum is not None else 0} bins")

    # Test different SBA configurations
    configs = [
        {"name": "baseline_4tta", "sba": False, "n_sba": 0},
        {"name": "sba3_noise_only", "sba": True, "n_sba": 3, "strategies": ["spectral_noise"], "strength": 0.1, "p": 1.0},
        {"name": "sba3_band_only", "sba": True, "n_sba": 3, "strategies": ["band_adversarial"], "strength": 0.1, "p": 1.0},
        {"name": "sba3_both", "sba": True, "n_sba": 3, "strategies": ["spectral_noise", "band_adversarial"], "strength": 0.1, "p": 1.0},
        {"name": "sba5_both", "sba": True, "n_sba": 5, "strategies": ["spectral_noise", "band_adversarial"], "strength": 0.1, "p": 1.0},
        {"name": "sba3_strong", "sba": True, "n_sba": 3, "strategies": ["spectral_noise", "band_adversarial"], "strength": 0.2, "p": 1.0},
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg['name']}")
        print(f"{'='*50}")

        sba = SpectralAugmentation(
            shift_spectrum=shift_spectrum,
            strength=cfg.get("strength", 0.5),
            strategies=cfg.get("strategies", ["spectral_noise", "band_adversarial"]),
            p=cfg.get("p", 1.0),
        ) if cfg["sba"] else None

        n_sba = cfg.get("n_sba", 0)
        macro, per_day = evaluate(models, classes, device, sba, n_sba)
        delta = (macro - 0.8305) * 100
        results[cfg["name"]] = {"macro": float(macro), "per_day": per_day}
        print(f"  Macro: {macro:.4f}  vs Chen: {delta:+.2f}%")
        print(f"  Per-day: { {k: round(v, 3) for k, v in sorted(per_day.items())} }")

    print(f"\n{'='*60}")
    print("SUMMARY — Test-time SBA")
    print(f"{'='*60}")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["macro"]):
        delta = (res["macro"] - 0.8305) * 100
        marker = " ★" if res["macro"] > 0.8305 else ""
        print(f"  {name:<25} Macro: {res['macro']:.4f}  {delta:+.2f}%{marker}")

    out_path = ROOT / "results" / "beat_83" / "test_time_sba.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
