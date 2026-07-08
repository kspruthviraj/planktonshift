"""
feature_beit_ensemble.py
========================
Ensemble a feature-based classifier with BEiT predictions.

Hypothesis: morphological features (shape, color, texture) are more
domain-invariant than pixel-level features. Combining them with BEiT
should improve OOD robustness.

Approach:
1. Extract Chen's 67 morphological features from each image
2. Train a Random Forest on ZooLake2 training data
3. Get RF predictions on OOD images
4. Geometric-ensemble RF + BEiT (with TTA)
"""

import sys, json, argparse, os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import timm
from scipy.stats import gmean
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import cv2

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
ZOOlAKE_DIR = ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0"
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
CLASSES_PATH = MODEL_DIR / "classes.npy"
RESULTS_DIR = ROOT / "results" / "feature_beit_ensemble"

CHEN_MODEL_FILES = [
    MODEL_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
    MODEL_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
]


# ── Chen's feature extraction (from utils_analysis/lib/feature_extraction.py) ──
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


def extract_features(img_path):
    """Extract 67 morphological features from a single image."""
    try:
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = resize_with_proportions(pil_img, 224)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None

    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (2, 2))
    ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None

    x, y, width, height = cv2.boundingRect(cnt)
    rot_rect = cv2.minAreaRect(cnt)
    w_rot = rot_rect[1][0]
    h_rot = rot_rect[1][1]
    angle_rot = rot_rect[2]

    M = cv2.moments(cnt)
    H = cv2.HuMoments(M)

    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    aspect_ratio = float(width) / height if height > 0 else 0
    rect_area = width * height
    contour_area = cv2.contourArea(cnt)
    contour_perimeter = cv2.arcLength(cnt, True)
    extent = float(contour_area) / rect_area if rect_area > 0 else 0
    compactness = (contour_perimeter ** 2) / (4 * np.pi * contour_area) if contour_area > 0 else 0
    formfactor = (4 * np.pi * contour_area) / (contour_perimeter ** 2) if contour_perimeter > 0 else 0
    hull_2 = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull_2)
    solidity = float(contour_area) / hull_area if hull_area > 0 else 0
    hull_perimeter = cv2.arcLength(hull_2, True)
    ESD = np.sqrt(4 * contour_area / np.pi) if contour_area > 0 else 0

    try:
        (x1, y1), (Major_Axis, Minor_Axis), angle = cv2.fitEllipse(cnt)
        Eccentricity1 = Minor_Axis / Major_Axis if Major_Axis > 0 else 0
    except Exception:
        angle, Eccentricity1, Major_Axis, Minor_Axis = 0, 0, 0, 0

    Mu02 = M['m02'] - (cy * M['m01'])
    Mu20 = M['m20'] - (cx * M['m10'])
    Mu11 = M['m11'] - (cx * M['m01'])
    Eccentricity2 = (Mu02 - Mu20) ** 2 + 4 * Mu11 / contour_area if contour_area > 0 else 0
    Convexity = hull_perimeter / contour_perimeter if contour_perimeter > 0 else 0
    Roundness = (4 * np.pi * contour_area) / (hull_perimeter ** 2) if hull_perimeter > 0 else 0

    BGR_mean_std = cv2.meanStdDev(image, mask=thresh)
    HSV_mean_std = cv2.meanStdDev(HSV, mask=thresh)

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    blurriness = np.mean(np.abs(laplacian))
    noise = np.std(image)

    features = [
        width, height, w_rot, h_rot, angle_rot, aspect_ratio,
        rect_area, contour_area, contour_perimeter, extent,
        compactness, formfactor, hull_area, solidity, hull_perimeter,
        ESD, Major_Axis, Minor_Axis,
        angle, Eccentricity1, Eccentricity2, Convexity, Roundness,
        BGR_mean_std[0][2][0], BGR_mean_std[0][1][0], BGR_mean_std[0][0][0],  # R, G, B mean
        BGR_mean_std[1][2][0], BGR_mean_std[1][1][0], BGR_mean_std[1][0][0],  # R, G, B std
        HSV_mean_std[0][0][0], HSV_mean_std[0][1][0], HSV_mean_std[0][2][0],  # H, S, V mean
        HSV_mean_std[1][0][0], HSV_mean_std[1][1][0], HSV_mean_std[1][2][0],  # H, S, V std
        blurriness, noise,
        M['m00'], M['m10'], M['m01'], M['m20'], M['m11'], M['m02'],
        M['m30'], M['m21'], M['m12'], M['m03'],
        H[0][0], H[1][0], H[2][0], H[3][0], H[4][0], H[5][0], H[6][0],
    ]

    return features


FEATURE_NAMES = [
    'width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio',
    'rect_area', 'contour_area', 'contour_perimeter', 'extent',
    'compactness', 'formfactor', 'hull_area', 'solidity', 'hull_perimeter',
    'ESD', 'major_axis', 'minor_axis', 'angle', 'eccentricity',
    'eccentricity2', 'convexity', 'roundness',
    'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
    'hue_mean', 'saturation_mean', 'brightness_mean',
    'hue_std', 'saturation_std', 'brightness_std',
    'blurriness', 'noise',
    'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
    'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6',
]


# ── Data loading ──
def load_dataset_features(data_dir, classes, max_per_class=None):
    """Extract features from all images in a directory."""
    features_list, labels_list, paths_list = [], [], []
    for cls_dir in sorted(Path(data_dir).iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in classes:
            continue
        cls_idx = np.where(classes == cls_dir.name)[0][0]
        count = 0
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                continue
            feat = extract_features(str(img_path))
            if feat is not None:
                features_list.append(feat)
                labels_list.append(cls_idx)
                paths_list.append(str(img_path))
                count += 1
            if max_per_class and count >= max_per_class:
                break
    return np.array(features_list), np.array(labels_list), paths_list


# ── BEiT prediction ──
def resize_with_proportions_224(im):
    return resize_with_proportions(im, 224)


@torch.no_grad()
def predict_beit_tta(models, im_pil, device, angles=[0, 90, 180, 270]):
    all_model_probs = []
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
        all_model_probs.append(np.mean(tta_probs, axis=0))
    return gmean(all_model_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-weight", type=float, default=0.3, help="Weight for RF in ensemble (0-1)")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    rf_path = RESULTS_DIR / "rf_model.pkl"
    scaler_path = RESULTS_DIR / "scaler.pkl"

    if not args.eval_only:
        # ── Step 1: Extract features from training data ──
        print("\nStep 1: Extracting features from ZooLake2 training data...")
        X_train, y_train, _ = load_dataset_features(ZOOlAKE_DIR, classes)
        print(f"  Training features: {X_train.shape}")

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # ── Step 2: Train Random Forest ──
        print("\nStep 2: Training Random Forest...")
        rf = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        train_acc = rf.score(X_train_scaled, y_train)
        print(f"  RF train accuracy: {train_acc:.4f}")

        # Save
        import pickle
        with open(rf_path, 'wb') as f:
            pickle.dump(rf, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        import pickle
        with open(rf_path, 'rb') as f:
            rf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    # ── Step 3: Load BEiT models ──
    print("\nStep 3: Loading BEiT models...")
    beit_models = []
    for p in CHEN_MODEL_FILES:
        m = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=False, num_classes=num_classes)
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
        m.to(device).eval()
        beit_models.append(m)

    # ── Step 4: Evaluate on OOD ──
    print(f"\nStep 4: Evaluating on OOD (RF weight={args.rf_weight}, BEiT weight={1-args.rf_weight})...")
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
            # BEiT prediction
            im = Image.open(img_path).convert("RGB")
            im = resize_with_proportions_224(im)
            beit_probs = predict_beit_tta(beit_models, im, device)

            # RF prediction
            feat = extract_features(img_path)
            if feat is not None:
                feat_scaled = scaler.transform(np.array([np.nan_to_num(feat, nan=0, posinf=0, neginf=0)]))
                rf_probs = rf.predict_proba(feat_scaled)[0]
                # Ensure same number of classes
                if len(rf_probs) == num_classes:
                    ensemble_probs = gmean([beit_probs ** (1 - args.rf_weight), rf_probs ** args.rf_weight])
                else:
                    ensemble_probs = beit_probs
            else:
                ensemble_probs = beit_probs

            pred = np.argmax(ensemble_probs)
            if pred == labels[i]:
                correct += 1

        acc = correct / len(labels)
        per_day[ood_name] = float(acc)
        print(f"  {ood_name}: {acc:.4f} ({correct}/{len(labels)})")

    macro = np.mean(list(per_day.values()))
    delta = (macro - 0.8305) * 100

    print(f"\n{'='*60}")
    print(f"Macro-OOD: {macro:.4f}  vs Chen 83.05%: {delta:+.2f}%")
    print(f"RF weight: {args.rf_weight}  BEiT weight: {1-args.rf_weight}")
    print(f"Per-day: { {k: round(v, 3) for k, v in sorted(per_day.items())} }")

    if delta > 0:
        print(f"★ BEATS CHEN by {delta:.2f}%!")
    print(f"{'='*60}")

    output = {"macro": float(macro), "vs_chen": float(delta), "per_day": per_day,
              "config": {"rf_weight": args.rf_weight, "beit_weight": 1 - args.rf_weight,
                         "rf_model": "GradientBoosting(300, depth=6)",
                         "features": len(FEATURE_NAMES),
                         "ensemble": "geometric_mean"}}
    out_path = args.output or str(RESULTS_DIR / "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
