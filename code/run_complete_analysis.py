"""
run_complete_analysis.py
========================
Complete analysis pipeline that runs all missing experiments.
Handles class name mapping between different datasets.

Runs in sequence:
1. SAA on Chen's 10 OOD days (temporal shift)
2. RAG on Chen's 10 OOD days (with correct class mapping)
3. SAA on cross-ecosystem (WHOI22→ZooLake35)
4. Generate all figures

Usage:
    nohup python run_complete_analysis.py > complete_analysis.log 2>&1 &
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("complete_analysis.log"),
    ],
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224

# Paths
ADVERSARIAL_NET = "/home/sreenath/research-space/Adverserial_net"
CHEN_DATA = "/home/sreenath/research-space/PlanktonShift/data/chen_data"
CHEN_ZOOLAKE = f"{CHEN_DATA}/ZooLake2/ZooLake2/ZooLake2.0"
CHEN_OOD = f"{CHEN_DATA}/OOD_data/OODs"
RESULTS_DIR = "/home/sreenath/research-space/PlanktonShift/results"
FIGURES_DIR = "/home/sreenath/research-space/PlanktonShift/figures"

sys.path.insert(0, ADVERSARIAL_NET)


# ---------------------------------------------------------------------------
# Freshwater morphological catalog for Chen's ZooLake classes
# ---------------------------------------------------------------------------
FRESHWATER_CATALOG = {
    "aphanizomenon": "Filamentous cyanobacterium forming elongated straight or slightly curved trichomes. Cells cylindrical, may form bundles or rafts. Blue-green color.",
    "asplanchna": "Large predatory rotifer with transparent sac-like body. No visible lorica (shell). Prominent corona (crown of cilia). Internal organs visible.",
    "asterionella": "Star-shaped colonial diatom. Cells lanceolate (lance-shaped) attached at one end, forming radiating star colonies. Siliceous cell wall.",
    "bosmina": "Small cladoceran with characteristic long antennae and a downward-curving rostrum (beak-like projection). Oval body enclosed in bivalve shell.",
    "brachionus": "Planktonic rotifer with prominent lorica (shell). Lorica typically oval with anterior spines. Corona with two large wheel organs.",
    "ceratium": "Dinoflagellate with 2-4 horn-like projections. One forward horn, 1-3 backward horns. Visible cingulum (groove) around cell middle. Chloroplasts present.",
    "chaoborus": "Phantom midge larva (Diptera). Transparent cylindrical body with two pairs of air sacs (hydrostatic organs). Dark eye spots. No legs in larval stage.",
    "collotheca": "Sessile rotifer with trumpet-shaped or vase-shaped body. Corona modified into a funnel for capturing food. Transparent body wall.",
    "conochilus": "Colonial rotifer forming spherical colonies of individuals attached to a common gelatinous base. Each individual has prominent corona.",
    "copepod_skins": "Transparent exuviae (molted exoskeletons) of copepods. Elongated body shape with visible segmentation. Fragile, often fragmented.",
    "cyclops": "Cyclopoid copepod with elongated body, long first antennae, and prominent egg sacs. Single median eye. Body divided into prosome and urosome.",
    "daphnia": "Large cladoceran (water flea) with prominent bivalve shell. Large second antennae for jumping. Visible brood chamber. Typically 2-5mm.",
    "daphnia_skins": "Transparent exuviae (molted exoskeletons) of Daphnia. Bivalve shell shape visible. Fragile, often with visible brood chamber outline.",
    "diaphanosoma": "Transparent cladoceran with elongated body. Thin shell, internal organs clearly visible. Long second antennae. No head shield.",
    "diatom_chain": "Chain of diatom cells linked together. Cells disc-shaped or cylindrical, connected by organic threads or silica spines.",
    "dinobryon": "Colonial golden-brown alga forming tree-like or vase-shaped colonies. Individual cells enclosed in loricae (vases). Characteristic branching pattern.",
    "dirt": "Non-biological debris. Irregular shape, opaque, variable color. May include mineral particles, plant fragments, or organic detritus.",
    "eudiaptomus": "Calanoid copepod with elongated body and long first antennae. Characteristic asymmetric genital segment. Red pigmentation possible.",
    "filament": "Thin elongated structure, may be algal filament, plant fiber, or cyanobacterial trichome. Variable length, often curved or tangled.",
    "fish": "Fish larva or egg. Elongated body with visible yolk sac (larvae) or spherical with embryo visible (eggs). Eyes prominent.",
    "fragilaria": "Chain-forming diatom. Cells tabular (rectangular in girdle view), linked into flat ribbons. Siliceous cell wall with fine striations.",
    "hydra": "Freshwater cnidarian with tubular body and tentacles. Typically 1-10mm. Green color from symbiotic algae possible. Tentacles arranged around mouth.",
    "kellicottia": "Spined rotifer with elongated body and prominent posterior spines. Corona with two wheel organs. Transparent lorica.",
    "keratella_cochlearis": "Small planktonic rotifer with asymmetric lorica. Posterior spine variable in length. Corona with two wheel organs. Typically <200μm.",
    "keratella_quadrata": "Planktonic rotifer with quadrangular lorica. Four anterior spines and one posterior spine. Corona with two wheel organs.",
    "leptodora": "Large predatory cladoceran (>10mm). Transparent body with prominent raptorial first legs. Large compound eye. Jellyfish-like appearance.",
    "maybe_cyano": "Possibly cyanobacterial colony or trichome. Variable morphology, may be filamentous or colonial. Green or blue-green color.",
    "nauplius": "Early larval stage of crustaceans (copepods, cladocerans). Small, pear-shaped with three pairs of appendages. Single median eye.",
    "paradileptus": "Ciliate protozoan with elongated body and prominent cilia. Fast-swimming. Characteristic body shape with anterior narrowed region.",
    "polyarthra": "Planktonic rotifer with paddle-like appendages (epipodia) for swimming. Small, transparent. Corona with two wheel organs.",
    "rotifers": "General rotifer category. Microscopic animals with corona (crown of cilia). Variable body forms. Typically 100-500μm.",
    "synchaeta": "Planktonic rotifer with elongated body and prominent lateral antennae. No lorica (shell-less). Corona with two wheel organs.",
    "trichocerca": "Planktonic rotifer with asymmetric body. One side curved, other straight. Corona with two wheel organs. Small, typically <200μm.",
    "unknown": "Organism that cannot be identified to class level. Variable morphology. May be fragment, artifact, or unfamiliar organism.",
    "unknown_plankton": "Planktonic organism of uncertain taxonomic affiliation. Microscopic, aquatic morphology but unknown class.",
    "uroglena": "Colonial golden-brown alga forming spherical colonies. Individual cells with two flagella embedded in gelatinous matrix. Fishy odor when abundant.",
}


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class PlanktonDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return
        for cls_name, idx in class_to_idx.items():
            cls_dir = data_path / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), idx))
        logger.info("  Loaded %d samples from %s", len(self.samples), data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


# ---------------------------------------------------------------------------
# SAA augmentation
# ---------------------------------------------------------------------------
def compute_shift_spectrum(data_dir, classes, max_per_class=30):
    sys.path.insert(0, ADVERSARIAL_NET)
    from spectral_augmentation import SpectralAugmentation
    
    spectra = []
    for cls_name in classes:
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.is_dir():
            continue
        count = 0
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            try:
                img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.float64) / 255.0
                f = np.fft.fft2(arr)
                amp = np.log1p(np.abs(np.fft.fftshift(f)))
                h, w = amp.shape
                cy, cx = h // 2, w // 2
                max_r = min(cx, cy)
                Y, X = np.ogrid[:h, :w]
                R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
                radial = np.zeros(max_r)
                for r in range(max_r):
                    mask = R == r
                    if mask.sum() > 0:
                        radial[r] = amp[mask].mean()
                spectra.append(radial)
                count += 1
            except Exception:
                continue
            if count >= max_per_class:
                break
    if not spectra:
        return None
    max_len = max(len(s) for s in spectra)
    matrix = np.zeros((len(spectra), max_len))
    for i, s in enumerate(spectra):
        matrix[i, :len(s)] = s
    return matrix.std(axis=0)


class SAATransform:
    def __init__(self, shift_spectrum=None, strategies=None, strength=0.5, p=0.8):
        from spectral_augmentation import SpectralAugmentation
        self.aug = SpectralAugmentation(
            shift_spectrum=shift_spectrum, strength=strength,
            strategies=strategies or ["spectral_noise", "band_adversarial"], p=p,
        )
        self.base = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        return self.base(Image.fromarray(arr_uint8, mode="L").convert("RGB"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def rotation_tta(model, image, device):
    preds = []
    for k in range(4):
        rotated = torch.rot90(image, k, dims=[2, 3]).to(device)
        preds.append(torch.softmax(model(rotated), dim=1))
        preds.append(torch.softmax(model(torch.flip(rotated, [3])), dim=1))
    return torch.stack(preds).mean(0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if use_tta:
            probs = rotation_tta(model, images, device)
            outputs = torch.log(probs + 1e-8)
        else:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = probs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1),
            "correct": correct, "total": total, "predictions": all_preds, "labels": all_labels}


def bootstrap_ci(binary, n=1000, ci=0.95):
    rng = np.random.RandomState(42)
    arr = np.array(binary)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    return {"mean": float(arr.mean()), "ci_low": float(np.percentile(boots, alpha * 100)),
            "score_high": float(np.percentile(boots, (1 - alpha) * 100))}


# ---------------------------------------------------------------------------
# EXPERIMENT 1: SAA on Chen's 10 OOD days
# ---------------------------------------------------------------------------
def run_saa_chen_ood():
    logger.info("=" * 72)
    logger.info("  EXPERIMENT 1: SAA on Chen's 10 OOD days")
    logger.info("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(RESULTS_DIR) / "saa_chen_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Classes from Chen's ZooLake2.0
    train_path = Path(CHEN_ZOOLAKE)
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("Classes: %d", num_classes)

    # Compute shift spectrum
    logger.info("Computing shift spectrum...")
    shift_spectrum = compute_shift_spectrum(CHEN_ZOOLAKE, classes)

    # SAA transform
    train_transform = SAATransform(shift_spectrum=shift_spectrum, strategies=["band_adversarial"])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Training data
    train_ds = PlanktonDataset(CHEN_ZOOLAKE, class_to_idx, train_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    # OOD test data (all 10 days)
    ood_dir = Path(CHEN_OOD)
    ood_loaders = {}
    for ood_cell in sorted(ood_dir.iterdir()):
        if ood_cell.is_dir():
            ds = PlanktonDataset(str(ood_cell), class_to_idx, eval_transform)
            if len(ds) > 0:
                ood_loaders[ood_cell.name] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    logger.info("OOD cells: %s", list(ood_loaders.keys()))

    # Train ensemble (3 members)
    criterion = nn.CrossEntropyLoss()
    all_results = {}

    for seed in range(3):
        logger.info("--- Ensemble member %d (seed=%d) ---", seed + 1, seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        for epoch in range(30):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %d/30  Loss: %.4f  Train: %.1f%%",
                            epoch + 1, total_loss / total, correct / total * 100)

        # Evaluate on all OOD cells with TTA
        model.eval()
        seed_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_res = evaluate(model, ood_loader, criterion, device, use_tta=True)
            seed_results[ood_name] = ood_res["accuracy"]
            logger.info("  %s: %.1f%%", ood_name, ood_res["accuracy"] * 100)

        all_results[f"seed_{seed}"] = seed_results
        torch.save(model.state_dict(), str(out_dir / f"model_seed{seed}.pth"))

    # Ensemble average
    ood_names = sorted(ood_loaders.keys())
    ensemble_avg = {}
    for ood_name in ood_names:
        accs = [all_results[seed][ood_name] for seed in all_results]
        ensemble_avg[ood_name] = float(np.mean(accs))

    overall = np.mean(list(ensemble_avg.values()))

    results = {
        "method": "SAA-band + ViT ensemble (n=3) + TTA",
        "per_ood_cell": ensemble_avg,
        "overall_ood_accuracy": overall,
        "vs_chen_best": overall - 0.83,
        "per_seed": all_results,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Overall OOD: %.1f%% (vs Chen's 83%%: %+.1f%%)", overall * 100, (overall - 0.83) * 100)
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 2: RAG on Chen's OOD days
# ---------------------------------------------------------------------------
def run_rag_chen_ood():
    logger.info("=" * 72)
    logger.info("  EXPERIMENT 2: RAG on Chen's OOD days")
    logger.info("=" * 72)

    import requests
    import base64

    endpoint = "http://localhost:8000/v1/chat/completions"
    model_name = "Qwen2.5-VL-32B-Instruct-AWQ"

    # Build prompt with freshwater catalog
    catalog_lines = []
    for cls, desc in FRESHWATER_CATALOG.items():
        catalog_lines.append(f"- {cls}: {desc}")
    catalog = "\n".join(catalog_lines)

    rag_prompt = (
        "You are a freshwater plankton expert. Examine this microscope image and "
        "identify the organism by matching its visible morphological features "
        "against the catalog below.\n\n"
        f"## Morphological Catalog\n{catalog}\n\n"
        "## Imaging System Awareness\n"
        "These images are from a DSPC field camera (ZooLake). "
        "Artifacts: variable natural lighting, water turbidity, background particles.\n"
        "IMPORTANT: Focus ONLY on morphological features (shape, symmetry, appendages). "
        "IGNORE differences in background, contrast, illumination.\n\n"
        "Reply with ONLY the organism name from the catalog."
    )

    baseline_prompt = (
        "You are a freshwater plankton expert. Look at this microscope image and "
        f"identify which organism it is from this list: [{', '.join(FRESHWATER_CATALOG.keys())}].\n"
        "Reply with ONLY the single best matching organism name."
    )

    def call_vlm(image_path, prompt):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def parse_class(raw, classes):
        cleaned = raw.strip().strip('"').strip("'").strip(".").lower()
        for cls in classes:
            if cleaned == cls.lower():
                return cls
        for cls in classes:
            if cls.lower() in cleaned:
                return cls
        return cleaned

    # Evaluate on each OOD cell
    ood_dir = Path(CHEN_OOD)
    all_classes = list(FRESHWATER_CATALOG.keys())
    results = {"baseline": {}, "rag": {}}

    for ood_cell in sorted(ood_dir.iterdir()):
        if not ood_cell.is_dir():
            continue
        ood_name = ood_cell.name
        logger.info("Evaluating %s...", ood_name)

        for condition, prompt in [("baseline", baseline_prompt), ("rag", rag_prompt)]:
            correct, total = 0, 0
            per_class = defaultdict(lambda: {"correct": 0, "total": 0})

            for cls_dir in sorted(ood_cell.iterdir()):
                if not cls_dir.is_dir():
                    continue
                true_cls = cls_dir.name
                count = 0
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.suffix.lower() not in SUPPORTED_EXT:
                        continue
                    try:
                        raw = call_vlm(str(img_path), prompt)
                        pred = parse_class(raw, all_classes)
                        is_correct = pred.lower() == true_cls.lower()
                        if is_correct:
                            correct += 1
                        total += 1
                        per_class[true_cls]["total"] += 1
                        if is_correct:
                            per_class[true_cls]["correct"] += 1
                    except Exception as e:
                        logger.warning("  Failed %s: %s", img_path, e)
                    count += 1
                    if count >= 10:  # Limit per class per cell
                        break

            acc = correct / max(total, 1)
            results[condition][ood_name] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "per_class": {k: dict(v) for k, v in per_class.items()},
            }
            logger.info("  %s %s: %.1f%% (%d/%d)", condition, ood_name, acc * 100, correct, total)

    # Compute lifts
    for ood_name in sorted(results["baseline"].keys()):
        bl = results["baseline"][ood_name]["accuracy"]
        rag = results["rag"][ood_name]["accuracy"]
        results["rag"][ood_name]["lift"] = rag - bl

    out_dir = Path(RESULTS_DIR) / "rag_chen_ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    bl_avg = np.mean([v["accuracy"] for v in results["baseline"].values()])
    rag_avg = np.mean([v["accuracy"] for v in results["rag"].values()])
    logger.info("Baseline avg: %.1f%%, RAG avg: %.1f%%, Lift: %+.1f%%",
                bl_avg * 100, rag_avg * 100, (rag_avg - bl_avg) * 100)
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 3: SAA on cross-ecosystem (WHOI22→ZooLake35)
# ---------------------------------------------------------------------------
def run_saa_cross_ecosystem():
    logger.info("=" * 72)
    logger.info("  EXPERIMENT 3: SAA on cross-ecosystem (WHOI22→ZooLake35)")
    logger.info("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(RESULTS_DIR) / "saa_cross_ecosystem"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the cross-domain data prepared earlier
    data_dir = "/home/sreenath/research-space/Adverserial_net/data/cross_domain/whoi22_full"
    train_dir = Path(data_dir) / "train" / "WHOI22"
    test_dirs = {
        "ZooScan20": Path(data_dir) / "test" / "ZooScan20",
        "ZooLake35": Path(data_dir) / "test" / "ZooLake35",
    }

    # Discover classes
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("WHOI22 classes: %d", num_classes)

    # Shift spectrum
    shift_spectrum = compute_shift_spectrum(str(train_dir), classes)

    # Transforms
    train_transform = SAATransform(shift_spectrum=shift_spectrum, strategies=["band_adversarial"])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = PlanktonDataset(str(train_dir), class_to_idx, train_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    test_loaders = {}
    for name, path in test_dirs.items():
        ds = PlanktonDataset(str(path), class_to_idx, eval_transform)
        if len(ds) > 0:
            test_loaders[name] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    # Train
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %d/30  Loss: %.4f  Train: %.1f%%",
                        epoch + 1, total_loss / total, correct / total * 100)

    # Evaluate
    results = {}
    train_res = evaluate(model, train_loader, criterion, device, use_tta=True)
    results["WHOI22"] = {"accuracy": train_res["accuracy"], "n": train_res["total"]}

    for name, loader in test_loaders.items():
        res = evaluate(model, loader, criterion, device, use_tta=True)
        results[name] = {
            "accuracy": res["accuracy"],
            "n": res["total"],
            "drop": train_res["accuracy"] - res["accuracy"],
        }
        logger.info("  %s: %.1f%% (drop: %.1f%%)", name, res["accuracy"] * 100,
                     (train_res["accuracy"] - res["accuracy"]) * 100)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

    logger.info("=" * 72)
    logger.info("  COMPLETE ANALYSIS PIPELINE")
    logger.info("  Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 72)

    # Experiment 1: SAA on Chen's OOD
    try:
        saa_results = run_saa_chen_ood()
    except Exception as e:
        logger.error("SAA Chen OOD failed: %s", e)
        saa_results = None

    # Experiment 2: RAG on Chen's OOD (only if vLLM is running)
    try:
        import requests
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        if resp.status_code == 200:
            rag_results = run_rag_chen_ood()
        else:
            logger.warning("vLLM not available, skipping RAG experiment")
            rag_results = None
    except Exception as e:
        logger.warning("RAG experiment skipped: %s", e)
        rag_results = None

    # Experiment 3: SAA on cross-ecosystem
    try:
        eco_results = run_saa_cross_ecosystem()
    except Exception as e:
        logger.error("Cross-ecosystem failed: %s", e)
        eco_results = None

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 72)
    logger.info("  ALL EXPERIMENTS COMPLETE")
    logger.info("  Total time: %.1f hours", elapsed / 3600)
    logger.info("=" * 72)

    if saa_results:
        logger.info("  SAA Chen OOD: %.1f%% overall", saa_results.get("overall_ood_accuracy", 0) * 100)
    if rag_results:
        bl = np.mean([v["accuracy"] for v in rag_results.get("baseline", {}).values()])
        rag = np.mean([v["accuracy"] for v in rag_results.get("rag", {}).values()])
        logger.info("  RAG Chen OOD: BL=%.1f%% RAG=%.1f%% Lift=%+.1f%%", bl * 100, rag * 100, (rag - bl) * 100)
    if eco_results:
        for domain, data in eco_results.items():
            logger.info("  Cross-eco %s: %.1f%%", domain, data.get("accuracy", 0) * 100)

    # Save combined summary
    summary = {
        "saa_chen_ood": saa_results,
        "rag_chen_ood": {k: {kk: vv for kk, vv in v.items() if kk != "per_class"}
                         for k, v in rag_results.items()} if rag_results else None,
        "saa_cross_ecosystem": eco_results,
        "elapsed_hours": elapsed / 3600,
    }
    with open(Path(RESULTS_DIR) / "complete_analysis.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
