"""
prepare_data_v2.py
==================
Improved data preparation with quality filtering.

Strategy:
- Download images >= 64px from Planktonzilla-17M
- Use classes that have sufficient resolution in BOTH domains
- Fall back to smaller threshold if needed for rare classes
"""

import argparse
import io
import json
import logging
import os
import random
import shutil

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Classes with good cross-domain representation
CLASSES = ["Chaetognatha", "Amphipoda", "Annelida", "Ceratium", "Calanoida", "Oithonidae"]

# IFCB label -> canonical
WHOI_LABELS = {
    0: "Amphidinium_sp", 1: "Asterionellopsis", 2: "Bacillaria",
    3: "Bidulphia", 4: "Cerataulina", 5: "Cerataulina_flagellate",
    6: "Ceratium", 7: "Chaetoceros", 8: "Chaetoceros_didymus",
    9: "Chaetoceros_didymus_flagellate", 10: "Chaetoceros_flagellate",
    11: "Chaetoceros_other", 12: "Chaetoceros_pennate",
    13: "Chrysochromulina", 14: "Ciliate_mix", 15: "Cochlodinium",
    16: "Corethron", 17: "Coscinodiscus", 18: "Cylindrotheca",
    19: "DactFragCerataul", 20: "Dactyliosolen", 21: "Delphineis",
    22: "Dictyocha", 23: "Didinium_sp", 24: "Dinobryon",
    25: "Dinophysis", 26: "Ditylum", 27: "Ditylum_parasite",
    28: "Emiliania_huxleyi", 29: "Ephemera", 30: "Eucampia",
    31: "Euglena", 32: "Euplotes_sp", 33: "G_delicatula_detritus",
    34: "G_delicatula_external_parasite", 35: "G_delicatula_parasite",
    36: "Gonyaulax", 37: "Guinardia_delicatula", 38: "Guinardia_flaccida",
    39: "Guinardia_striata", 40: "Gyrodinium", 41: "Hemiaulus",
    42: "Heterocapsa_triquetra", 43: "Katodinium_or_Torodinium",
    44: "Laboea_strobila", 45: "Lauderia", 46: "Leegaardiella_ovalis",
    47: "Leptocylindrus", 48: "Leptocylindrus_mediterraneus",
    49: "Licmophora", 50: "Mesodinium_sp", 51: "Odontella",
    52: "Paralia", 53: "Parvicorbicula_socialis", 54: "Phaeocystis",
    55: "Pleuronema_sp", 56: "Pleurosigma", 57: "Prorocentrum",
    58: "Proterythropsis_sp", 59: "Protoperidinium",
    60: "Pseudochattonella_farcimen", 61: "Pseudonitzschia",
    62: "Pyramimonas_longicauda", 63: "Rhizosolenia", 64: "Skeletonema",
    65: "Stephanopyxis", 66: "Strobilidium_morphotype1",
    67: "Strombidium_capitatum", 68: "Strombidium_conicum",
    69: "Strombidium_inclinatum", 70: "Strombidium_morphotype1",
    71: "Strombidium_morphotype2", 72: "Strombidium_oculatum",
    73: "Strombidium_wulffi", 74: "Thalassionema", 75: "Thalassiosira",
    76: "Thalassiosira_dirty", 77: "Tiarina_fusus", 78: "Tintinnid",
    79: "Tontonia_appendiculariformis", 80: "Tontonia_gracillima",
    81: "amoeba", 82: "bad", 83: "bead", 84: "bubble",
    85: "clusterflagellate", 86: "detritus", 87: "diatom_flagellate",
    88: "dino30", 89: "dino_large1", 90: "flagellate_sp3",
    91: "kiteflagellates", 92: "mix_elongated", 93: "other_interaction",
    94: "pennate", 95: "pennate_morphotype1", 96: "pennates_on_diatoms",
    97: "pollen", 98: "spore", 99: "zooplankton",
}

WHOI_TO_CANONICAL = {
    "Ceratium": "Ceratium",
}

WHOI_PZ_TO_CANONICAL = {
    "ceratium": "Ceratium",
    "chaetognatha": "Chaetognatha",
    "calanoida": "Calanoida",
    "oithonidae": "Oithonidae",
    "amphipoda": "Amphipoda",
    "annelida": "Annelida",
}

ZOOSCAN_TO_CANONICAL = {
    "chaetognatha": "Chaetognatha",
    "calanoida": "Calanoida",
    "oithonidae": "Oithonidae",
    "amphipoda": "Amphipoda",
    "annelida": "Annelida",
    "tripos": "Ceratium",
}

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_SIZE = 48  # More permissive threshold


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, "PNG")


def decode_parquet_image(img_data):
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"]))
    elif isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data))
    return None


def prepare_ifcb(output_dir, max_per_class, seed=42):
    random.seed(seed)
    counts = {}
    saved = 0

    # From WHOI-Plankton-small
    logger.info("Loading WHOI-Plankton-small for IFCB…")
    ds = load_dataset("nf-whoi/whoi-plankton-small", split="test")
    for sample in ds:
        label_name = WHOI_LABELS.get(sample["label"], "")
        canonical = WHOI_TO_CANONICAL.get(label_name)
        if canonical is None or canonical not in CLASSES:
            continue
        if counts.get(canonical, 0) >= max_per_class:
            continue
        img = sample["image"]
        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        idx = counts.get(canonical, 0)
        save_image(img, f"{output_dir}/IFCB/{canonical}/img_{idx:05d}.png")
        counts[canonical] = idx + 1
        saved += 1

    # From Planktonzilla-17M WHOI shards
    logger.info("Loading Planktonzilla-17M WHOI shards for additional IFCB classes…")
    for shard_id in range(120, 160):
        if all(counts.get(c, 0) >= max_per_class for c in CLASSES):
            break
        fname = f"data/train-{shard_id:05d}-of-00187.parquet"
        path = hf_hub_download("project-oceania/planktonzilla-17M", fname, repo_type="dataset")
        t = pq.read_table(path, columns=["image", "proposed_label"])
        for row_idx in range(len(t)):
            label = t.column("proposed_label")[row_idx].as_py()
            canonical = WHOI_PZ_TO_CANONICAL.get(label)
            if canonical is None or canonical not in CLASSES:
                continue
            if counts.get(canonical, 0) >= max_per_class:
                continue
            img = decode_parquet_image(t.column("image")[row_idx].as_py())
            if img is None:
                continue
            w, h = img.size
            if w < MIN_SIZE or h < MIN_SIZE:
                continue
            idx = counts.get(canonical, 0)
            save_image(img, f"{output_dir}/IFCB/{canonical}/img_{idx:05d}.png")
            counts[canonical] = idx + 1
            saved += 1

    logger.info("IFCB total: %d images across %d classes: %s", saved, len(counts), counts)
    return counts


def prepare_zooscan(output_dir, max_per_class, seed=42):
    random.seed(seed)
    counts = {}
    saved = 0

    for shard_id in range(172, 187):
        if all(counts.get(c, 0) >= max_per_class for c in CLASSES):
            break
        fname = f"data/train-{shard_id:05d}-of-00187.parquet"
        logger.info("Downloading ZooScan shard %d…", shard_id)
        path = hf_hub_download("project-oceania/planktonzilla-17M", fname, repo_type="dataset")
        t = pq.read_table(path, columns=["image", "proposed_label"])
        for row_idx in range(len(t)):
            label = t.column("proposed_label")[row_idx].as_py()
            canonical = ZOOSCAN_TO_CANONICAL.get(label)
            if canonical is None or canonical not in CLASSES:
                continue
            if counts.get(canonical, 0) >= max_per_class:
                continue
            img = decode_parquet_image(t.column("image")[row_idx].as_py())
            if img is None:
                continue
            w, h = img.size
            if w < MIN_SIZE or h < MIN_SIZE:
                continue
            idx = counts.get(canonical, 0)
            save_image(img, f"{output_dir}/ZooScan/{canonical}/img_{idx:05d}.png")
            counts[canonical] = idx + 1
            saved += 1

    logger.info("ZooScan total: %d images across %d classes: %s", saved, len(counts), counts)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/eval_v2")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    ifcb = prepare_ifcb(args.output_dir, args.max_per_class, args.seed)
    zooscan = prepare_zooscan(args.output_dir, args.max_per_class, args.seed)

    manifest = {
        "classes": CLASSES,
        "min_size": MIN_SIZE,
        "ifcb_classes": ifcb,
        "zooscan_classes": zooscan,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved.")


if __name__ == "__main__":
    main()
