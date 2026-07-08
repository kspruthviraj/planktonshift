"""
prepare_data.py
===============
Download and prepare evaluation subsets from two domains:

  - IFCB domain:   WHOI-Plankton-small (nf-whoi/whoi-plankton-small)
  - ZooScan domain: Planktonzilla-17M ZooScan shards (project-oceania/planktonzilla-17M)

Saves images to disk organized as:
    data/eval/
        IFCB/<class>/img_00000.png ...
        ZooScan/<class>/img_00000.png ...
"""

import argparse
import io
import json
import logging
import os
import random

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# WHOI-Plankton label map (IFCB domain)
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

# Mapping from WHOI label -> canonical class name
WHOI_TO_CANONICAL = {
    "Coscinodiscus": "Coscinodiscus",
    "Ceratium": "Ceratium",
    "Chaetoceros": "Chaetoceros",
    "Chaetoceros_didymus": "Chaetoceros",
    "Chaetoceros_other": "Chaetoceros",
    "Thalassiosira": "Thalassiosira",
    "Protoperidinium": "Protoperidinium",
    "Rhizosolenia": "Rhizosolenia",
    "Pseudonitzschia": "Pseudo-nitzschia",
    "Guinardia_delicatula": "Guinardia",
    "Guinardia_flaccida": "Guinardia",
    "Guinardia_striata": "Guinardia",
    "Euplotes_sp": "Euplotes",
    "Strombidium_capitatum": "Strombidium",
    "Strombidium_conicum": "Strombidium",
    "Strombidium_oculatum": "Strombidium",
    "Strombidium_inclinatum": "Strombidium",
    "Strombidium_wulffi": "Strombidium",
    "Mesodinium_sp": "Mesodinium",
    "Tintinnid": "Tintinnopsis",
}

# Mapping from Planktonzilla-17M WHOI proposed_label -> canonical class name
# These labels come from shards 120-159 (the 'whoi' domain in Planktonzilla-17M)
WHOI_PZ_TO_CANONICAL = {
    "coscinodiscus": "Coscinodiscus",
    "ceratium": "Ceratium",
    "chaetognatha": "Chaetognatha",
    "calanoida": "Calanoida",
    "oithonidae": "Oithonidae",
    "appendicularia": "Appendicularia",
    "ostracoda": "Ostracoda",
    "doliolida": "Doliolida",
    "salpida": "Salpida",
    "amphipoda": "Amphipoda",
    "annelida": "Annelida",
    "noctiluca": "Noctiluca",
}

# Mapping from ZooScan proposed_label -> canonical class name
# These are the labels found in Planktonzilla-17M ZooScan shards (172-186)
ZOOSCAN_TO_CANONICAL = {
    "coscinodiscus": "Coscinodiscus",
    "noctiluca": "Noctiluca",
    "tripos": "Ceratium",  # Ceratium was reclassified as Tripos
    "chaetognatha": "Chaetognatha",
    "calanoida": "Calanoida",
    "oithonidae": "Oithonidae",
    "appendicularia": "Appendicularia",
    "ostracoda": "Ostracoda",
    "doliolida": "Doliolida",
    "salpida": "Salpida",
    "amphipoda": "Amphipoda",
    "annelida": "Annelida",
}

# Canonical classes present in BOTH domains (true cross-domain overlap)
OVERLAPPING_CLASSES = {
    "Coscinodiscus",
    "Ceratium",
    "Chaetognatha",
    "Calanoida",
    "Oithonidae",
    "Appendicularia",
    "Ostracoda",
    "Doliolida",
    "Salpida",
    "Amphipoda",
    "Annelida",
    "Noctiluca",
}


def save_image(image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, "PNG")


def prepare_ifcb(output_dir: str, max_per_class: int, seed: int = 42) -> dict:
    """Download WHOI-Plankton-small (IFCB domain) and save overlapping classes."""
    logger.info("Loading WHOI-Plankton dataset (IFCB domain)…")
    ds = load_dataset("nf-whoi/whoi-plankton-small", split="test")

    random.seed(seed)
    class_counts: dict[str, int] = {}
    saved = 0

    for sample in ds:
        label_id = sample["label"]
        label_name = WHOI_LABELS.get(label_id, f"unknown_{label_id}")
        canonical = WHOI_TO_CANONICAL.get(label_name)
        if canonical is None or canonical not in OVERLAPPING_CLASSES:
            continue
        if class_counts.get(canonical, 0) >= max_per_class:
            continue

        idx = class_counts.get(canonical, 0)
        img_path = os.path.join(output_dir, "IFCB", canonical, f"img_{idx:05d}.png")
        save_image(sample["image"], img_path)
        class_counts[canonical] = idx + 1
        saved += 1

    logger.info("IFCB (WHOI-small): saved %d images across %d classes", saved, len(class_counts))

    # Also pull zooplankton classes from Planktonzilla-17M WHOI shards (120-159)
    # to increase cross-domain overlap
    logger.info("Loading additional IFCB data from Planktonzilla-17M WHOI shards…")
    for shard_id in range(120, 160):
        needed = any(
            class_counts.get(cls, 0) < max_per_class
            for cls in OVERLAPPING_CLASSES
        )
        if not needed:
            break

        fname = f"data/train-{shard_id:05d}-of-00187.parquet"
        logger.info("Downloading IFCB shard %d (%s)…", shard_id, fname)
        path = hf_hub_download(
            "project-oceania/planktonzilla-17M", fname, repo_type="dataset"
        )

        t = pq.read_table(path, columns=["image", "proposed_label"])
        for row_idx in range(len(t)):
            raw_label = t.column("proposed_label")[row_idx].as_py()
            canonical = WHOI_PZ_TO_CANONICAL.get(raw_label)
            if canonical is None or canonical not in OVERLAPPING_CLASSES:
                continue
            if class_counts.get(canonical, 0) >= max_per_class:
                continue

            img_data = t.column("image")[row_idx].as_py()
            if isinstance(img_data, dict) and "bytes" in img_data:
                image = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, bytes):
                image = Image.open(io.BytesIO(img_data))
            else:
                continue

            idx = class_counts.get(canonical, 0)
            img_path = os.path.join(output_dir, "IFCB", canonical, f"img_{idx:05d}.png")
            save_image(image, img_path)
            class_counts[canonical] = idx + 1
            saved += 1

        logger.info("  IFCB shard %d done. Total saved: %d", shard_id, saved)

    logger.info("IFCB total: saved %d images across %d classes", saved, len(class_counts))
    return class_counts


def prepare_zooscan(output_dir: str, max_per_class: int, seed: int = 42) -> dict:
    """Download ZooScan data from Planktonzilla-17M shards 172-186."""
    logger.info("Loading ZooScan data from Planktonzilla-17M…")
    random.seed(seed)

    class_counts: dict[str, int] = {}
    saved = 0

    for shard_id in range(172, 187):
        # Check if we already have enough for all classes
        needed = any(
            class_counts.get(cls, 0) < max_per_class
            for cls in OVERLAPPING_CLASSES
        )
        if not needed:
            break

        fname = f"data/train-{shard_id:05d}-of-00187.parquet"
        logger.info("Downloading shard %d (%s)…", shard_id, fname)
        path = hf_hub_download(
            "project-oceania/planktonzilla-17M", fname, repo_type="dataset"
        )

        t = pq.read_table(path, columns=["image", "proposed_label"])

        for row_idx in range(len(t)):
            raw_label = t.column("proposed_label")[row_idx].as_py()
            canonical = ZOOSCAN_TO_CANONICAL.get(raw_label)
            if canonical is None or canonical not in OVERLAPPING_CLASSES:
                continue
            if class_counts.get(canonical, 0) >= max_per_class:
                continue

            # Decode image from parquet binary
            img_data = t.column("image")[row_idx].as_py()
            if isinstance(img_data, dict) and "bytes" in img_data:
                image = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, bytes):
                image = Image.open(io.BytesIO(img_data))
            else:
                continue

            idx = class_counts.get(canonical, 0)
            img_path = os.path.join(
                output_dir, "ZooScan", canonical, f"img_{idx:05d}.png"
            )
            save_image(image, img_path)
            class_counts[canonical] = idx + 1
            saved += 1

        logger.info(
            "  Shard %d done. Total ZooScan saved so far: %d", shard_id, saved
        )

    logger.info("ZooScan: saved %d images across %d classes", saved, len(class_counts))
    return class_counts


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation data subsets.")
    parser.add_argument("--output-dir", type=str, default="data/eval")
    parser.add_argument(
        "--max-per-class", type=int, default=50, help="Max images per class per domain."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ifcb_counts = prepare_ifcb(args.output_dir, args.max_per_class, args.seed)
    zooscan_counts = prepare_zooscan(args.output_dir, args.max_per_class, args.seed)

    manifest = {
        "ifcb_classes": ifcb_counts,
        "zooscan_classes": zooscan_counts,
        "overlapping_classes": sorted(OVERLAPPING_CLASSES),
        "max_per_class": args.max_per_class,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
