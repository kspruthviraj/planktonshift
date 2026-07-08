"""
prepare_multi_domain.py
=======================
Download overlapping classes from multiple plankton datasets to create a
comprehensive multi-domain evaluation:

  Domain 1: IFCB-WHOI (already in data/eval_v2/IFCB)
  Domain 2: ZooScan (already in data/eval_v2/ZooScan)
  Domain 3: ZooLake (already in data/eval_v2/ZooLake)
  Domain 4: IFCB-NES (NES plankton, NOAA Northeast Shelf, DIFFERENT IFCB)
  Domain 5: PlanktonSet (NDSB, IFCB from Florida Straits, DIFFERENT IFCB)

Key insight: Domains 1, 4, 5 are ALL IFCB but from different deployments.
This tests whether RAG helps within the same modality across systems.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Target classes that appear in at least 2 datasets
TARGET_CLASSES = [
    "Amphipoda", "Annelida", "Appendicularia", "Calanoida", "Ceratium",
    "Chaetognatha", "Coscinodiscus", "Fragilaria", "Oithonidae",
]

# NES class name → canonical
NES_TO_CANONICAL = {
    "Tripos": "Ceratium", "Tripos_furca": "Ceratium", "Tripos_fusus": "Ceratium",
    "Tripos_lineatus": "Ceratium",
    "Coscinodiscus": "Coscinodiscus",
    "Chaetognatha": "Chaetognatha",  # if present
    "polychaete": "Annelida",
    "amphipods": "Amphipoda",
    "copepod_calanoid": "Calanoida",
    "copepod_cyclopoid_oithona": "Oithonidae",
    "diatom_chain_string": "Fragilaria", "diatom_chain_tube": "Fragilaria",
    "appendicularian_fritillaridae": "Appendicularia",
    "appendicularian_s_shape": "Appendicularia",
    "appendicularian_slight_curve": "Appendicularia",
    "appendicularian_straight": "Appendicularia",
}

# PlanktonSet class name → canonical
PS_TO_CANONICAL = {
    "amphipods": "Amphipoda",
    "chaetognath_non_sagitta": "Chaetognatha", "chaetognath_other": "Chaetognatha",
    "chaetognath_sagitta": "Chaetognatha",
    "copepod_calanoid": "Calanoida", "copepod_calanoid_eggs": "Calanoida",
    "copepod_calanoid_large": "Calanoida",
    "copepod_cyclopoid_oithona": "Oithonidae", "copepod_cyclopoid_oithona_eggs": "Oithonidae",
    "polychaete": "Annelida",
    "diatom_chain_string": "Fragilaria", "diatom_chain_tube": "Fragilaria",
    "appendicularian_fritillaridae": "Appendicularia",
    "appendicularian_s_shape": "Appendicularia",
    "appendicularian_slight_curve": "Appendicularia",
    "appendicularian_straight": "Appendicularia",
}

MIN_SIZE = 48


def save_sample(img, dst_dir, idx):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w < MIN_SIZE or h < MIN_SIZE:
        return False
    os.makedirs(dst_dir, exist_ok=True)
    img.save(f"{dst_dir}/img_{idx:05d}.png", "PNG")
    return True


def download_nes(output_dir, max_per_class, seed=42):
    """Download from NES plankton dataset (NOAA Northeast Shelf IFCB)."""
    random.seed(seed)
    logger.info("Loading NES plankton dataset…")
    ds = load_dataset("sbatchelder/NES-plankton-classifier-2022-dataset", split="train")
    
    counts = {}
    for sample in ds:
        label_name = ds.features["label"].int2str(sample["label"])
        canonical = NES_TO_CANONICAL.get(label_name)
        if canonical is None or canonical not in TARGET_CLASSES:
            continue
        if counts.get(canonical, 0) >= max_per_class:
            continue
        
        dst = Path(output_dir) / "IFCB-NES" / canonical
        if save_sample(sample["image"], dst, counts.get(canonical, 0)):
            counts[canonical] = counts.get(canonical, 0) + 1
    
    logger.info("IFCB-NES: %d images across %d classes: %s", sum(counts.values()), len(counts), counts)
    return counts


def download_planktonset(output_dir, max_per_class, seed=42):
    """Download from PlanktonSet 1.0 (NDSB, Florida Straits IFCB)."""
    random.seed(seed)
    logger.info("Loading PlanktonSet 1.0…")
    ds = load_dataset("project-oceania/planktonset1.0", split="train")
    
    counts = {}
    for sample in ds:
        label_name = ds.features["label"].int2str(sample["label"])
        canonical = PS_TO_CANONICAL.get(label_name)
        if canonical is None or canonical not in TARGET_CLASSES:
            continue
        if counts.get(canonical, 0) >= max_per_class:
            continue
        
        dst = Path(output_dir) / "PlanktonSet" / canonical
        if save_sample(sample["image"], dst, counts.get(canonical, 0)):
            counts[canonical] = counts.get(canonical, 0) + 1
    
    logger.info("PlanktonSet: %d images across %d classes: %s", sum(counts.values()), len(counts), counts)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/eval_v2")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nes_counts = download_nes(args.output_dir, args.max_per_class, args.seed)
    ps_counts = download_planktonset(args.output_dir, args.max_per_class, args.seed)

    # Update manifest
    manifest_path = f"{args.output_dir}/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest["ifcb_nes_classes"] = nes_counts
    manifest["ifcb_nes_source"] = "NES Plankton (NOAA, IFCB Northeast Shelf)"
    manifest["planktonset_classes"] = ps_counts
    manifest["planktonset_source"] = "PlanktonSet 1.0 (NDSB, IFCB Florida Straits)"
    manifest["target_classes"] = TARGET_CLASSES
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest updated with 5 domains.")


if __name__ == "__main__":
    main()
