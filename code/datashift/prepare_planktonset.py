"""
prepare_planktonset.py
======================
Download PlanktonSet 1.0 (National Data Science Bowl, IFCB from F.G. Walton Smith)
as a fourth domain. This is a DIFFERENT IFCB deployment than WHOI, providing
within-modality cross-system evaluation.

Overlapping classes with our existing taxonomy:
  amphipods → Amphipoda
  chaetognath_* → Chaetognatha
  copepod_calanoid → Calanoida
  copepod_cyclopoid_oithona → Oithonidae
  polychaete → Annelida
  diatom_chain_* → Fragilaria
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# PlanktonSet class name → canonical name
PS_TO_CANONICAL = {
    "amphipods": "Amphipoda",
    "chaetognath_non_sagitta": "Chaetognatha",
    "chaetognath_other": "Chaetognatha",
    "chaetognath_sagitta": "Chaetognatha",
    "copepod_calanoid": "Calanoida",
    "copepod_calanoid_eggs": "Calanoida",
    "copepod_calanoid_large": "Calanoida",
    "copepod_calanoid_small_longantennae": "Calanoida",
    "copepod_cyclopoid_oithona": "Oithonidae",
    "copepod_cyclopoid_oithona_eggs": "Oithonidae",
    "polychaete": "Annelida",
    "diatom_chain_string": "Fragilaria",
    "diatom_chain_tube": "Fragilaria",
    "protist_noctiluca": "Noctiluca",
    "tunicate_doliolid": "Doliolida",
    "tunicate_doliolid_nurse": "Doliolida",
    "tunicate_salp": "Salpida",
    "tunicate_salp_chains": "Salpida",
    "appendicularian_fritillaridae": "Appendicularia",
    "appendicularian_s_shape": "Appendicularia",
    "appendicularian_slight_curve": "Appendicularia",
    "appendicularian_straight": "Appendicularia",
    "euphausiids": "Euphausiacea",
    "euphausiids_young": "Euphausiacea",
    "stomatopod": "Stomatopoda",
    "decapods": "Decapoda",
    "ctenophore_cestid": "Ctenophora",
    "ctenophore_cydippid_no_tentacles": "Ctenophora",
    "ctenophore_cydippid_tentacles": "Ctenophora",
    "ctenophore_lobate": "Ctenophora",
}

CANONICAL_CLASSES = sorted(set(PS_TO_CANONICAL.values()))
MIN_SIZE = 48
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/eval_v2")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("Loading PlanktonSet 1.0 from HuggingFace…")
    ds = load_dataset("project-oceania/planktonset1.0", split="train")

    # Count label distribution
    label_counts = {}
    for sample in ds:
        label = sample["label"]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Map to canonical
    canonical_counts = {}
    saved_per_class = {}

    for sample in ds:
        label_id = sample["label"]
        label_name = ds.features["label"].int2str(label_id)

        canonical = PS_TO_CANONICAL.get(label_name)
        if canonical is None:
            continue
        if saved_per_class.get(canonical, 0) >= args.max_per_class:
            continue

        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            continue

        dst_dir = Path(args.output_dir) / "PlanktonSet" / canonical
        dst_dir.mkdir(parents=True, exist_ok=True)
        idx = saved_per_class.get(canonical, 0)
        img.save(dst_dir / f"img_{idx:05d}.png", "PNG")
        saved_per_class[canonical] = idx + 1

    # Report
    total = sum(saved_per_class.values())
    logger.info("PlanktonSet: %d images across %d classes", total, len(saved_per_class))
    for cls, n in sorted(saved_per_class.items()):
        logger.info("  %s: %d", cls, n)

    # Update manifest
    manifest_path = f"{args.output_dir}/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest["planktonset_classes"] = saved_per_class
    manifest["planktonset_source"] = "PlanktonSet 1.0 (NDSB, IFCB F.G. Walton Smith, Florida Straits)"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest updated.")


if __name__ == "__main__":
    main()
