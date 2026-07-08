"""
prepare_zoolake.py
==================
Prepare ZooLake data as a third evaluation domain (freshwater DSPC camera).

Uses ZooLake35 from Eawag with classes that overlap with IFCB (Ceratium) plus
freshwater-specific classes for cross-taxonomy evaluation.
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ZOOLAKE35_DIR = "/home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed"

# Classes from ZooLake35 that we'll use:
# - ceratium: overlaps with IFCB (cross-system validation)
# - daphnia, bosmina, conochilus, keratella-quadrata: freshwater zooplankton
# - fragilaria: freshwater diatom (overlaps with WHOI diatoms at genus level)
CLASSES = ["ceratium", "daphnia", "bosmina", "conochilus", "keratella-quadrata", "fragilaria"]

# Canonical name mapping
CANONICAL = {
    "ceratium": "Ceratium",
    "daphnia": "Daphnia",
    "bosmina": "Bosmina",
    "conochilus": "Conochilus",
    "keratella-quadrata": "Keratella",
    "fragilaria": "Fragilaria",
}

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_SIZE = 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/eval_v2")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    counts = {}

    for cls_name in CLASSES:
        canonical = CANONICAL[cls_name]
        src_dir = Path(ZOOLAKE35_DIR) / cls_name
        if not src_dir.is_dir():
            logger.warning("Class directory not found: %s", src_dir)
            continue

        # Get all valid images
        valid_imgs = []
        for img_path in sorted(src_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            try:
                img = Image.open(img_path)
                w, h = img.size
                if w >= MIN_SIZE and h >= MIN_SIZE:
                    valid_imgs.append(img_path)
            except Exception:
                continue

        # Sample up to max_per_class
        if len(valid_imgs) > args.max_per_class:
            valid_imgs = random.sample(valid_imgs, args.max_per_class)

        # Copy to output
        dst_dir = Path(args.output_dir) / "ZooLake" / canonical
        dst_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_path in enumerate(valid_imgs):
            img = Image.open(img_path).convert("RGB")
            img.save(dst_dir / f"img_{idx:05d}.png", "PNG")

        counts[canonical] = len(valid_imgs)
        logger.info("ZooLake/%s: %d images saved", canonical, len(valid_imgs))

    # Update manifest
    manifest_path = f"{args.output_dir}/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest["zoolake_classes"] = counts
    manifest["zoolake_source"] = "ZooLake35 (Eawag, DSPC field camera)"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Total ZooLake: %d images across %d classes", sum(counts.values()), len(counts))
    logger.info("Manifest updated.")


if __name__ == "__main__":
    main()
