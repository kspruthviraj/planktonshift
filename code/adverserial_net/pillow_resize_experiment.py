"""
pillow_resize_experiment.py
===========================
Process plankton images with the CURRENT Pillow version, saving numpy arrays
for later comparison. Does NOT specify resampling filter in resize() — this
is the key: Pillow 6.x defaults to NEAREST, Pillow 7.x defaults to BICUBIC.

Run this script from TWO different venvs:
  - .venv_pillow6 (Pillow 6.2.2): uses NEAREST by default
  - .venv_pillow7 (Pillow 7.0.0): uses BICUBIC by default

Usage:
    .venv_pillow6/bin/python pillow_resize_experiment.py --data-dir ... --output-dir results/pillow_noise/v6.2.2
    .venv_pillow7/bin/python pillow_resize_experiment.py --data-dir ... --output-dir results/pillow_noise/v7.0.0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    import PIL
    logger.info("Pillow version: %s", PIL.__version__)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for cls_dir in sorted(Path(args.data_dir).iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED_EXT:
                images.append((cls_dir.name, img_path))
            if len(images) >= args.max_images:
                break

    results = []
    for cls_name, img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            # CRITICAL: no resampling filter specified
            # Pillow 6.x: Image.NEAREST (nearest neighbor)
            # Pillow 7.x: Image.BICUBIC (bicubic convolution)
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img_resized, dtype=np.float64) / 255.0
            np.save(str(out_dir / f"{cls_name}_{img_path.stem}.npy"), arr)
            results.append({"class": cls_name, "file": img_path.name})
        except Exception as e:
            logger.warning("Failed %s: %s", img_path, e)

    metadata = {
        "pillow_version": PIL.__version__,
        "data_dir": args.data_dir,
        "img_size": IMG_SIZE,
        "images": results,
        "resize_filter": "NEAREST (default)" if PIL.__version__ < "7.0.0" else "BICUBIC (default)",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved %d images to %s", len(results), args.output_dir)


if __name__ == "__main__":
    main()
