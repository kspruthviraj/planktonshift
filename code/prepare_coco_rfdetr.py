"""
prepare_coco_rfdetr.py
======================
Prepare COCO-format dataset from v2a-segmented plankton data for RF-DETR training.

Creates:
  data_rfdetr/
    train/
      images/        ← cropped organisms (224×224)
      _annotations.json
    test/
      images/        ← OOD crops
      _annotations.json

Usage:
    python prepare_coco_rfdetr.py
"""

import json
import shutil
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}
ZOOlAKE2_DIR = "/home/sreenath/research-space/PlanktonShift/data_segmentation/zoolake2"
OOD_DIR = "/home/sreenath/research-space/PlanktonShift/data_segmentation/ood"
OUTPUT_DIR = "/home/sreenath/research-space/PlanktonShift/data_rfdetr"


def build_coco(split_dir, image_dir, class_to_id):
    """Build COCO annotations from segmented data."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k, "supercategory": "plankton"} for k, v in class_to_id.items()],
    }

    img_id = 1
    ann_id = 1
    images_path = Path(image_dir)
    out_img_dir = Path(split_dir) / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(images_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in class_to_id:
            continue

        for img_path in sorted(cls_dir.glob("*_crop.png")):
            # Copy image
            dst = out_img_dir / f"{img_id:06d}.png"
            shutil.copy2(img_path, dst)

            # Get image size
            img = Image.open(img_path)
            w, h = img.size

            # Read mask to get accurate bbox
            mask_path = str(img_path).replace("_crop.png", "_mask.png")
            if Path(mask_path).exists():
                import numpy as np
                mask = np.array(Image.open(mask_path))
                rows = np.any(mask > 127, axis=1)
                cols = np.any(mask > 127, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
                else:
                    bbox = [0, 0, w, h]
            else:
                bbox = [0, 0, w, h]

            coco["images"].append({
                "id": img_id,
                "file_name": f"images/{img_id:06d}.png",
                "width": w,
                "height": h,
            })
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })

            img_id += 1
            ann_id += 1

    # Save annotations
    ann_path = Path(split_dir) / "_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info("  %s: %d images, %d annotations, %d classes",
                split_dir, len(coco["images"]), len(coco["annotations"]), len(class_to_id))
    return coco


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoolake2-dir", type=str, default=ZOOlAKE2_DIR)
    parser.add_argument("--ood-dir", type=str, default=OOD_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    # Discover classes from ZooLake2
    classes = sorted([d.name for d in Path(args.zoolake2_dir).iterdir() if d.is_dir()])
    class_to_id = {c: i + 1 for i, c in enumerate(classes)}  # COCO categories are 1-indexed
    logger.info("Classes (%d): %s", len(classes), classes[:10])

    # Prepare train split (ZooLake2)
    logger.info("Preparing train split...")
    train_dir = Path(args.output_dir) / "train"
    build_coco(str(train_dir), args.zoolake2_dir, class_to_id)

    # Prepare test split (OOD — all days combined)
    logger.info("Preparing test split...")
    test_dir = Path(args.output_dir) / "test"

    # OOD has subdirectories for each day — combine all
    ood_combined = Path(args.output_dir) / "ood_combined"
    ood_combined.mkdir(parents=True, exist_ok=True)

    ood_path = Path(args.ood_dir)
    for ood_day in sorted(ood_path.iterdir()):
        if not ood_day.is_dir():
            continue
        for cls_dir in sorted(ood_day.iterdir()):
            if not cls_dir.is_dir():
                continue
            dst_cls = ood_combined / cls_dir.name
            dst_cls.mkdir(parents=True, exist_ok=True)
            for img_path in sorted(cls_dir.glob("*_crop.png")):
                # Copy with unique name to avoid collisions
                dst = dst_cls / f"{ood_day.name}_{img_path.name}"
                if not dst.exists():
                    import shutil
                    shutil.copy2(img_path, dst)

    build_coco(str(test_dir), str(ood_combined), class_to_id)

    # Also prepare per-OOD-day test sets for detailed analysis
    for ood_day in sorted(ood_path.iterdir()):
        if not ood_day.is_dir():
            continue
        day_test_dir = Path(args.output_dir) / f"test_{ood_day.name}"
        build_coco(str(day_test_dir), str(ood_day), class_to_id)

    logger.info("All splits prepared in %s", args.output_dir)


if __name__ == "__main__":
    main()
