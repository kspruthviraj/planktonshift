"""
prepare_coco_rfdetr_seg.py
==========================
Prepare COCO-format dataset with SEGMENTATION MASKS for RF-DETR-Seg.

RF-DETR-Seg requires:
- images/ directory with images
- _annotations.coco.json with:
  - images: [{id, file_name, width, height}]
  - annotations: [{id, image_id, category_id, bbox, area, iscrowd, segmentation}]
  - categories: [{id, name}]

segmentation must be in RLE format (from pycocotools).

Usage:
    python prepare_coco_rfdetr_seg.py
"""

import json
import shutil
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}
ZOOlAKE2_DIR = "/home/sreenath/research-space/PlanktonShift/data_segmentation/zoolake2"
OOD_DIR = "/home/sreenath/research-space/PlanktonShift/data_segmentation/ood"
OUTPUT_DIR = "/home/sreenath/research-space/PlanktonShift/data_rfdetr_seg"


def mask_to_rle(mask):
    """Convert binary mask to COCO RLE format."""
    from pycocotools import mask as mask_utils
    # mask should be 2D numpy array of 0s and 1s
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    # Convert bytes to list for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_coco(split_dir, image_dir, class_to_id, include_masks=True):
    """Build COCO annotations with segmentation masks."""
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

            # Read mask
            mask_path = str(img_path).replace("_crop.png", "_mask.png")
            bbox = [0, 0, w, h]
            segmentation = None
            area = w * h

            if Path(mask_path).exists():
                mask = np.array(Image.open(mask_path))
                mask_binary = (mask > 127).astype(np.uint8)

                # Get accurate bbox from mask
                rows = np.any(mask_binary, axis=1)
                cols = np.any(mask_binary, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
                    area = int(mask_binary.sum())

                # Convert mask to RLE for segmentation
                if include_masks:
                    try:
                        segmentation = mask_to_rle(mask_binary)
                    except Exception:
                        segmentation = None

            coco["images"].append({
                "id": img_id,
                "file_name": f"images/{img_id:06d}.png",
                "width": w,
                "height": h,
            })

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
            if segmentation is not None:
                ann["segmentation"] = segmentation
            else:
                # Empty segmentation as fallback
                ann["segmentation"] = []

            coco["annotations"].append(ann)
            img_id += 1
            ann_id += 1

    # Save annotations
    ann_path = Path(split_dir) / "_annotations.coco.json"
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
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    # Clean old output
    import shutil as sh
    if Path(args.output_dir).exists():
        sh.rmtree(args.output_dir)

    # Discover classes
    classes = sorted([d.name for d in Path(args.zoolake2_dir).iterdir() if d.is_dir()])
    class_to_id = {c: i + 1 for i, c in enumerate(classes)}
    logger.info("Classes (%d): %s", len(classes), classes[:10])

    # Build train + valid from ZooLake2 (80/20 split)
    logger.info("Preparing train + valid splits...")
    random.seed(42)

    # First build full train
    full_train_dir = Path(args.output_dir) / "full_train"
    build_coco(str(full_train_dir), args.zoolake2_dir, class_to_id)

    # Load and split
    with open(full_train_dir / "_annotations.coco.json") as f:
        full_coco = json.load(f)

    images = full_coco["images"]
    random.shuffle(images)
    split = int(len(images) * args.train_ratio)
    train_imgs = images[:split]
    val_imgs = images[split:]
    train_ids = {img["id"] for img in train_imgs}
    val_ids = {img["id"] for img in val_imgs}

    # Create train directory
    train_dir = Path(args.output_dir) / "train"
    train_img_dir = train_dir / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)

    # Create valid directory
    valid_dir = Path(args.output_dir) / "valid"
    valid_img_dir = valid_dir / "images"
    valid_img_dir.mkdir(parents=True, exist_ok=True)

    # Move images to correct splits
    for img in train_imgs:
        src = full_train_dir / img["file_name"]
        dst = train_img_dir / Path(img["file_name"]).name
        if src.exists():
            shutil.copy2(str(src), str(dst))

    for img in val_imgs:
        src = full_train_dir / img["file_name"]
        dst = valid_img_dir / Path(img["file_name"]).name
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # Save split annotations
    train_anns = [a for a in full_coco["annotations"] if a["image_id"] in train_ids]
    val_anns = [a for a in full_coco["annotations"] if a["image_id"] in val_ids]

    train_coco = {"images": train_imgs, "annotations": train_anns, "categories": full_coco["categories"]}
    val_coco = {"images": val_imgs, "annotations": val_anns, "categories": full_coco["categories"]}

    with open(train_dir / "_annotations.coco.json", "w") as f:
        json.dump(train_coco, f, indent=2)
    with open(valid_dir / "_annotations.coco.json", "w") as f:
        json.dump(val_coco, f, indent=2)

    logger.info("  Train: %d images, %d annotations", len(train_imgs), len(train_anns))
    logger.info("  Valid: %d images, %d annotations", len(val_imgs), len(val_anns))

    # Clean up full_train
    shutil.rmtree(str(full_train_dir))

    # Build test split (OOD — all days combined)
    logger.info("Preparing test split...")
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
                dst = dst_cls / f"{ood_day.name}_{img_path.name}"
                if not dst.exists():
                    shutil.copy2(img_path, dst)

    test_dir = Path(args.output_dir) / "test"
    build_coco(str(test_dir), str(ood_combined), class_to_id)

    # Clean up ood_combined
    shutil.rmtree(str(ood_combined))

    logger.info("All splits prepared in %s", args.output_dir)


if __name__ == "__main__":
    main()
