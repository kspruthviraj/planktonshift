"""
train_rfdetr.py
===============
Train RF-DETR (small) on SAM-segmented plankton data.

Flow:
1. SAM segments organisms → bounding boxes + crops
2. RF-DETR trains on cropped organisms (background removed)
3. Evaluate on Chen's OOD benchmark

This should beat 83% because:
- Background artifacts (main source of domain shift) are removed
- RF-DETR is trained on clean organism crops
- At inference: SAM segments → RF-DETR classifies

Usage:
    python train_rfdetr.py \
        --data-dir data_segmentation/ZooLake2 \
        --ood-dir data_segmentation/OOD \
        --output-dir results/rfdetr
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}


def prepare_coco_dataset(data_dir, output_dir, split="train"):
    """Prepare COCO-format dataset from SAM segmentation output."""
    data_path = Path(data_dir)
    out_path = Path(output_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all crop images
    images = []
    categories = {}
    cat_id = 1

    for cls_dir in sorted(data_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name

        # Check if this is a class directory (has crop images)
        crops = list(cls_dir.glob("*_crop.png"))
        if not crops:
            # Might be a subdirectory (OOD days)
            for sub_dir in sorted(cls_dir.iterdir()):
                if sub_dir.is_dir():
                    for sub_cls in sorted(sub_dir.iterdir()):
                        if sub_cls.is_dir():
                            sub_crops = list(sub_cls.glob("*_crop.png"))
                            if sub_crops:
                                if sub_cls.name not in categories:
                                    categories[sub_cls.name] = cat_id
                                    cat_id += 1
                                for crop in sub_crops:
                                    images.append({
                                        "path": str(crop),
                                        "class": sub_cls.name,
                                        "cat_id": categories[sub_cls.name],
                                    })
            continue

        if cls_name not in categories:
            categories[cls_name] = cat_id
            cat_id += 1

        for crop in crops:
            images.append({
                "path": str(crop),
                "class": cls_name,
                "cat_id": categories[cls_name],
            })

    logger.info("Found %d images, %d classes for %s", len(images), len(categories), split)

    # Copy images and create COCO annotations
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in categories.items()],
    }

    import shutil
    for img_id, img_info in enumerate(images, 1):
        # Copy image
        dst = out_path / f"{img_id:06d}.png"
        shutil.copy2(img_info["path"], dst)

        # Get image size
        from PIL import Image
        im = Image.open(img_info["path"])
        w, h = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": f"{img_id:06d}.png",
            "width": w,
            "height": h,
        })
        coco["annotations"].append({
            "id": img_id,
            "image_id": img_id,
            "category_id": img_info["cat_id"],
            "bbox": [0, 0, w, h],  # Full image (organism fills the crop)
            "area": w * h,
            "iscrowd": 0,
        })

    # Save annotations
    ann_path = out_path.parent / f"{split}_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info("Saved %s: %d images, %d annotations → %s", split, len(images), len(coco["annotations"]), ann_path)
    return ann_path, categories


def train_rfdetr(train_ann, val_ann, categories, output_dir, epochs=50):
    """Train RF-DETR small model."""
    from rfdetr import RFDETRSmall

    num_classes = len(categories)
    logger.info("Training RF-DETR Small with %d classes", num_classes)

    model = RFDETRSmall()

    model.train(
        dataset_dir=str(Path(train_ann).parent),
        epochs=epochs,
        batch_size=16,
        grad_accum_steps=1,
        lr=1e-4,
        output_dir=str(output_dir),
        num_classes=num_classes,
    )

    return model


def evaluate_rfdetr(model, test_dir, categories, device="cuda"):
    """Evaluate RF-DETR on test images."""
    from PIL import Image
    
    test_path = Path(test_dir)
    cat_to_name = {v: k for k, v in categories.items()}
    
    correct, total = 0, 0
    per_class = {}
    
    for cls_dir in sorted(test_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for img_path in sorted(cls_dir.glob("*_crop.png")):
            try:
                # Predict
                detections = model.predict(str(img_path))
                
                if len(detections) > 0:
                    pred_id = int(detections.class_id[0])
                    pred_name = cat_to_name.get(pred_id, "unknown")
                else:
                    pred_name = "unknown"
                
                is_correct = pred_name.lower() == cls_name.lower()
                if is_correct:
                    correct += 1
                total += 1
                
                if cls_name not in per_class:
                    per_class[cls_name] = {"correct": 0, "total": 0}
                per_class[cls_name]["total"] += 1
                if is_correct:
                    per_class[cls_name]["correct"] += 1
                    
            except Exception as e:
                logger.warning("Failed %s: %s", img_path, e)
    
    accuracy = correct / max(total, 1)
    return accuracy, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_segmentation/ZooLake2")
    parser.add_argument("--ood-dir", type=str, default="data_segmentation/OOD")
    parser.add_argument("--output-dir", type=str, default="results/rfdetr")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare training data
    logger.info("Preparing training data...")
    train_ann, categories = prepare_coco_dataset(args.data_dir, args.output_dir, "train")

    # Prepare OOD test data
    logger.info("Preparing OOD test data...")
    test_ann, _ = prepare_coco_dataset(args.ood_dir, args.output_dir, "test")

    # Train
    logger.info("Training RF-DETR...")
    model = train_rfdetr(train_ann, test_ann, categories, args.output_dir, args.epochs)

    # Evaluate
    logger.info("Evaluating on OOD...")
    accuracy, per_class = evaluate_rfdetr(model, args.ood_dir + "/test", categories)

    logger.info("=" * 60)
    logger.info("  RF-DETR Results")
    logger.info("=" * 60)
    logger.info("  OOD Accuracy: %.1f%%", accuracy * 100)
    logger.info("  Chen BEsT: 83.0%%")
    logger.info("  vs Chen: %+.1f%%", accuracy * 100 - 83)
    for cls, stats in sorted(per_class.items(), key=lambda x: -x[1]["correct"]/max(x[1]["total"],1)):
        acc = stats["correct"] / max(stats["total"], 1)
        logger.info("    %-25s %.1f%% (n=%d)", cls, acc * 100, stats["total"])

    # Save
    results = {
        "accuracy": accuracy,
        "per_class": per_class,
        "vs_chen": accuracy - 0.83,
    }
    with open(f"{args.output_dir}/rfdetr_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
