"""
train_rfdetr_seg.py
===================
Train RF-DETR-Seg Medium on v2a-segmented plankton data.

Uses COCO-format dataset from data_rfdetr/ (prepared by prepare_coco_rfdetr.py).

Usage:
    python train_rfdetr_seg.py \
        --dataset-dir data_rfdetr \
        --epochs 50 \
        --batch-size 8 \
        --output-dir results/rfdetr_seg_medium
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rfdetr_seg.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}


def evaluate_rfdetr(model, test_dir, class_names, device="cuda"):
    """Evaluate RF-DETR-Seg on test images."""
    from PIL import Image

    test_path = Path(test_dir)
    images_dir = test_path / "images"
    ann_path = test_path / "_annotations.coco.json"

    if not ann_path.exists():
        logger.warning("No annotations found at %s", ann_path)
        return 0, {}

    with open(ann_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    correct, total = 0, 0
    per_class = {}

    for img_info in coco["images"]:
        img_path = images_dir / Path(img_info["file_name"]).name
        if not img_path.exists():
            continue

        # Get ground truth
        ann = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
        if not ann:
            continue
        true_cat = ann[0]["category_id"]
        true_name = cat_id_to_name.get(true_cat, "unknown")

        try:
            detections = model.predict(str(img_path))

            if len(detections) > 0:
                pred_id = int(detections.class_id[0])
                pred_name = cat_id_to_name.get(pred_id, "unknown")
            else:
                pred_name = "unknown"

            is_correct = pred_name.lower() == true_name.lower()
            if is_correct:
                correct += 1
            total += 1

            if true_name not in per_class:
                per_class[true_name] = {"correct": 0, "total": 0}
            per_class[true_name]["total"] += 1
            if is_correct:
                per_class[true_name]["correct"] += 1

        except Exception as e:
            logger.warning("Failed %s: %s", img_path, e)

    accuracy = correct / max(total, 1)
    return accuracy, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="data_rfdetr")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="results/rfdetr_seg_medium")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load class names from annotations
    ann_path = Path(args.dataset_dir) / "train" / "_annotations.coco.json"
    with open(ann_path) as f:
        train_coco = json.load(f)
    class_names = [c["name"] for c in sorted(train_coco["categories"], key=lambda x: x["id"])]
    num_classes = len(class_names)
    logger.info("Classes (%d): %s", num_classes, class_names[:10])

    # Create RF-DETR-Seg Medium
    from rfdetr import RFDETRSegMedium
    logger.info("Creating RF-DETR-Seg Medium...")
    model = RFDETRSegMedium()
    logger.info("Model created")

    # Train
    logger.info("Starting training: %d epochs, batch=%d, lr=%s, resolution=%d",
                args.epochs, args.batch_size, args.lr, args.resolution)

    train_dir = str(Path(args.dataset_dir))

    model.train(
        dataset_dir=train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=4,
        lr=args.lr,
        output_dir=args.output_dir,
        num_classes=num_classes,
        resolution=args.resolution,
        checkpoint_interval=5,
        multi_scale=False,  # Fixed resolution for consistency
        do_random_resize_via_padding=False,
    )

    logger.info("Training complete!")

    # Evaluate on all OOD test sets
    logger.info("=" * 60)
    logger.info("  Evaluating on OOD test sets")
    logger.info("=" * 60)

    test_dir = Path(args.dataset_dir) / "test"
    accuracy, per_class = evaluate_rfdetr(model, str(test_dir), class_names, args.device)

    logger.info("  Overall OOD accuracy: %.1f%%", accuracy * 100)
    logger.info("  Chen's BEsT: 83.0%%")
    logger.info("  vs Chen: %+.1f%%", accuracy * 100 - 83)

    # Per-class results
    logger.info("  Per-class OOD accuracy:")
    for cls, stats in sorted(per_class.items(), key=lambda x: -x[1]["correct"] / max(x[1]["total"], 1)):
        acc = stats["correct"] / max(stats["total"], 1)
        logger.info("    %-25s  %.1f%%  (n=%d)", cls, acc * 100, stats["total"])

    # Per-OOD-day results
    logger.info("  Per-OOD-day accuracy:")
    per_day = {}
    for ood_day in sorted(Path(args.dataset_dir).glob("test_OOD*")):
        day_name = ood_day.name.replace("test_", "")
        day_acc, day_cls = evaluate_rfdetr(model, str(ood_day), class_names, args.device)
        per_day[day_name] = {"accuracy": day_acc, "n": sum(s["total"] for s in day_cls.values())}
        logger.info("    %s: %.1f%%  (n=%d)", day_name, day_acc * 100, per_day[day_name]["n"])

    # Save results
    results = {
        "model": "RF-DETR-Seg Medium",
        "augmentation": "v2a segmentation (background removed)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "num_classes": num_classes,
        "overall_ood_accuracy": accuracy,
        "per_class": per_class,
        "per_day": per_day,
        "vs_chen_best": accuracy - 0.83,
    }

    with open(Path(args.output_dir) / "rfdetr_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s/rfdetr_results.json", args.output_dir)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    logger.info("  RF-DETR-Seg Medium + v2a segmentation")
    logger.info("  OOD Accuracy: %.1f%%", accuracy * 100)
    logger.info("  Chen's BEsT: 83.0%%")
    logger.info("  BEiT + SAA: 82.1%%")
    logger.info("  vs Chen: %+.1f%%", accuracy * 100 - 83)
    logger.info("  vs BEiT+SAA: %+.1f%%", accuracy * 100 - 82.1)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
