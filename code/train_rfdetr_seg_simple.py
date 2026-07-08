"""
train_rfdetr_seg_simple.py
===========================
RF-DETR-Seg training using the same approach as the eagle project.

Key differences from previous attempt:
1. Uses default resolution (no explicit override)
2. Simpler API (just dataset_dir, epochs, batch_size)
3. Lets RF-DETR auto-detect num_classes from dataset
4. Uses Roboflow COCO format (auto-detected)

Usage:
    python train_rfdetr_seg_simple.py \
        --dataset-dir data_rfdetr_seg \
        --model-size nano \
        --epochs 50 \
        --output-dir results/rfdetr_seg_nano
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rfdetr_simple.log")],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="data_rfdetr_seg")
    parser.add_argument("--model-size", type=str, default="nano",
                        choices=["nano", "small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results/rfdetr_seg_nano")
    args = parser.parse_args()

    # Load class info
    ann_path = Path(args.dataset_dir) / "train" / "_annotations.coco.json"
    with open(ann_path) as f:
        coco = json.load(f)
    num_classes = len(coco["categories"])
    class_names = [c["name"] for c in sorted(coco["categories"], key=lambda x: x["id"])]
    logger.info("Classes (%d): %s", num_classes, class_names[:10])

    # Create model (same approach as eagle project)
    if args.model_size == "nano":
        from rfdetr import RFDETRSegNano
        model = RFDETRSegNano()
    elif args.model_size == "small":
        from rfdetr import RFDETRSegSmall
        model = RFDETRSegSmall()
    elif args.model_size == "medium":
        from rfdetr import RFDETRSegMedium
        model = RFDETRSegMedium()
    else:
        from rfdetr import RFDETRSegLarge
        model = RFDETRSegLarge()

    logger.info("Model: RFDETRSeg%s", args.model_size.capitalize())
    logger.info("Default resolution: %d", model.model_config.resolution)
    logger.info("Patch size: %d", model.model_config.patch_size)

    # Train (simple API — same as eagle project notebook)
    logger.info("Starting training: %d epochs, batch=%d", args.epochs, args.batch_size)
    model.train(
        dataset_dir=str(Path(args.dataset_dir).absolute()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=4,
        output_dir=args.output_dir,
    )

    logger.info("Training complete! Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
