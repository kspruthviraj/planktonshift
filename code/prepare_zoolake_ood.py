"""
prepare_zoolake_ood.py
======================
Prepare ZooLake35 temporal OOD splits from deployment dates in filenames.

Replicates and extends Chen et al.'s (2025) 10 OOD deployment day evaluation
using the ZooLake35 dataset available on disk.

Image filenames contain timestamps:
  SPC-EAWAG-0P5X-{microseconds}-{...}.jpeg

We extract deployment dates and create temporal splits:
- Train on early dates
- Test on later dates (OOD)

Usage:
    python prepare_zoolake_ood.py \
        --data-dir /home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed \
        --output-dir data/zoolake_ood \
        --min-samples-per-class 10
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TS_PATTERN = re.compile(r"SPC-EAWAG-0P5X-(\d+)-")


def extract_date(filename: str) -> str:
    m = TS_PATTERN.search(filename)
    if m:
        ts_us = int(m.group(1))
        try:
            return datetime.fromtimestamp(ts_us / 1_000_000).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed")
    parser.add_argument("--output-dir", type=str, default="data/zoolake_ood")
    parser.add_argument("--classes", nargs="+",
                        default=["ceratium", "daphnia", "bosmina", "conochilus",
                                 "keratella-quadrata", "fragilaria"])
    parser.add_argument("--min-samples-per-class", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all images with dates
    logger.info("Collecting images from %s...", args.data_dir)
    all_images = []  # (path, class, date)
    for cls_name in args.classes:
        cls_dir = Path(args.data_dir) / cls_name
        if not cls_dir.is_dir():
            logger.warning("Class directory not found: %s", cls_dir)
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            date = extract_date(img_path.name)
            if date != "unknown":
                all_images.append((str(img_path), cls_name, date))

    logger.info("Found %d images across %d classes", len(all_images), len(args.classes))

    # Group by date
    date_images = defaultdict(list)
    for path, cls, date in all_images:
        date_images[date].append((path, cls))

    # Find dates with enough samples per class
    valid_dates = []
    for date in sorted(date_images.keys()):
        cls_counts = defaultdict(int)
        for _, cls in date_images[date]:
            cls_counts[cls] += 1
        n_valid = sum(1 for c in args.classes if cls_counts.get(c, 0) >= args.min_samples_per_class)
        if n_valid >= len(args.classes) // 2:  # At least half the classes
            valid_dates.append(date)

    logger.info("Valid dates (>= %d samples in >= %d classes): %d",
                args.min_samples_per_class, len(args.classes) // 2, len(valid_dates))

    if len(valid_dates) < 2:
        # Fallback: use all dates with year-based split
        logger.warning("Not enough valid dates. Using year-based split.")
        all_dates = sorted(date_images.keys())
        train_dates = [d for d in all_dates if d.startswith("2018")]
        test_dates = [d for d in all_dates if d.startswith("2019") or d.startswith("2020")]
        if not train_dates or not test_dates:
            split = int(len(all_dates) * args.train_ratio)
            train_dates = all_dates[:split]
            test_dates = all_dates[split:]
    else:
        split = int(len(valid_dates) * args.train_ratio)
        train_dates = valid_dates[:split]
        test_dates = valid_dates[split:]

    logger.info("Train dates (%d): %s ... %s", len(train_dates), train_dates[0], train_dates[-1])
    logger.info("Test dates (%d): %s ... %s", len(test_dates), test_dates[0], test_dates[-1])

    # Create directory structure
    class_to_idx = {c: i for i, c in enumerate(args.classes)}

    # Training set
    train_dir = out / "train"
    train_stats = defaultdict(int)
    for date in train_dates:
        for path, cls in date_images[date]:
            if cls not in class_to_idx:
                continue
            dst = train_dir / cls
            dst.mkdir(parents=True, exist_ok=True)
            try:
                img = Image.open(path).convert("RGB")
                img.save(dst / f"{date}_{Path(path).stem}.png", "PNG")
                train_stats[cls] += 1
            except Exception as e:
                logger.warning("Failed %s: %s", path, e)

    # Test set (OOD)
    test_dir = out / "test"
    test_stats = defaultdict(int)
    test_date_stats = defaultdict(lambda: defaultdict(int))
    for date in test_dates:
        for path, cls in date_images[date]:
            if cls not in class_to_idx:
                continue
            dst = test_dir / cls
            dst.mkdir(parents=True, exist_ok=True)
            try:
                img = Image.open(path).convert("RGB")
                img.save(dst / f"{date}_{Path(path).stem}.png", "PNG")
                test_stats[cls] += 1
                test_date_stats[date][cls] += 1
            except Exception as e:
                logger.warning("Failed %s: %s", path, e)

    # Save manifest
    manifest = {
        "train_dates": train_dates,
        "test_dates": test_dates,
        "classes": args.classes,
        "class_to_idx": class_to_idx,
        "train_stats": dict(train_stats),
        "test_stats": dict(test_stats),
        "test_date_stats": {d: dict(v) for d, v in test_date_stats.items()},
        "total_train": sum(train_stats.values()),
        "total_test": sum(test_stats.values()),
    }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("  ZOO LAKE OOD SPLITS")
    logger.info("=" * 60)
    logger.info("  Train: %d images", sum(train_stats.values()))
    for cls in args.classes:
        logger.info("    %-25s %d", cls, train_stats.get(cls, 0))
    logger.info("  Test (OOD): %d images across %d dates",
                sum(test_stats.values()), len(test_dates))
    for cls in args.classes:
        logger.info("    %-25s %d", cls, test_stats.get(cls, 0))
    logger.info("  Per-date test distribution:")
    for date in sorted(test_date_stats.keys()):
        total = sum(test_date_stats[date].values())
        logger.info("    %s: %d images", date, total)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
