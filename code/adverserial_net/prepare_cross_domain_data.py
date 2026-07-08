"""
prepare_cross_domain_data.py
============================
Prepare unified cross-domain evaluation datasets from WHOI22, ZooScan20,
ZooLake35, and DataShift eval. Handles the fact that class name overlap
across datasets is limited.

Three experimental configurations:
  A. Cross-instrument (same ecosystem): DataShift IFCB ↔ ZooScan (5 overlapping classes)
  B. Cross-instrument + ecosystem: WHOI22 → ZooLake35 (Dinobryon + all for Fourier)
  C. Temporal OOD: ZooLake35 → Chen's 10 OOD days

The Fourier analysis uses ALL images from each domain regardless of class.

Usage:
    python prepare_cross_domain_data.py --output-dir data/cross_domain --all
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source dataset paths
# ---------------------------------------------------------------------------
SOURCES = {
    "WHOI22": "/home/sreenath/research-space/Traidmind/data/whoi22-preprocessed",
    "ZooScan20": "/home/sreenath/research-space/Traidmind/data/zooscan20-preprocessed",
    "ZooLake35": "/home/sreenath/research-space/Traidmind/data/zoolake35-preprocessed",
    "DataShift_IFCB": "/home/sreenath/research-space/DataShift/data/eval/IFCB",
    "DataShift_ZooScan": "/home/sreenath/research-space/DataShift/data/eval/ZooScan",
    "DataShift_ZooLake": "/home/sreenath/research-space/DataShift/data/eval_v2/ZooLake",
}

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_SIZE = 48

# ---------------------------------------------------------------------------
# Experimental configurations
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    # Experiment A: Cross-instrument (marine IFCB ↔ marine ZooScan)
    # DataShift eval has good overlap
    "cross_instrument": {
        "description": "IFCB vs ZooScan on overlapping marine plankton classes",
        "train_domain": "DataShift_IFCB",
        "test_domains": ["DataShift_ZooScan"],
        "overlapping_classes": ["Amphipoda", "Annelida", "Appendicularia", "Calanoida", "Ceratium"],
        "all_classes": [
            "Amphipoda", "Annelida", "Appendicularia", "Calanoida", "Ceratium",
            "Chaetognatha", "Coscinodiscus", "Doliolida", "Noctiluca",
            "Oithonidae", "Ostracoda", "Salpida",
        ],
    },
    # Experiment B: Cross-ecosystem (WHOI22 marine → ZooLake35 freshwater)
    # Uses all images for Fourier analysis, Dinobryon for classification
    "cross_ecosystem": {
        "description": "Marine IFCB vs freshwater DSPC — Fourier analysis + Dinobryon classification",
        "train_domain": "WHOI22",
        "test_domains": ["ZooLake35"],
        "overlapping_classes": ["Dinobryon"],
        "fourier_classes": "all",  # Use all classes for Fourier analysis
    },
    # Experiment C: Full WHOI22 baseline (22 marine phytoplankton classes)
    "whoi22_full": {
        "description": "Full WHOI22 dataset for baseline training and cross-domain evaluation",
        "train_domain": "WHOI22",
        "test_domains": ["ZooScan20", "ZooLake35"],
        "overlapping_classes": ["Dinobryon"],
    },
}


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------
def collect_images(source_path: str, max_per_class: int = 0, min_size: int = MIN_SIZE):
    """Collect valid images from a source directory."""
    src = Path(source_path)
    if not src.is_dir():
        logger.warning("Source not found: %s", source_path)
        return {}

    classes = {}
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        valid = []
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            try:
                img = Image.open(img_path)
                w, h = img.size
                if w >= min_size and h >= min_size:
                    valid.append(img_path)
            except Exception:
                continue
        if max_per_class > 0 and len(valid) > max_per_class:
            random.seed(42)
            valid = random.sample(valid, max_per_class)
        if valid:
            classes[cls_dir.name] = valid
    return classes


def copy_images(images: list, dst_dir: Path, label: str):
    """Copy images to destination, converting to RGB PNG."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            img.save(dst_dir / f"{label}_{idx:05d}.png", "PNG")
        except Exception as e:
            logger.warning("Failed %s: %s", img_path, e)
    return len(images)


# ---------------------------------------------------------------------------
# Experiment preparation
# ---------------------------------------------------------------------------
def prepare_experiment(exp_name: str, config: dict, output_dir: str, max_per_class: int):
    """Prepare a single experiment's data."""
    exp_dir = Path(output_dir) / exp_name
    stats = {"experiment": exp_name, "description": config["description"], "domains": {}}

    # Collect train domain
    train_src = SOURCES[config["train_domain"]]
    train_classes = collect_images(train_src, max_per_class)
    logger.info("Train domain %s: %d classes, %d images",
                config["train_domain"],
                len(train_classes),
                sum(len(v) for v in train_classes.values()))

    # Copy train data
    for cls_name, images in train_classes.items():
        dst = exp_dir / "train" / config["train_domain"] / cls_name
        copy_images(images, dst, cls_name)

    stats["domains"][config["train_domain"]] = {
        "role": "train",
        "n_classes": len(train_classes),
        "n_images": sum(len(v) for v in train_classes.values()),
        "classes": {k: len(v) for k, v in train_classes.items()},
    }

    # Collect test domains
    for test_domain in config["test_domains"]:
        test_src = SOURCES[test_domain]
        test_classes = collect_images(test_src, max_per_class)
        logger.info("Test domain %s: %d classes, %d images",
                    test_domain,
                    len(test_classes),
                    sum(len(v) for v in test_classes.values()))

        for cls_name, images in test_classes.items():
            dst = exp_dir / "test" / test_domain / cls_name
            copy_images(images, dst, cls_name)

        stats["domains"][test_domain] = {
            "role": "test",
            "n_classes": len(test_classes),
            "n_images": sum(len(v) for v in test_classes.values()),
            "classes": {k: len(v) for k, v in test_classes.items()},
        }

    # Identify overlapping classes
    all_domains = [config["train_domain"]] + config["test_domains"]
    all_domain_classes = {}
    for domain in all_domains:
        if domain in train_classes:
            all_domain_classes[domain] = set(train_classes.keys())
        for test_domain in config["test_domains"]:
            if domain == test_domain and test_domain in [d for d in stats["domains"]]:
                tc = collect_images(SOURCES[test_domain], max_per_class)
                all_domain_classes[domain] = set(tc.keys())

    stats["overlapping"] = config.get("overlapping_classes", [])
    stats["total_images"] = sum(d["n_images"] for d in stats["domains"].values())

    # Save manifest
    with open(exp_dir / "manifest.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Experiment '%s' prepared: %d total images", exp_name, stats["total_images"])
    return stats


# ---------------------------------------------------------------------------
# Dataset statistics table
# ---------------------------------------------------------------------------
def generate_statistics_table(all_stats: dict, output_dir: str):
    """Generate LaTeX table with dataset statistics."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-domain plankton dataset overview. "
        "IFCB = Imaging FlowCytobot (dark-field cytometer), "
        "ZooScan = flatbed scanner, "
        "DSPC = Digital Scientific Plankton Camera (field camera).}",
        "\\label{tab:datasets}",
        "\\begin{tabular}{llrrl}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Imaging System} & \\textbf{Classes} & "
        "\\textbf{Images} & \\textbf{Ecosystem} \\\\",
        "\\midrule",
        "WHOI22 & IFCB & 22 & 6,598 & Marine \\\\",
        "ZooScan20 & ZooScan & 20 & 4,066 & Marine \\\\",
        "ZooLake35 & DSPC & 35 & 24,242 & Freshwater \\\\",
        "DataShift-IFCB & IFCB & 16 & 450 & Marine \\\\",
        "DataShift-ZooScan & ZooScan & 12 & 360 & Marine \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{Cross-domain evaluation configurations}} \\\\",
    ]

    for exp_name, stats in all_stats.items():
        desc = stats.get("description", exp_name)
        n_img = stats.get("total_images", 0)
        lines.append(f"  {exp_name} & --- & --- & {n_img} & {desc[:50]} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_path = Path(output_dir) / "dataset_table.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Dataset table saved to %s", table_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare cross-domain plankton data.")
    parser.add_argument("--output-dir", type=str, default="data/cross_domain")
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--experiments", nargs="+",
                        default=["cross_instrument", "cross_ecosystem", "whoi22_full"])
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if args.all:
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = args.experiments

    all_stats = {}
    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            logger.warning("Unknown experiment: %s", exp_name)
            continue
        logger.info("=" * 60)
        logger.info("Preparing: %s", exp_name)
        logger.info("=" * 60)
        all_stats[exp_name] = prepare_experiment(
            exp_name, EXPERIMENTS[exp_name], args.output_dir, args.max_per_class
        )

    generate_statistics_table(all_stats, args.output_dir)

    global_manifest = {
        "experiments": all_stats,
        "sources": SOURCES,
        "max_per_class": args.max_per_class,
    }
    with open(Path(args.output_dir) / "global_manifest.json", "w") as f:
        json.dump(global_manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS PREPARED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
