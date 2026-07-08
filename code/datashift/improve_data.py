"""
improve_data.py
===============
Filter evaluation data for image quality and re-prepare with better class
selection. Also prepares a "hard" and "easy" subset for analysis.
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import random

random.seed(42)

CLASSES_6 = ["Amphipoda", "Annelida", "Ceratium", "Chaetognatha", "Coscinodiscus", "Noctiluca"]
MIN_SIZE = 64  # Minimum width or height in pixels
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

SRC = "data/eval"
DST = "data/eval_filtered"


def filter_and_copy():
    """Filter images by minimum size, copy to new directory."""
    stats = {}
    for domain in ["IFCB", "ZooScan"]:
        stats[domain] = {}
        for cls in CLASSES_6:
            src_dir = Path(SRC) / domain / cls
            dst_dir = Path(DST) / domain / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            kept = 0
            total = 0
            if not src_dir.is_dir():
                stats[domain][cls] = {"kept": 0, "total": 0, "note": "class not in domain"}
                continue
            for img_path in sorted(src_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_EXT:
                    continue
                total += 1
                img = Image.open(img_path)
                w, h = img.size
                if w >= MIN_SIZE and h >= MIN_SIZE:
                    shutil.copy2(img_path, dst_dir / img_path.name)
                    kept += 1
            
            stats[domain][cls] = {"kept": kept, "total": total}
            print(f"  {domain}/{cls}: {kept}/{total} images >= {MIN_SIZE}px")
    
    # Save manifest
    with open(f"{DST}/manifest.json", "w") as f:
        json.dump({"min_size": MIN_SIZE, "classes": CLASSES_6, "stats": stats}, f, indent=2)
    
    return stats


def count_total():
    """Count total images after filtering."""
    total = 0
    for domain in ["IFCB", "ZooScan"]:
        domain_total = 0
        for cls in CLASSES_6:
            cls_dir = Path(DST) / domain / cls
            if cls_dir.is_dir():
                n = len([f for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT])
                domain_total += n
        print(f"  {domain}: {domain_total} images")
        total += domain_total
    print(f"  Total: {total} images")
    return total


if __name__ == "__main__":
    print("Filtering images by quality (min_size={MIN_SIZE}px)...")
    stats = filter_and_copy()
    print("\nFiltered dataset:")
    count_total()
