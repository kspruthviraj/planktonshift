"""
run_sam_segmentation.py
=======================
Run SAM on all plankton datasets to generate:
1. Binary masks (organism vs background)
2. Bounding boxes (for RF-DETR detection training)
3. Cropped organisms (organism on white/black background)
4. COCO-format annotations (for RF-DETR)

Saves everything to data_segmentation/ for reproducibility.

Usage:
    python run_sam_segmentation.py \
        --dataset zoolake2 \
        --max-per-class 500 \
        --output-dir data_segmentation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SAM_CHECKPOINT = "/home/sreenath/research-space/PlanktonShift/models/sam_vit_b.pth"

# Dataset paths
DATASET_PATHS = {
    "zoolake2": "/home/sreenath/research-space/PlanktonShift/data/chen_data/ZooLake2/ZooLake2/ZooLake2.0",
    "whoi22": "/home/sreenath/research-space/Adverserial_net/data/cross_domain/whoi22_full/train/WHOI22",
    "zooscan20": "/home/sreenath/research-space/Adverserial_net/data/cross_domain/whoi22_full/test/ZooScan20",
    "ood": "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs",
}


def load_sam():
    """Load SAM model."""
    from segment_anything import sam_model_registry, SamPredictor
    logger.info("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    logger.info("SAM loaded")
    return predictor


def segment_organism(predictor, image_np):
    """Segment the main organism from a plankton image.
    
    Strategy: Use automatic mask generation with center point prompt.
    Plankton images typically have one organism centered in the frame.
    """
    h, w = image_np.shape[:2]
    
    # Set image
    predictor.set_image(image_np)
    
    # Prompt: center point (plankton is usually centered)
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])  # 1 = foreground
    
    # Also try 4 corner points as background prompts
    corner_points = np.array([
        [5, 5], [w - 5, 5], [5, h - 5], [w - 5, h - 5]
    ])
    corner_labels = np.array([0, 0, 0, 0])  # 0 = background
    
    all_points = np.concatenate([center_point, corner_points], axis=0)
    all_labels = np.concatenate([center_label, corner_labels], axis=0)
    
    masks, scores, _ = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        multimask_output=True,
    )
    
    # Select best mask (highest score)
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    
    return mask


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x, y, w, h]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]


def crop_organism(image_np, mask, bbox, padding=10):
    """Crop organism from image using mask and bbox."""
    h, w = image_np.shape[:2]
    x, y, bw, bh = bbox
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    
    # Crop
    cropped = image_np[y1:y2, x1:x2].copy()
    
    # Apply mask (set background to white)
    mask_crop = mask[y1:y2, x1:x2]
    cropped[~mask_crop] = 255  # White background
    
    return cropped


def process_dataset(dataset_name, predictor, output_dir, max_per_class=500):
    """Process a single dataset with SAM."""
    if dataset_name == "ood":
        # OOD has subdirectories for each day
        for ood_day in sorted(Path(DATASET_PATHS["ood"]).iterdir()):
            if ood_day.is_dir():
                logger.info("Processing OOD day: %s", ood_day.name)
                process_directory(str(ood_day), predictor, 
                                Path(output_dir) / "OOD" / ood_day.name,
                                max_per_class=max_per_class)
    else:
        src = DATASET_PATHS[dataset_name]
        logger.info("Processing %s: %s", dataset_name, src)
        process_directory(src, predictor, Path(output_dir) / dataset_name,
                         max_per_class=max_per_class)


def process_directory(src_dir, predictor, out_dir, max_per_class=500):
    """Process all images in a directory."""
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    
    # COCO annotation structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    ann_id = 1
    img_id = 1
    class_to_id = {}
    
    for cls_dir in sorted(src_path.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        
        cls_name = cls_dir.name
        if cls_name not in class_to_id:
            class_to_id[cls_name] = len(class_to_id) + 1
            coco["categories"].append({
                "id": class_to_id[cls_name],
                "name": cls_name,
            })
        
        cls_out = out_path / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            if max_per_class > 0 and count >= max_per_class:
                break
            
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image)
                
                # Resize for SAM (keep aspect ratio, max 1024)
                h, w = image_np.shape[:2]
                scale = min(1024 / h, 1024 / w, 1.0)
                if scale < 1.0:
                    new_h, new_w = int(h * scale), int(w * scale)
                    image_resized = np.array(Image.fromarray(image_np).resize((new_w, new_h)))
                else:
                    image_resized = image_np
                
                # Segment
                mask = segment_organism(predictor, image_resized)
                
                # Scale mask back to original size
                if scale < 1.0:
                    mask_orig = np.array(Image.fromarray(mask.astype(np.uint8)).resize((w, h))) > 0
                else:
                    mask_orig = mask
                
                # Get bounding box
                bbox = mask_to_bbox(mask_orig)
                if bbox is None:
                    continue
                
                # Crop organism
                cropped = crop_organism(image_np, mask_orig, bbox)
                
                # Save
                stem = img_path.stem
                Image.fromarray(image_np).save(cls_out / f"{stem}_original.png")
                Image.fromarray((mask_orig * 255).astype(np.uint8)).save(cls_out / f"{stem}_mask.png")
                Image.fromarray(cropped).save(cls_out / f"{stem}_crop.png")
                
                # COCO annotation
                coco["images"].append({
                    "id": img_id,
                    "file_name": f"{cls_name}/{stem}_crop.png",
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
                    "segmentation": [],  # Could add mask polygon
                })
                
                ann_id += 1
                img_id += 1
                count += 1
                
                if count % 100 == 0:
                    logger.info("    %s: %d images processed", cls_name, count)
                    
            except Exception as e:
                logger.warning("Failed %s: %s", img_path, e)
        
        logger.info("  %s: %d images done", cls_name, count)
    
    # Save COCO annotations
    coco_path = out_path / "annotations.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    
    logger.info("  Saved %d images, %d annotations to %s", 
                len(coco["images"]), len(coco["annotations"]), coco_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["zoolake2", "whoi22", "zooscan20", "ood", "all"])
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="data_segmentation")
    args = parser.parse_args()
    
    predictor = load_sam()
    
    if args.dataset == "all":
        datasets = ["zoolake2", "whoi22", "zooscan20", "ood"]
    else:
        datasets = [args.dataset]
    
    for ds in datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds)
        logger.info("=" * 60)
        process_dataset(ds, predictor, args.output_dir, args.max_per_class)
    
    logger.info("All done!")


if __name__ == "__main__":
    main()
