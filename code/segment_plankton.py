"""
segment_plankton.py
===================
Segment plankton organisms using Otsu thresholding + morphological cleanup.

This is more reliable than SAM for plankton images because:
1. Plankton images have relatively uniform backgrounds
2. Organisms are darker/lighter than background
3. Simple thresholding works well for single-organism images

Flow:
1. Convert to grayscale
2. Otsu threshold → binary mask
3. Morphological close → fill small holes
4. Find largest connected component → organism mask
5. Get bounding box + crop with padding
6. Save: original, mask, crop (organism on white background)

Usage:
    python segment_plankton.py \
        --data-dir /path/to/zoolake2 \
        --output-dir data_segmentation/zoolake2 \
        --max-per-class 500
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def segment_organism(image_np):
    """Segment organism using corner-background subtraction + morphological cleanup (v2a).
    
    v2a: threshold at 2.5σ from corner background, erode(1) + dilate(2).
    Best balance of tight segmentation and tail preservation for ZooLake2.
    
    Returns:
        mask: binary mask (True = organism)
        bbox: [x, y, w, h] bounding box
    """
    from skimage.morphology import disk, remove_small_objects, binary_erosion, binary_dilation
    
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = np.mean(image_np[:, :, :3], axis=2)
    else:
        gray = image_np
    
    h, w = gray.shape
    
    # Sample background from 4 corners
    cs = min(8, h // 3, w // 3)
    corners = [gray[:cs, :cs], gray[:cs, -cs:], gray[-cs:, :cs], gray[-cs:, -cs:]]
    bg_pixels = np.concatenate([c.flatten() for c in corners])
    bg_mean = bg_pixels.mean()
    bg_std = bg_pixels.std()
    
    # Background subtraction: pixels significantly different from corner background
    diff = np.abs(gray.astype(float) - bg_mean)
    mask = (diff > bg_std * 2.5) & (diff > 5)
    
    # Morphological cleanup (v2a: less aggressive erosion)
    mask = binary_erosion(mask, disk(1))
    mask = binary_dilation(mask, disk(2))
    mask = ndimage.binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=max(20, h * w // 500))
    
    # Keep only largest connected component (the organism)
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None, None
    
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = labeled == largest_component
    
    # Get bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None, None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
    
    return mask, bbox


def crop_organism(image_np, mask, bbox, padding=5, crop_size=None):
    """Crop organism from image using mask and bbox.
    
    Args:
        image_np: original image
        mask: binary mask
        bbox: [x, y, w, h]
        padding: pixels to add around bbox
        crop_size: if set, resize crop to this size (h, w)
    """
    h, w = image_np.shape[:2]
    x, y, bw, bh = bbox
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    
    # Crop
    cropped = image_np[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    
    # Set background to white
    if len(cropped.shape) == 3:
        cropped[~mask_crop] = [255, 255, 255]
    else:
        cropped[~mask_crop] = 255
    
    # Resize if requested
    if crop_size:
        from PIL import Image as PILImage
        cropped = np.array(PILImage.fromarray(cropped).resize(
            (crop_size[1], crop_size[0]), PILImage.BILINEAR
        ))
    
    return cropped


def process_directory(src_dir, out_dir, max_per_class=500, crop_size=None):
    """Process all images in a directory."""
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    
    coco = {"images": [], "annotations": [], "categories": []}
    ann_id = 1
    img_id = 1
    class_to_id = {}
    
    total_processed = 0
    total_failed = 0
    
    for cls_dir in sorted(src_path.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        
        cls_name = cls_dir.name
        if cls_name not in class_to_id:
            class_to_id[cls_name] = len(class_to_id) + 1
            coco["categories"].append({"id": class_to_id[cls_name], "name": cls_name})
        
        cls_out = out_path / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            if max_per_class > 0 and count >= max_per_class:
                break
            
            try:
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image)
                
                # Segment
                mask, bbox = segment_organism(image_np)
                if mask is None or bbox is None:
                    total_failed += 1
                    continue
                
                # Crop
                cropped = crop_organism(image_np, mask, bbox, padding=5, crop_size=crop_size)
                
                # Save
                stem = img_path.stem
                Image.fromarray(image_np).save(cls_out / f"{stem}_original.png")
                Image.fromarray((mask * 255).astype(np.uint8)).save(cls_out / f"{stem}_mask.png")
                Image.fromarray(cropped).save(cls_out / f"{stem}_crop.png")
                
                # COCO annotation
                h, w = image_np.shape[:2]
                coco["images"].append({
                    "id": img_id,
                    "file_name": f"{cls_name}/{stem}_crop.png",
                    "width": w, "height": h,
                })
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_to_id[cls_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                })
                
                ann_id += 1
                img_id += 1
                count += 1
                total_processed += 1
                
                if count % 100 == 0:
                    logger.info("    %s: %d images processed", cls_name, count)
                    
            except Exception as e:
                logger.warning("Failed %s: %s", img_path, e)
                total_failed += 1
        
        logger.info("  %s: %d images done", cls_name, count)
    
    # Save COCO annotations
    coco_path = out_path / "annotations.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    
    logger.info("  Saved %d images, %d annotations (%d failed)",
                len(coco["images"]), len(coco["annotations"]), total_failed)
    return coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Resize crops to this size (0 = no resize)")
    args = parser.parse_args()
    
    crop_size = (args.crop_size, args.crop_size) if args.crop_size > 0 else None
    
    logger.info("Processing %s → %s", args.data_dir, args.output_dir)
    logger.info("Max per class: %d, Crop size: %s", args.max_per_class, crop_size)
    
    process_directory(args.data_dir, args.output_dir, args.max_per_class, crop_size)
    logger.info("Done!")


if __name__ == "__main__":
    main()
