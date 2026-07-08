"""
segment_plankton_v3.py — Improved v3 segmentation for full ZooLake2 dataset.

v3 improvements over v2a (segment_plankton.py):
  1. Border-background estimation (all 4 edges, not just corners).
     Handles organisms near edges and ZooScan flatbed backgrounds.
  2. Removed 500-image per-class cap — uses ALL images.
  3. Lower min_size filter (h*w//2000 instead of h*w//500).
     Preserves small organisms (nauplius, dinobryon, small rotifers).

Usage:
    python code/segment_plankton_v3.py \
        --data-dir data/chen_data/ZooLake2/ZooLake2/ZooLake2.0 \
        --output-dir data_segmentation_v3/zoolake2 \
        --crop-size 224
"""
import argparse, json, logging
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import disk, remove_small_objects, binary_erosion, binary_dilation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def segment_organism_v3(image_np):
    """v3: border-background subtraction + morphological cleanup.

    Returns (mask, bbox) or (None, None) if segmentation fails.
    """
    if len(image_np.shape) == 3:
        gray = np.mean(image_np[:, :, :3], axis=2)
    else:
        gray = image_np.astype(float)

    h, w = gray.shape

    # v3: use ALL border pixels, not just corners
    border_pixels = np.concatenate([
        gray[0, :].flatten(), gray[-1, :].flatten(),
        gray[:, 0].flatten(), gray[:, -1].flatten(),
    ])
    bg_mean = np.median(border_pixels)
    bg_std = np.std(border_pixels)

    # Statistical threshold
    diff = np.abs(gray.astype(float) - bg_mean)
    mask = (diff > bg_std * 2.0) & (diff > 3)  # v3: slightly looser threshold

    # Morphological cleanup (v2a parameters)
    mask = binary_erosion(mask, disk(1))
    mask = binary_dilation(mask, disk(2))
    mask = ndimage.binary_fill_holes(mask)

    # v3: lower min_size — preserve small organisms
    mask = remove_small_objects(mask, min_size=max(20, h * w // 2000))

    # Keep only largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None, None
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = labeled == largest_component

    # Bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None, None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]

    return mask, bbox


def crop_organism(image_np, mask, bbox, padding=5, crop_size=None):
    h, w = image_np.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    cropped = image_np[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    if len(cropped.shape) == 3:
        cropped[~mask_crop] = [255, 255, 255]
    else:
        cropped[~mask_crop] = 255
    if crop_size:
        cropped = np.array(Image.fromarray(cropped).resize(
            (crop_size[1], crop_size[0]), Image.BILINEAR))
    return cropped


def process_directory(src_dir, out_dir, crop_size=None):
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    clase_to_id = {}
    total_processed = 0
    total_failed = 0
    total_skipped = 0  # images where segmentation produced nothing

    for cls_dir in sorted(src_path.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        cls_name = cls_dir.name
        cls_out = out_path / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXT]
        logger.info("  %s: %d images", cls_name, len(imgs))

        for img_path in imgs:
            try:
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image)

                mask, bbox = segment_organism_v3(image_np)
                if mask is None or bbox is None:
                    total_skipped += 1
                    continue

                cropped = crop_organism(image_np, mask, bbox, padding=5, crop_size=crop_size)
                stem = img_path.stem
                Image.fromarray(image_np).save(cls_out / f"{stem}_original.png")
                Image.fromarray((mask * 255).astype(np.uint8)).save(cls_out / f"{stem}_mask.png")
                Image.fromarray(cropped).save(cls_out / f"{stem}_crop.png")
                total_processed += 1

            except Exception as e:
                logger.warning("Failed %s: %s", img_path, e)
                total_failed += 1

    logger.info("Done: %d processed, %d skipped (no mask), %d errors",
                total_processed, total_skipped, total_failed)
    return total_processed, total_skipped, total_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()

    crop_size = (args.crop_size, args.crop_size) if args.crop_size > 0 else None
    logger.info("Segmentation v3: border-background, no cap, low min_size")
    logger.info("Source: %s", args.data_dir)
    logger.info("Output: %s", args.output_dir)
    logger.info("Crop size: %s", crop_size)

    process_directory(args.data_dir, args.output_dir, crop_size)


if __name__ == "__main__":
    main()
