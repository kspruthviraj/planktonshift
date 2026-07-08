"""
train_rfdetr_large_3phase.py — RF-DETR-Large 3-phase training on v3 segmentation.

Phase 1 (50 epochs, lr=1e-4): Full training from pretrained checkpoint.
Phase 2 (20 epochs, lr=1e-5): Tune — lower LR on same model object.
Phase 3 (10 epochs, lr=1e-6): Finetune — minimal LR for final refinement.

The model object retains weights between train() calls (Lightning updates in-place),
so each phase continues from the previous phase's trained weights.

Usage:
    python code/train_rfdetr_large_3phase.py
"""
import argparse, json, logging, os, shutil
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "rfdetr_large_3phase.log"), mode="a")],
)
logger = logging.getLogger(__name__)


def prepare_coco(seg_dir, rfdetr_dir):
    """Convert v3 segmented data to RF-DETR COCO format with train/valid split."""
    seg_path = Path(seg_dir)
    out_path = Path(rfdetr_dir)

    classes = sorted([d.name for d in seg_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_id = {c: i + 1 for i, c in enumerate(classes)}
    logger.info("Classes (%d): %s", len(classes), classes[:5])

    # Collect all crop files
    all_samples = []
    for cls_name in classes:
        cls_dir = seg_path / cls_name
        crops = sorted(cls_dir.glob("*_crop.png"))
        for cp in crops:
            all_samples.append((str(cp), cls_name))
    logger.info("Total crop images: %d", len(all_samples))

    # Shuffle and split 80/20
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_samples))
    split = int(len(all_samples) * 0.8)
    samples = {"train": [all_samples[i] for i in idx[:split]],
               "valid": [all_samples[i] for i in idx[split:]]}
    logger.info("Split: train=%d valid=%d", len(samples["train"]), len(samples["valid"]))

    for split_name, split_samples in samples.items():
        split_dir = out_path / split_name
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        coco = {"images": [], "annotations": [], "categories": [
            {"id": v, "name": k, "supercategory": "plankton"} for k, v in class_to_id.items()]}

        for img_id, (src_path, cls_name) in enumerate(split_samples, 1):
            dst = img_dir / f"{img_id:06d}.png"
            if not dst.exists():
                shutil.copy2(src_path, dst)

            im = Image.open(src_path)
            w, h = im.size

            coco["images"].append({
                "id": img_id, "file_name": f"images/{img_id:06d}.png",
                "width": w, "height": h,
            })
            coco["annotations"].append({
                "id": img_id, "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0,
            })

        ann_path = split_dir / "_annotations.coco.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)
        logger.info("  %s: %d images → %s", split_name, len(coco["images"]), ann_path)

    return len(classes), classes


def train_phase(model, dataset_dir, output_dir, epochs, lr, phase_name, batch_size=4, grad_accum_steps=4):
    logger.info("=== PHASE %s: %d epochs, lr=%s ===", phase_name, epochs, lr)
    model.train(
        dataset_dir=str(Path(dataset_dir).absolute()),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=str(output_dir),
        do_random_resize_via_padding=False,
        multi_scale=False,
    )
    logger.info("Phase %s complete", phase_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-dir", type=str, default="data_segmentation_v3/zoolake2")
    parser.add_argument("--rfdetr-dir", type=str, default="data_rfdetr_v3")
    parser.add_argument("--phase1-epochs", type=int, default=50)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--phase3-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    # Step 1: Prepare COCO data
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing COCO dataset from v3 segmentation")
    logger.info("=" * 60)
    num_classes, class_names = prepare_coco(args.seg_dir, args.rfdetr_dir)

    # Step 2: Create model
    logger.info("=" * 60)
    logger.info("STEP 2: Creating RF-DETR-Large")
    logger.info("=" * 60)
    from rfdetr import RFDETRLarge
    model = RFDETRLarge()
    logger.info("Model created. Resolution: %d, Patch: %d",
                model.model_config.resolution, model.model_config.patch_size)

    # Step 3: Phase 1 — Training (50 epochs, lr=1e-4)
    logger.info("=" * 60)
    logger.info("STEP 3: Phase 1 — Training")
    logger.info("=" * 60)
    p1_out = ROOT / "results" / "rfdetr_large_3phase"
    train_phase(model, args.rfdetr_dir, p1_out, args.phase1_epochs,
                1e-4, "1 (train)", args.batch_size, args.grad_accum)

    # Step 4: Phase 2 — Tuning (20 epochs, lr=1e-5)
    logger.info("=" * 60)
    logger.info("STEP 4: Phase 2 — Tuning")
    logger.info("=" * 60)
    train_phase(model, args.rfdetr_dir, p1_out, args.phase2_epochs,
                1e-5, "2 (tune)", args.batch_size, args.grad_accum)

    # Step 5: Phase 3 — Finetuning (10 epochs, lr=1e-6)
    logger.info("=" * 60)
    logger.info("STEP 5: Phase 3 — Finetuning")
    logger.info("=" * 60)
    train_phase(model, args.rfdetr_dir, p1_out, args.phase3_epochs,
                1e-6, "3 (finetune)", args.batch_size, args.grad_accum)

    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETE. Model in: %s", p1_out)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
