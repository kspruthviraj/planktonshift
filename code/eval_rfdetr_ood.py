"""
eval_rfdetr_ood.py — Evaluate best RF-DETR checkpoint on OOD classification.

Loads the best EMA checkpoint from epoch 5 and measures per-day classification
accuracy on the v2a-prepped OOD test sets (data_rfdetr/test_OOD1-10).
"""
import json, logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path):
    from rfdetr import RFDETRLarge
    import torch

    # Use built-in from_checkpoint class method
    model = RFDETRLarge.from_checkpoint(str(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loaded from %s", checkpoint_path)
    return model, device


def evaluate_day(model, device, test_dir, cat_id_to_name):
    test_path = Path(test_dir)
    ann_path = test_path / "_annotations.coco.json"
    if not ann_path.exists():
        logger.warning("No annotations: %s", ann_path)
        return None

    with open(ann_path) as f:
        coco = json.load(f)

    images_dir = test_path / "images"
    correct, total = 0, 0
    per_class = {}
    skipped = 0

    for img_info in tqdm(coco["images"], desc=f"  {test_path.name}", leave=False):
        img_path = images_dir / img_info["file_name"].split("/")[-1]
        if not img_path.exists():
            skipped += 1
            continue

        anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
        if not anns:
            skipped += 1
            continue
        true_cat = anns[0]["category_id"]
        true_name = cat_id_to_name.get(true_cat, "unknown").lower()

        try:
            detections = model.predict(str(img_path))
            if len(detections) > 0:
                pred_id = int(detections.class_id[0])
                pred_name = cat_id_to_name.get(pred_id, "unknown").lower()
            else:
                pred_name = "unknown"
        except Exception:
            pred_name = "unknown"

        is_correct = pred_name == true_name
        if is_correct:
            correct += 1
        total += 1

        if true_name not in per_class:
            per_class[true_name] = {"correct": 0, "total": 0}
        per_class[true_name]["total"] += 1
        if is_correct:
            per_class[true_name]["correct"] += 1

    acc = correct / max(total, 1)
    return {"accuracy": float(acc), "correct": correct, "total": total,
            "skipped": skipped, "per_class": per_class}


def main():
    checkpoint = ROOT / "results" / "rfdetr_large_3phase" / "checkpoint_best_ema.pth"
    logger.info("Loading %s", checkpoint)
    model, device = load_model(checkpoint)

    # Load class mapping from OOD1
    ann = json.load(open(ROOT / "data_rfdetr" / "test_OOD1" / "_annotations.coco.json"))
    cat_to_name = {c["id"]: c["name"] for c in ann["categories"]}

    results = {}
    overall_correct, overall_total = 0, 0

    for ood_dir in sorted(Path(ROOT / "data_rfdetr").glob("test_OOD*")):
        day_name = ood_dir.name.replace("test_", "")
        logger.info("Evaluating %s...", day_name)
        day_res = evaluate_day(model, device, str(ood_dir), cat_to_name)
        if day_res:
            results[day_name] = day_res
            overall_correct += day_res["correct"]
            overall_total += day_res["total"]
            logger.info("  %s: %.1f%% (%d/%d, skipped=%d)",
                        day_name, day_res["accuracy"] * 100,
                        day_res["correct"], day_res["total"], day_res["skipped"])

    overall_acc = overall_correct / max(overall_total, 1)
    logger.info("=" * 50)
    logger.info("OVERALL OOD: %.1f%% (%d/%d)", overall_acc * 100, overall_correct, overall_total)

    # Per-class summary
    all_classes = {}
    for day_res in results.values():
        for cls_name, stats in day_res["per_class"].items():
            if cls_name not in all_classes:
                all_classes[cls_name] = {"correct": 0, "total": 0}
            all_classes[cls_name]["correct"] += stats["correct"]
            all_classes[cls_name]["total"] += stats["total"]

    logger.info("Per-class (>10 samples):")
    for cls_name in sorted(all_classes):
        s = all_classes[cls_name]
        if s["total"] >= 10:
            logger.info("  %-25s %.1f%% (n=%d)", cls_name,
                        100 * s["correct"] / max(s["total"], 1), s["total"])

    # Save
    out = {
        "checkpoint": str(checkpoint),
        "epoch": 5,
        "overall_accuracy": float(overall_acc),
        "per_day": results,
    }
    out_path = ROOT / "results" / "rfdetr_large_3phase" / "ood_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
