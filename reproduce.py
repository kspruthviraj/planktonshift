#!/usr/bin/env python3
"""
reproduce.py — Master reproduction script for PlanktonShift.

Reproduces ALL key results from the paper in order.
Uses centralized config.py for paths and random seeds.
Checkpoints automatically — re-run to resume from where it left off.

Usage:
    python reproduce.py              # Run everything
    python reproduce.py --phase 1    # Fourier analysis only
    python reproduce.py --phase 2    # SAA ablation only
    python reproduce.py --phase 3    # Chen OOD benchmark only
    python reproduce.py --status     # Show what's done
    python reproduce.py --dry-run    # Show what would run
"""

import sys, os, json, time, logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "code"))
import config

RESULTS = config.RESULTS
ROOT = config.ROOT

CHECKPOINT_FILE = RESULTS / ".reproduce_checkpoints.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ckpt():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_ckpt(name, status="done"):
    ckpt = load_ckpt()
    ckpt[name] = {"status": status, "time": str(datetime.now())}
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


def is_done(name):
    return load_ckpt().get(name, {}).get("status") == "done"


def run(cmd, name, cwd=None):
    """Run a command with logging and checkpoint."""
    if is_done(name):
        logger.info("[SKIP] %s (already done)", name)
        return True
    logger.info("[RUN] %s", name)
    logger.info("  CMD: %s", cmd[:120])
    t0 = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0
    if ret == 0:
        save_ckpt(name)
        logger.info("[DONE] %s (%.0fs)", name, elapsed)
        return True
    logger.error("[FAIL] %s (%.0fs)", name, elapsed)
    return False


# ═══════════════════════════════════════════════
# PHASE 1: Fourier Analysis
# ═══════════════════════════════════════════════
def phase1_fourier():
    """Reproduce Fourier shift characterization (83.1% domain classifier)."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Fourier-Domain Shift Analysis")
    logger.info("=" * 60)

    # Prepare data symlinks
    os.makedirs(ROOT / "data" / "fourier_input" / "WHOI22", exist_ok=True)
    os.makedirs(ROOT / "data" / "fourier_input" / "ZooScan20", exist_ok=True)
    os.makedirs(ROOT / "data" / "fourier_input" / "ZooLake2", exist_ok=True)

    # Run Fourier analysis (using adverserial_net script)
    cmd = (
        f"cd {ROOT} && python code/adverserial_net/fourier_shift_analysis.py "
        f"--data-root data/fourier_input "
        f"--domains WHOI22 ZooScan20 ZooLake2 "
        f"--output-dir results/fourier_analysis "
        f"--max-per-class 30"
    )
    if not run(cmd, "phase1_fourier"):
        return False

    # Verify result
    result_path = RESULTS / "fourier_analysis" / "fourier_analysis.json"
    if result_path.exists():
        with open(result_path) as f:
            d = json.load(f)
        dc = d.get("domain_classifier", {}).get("accuracy", 0)
        logger.info("  Domain classifier accuracy: %.1f%%", dc * 100)
    return True


# ═══════════════════════════════════════════════
# PHASE 2: SAA Cross-Instrument Ablation
# ═══════════════════════════════════════════════
def phase2_saa_ablation():
    """Reproduce SAA ablation on cross-instrument benchmark."""
    logger.info("=" * 60)
    logger.info("PHASE 2: SAA Cross-Instrument Ablation")
    logger.info("=" * 60)

    augmentations = ["standard", "heavy", "saa_amplitude", "saa_noise",
                     "saa_band", "saa_phase"]

    for aug in augmentations:
        name = f"phase2_saa_{aug}"
        cmd = (
            f"cd {ROOT} && python code/train_with_saa.py "
            f"--data-dir data/cross_domain/cross_instrument "
            f"--source-domain train/DataShift_IFCB "
            f"--target-domains test/DataShift_ZooScan "
            f"--architectures vit_b_16 "
            f"--augmentation {aug} "
            f"--epochs 30 --batch-size 16 --seeds 42 "
            f"--output results/saa_ablation_{aug}.json"
        )
        run(cmd, name)

    cmd_tta = (
        f"cd {ROOT} && python code/train_with_saa.py "
        f"--data-dir data/cross_domain/cross_instrument "
        f"--source-domain train/DataShift_IFCB "
        f"--target-domains test/DataShift_ZooScan "
        f"--architectures vit_b_16 "
        f"--augmentation saa_band --tta "
        f"--epochs 30 --batch-size 16 --seeds 42 "
        f"--output results/saa_ablation_band_tta.json"
    )
    run(cmd_tta, "phase2_saa_band_tta")
    return True


# ═══════════════════════════════════════════════
# PHASE 3: Chen OOD Benchmark (BEiT + SAA)
# ═══════════════════════════════════════════════
def phase3_chen_ood():
    """Reproduce Chen OOD benchmark with BEiT + SAA fine-tuning."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Chen OOD Benchmark (BEiT + SAA)")
    logger.info("=" * 60)

    # Approach A: Train from ImageNet-22K with Chen aug + SAA
    cmd_a = (
        f"cd {ROOT} && python code/final_chen_saa.py"
    )
    run(cmd_a, "phase3_approach_a")

    # Approach B: Fine-tune Chen's model with SAA
    cmd_b = (
        f"cd {ROOT} && python code/approach_B_only.py"
    )
    run(cmd_b, "phase3_approach_b")

    # Verify result
    result_path = RESULTS / "finetune_chen_saa" / "finetune_results_v4.json"
    if result_path.exists():
        with open(result_path) as f:
            d = json.load(f)
        ensemble = d.get("finetune_ensemble_geometric", {}).get("overall", 0)
        logger.info("  SBA fine-tuned ensemble OOD: %.1f%%", ensemble * 100)
    return True


# ═══════════════════════════════════════════════
# PHASE 4: Generate Paper Figures
# ═══════════════════════════════════════════════
def phase4_figures():
    """Generate all publication figures."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Generate Paper Figures")
    logger.info("=" * 60)

    cmd = f"cd {ROOT} && python code/generate_final_results.py"
    run(cmd, "phase4_figures")
    return True


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PlanktonShift Reproduction Script")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run specific phase")
    parser.add_argument("--status", action="store_true", help="Show reproduction status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--reset", type=str, help="Reset a specific checkpoint")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.reset:
        ckpt = load_ckpt()
        if args.reset in ckpt:
            del ckpt[args.reset]
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(ckpt, f, indent=2)
            logger.info("Reset: %s", args.reset)
        return

    if args.status:
        ckpt = load_ckpt()
        phases = {
            "phase1_fourier": "Fourier shift analysis (83.1% domain classifier)",
            "phase2_saa_standard": "SAA ablation — standard",
            "phase2_saa_band_tta": "SAA ablation — band + TTA (52.6%)",
            "phase3_approach_a": "Approach A: ImageNet-22K + Chen aug + SAA",
            "phase3_approach_b": "Approach B: Chen model + SAA fine-tune (82.1%)",
            "phase4_figures": "Publication figures",
        }
        logger.info("=" * 60)
        logger.info("REPRODUCTION STATUS")
        logger.info("=" * 60)
        done = 0
        for name, desc in phases.items():
            status = ckpt.get(name, {}).get("status", "pending")
            icon = "DONE" if status == "done" else "pending"
            logger.info("  [%s] %s", icon, desc)
            if status == "done":
                done += 1
        logger.info("-" * 60)
        logger.info("  %d/%d steps completed", done, len(phases))
        return

    if args.dry_run:
        phases = ["phase1_fourier", "phase2_saa_standard", "phase2_saa_band_tta",
                  "phase3_approach_a", "phase3_approach_b", "phase4_figures"]
        for name in phases:
            status = "SKIP (done)" if is_done(name) else "WOULD RUN"
            logger.info("  [%s] %s", status, name)
        return

    if args.phase == 1:
        phase1_fourier()
    elif args.phase == 2:
        phase2_saa_ablation()
    elif args.phase == 3:
        phase3_chen_ood()
    elif args.phase == 4:
        phase4_figures()
    else:
        phase1_fourier()
        phase2_saa_ablation()
        phase3_chen_ood()
        phase4_figures()

    logger.info("=" * 60)
    logger.info("REPRODUCTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
