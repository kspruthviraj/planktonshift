"""
run_all.py
==========
Master runner for the full paper pipeline with:
  - Resume capability (skips completed experiments)
  - Live progress dashboard
  - Time estimates
  - Checkpoint tracking

Usage:
    python run_all.py                     # Run everything
    python run_all.py --phase 2           # Run only phase 2
    python run_all.py --skip-vlm          # Skip VLM (needs endpoint)
    python run_all.py --dry-run           # Show what would run
    python run_all.py --status            # Show current status only
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_all.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint system
# ---------------------------------------------------------------------------
CHECKPOINT_FILE = "results/.checkpoints.json"


def load_checkpoints() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(step: str, status: str = "done", details: dict = None):
    ckpts = load_checkpoints()
    ckpts[step] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
    }
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpts, f, indent=2)


def is_done(step: str) -> bool:
    ckpts = load_checkpoints()
    return ckpts.get(step, {}).get("status") == "done"


# ---------------------------------------------------------------------------
# Progress dashboard
# ---------------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.completed = 0
        self.start_time = time.time()
        self.step_times = []

    def start_step(self, step_name: str):
        self.current = step_name
        self.step_start = time.time()
        logger.info("")
        logger.info("=" * 72)
        logger.info("  STEP %d/%d: %s", self.completed + 1, self.total, step_name)
        logger.info("  Started: %s", datetime.now().strftime("%H:%M:%S"))
        elapsed = time.time() - self.start_time
        if self.completed > 0:
            avg = elapsed / self.completed
            remaining = avg * (self.total - self.completed)
            logger.info("  Est. remaining: %s", str(timedelta(seconds=int(remaining))))
        logger.info("=" * 72)

    def end_step(self, success: bool = True):
        duration = time.time() - self.step_start
        self.step_times.append(duration)
        self.completed += 1
        status = "OK" if success else "FAILED"
        logger.info("  Completed in %.1fs [%s]", duration, status)
        logger.info("  Progress: %d/%d (%.0f%%)", self.completed, self.total,
                     self.completed / self.total * 100)

    def summary(self):
        total_time = time.time() - self.start_time
        logger.info("")
        logger.info("=" * 72)
        logger.info("  PIPELINE COMPLETE")
        logger.info("  Total time: %s", str(timedelta(seconds=int(total_time))))
        logger.info("  Steps completed: %d/%d", self.completed, self.total)
        logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
def get_steps(skip_vlm: bool = False) -> list:
    steps = [
        # Phase 1: Data
        ("data_prep", "Phase 1: Prepare cross-domain datasets", 1),
        # Phase 2: Analysis
        ("pillow_noise", "Phase 2: Pillow 6v7 resize noise experiment", 2),
        ("imaging_artifacts", "Phase 2: Imaging artifact analysis (15 types)", 2),
        ("fourier_analysis", "Phase 2: Fourier shift characterization (3 domains)", 2),
        ("ood_detection", "Phase 2: Spectral OOD detection", 2),
        # Phase 3: Training — baselines
        ("train_vit_baseline", "Phase 3: Train ViT-B/16 baseline", 3),
        ("train_resnet_baseline", "Phase 3: Train ResNet-50 baseline", 3),
        ("train_convnext_baseline", "Phase 3: Train ConvNeXt-Tiny baseline", 3),
        # Phase 3: Training — SAA ablation
        ("train_saa_amplitude", "Phase 3: Train ViT + SAA (amplitude mix)", 3),
        ("train_saa_noise", "Phase 3: Train ViT + SAA (spectral noise)", 3),
        ("train_saa_phase", "Phase 3: Train ViT + SAA (phase preserve)", 3),
        ("train_saa_band", "Phase 3: Train ViT + SAA (band adversarial)", 3),
        ("train_saa_all", "Phase 3: Train ViT + SAA (all combined)", 3),
        # Phase 3: Training — ensemble
        ("train_ensemble_seed1", "Phase 3: Train ensemble member 1 (seed 42)", 3),
        ("train_ensemble_seed2", "Phase 3: Train ensemble member 2 (seed 123)", 3),
        ("train_ensemble_seed3", "Phase 3: Train ensemble member 3 (seed 456)", 3),
        ("train_ensemble_seed4", "Phase 3: Train ensemble member 4 (seed 789)", 3),
        ("train_ensemble_seed5", "Phase 3: Train ensemble member 5 (seed 999)", 3),
        # Phase 3: Training — WHOI22 full
        ("train_whoi22_vit", "Phase 3: Train ViT-B/16 on WHOI22 (22 classes)", 3),
        ("train_whoi22_resnet", "Phase 3: Train ResNet-50 on WHOI22 (22 classes)", 3),
    ]

    if not skip_vlm:
        steps.extend([
            # Phase 4: VLM
            ("vlm_baseline", "Phase 4: VLM baseline evaluation", 4),
            ("vlm_morphological_rag", "Phase 4: VLM + morphological RAG", 4),
            ("vlm_aa_rag", "Phase 4: VLM + AA-RAG (artifact-aware)", 4),
            ("vlm_aa_rag_ref", "Phase 4: VLM + AA-RAG + reference images", 4),
        ])

    # Phase 5: Figures
    steps.append(("generate_figures", "Phase 5: Generate all paper figures and tables", 5))

    return steps


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------
def run_cmd(cmd: str, desc: str) -> bool:
    logger.info("  Running: %s", desc)
    logger.info("  Command: %s", cmd[:100])
    t0 = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0
    if ret != 0:
        logger.error("  FAILED (exit %d) after %.1fs", ret, elapsed)
        return False
    logger.info("  Done in %.1fs", elapsed)
    return True


def step_data_prep():
    return run_cmd(
        "python prepare_cross_domain_data.py --output-dir data/cross_domain --all --max-per-class 300",
        "Preparing cross-domain datasets"
    )


def step_pillow_noise():
    return run_cmd("make pillow_noise 2>&1 | tail -5", "Pillow 6v7 noise experiment")


def step_imaging_artifacts():
    return run_cmd(
        "python imaging_artifact_experiment.py --data-dir data/cross_domain/cross_instrument/train/DataShift_IFCB "
        "--output-dir results/imaging_artifacts --max-images 50",
        "Imaging artifact analysis"
    )


def step_fourier_analysis():
    cmds = [
        "mkdir -p data/fourier_input/WHOI22 data/fourier_input/ZooScan20 data/fourier_input/ZooLake35",
        "ln -sf $(pwd)/data/cross_domain/whoi22_full/train/WHOI22/* data/fourier_input/WHOI22/ 2>/dev/null",
        "ln -sf $(pwd)/data/cross_domain/whoi22_full/test/ZooScan20/* data/fourier_input/ZooScan20/ 2>/dev/null",
        "ln -sf $(pwd)/data/cross_domain/whoi22_full/test/ZooLake35/* data/fourier_input/ZooLake35/ 2>/dev/null",
    ]
    for cmd in cmds:
        os.system(cmd)
    return run_cmd(
        "python fourier_shift_analysis.py --data-root data/fourier_input "
        "--domains WHOI22 ZooScan20 ZooLake35 "
        "--output-dir results/fourier_analysis/cross_domain --max-per-class 30",
        "Fourier shift characterization"
    )


def step_ood_detection():
    return run_cmd(
        "python spectral_ood_detection.py "
        "--train-dir data/cross_domain/cross_instrument/train/DataShift_IFCB "
        "--test-dirs data/cross_domain/cross_instrument/test/DataShift_ZooScan "
        "data/cross_domain/whoi22_full/test/ZooLake35 "
        "--output-dir results/ood_detection",
        "Spectral OOD detection"
    )


def step_train(arch: str, aug: str, output: str, seed: int = 42, epochs: int = 30,
               source: str = "train/DataShift_IFCB",
               targets: str = "test/DataShift_ZooScan",
               data_dir: str = "data/cross_domain/cross_instrument"):
    return run_cmd(
        f"python train_with_saa.py "
        f"--data-dir {data_dir} "
        f"--source-domain {source} "
        f"--target-domains {targets} "
        f"--architectures {arch} "
        f"--augmentation {aug} "
        f"--epochs {epochs} "
        f"--batch-size 16 "
        f"--seeds {seed} "
        f"--output {output}",
        f"Train {arch} + {aug} (seed={seed})"
    )


def step_ensemble_member(seed: int):
    return step_train(
        "vit_b_16", "saa_all",
        f"results/ensemble_member_seed{seed}.json",
        seed=seed, epochs=30
    )


def step_vlm(condition: str):
    return run_cmd(
        f"python artifact_aware_rag.py "
        f"--data-dir data/cross_domain/cross_instrument "
        f"--output results/vlm_{condition}.json "
        f"--max-samples 50",
        f"VLM evaluation: {condition}"
    )


def step_figures():
    return run_cmd(
        "python generate_paper_figures.py --output-dir figures",
        "Generate paper figures and tables"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Master pipeline runner.")
    parser.add_argument("--phase", type=int, help="Run only this phase (1-5)")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--reset", type=str, help="Reset a specific checkpoint")
    args = parser.parse_args()

    # Reset checkpoint if requested
    if args.reset:
        ckpts = load_checkpoints()
        if args.reset in ckpts:
            del ckpts[args.reset]
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(ckpts, f, indent=2)
            logger.info("Reset checkpoint: %s", args.reset)
        return

    steps = get_steps(skip_vlm=args.skip_vlm)

    # Filter by phase
    if args.phase:
        steps = [(n, d, p) for n, d, p in steps if p == args.phase]

    # Status only
    if args.status:
        ckpts = load_checkpoints()
        logger.info("=" * 72)
        logger.info("  PIPELINE STATUS")
        logger.info("=" * 72)
        for name, desc, phase in steps:
            status = ckpts.get(name, {}).get("status", "pending")
            icon = "DONE" if status == "done" else "RUNNING" if status == "running" else "pending"
            ts = ckpts.get(name, {}).get("timestamp", "")
            logger.info("  [%s] %s  %s", icon, desc, ts)
        done = sum(1 for n, _, _ in steps if ckpts.get(n, {}).get("status") == "done")
        logger.info("-" * 72)
        logger.info("  %d/%d steps completed", done, len(steps))
        logger.info("=" * 72)
        return

    # Dry run
    if args.dry_run:
        logger.info("=" * 72)
        logger.info("  DRY RUN — Steps that would execute:")
        logger.info("=" * 72)
        for name, desc, phase in steps:
            status = "SKIP (done)" if is_done(name) else "WOULD RUN"
            logger.info("  [%s] P%d: %s", status, phase, desc)
        return

    # Run
    tracker = ProgressTracker(len(steps))

    STEP_IMPLS = {
        "data_prep": step_data_prep,
        "pillow_noise": step_pillow_noise,
        "imaging_artifacts": step_imaging_artifacts,
        "fourier_analysis": step_fourier_analysis,
        "ood_detection": step_ood_detection,
        "train_vit_baseline": lambda: step_train("vit_b_16", "standard", "results/baselines_vit.json"),
        "train_resnet_baseline": lambda: step_train("resnet50", "standard", "results/baselines_resnet.json"),
        "train_convnext_baseline": lambda: step_train("convnext_tiny", "standard", "results/baselines_convnext.json"),
        "train_saa_amplitude": lambda: step_train("vit_b_16", "saa_amplitude", "results/saa_amplitude.json"),
        "train_saa_noise": lambda: step_train("vit_b_16", "saa_noise", "results/saa_noise.json"),
        "train_saa_phase": lambda: step_train("vit_b_16", "saa_phase", "results/saa_phase.json"),
        "train_saa_band": lambda: step_train("vit_b_16", "saa_band", "results/saa_band.json"),
        "train_saa_all": lambda: step_train("vit_b_16", "saa_all", "results/saa_all.json"),
        "train_ensemble_seed1": lambda: step_ensemble_member(42),
        "train_ensemble_seed2": lambda: step_ensemble_member(123),
        "train_ensemble_seed3": lambda: step_ensemble_member(456),
        "train_ensemble_seed4": lambda: step_ensemble_member(789),
        "train_ensemble_seed5": lambda: step_ensemble_member(999),
        "train_whoi22_vit": lambda: step_train(
            "vit_b_16", "standard", "results/whoi22_vit.json",
            source="train/WHOI22", targets="test/ZooScan20 test/ZooLake35",
            data_dir="data/cross_domain/whoi22_full", epochs=30
        ),
        "train_whoi22_resnet": lambda: step_train(
            "resnet50", "standard", "results/whoi22_resnet.json",
            source="train/WHOI22", targets="test/ZooScan20 test/ZooLake35",
            data_dir="data/cross_domain/whoi22_full", epochs=30
        ),
        "vlm_baseline": lambda: step_vlm("baseline"),
        "vlm_morphological_rag": lambda: step_vlm("morphological_rag"),
        "vlm_aa_rag": lambda: step_vlm("aa_rag"),
        "vlm_aa_rag_ref": lambda: step_vlm("aa_rag_ref"),
        "generate_figures": step_figures,
    }

    for name, desc, phase in steps:
        if is_done(name):
            logger.info("  [SKIP] %s (already done)", desc)
            tracker.completed += 1
            continue

        tracker.start_step(desc)
        save_checkpoint(name, "running")

        impl = STEP_IMPLS.get(name)
        if impl is None:
            logger.warning("  No implementation for %s", name)
            tracker.end_step(False)
            continue

        try:
            success = impl()
        except Exception as e:
            logger.error("  Exception: %s", e)
            success = False

        if success:
            save_checkpoint(name, "done")
        else:
            save_checkpoint(name, "failed")
            logger.error("  STEP FAILED — continuing with next step")

        tracker.end_step(success)

    tracker.summary()


if __name__ == "__main__":
    main()
