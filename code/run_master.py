#!/usr/bin/env python3
"""
run_master.py — Run all corrected experiments in order.

This script orchestrates the entire analysis pipeline. Each step produces
results that the next step may depend on (e.g., the Fourier analysis produces
the shift spectrum that SBA uses).

Steps are grouped into phases by resource requirement:
  Phase 1 (CPU only, ~5 min):  Fourier analysis + information allocation figure
  Phase 2 (GPU, ~1-2 hours):   Frequency masking, amplitude vs phase, DA baselines
  Phase 3 (GPU, ~6 hours):     SBA cross-instrument (5 seeds × 3 strategies)
  Phase 4 (GPU, ~2 hours):     Pillow impact (10 OOD days × 2 resize methods)
  Phase 5 (CPU, ~1 min):       Aggregate statistics

Usage:
    python code/run_master.py              # Run everything
    python code/run_master.py --phase 1    # Run only phase 1
    python code/run_master.py --dry-run    # Show what would run
"""
import sys, os, time, argparse, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
LOG_DIR = ROOT / "results" / "tier1_corrected" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

PHASES = {
    1: [
        ("01 Fourier Analysis", "01_fourier_analysis.py"),
        ("07 Information Allocation Figure", "07_information_allocation.py"),
    ],
    2: [
        ("03 Frequency Masking", "03_frequency_masking.py"),
        ("02 Amplitude vs Phase", "02_amplitude_vs_phase.py"),
        ("05 DA Baselines", "05_da_baselines.py"),
    ],
    3: [
        ("04 SBA Cross-Instrument (5 seeds)", "04_sba_cross_instrument.py"),
    ],
    4: [
        ("06 Pillow Impact", "06_pillow_impact.py"),
    ],
    5: [
        ("08 Statistics Aggregation", "08_statistics.py"),
    ],
}


def run_script(name, script, dry_run=False):
    log_file = LOG_DIR / f"{script.replace('.py', '')}.log"
    cmd = [PYTHON, str(CODE / script)]
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Script:  code/{script}")
    print(f"  Log:     {log_file.relative_to(ROOT)}")
    print(f"{'='*60}")
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return 0
    t0 = time.time()
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(CODE)})
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    print(f"  {status} in {dt:.0f}s")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Run all corrected experiments")
    ap.add_argument("--phase", type=int, default=0, help="Run only phase N (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="Show commands without running")
    args = ap.parse_args()

    phases_to_run = [args.phase] if args.phase > 0 else sorted(PHASES.keys())
    print(f"PlanktonShift — Master Experiment Runner")
    print(f"Phases to run: {phases_to_run}")
    print(f"Dry run: {args.dry_run}")

    results = {}
    t_total = time.time()
    for phase in phases_to_run:
        print(f"\n{'#'*60}\n# PHASE {phase}\n{'#'*60}")
        for name, script in PHASES[phase]:
            rc = run_script(name, script, args.dry_run)
            results[script] = rc
            if rc != 0 and not args.dry_run:
                print(f"  WARNING: {script} failed (rc={rc}). Continuing...")

    dt = time.time() - t_total
    print(f"\n{'='*60}\nMASTER RUN COMPLETE ({dt:.0f}s)\n{'='*60}")
    for script, rc in results.items():
        status = "OK" if rc == 0 else "FAIL"
        print(f"  {status}: {script}")

    if all(rc == 0 for rc in results.values()):
        print("\nAll experiments completed successfully!")
    else:
        failed = [s for s, rc in results.items() if rc != 0]
        print(f"\nFailed: {failed}")


if __name__ == "__main__":
    main()
