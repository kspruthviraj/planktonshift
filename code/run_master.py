#!/usr/bin/env python3
"""
run_master.py — Master orchestrator for all corrected experiments.

Runs experiments in dependency order:
  Phase 1 (CPU, fast): Fourier analysis, information allocation figure
  Phase 2 (GPU, moderate): Frequency masking, amplitude vs phase, DA baselines
  Phase 3 (GPU, heavy): SBA cross-instrument (5 seeds × 3 strategies)
  Phase 4 (GPU, heavy): Pillow impact (evaluates on 10 OOD days × 2 resize methods)
  Phase 5 (CPU, fast): Statistics aggregation

Usage:
    python code/run_master.py [--phase N] [--dry-run]
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
        ("Fourier Analysis (Pipeline A)", "run_fourier_analysis_corrected.py"),
        ("Information Allocation Figure", "run_information_allocation_figure.py"),
    ],
    2: [
        ("Frequency Masking (corrected)", "run_frequency_masking_corrected.py"),
        ("Amplitude vs Phase", "run_amplitude_vs_phase.py"),
        ("DA Baselines", "run_da_baselines.py"),
    ],
    3: [
        ("SBA Cross-Instrument (5 seeds × 3 strategies)", "run_sba_cross_instrument_corrected.py"),
    ],
    4: [
        ("Pillow Impact (corrected weighting)", "run_pillow_impact_corrected.py"),
    ],
    5: [
        ("Statistics Aggregation", "run_statistics.py"),
    ],
}


def run_script(name, script, dry_run=False):
    log_file = LOG_DIR / f"{script.replace('.py', '')}.log"
    cmd = [PYTHON, str(CODE / script)]
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Script:  {script}")
    print(f"  Log:     {log_file}")
    print(f"{'='*60}")
    if dry_run:
        print(f"  [DRY RUN] {cmd}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0, help="Run only phase N (0=all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    phases_to_run = [args.phase] if args.phase > 0 else sorted(PHASES.keys())
    print(f"PlanktonShift — Master Experiment Runner")
    print(f"Phases: {phases_to_run}")
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
