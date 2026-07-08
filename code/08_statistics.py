"""
08_statistics.py — Collect all results into one summary with proper statistics.

STEP 8: STATISTICS AGGREGATION
==============================

After running Steps 01-07, the results are spread across multiple JSON files.
This script collects them into a single summary file with proper statistical
measures.

STATISTICAL METHODS USED:
=========================

1. BOOTSTRAP CONFIDENCE INTERVALS (for accuracy estimates):
   Given N binary correctness values c_1, ..., c_N (1=correct, 0=wrong):
     (a) Resample N values WITH REPLACEMENT, compute mean
     (b) Repeat B=2000 times
     (c) 95% CI = [2.5th percentile, 97.5th percentile] of bootstrap means

   This gives a honest interval for "if we repeated the experiment, where
   would the accuracy likely fall?" Unlike the binomial approximation,
   bootstrap makes NO assumptions about the distribution shape.

2. McNEMAR'S TEST (for comparing two classifiers):
   Given classifiers A and B tested on the SAME images:
     n01 = # images A correct, B wrong
     n10 = # images A wrong, B correct

   Under the null hypothesis (A and B are equally good):
     n01 ~ Binomial(n01 + n10, 0.5)

   p-value = P( Binomial <= min(n01, n10) | n01+n10, 0.5 )  (two-sided)

   If p < 0.05, the difference is statistically significant.
   If p >= 0.05, we cannot distinguish the two classifiers.

3. PER-SEED AGGREGATION (for multi-seed experiments):
   For each augmentation strategy, report:
     mean_accuracy = (1/K) * SUM_k accuracy_k    (K=5 seeds)
     std_accuracy  = std(accuracy_1, ..., accuracy_K)
     seed_ci_95    = bootstrap CI over the K seed accuracies

WHY IT MATTERS:
  A single summary file makes it easy to check all key numbers in the paper
  against the actual computed results. Every number in the paper should be
  traceable to this file.

Output: results/tier1_corrected/statistics_summary.json
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS
from utils_pipeline import bootstrap_ci, mcnemar_test

OUT = RESULTS / "tier1_corrected" / "statistics_summary.json"
BASE = RESULTS / "tier1_corrected"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1. Fourier analysis domain classifier
    fa_path = BASE / "fourier_analysis.json"
    if fa_path.exists():
        with open(fa_path) as f:
            fa = json.load(f)
        dc = fa.get("domain_classifier_gray", {})
        dc_rgb = fa.get("domain_classifier_rgb", {})
        summary["fourier_domain_classifier"] = {
            "gray": {"accuracy": dc.get("accuracy"), "ci_95": dc.get("ci_95"), "n": dc.get("n_samples")},
            "rgb": {"accuracy": dc_rgb.get("accuracy"), "ci_95": dc_rgb.get("ci_95"), "n": dc_rgb.get("n_samples")},
        }
        for pair, sh in fa.get("shift_spectra", {}).items():
            summary.setdefault("shift_spectra", {})[pair] = {
                "total_shift_energy": sh["total_shift_energy"],
                "most_shifted_band": max(sh["band_analysis"], key=lambda b: b["mean_abs_diff"])["band"],
            }

    # 2. Frequency masking
    fm_path = BASE / "frequency_masking.json"
    if fm_path.exists():
        with open(fm_path) as f:
            fm = json.load(f)
        summary["frequency_masking"] = {}
        for band in ["low", "mid", "high", "all"]:
            if band in fm:
                r = fm[band]
                summary["frequency_masking"][band] = {
                    "species_accuracy": r["species_accuracy"],
                    "species_ci95": r["species_ci95"],
                    "domain_accuracy_amplitude": r["domain_accuracy_amplitude"],
                    "domain_delta_vs_all": r["domain_delta_vs_all"],
                }

    # 3. SBA cross-instrument (all seeds)
    sba_path = BASE / "sba_cross_instrument.json"
    if sba_path.exists():
        with open(sba_path) as f:
            sba = json.load(f)
        summary["sba_cross_instrument"] = {}
        for aug in ["standard", "sba_band", "phase_preserve"]:
            if aug in sba:
                r = sba[aug]
                accs = [s["accuracy"] for s in r["per_seed"]]
                summary["sba_cross_instrument"][aug] = {
                    "mean": r["mean_accuracy"],
                    "std": r["std_accuracy"],
                    "seed_ci_95": r["seed_ci_95"],
                    "best_seed": r["best_seed"],
                    "worst_seed": r["worst_seed"],
                    "per_seed_accs": accs,
                }
        if "mcnemar_standard_vs_sba_seed42" in sba:
            summary["mcnemar_standard_vs_sba"] = sba["mcnemar_standard_vs_sba_seed42"]

    # 4. Pillow impact
    pi_path = BASE / "pillow_impact.json"
    if pi_path.exists():
        with open(pi_path) as f:
            pi = json.load(f)
        summary["pillow_impact"] = {
            "nearest_macro": pi["pillow_6_nearest_macro"],
            "nearest_micro": pi["pillow_6_nearest_micro"],
            "bicubic_macro": pi["pillow_7_bicubic_macro"],
            "bicubic_micro": pi["pillow_7_bicubic_micro"],
            "macro_diff": pi["macro_diff_percent"],
            "micro_diff": pi["micro_diff_percent"],
            "mcnemar": pi["mcnemar"],
        }

    # 5. Amplitude vs phase
    ap_path = BASE / "amplitude_vs_phase.json"
    if ap_path.exists():
        with open(ap_path) as f:
            ap = json.load(f)
        summary["amplitude_vs_phase"] = {}
        for cond in ["normal", "phase_scrambled", "amp_swapped", "both_scrambled"]:
            if cond in ap:
                summary["amplitude_vs_phase"][cond] = {
                    "species_accuracy": ap[cond]["species_accuracy"],
                    "ci_95": ap[cond]["ci_95"],
                }

    # 6. DA baselines
    da_path = BASE / "da_baselines.json"
    if da_path.exists():
        with open(da_path) as f:
            da = json.load(f)
        summary["da_baselines"] = {}
        for name in da:
            if name.startswith("_"):
                continue
            summary["da_baselines"][name] = {
                "accuracy": da[name]["accuracy"],
                "ci_95": da[name]["ci_95"],
            }

    # 7. Information allocation
    ia_path = BASE / "information_allocation.json"
    if ia_path.exists():
        with open(ia_path) as f:
            ia = json.load(f)
        summary["information_allocation"] = {
            "n_bands": ia["n_bands"],
            "domain_accuracy_per_band": ia["domain_accuracy_per_band"],
            "class_separability_per_band": ia["class_separability_per_band"],
        }

    # 8. Temporal OOD results (from existing results)
    pc_path = RESULTS / "perchannel_sba" / "results.json"
    if pc_path.exists():
        with open(pc_path) as f:
            pc = json.load(f)
        summary["temporal_ood_perchannel_sba"] = {
            "macro": pc["macro"],
            "vs_chen": pc["vs_chen"],
            "per_day": pc["per_day"],
            "config": pc["config"],
        }

    # Print summary
    print(f"{'='*60}\nSTATISTICS SUMMARY\n{'='*60}")
    for section, data in summary.items():
        print(f"\n--- {section} ---")
        print(json.dumps(data, indent=2, default=str)[:500])

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
