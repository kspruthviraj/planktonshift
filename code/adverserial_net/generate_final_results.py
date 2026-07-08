"""
generate_final_results.py
=========================
Generate all paper figures, tables, and summary statistics from the
completed experiments.

Usage:
    python generate_final_results.py --output-dir figures
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_all_results(results_dir="results"):
    """Load all experiment result files."""
    all_results = {}
    for f in sorted(glob.glob(f"{results_dir}/*.json")):
        name = os.path.basename(f).replace(".json", "")
        if any(skip in name for skip in ["checkpoints", "shift_spectrum", "fourier", "ood", "pillow", "imaging", "manifest"]):
            continue
        try:
            data = json.load(open(f))
            for key, val in data.items():
                if isinstance(val, dict) and "domains" in val:
                    all_results[f"{name}_{key}"] = {
                        "file": name,
                        "architecture": val.get("architecture", "?"),
                        "augmentation": val.get("augmentation", "?"),
                        "seed": val.get("seed", 0),
                        "domains": val["domains"],
                    }
        except Exception:
            continue
    return all_results


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.size": 10, "font.family": "serif",
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    out = Path("figures")
    out.mkdir(exist_ok=True)

    results = load_all_results()

    # ===================================================================
    # Table 1: Cross-domain accuracy (all methods)
    # ===================================================================
    rows = []
    for key, val in sorted(results.items(), key=lambda x: -x[1]["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0)):
        src = val["domains"].get("train/DataShift_IFCB", {}).get("accuracy", 0) * 100
        tgt = val["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0) * 100
        if src == 0 or tgt == 0:
            continue
        ci = val["domains"].get("test/DataShift_ZooScan", {}).get("ci_95", [0, 0])
        ece = val["domains"].get("test/DataShift_ZooScan", {}).get("ece", 0)
        drop = src - tgt
        aug = val["augmentation"]
        tta = "tta" in val["file"]
        rows.append({
            "name": val["file"],
            "aug": aug,
            "src": src,
            "tgt": tgt,
            "drop": drop,
            "ci": ci,
            "ece": ece,
            "tta": tta,
        })

    # Deduplicate by file name (keep best)
    seen = {}
    for r in rows:
        if r["name"] not in seen or r["tgt"] > seen[r["name"]]["tgt"]:
            seen[r["name"]] = r
    rows = sorted(seen.values(), key=lambda x: -x["tgt"])

    # LaTeX table
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-domain plankton classification accuracy (\\%). "
        "All models trained on DataShift IFCB (16 classes, 384 images), "
        "evaluated on DataShift ZooScan (12 classes, 137 images). "
        "Best in \\textbf{bold}.}",
        "\\label{tab:cross_domain}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Augmentation} & \\textbf{IFCB} & "
        "\\textbf{ZooScan} & \\textbf{Drop} & \\textbf{95\\% CI} \\\\",
        "\\midrule",
    ]

    for i, r in enumerate(rows[:15]):
        bold = "\\textbf{" if r == rows[0] else ""
        bold_end = "}" if r == rows[0] else ""
        tta_str = " + TTA" if r["tta"] else ""
        latex_lines.append(
            f"{r['name'].replace('_', '\\_')} & {r['aug']}{tta_str} & "
            f"{r['src']:.1f} & {bold}{r['tgt']:.1f}{bold_end} & "
            f"{r['drop']:.1f} & [{r['ci'][0]*100:.1f}, {r['ci'][1]*100:.1f}] \\\\"
        )

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ])

    with open(out / "table_cross_domain.tex", "w") as f:
        f.write("\n".join(latex_lines))

    # ===================================================================
    # Figure 1: Cross-domain accuracy comparison (bar chart)
    # ===================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    # Select key methods for comparison
    key_methods = [
        ("baseline", "ViT + standard", "#2196F3"),
        ("baseline_tta", "ViT + standard + TTA", "#64B5F6"),
        ("saa_noise_v2", "ViT + SAA noise", "#FF9800"),
        ("saa_band_v2", "ViT + SAA band", "#FF5722"),
        ("saa_band_tta", "ViT + SAA band + TTA", "#4CAF50"),
        ("ens_saa_band_tta_seed42", "ViT + SAA band + TTA (best seed)", "#388E3C"),
    ]

    names = []
    accs = []
    colors = []
    for method_key, label, color in key_methods:
        for key, val in results.items():
            if method_key in val["file"]:
                tgt = val["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0) * 100
                if tgt > 0:
                    names.append(label)
                    accs.append(tgt)
                    colors.append(color)
                    break

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ZooScan Accuracy (%)")
    ax.set_title("Cross-Domain Accuracy: IFCB → ZooScan")
    ax.set_ylim(40, 60)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=46.7, color="red", linestyle="--", alpha=0.5, label="Baseline (46.7%)")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(str(out / "fig_cross_domain_accuracy.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out / "fig_cross_domain_accuracy.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Figure 2: SAA ablation comparison
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    ablation_methods = [
        ("baselines_cross_instrument", "standard", "Baseline"),
        ("saa_amplitude_v2", "saa_amplitude", "SAA amplitude"),
        ("saa_noise_v2", "saa_noise", "SAA spectral noise"),
        ("saa_band_v2", "saa_band", "SAA band adversarial"),
        ("saa_fda_v2", "saa_fda", "FDA swap"),
        ("saa_phase_v2", "saa_phase", "SAA phase preserve"),
        ("saa_band_tta", "saa_band", "SAA band + TTA"),
    ]

    ab_names = []
    ab_accs = []
    ab_colors = []
    color_map = {
        "Baseline": "#9E9E9E",
        "SAA amplitude": "#FF9800",
        "SAA spectral noise": "#2196F3",
        "SAA band adversarial": "#4CAF50",
        "FDA swap": "#9C27B0",
        "SAA phase preserve": "#d32f2f",
        "SAA band + TTA": "#388E3C",
    }

    for method_key, aug, label in ablation_methods:
        for key, val in results.items():
            if method_key in val["file"]:
                tgt = val["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0) * 100
                if tgt > 0:
                    ab_names.append(label)
                    ab_accs.append(tgt)
                    ab_colors.append(color_map.get(label, "#607d8b"))
                    break

    bars = ax.bar(range(len(ab_names)), ab_accs, color=ab_colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(ab_names)))
    ax.set_xticklabels(ab_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ZooScan Accuracy (%)")
    ax.set_title("SAA Augmentation Ablation")
    ax.set_ylim(35, 58)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=46.7, color="red", linestyle="--", alpha=0.5)

    for bar, acc in zip(bars, ab_accs):
        diff = acc - 46.7
        sign = "+" if diff > 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%\n({sign}{diff:.1f}%)", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out / "fig_saa_ablation.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out / "fig_saa_ablation.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Figure 3: Ensemble seed diversity
    # ===================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    seed_results = []
    for key, val in results.items():
        if "ens_saa_band_tta" in val["file"]:
            seed = val.get("seed", 0)
            tgt = val["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0) * 100
            if tgt > 0:
                seed_results.append((seed, tgt))

    seed_results.sort()
    seeds = [s for s, _ in seed_results]
    accs = [a for _, a in seed_results]

    ax.bar(range(len(seeds)), accs, color="#4CAF50", alpha=0.85)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=9)
    ax.set_ylabel("ZooScan Accuracy (%)")
    ax.set_title("Ensemble Seed Diversity (SAA band + TTA)")
    ax.set_ylim(40, 58)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=np.mean(accs), color="blue", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(accs):.1f}%")
    ax.axhline(y=46.7, color="red", linestyle="--", alpha=0.5, label="Baseline: 46.7%")

    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.3, f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out / "fig_ensemble_diversity.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(out / "fig_ensemble_diversity.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Summary statistics
    # ===================================================================
    print("=" * 72)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Method':45s} {'IFCB':>6s} {'ZooScan':>8s} {'Drop':>6s}")
    print("  " + "-" * 68)
    for r in rows[:10]:
        tta = " + TTA" if r["tta"] else ""
        print(f"  {r['name']:45s} {r['src']:5.1f}% {r['tgt']:5.1f}% {r['drop']:5.1f}%")
    print("=" * 72)

    # Key comparisons
    baseline = next((r for r in rows if "baseline" in r["name"] and not r["tta"]), None)
    best = rows[0]
    if baseline:
        improvement = best["tgt"] - baseline["tgt"]
        print(f"\n  Baseline: {baseline['tgt']:.1f}%")
        print(f"  Best:     {best['tgt']:.1f}% ({best['name']})")
        print(f"  Improvement: +{improvement:.1f}%")
        print(f"  Drop reduction: {baseline['drop']:.1f}% → {best['drop']:.1f}%")

    print(f"\n  Figures and tables saved to: {out}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
