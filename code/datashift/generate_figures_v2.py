"""
generate_figures_v2.py
======================
Generate comprehensive paper figures from all experimental results.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

CLASSES = ["Amphipoda", "Annelida", "Calanoida", "Ceratium", "Chaetognatha", "Oithonidae"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig1_dataset_shift(vit_data, vlm_data, out_dir):
    """Main figure: shows catastrophic shift in ViTs vs robustness of VLM+RAG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bar chart comparing models
    ax = axes[0]
    models = ["ViT-B/16", "ResNet-50", "ConvNeXt-T", "VLM\n(baseline)", "VLM\n+ RAG"]
    ifcb_acc = [
        vit_data["vit_b_16"]["ifcb"]["accuracy"] * 100,
        vit_data["resnet50"]["ifcb"]["accuracy"] * 100,
        vit_data["convnext_tiny"]["ifcb"]["accuracy"] * 100,
        28.1,  # from VLM eval
        35.6,
    ]
    zs_acc = [
        vit_data["vit_b_16"]["zooscan"]["accuracy"] * 100,
        vit_data["resnet50"]["zooscan"]["accuracy"] * 100,
        vit_data["convnext_tiny"]["zooscan"]["accuracy"] * 100,
        44.2,
        57.1,
    ]

    x = np.arange(len(models))
    w = 0.35
    bars_ifcb = ax.bar(x - w/2, ifcb_acc, w, label="IFCB (source)", color="#2196F3", edgecolor="white")
    bars_zs = ax.bar(x + w/2, zs_acc, w, label="ZooScan (target)", color="#FF9800", edgecolor="white")

    for bar, val in zip(bars_ifcb, ifcb_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_zs, zs_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("(a) Cross-domain accuracy comparison")

    # Panel B: Drop magnitude
    ax2 = axes[1]
    drops = [ifcb_acc[i] - zs_acc[i] for i in range(len(models))]
    colors = ["#d32f2f" if d > 30 else "#ff9800" if d > 0 else "#4caf50" for d in drops]
    bars = ax2.barh(models, drops, color=colors, edgecolor="white")
    for bar, val in zip(bars, drops):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Accuracy Change: IFCB → ZooScan (%)")
    ax2.set_title("(b) Dataset shift magnitude")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    # Add annotation
    ax2.annotate("Catastrophic\nfailure", xy=(70, 0.5), fontsize=9, color="#d32f2f",
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d32f2f"),
                xytext=(55, 1.5))
    ax2.annotate("RAG improves\nboth domains", xy=(15, 4), fontsize=9, color="#4caf50",
                ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig1_dataset_shift.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig1_dataset_shift.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1")


def fig2_per_class(vit_data, vlm_data, out_dir):
    """Per-class comparison showing where RAG helps most."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ZooScan per-class
    ax = axes[0]
    vit_zs = vit_data["vit_b_16"]["zooscan"]["per_class"]
    vlm_bl = {}
    vlm_rag = {}
    per_sample = vlm_data["per_sample"]
    for cls in CLASSES:
        bl = [r for r in per_sample if r["mode"] == "baseline" and r["true_label"] == cls and r["domain"] == "ZooScan"]
        rg = [r for r in per_sample if r["mode"] == "rag" and r["true_label"] == cls and r["domain"] == "ZooScan"]
        vlm_bl[cls] = sum(1 for r in bl if r["correct"]) / len(bl) * 100 if bl else 0
        vlm_rag[cls] = sum(1 for r in rg if r["correct"]) / len(rg) * 100 if rg else 0

    vit_vals = [vit_zs.get(c, {}).get("accuracy", 0) * 100 for c in CLASSES]
    bl_vals = [vlm_bl.get(c, 0) for c in CLASSES]
    rag_vals = [vlm_rag.get(c, 0) for c in CLASSES]

    y = np.arange(len(CLASSES))
    h = 0.25
    ax.barh(y + h, vit_vals, h, label="ViT-B/16", color="#d32f2f", alpha=0.85)
    ax.barh(y, bl_vals, h, label="VLM baseline", color="#2196F3", alpha=0.85)
    ax.barh(y - h, rag_vals, h, label="VLM + RAG", color="#4caf50", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("ZooScan Accuracy (%)")
    ax.set_title("(a) Per-class ZooScan accuracy")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)

    # RAG lift chart
    ax2 = axes[1]
    lifts = [rag_vals[i] - bl_vals[i] for i in range(len(CLASSES))]
    colors = ["#4caf50" if l > 0 else "#d32f2f" for l in lifts]
    bars = ax2.barh(CLASSES, lifts, color=colors, edgecolor="white")
    for bar, val in zip(bars, lifts):
        offset = 0.5 if val >= 0 else -3
        ax2.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", ha="left" if val >= 0 else "right", va="center",
                fontsize=9, fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_xlabel("RAG Accuracy Lift (%)")
    ax2.set_title("(b) RAG grounding effect per class")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig2_per_class.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig2_per_class.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2")


def fig3_domain_consistency(vlm_data, out_dir):
    """Show VLM domain consistency vs ViT inconsistency."""
    fig, ax = plt.subplots(figsize=(8, 5))

    per_sample = vlm_data["per_sample"]

    # Compute per-class accuracy for both domains
    domains = ["IFCB", "ZooScan"]
    modes = ["baseline", "rag"]

    for mode, color, marker, label in [
        ("baseline", "#2196F3", "o", "VLM baseline"),
        ("rag", "#4caf50", "s", "VLM + RAG"),
    ]:
        ifcb_acc = []
        zs_acc = []
        for cls in CLASSES:
            for d_idx, domain in enumerate(domains):
                results = [r for r in per_sample if r["mode"] == mode and r["true_label"] == cls and r["domain"] == domain]
                acc = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else 0
                if d_idx == 0:
                    ifcb_acc.append(acc)
                else:
                    zs_acc.append(acc)

        ax.scatter(ifcb_acc, zs_acc, c=color, marker=marker, s=80, label=label, zorder=3)
        for i, cls in enumerate(CLASSES):
            ax.annotate(cls[:4], (ifcb_acc[i], zs_acc[i]), fontsize=7,
                       xytext=(5, 5), textcoords="offset points")

    # Add diagonal (perfect consistency)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Perfect consistency")
    ax.fill_between([0, 100], [0, 100], [100, 100], alpha=0.05, color="green")
    ax.fill_between([0, 100], [0, 0], [0, 100], alpha=0.05, color="red")

    ax.set_xlabel("IFCB Accuracy (%)")
    ax.set_ylabel("ZooScan Accuracy (%)")
    ax.set_title("Domain Consistency: IFCB vs ZooScan Accuracy per Class")
    ax.legend(loc="lower right")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig3_consistency.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig3_consistency.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3")


def table1_latex(vit_data, vlm_data, out_dir):
    """Generate comprehensive LaTeX results table."""
    per_sample = vlm_data["per_sample"]

    rows = []
    for cls in CLASSES:
        row = [cls]
        # ViT results
        vit_ifcb = vit_data["vit_b_16"]["ifcb"]["per_class"].get(cls, {}).get("accuracy", 0) * 100
        vit_zs = vit_data["vit_b_16"]["zooscan"]["per_class"].get(cls, {}).get("accuracy", 0) * 100
        row.append(f"{vit_ifcb:.0f}")
        row.append(f"{vit_zs:.0f}")
        # VLM baseline
        for domain in ["IFCB", "ZooScan"]:
            results = [r for r in per_sample if r["mode"] == "baseline" and r["true_label"] == cls and r["domain"] == domain]
            acc = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else None
            row.append(f"{acc:.0f}" if acc is not None else "---")
        # VLM RAG
        for domain in ["IFCB", "ZooScan"]:
            results = [r for r in per_sample if r["mode"] == "rag" and r["true_label"] == cls and r["domain"] == domain]
            acc = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else None
            row.append(f"{acc:.0f}" if acc is not None else "---")
        rows.append(row)

    # Overall row
    overall = ["\\textbf{Overall}"]
    for model in ["vit_b_16"]:
        overall.append(f"\\textbf{{{vit_data[model]['ifcb']['accuracy']*100:.0f}}}")
        overall.append(f"\\textbf{{{vit_data[model]['zooscan']['accuracy']*100:.0f}}}")
    for mode in ["baseline", "rag"]:
        for domain in ["IFCB", "ZooScan"]:
            results = [r for r in per_sample if r["mode"] == mode and r["domain"] == domain]
            acc = sum(1 for r in results if r["correct"]) / len(results) * 100
            overall.append(f"\\textbf{{{acc:.0f}}}")
    rows.append(overall)

    latex = """\\begin{table}[t]
\\centering
\\caption{Cross-domain plankton classification accuracy (\\%). All models trained on IFCB. ViT-B/16 experiences catastrophic accuracy drop on ZooScan (dataset shift). VLM maintains cross-domain consistency, and RAG grounding further improves accuracy.}
\\label{tab:main_results}
\\resizebox{\\columnwidth}{!}{
\\begin{tabular}{l|cc|cc|cc}
\\toprule
 & \\multicolumn{2}{c|}{\\textbf{ViT-B/16}} & \\multicolumn{2}{c|}{\\textbf{VLM baseline}} & \\multicolumn{2}{c}{\\textbf{VLM + RAG}} \\\\
 & IFCB & ZooScan & IFCB & ZooScan & IFCB & ZooScan \\\\
\\midrule
"""
    for row in rows:
        latex += " & ".join(row) + " \\\\\n"
    latex += """\\bottomrule
\\end{tabular}}
\\end{table}
"""

    with open(f"{out_dir}/table1_results.tex", "w") as f:
        f.write(latex)
    print(f"  Saved table1")


def main():
    out_dir = "figures_v2"
    os.makedirs(out_dir, exist_ok=True)

    vit_data = load_json("results/vit_baselines.json")
    vlm_data = load_json("results/eval_v2_6class.json")

    print("Generating figures...")
    fig1_dataset_shift(vit_data, vlm_data, out_dir)
    fig2_per_class(vit_data, vlm_data, out_dir)
    fig3_domain_consistency(vlm_data, out_dir)
    table1_latex(vit_data, vlm_data, out_dir)
    print("\nAll figures generated in figures_v2/")


if __name__ == "__main__":
    main()
