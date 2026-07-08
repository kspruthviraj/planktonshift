"""
generate_figures.py
===================
Generate paper-ready figures and tables from evaluation results.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fig1_cross_domain_accuracy(data: dict, out_dir: str) -> None:
    """Bar chart comparing baseline vs RAG accuracy across domains."""
    per_sample = data["per_sample"]

    domains = ["IFCB", "ZooScan"]
    modes = ["baseline", "rag"]
    mode_labels = {"baseline": "Unconstrained VLM", "rag": "RAG-Grounded VLM"}

    acc = {}
    for mode in modes:
        for domain in domains:
            results = [r for r in per_sample if r["mode"] == mode and r["domain"] == domain]
            correct = sum(1 for r in results if r["correct"])
            acc[(mode, domain)] = correct / len(results) if results else 0

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(domains))
    width = 0.32
    colors = {"baseline": "#d62728", "rag": "#2ca02c"}

    for i, mode in enumerate(modes):
        vals = [acc[(mode, d)] * 100 for d in domains]
        bars = ax.bar(x + i * width, vals, width, label=mode_labels[mode],
                      color=colors[mode], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax.set_xlabel("Imaging Domain", fontsize=11)
    ax.set_title("Cross-Domain Accuracy: Unconstrained vs. RAG-Grounded VLM", fontsize=12, pad=12)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_cross_domain_accuracy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_cross_domain_accuracy.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig2_per_class_lift(data: dict, out_dir: str) -> None:
    """Horizontal bar chart showing per-class accuracy lift from RAG."""
    per_sample = data["per_sample"]
    classes = sorted(set(r["true_label"] for r in per_sample))

    lifts = []
    bl_accs = []
    rag_accs = []
    for cls in classes:
        for mode in ["baseline", "rag"]:
            results = [r for r in per_sample if r["mode"] == mode and r["true_label"] == cls]
            correct = sum(1 for r in results if r["correct"])
            acc = correct / len(results) if results else 0
            if mode == "baseline":
                bl = acc
            else:
                rg = acc
        lifts.append((rg - bl) * 100)
        bl_accs.append(bl * 100)
        rag_accs.append(rg * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y = np.arange(len(classes))
    height = 0.35

    bars_bl = ax.barh(y + height / 2, bl_accs, height, label="Unconstrained VLM",
                       color="#d62728", alpha=0.85, edgecolor="white")
    bars_rag = ax.barh(y - height / 2, rag_accs, height, label="RAG-Grounded VLM",
                        color="#2ca02c", alpha=0.85, edgecolor="white")

    # Add lift annotations
    for i, (lift, bl, rg) in enumerate(zip(lifts, bl_accs, rag_accs)):
        color = "green" if lift > 0 else "red"
        sign = "+" if lift > 0 else ""
        ax.annotate(f"{sign}{lift:.1f}%", xy=(max(bl, rg) + 2, i),
                    fontsize=8, color=color, fontweight="bold", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy: RAG Grounding Effect", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 120)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_per_class_accuracy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_per_class_accuracy.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig3_confusion_matrices(data: dict, out_dir: str) -> None:
    """Side-by-side confusion matrices for baseline and RAG."""
    from collections import Counter

    per_sample = data["per_sample"]
    classes = sorted(set(r["true_label"] for r in per_sample))
    cls_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, mode in zip(axes, ["baseline", "rag"]):
        mode_results = [r for r in per_sample if r["mode"] == mode]
        cm = np.zeros((n, n), dtype=int)
        for r in mode_results:
            true_i = cls_idx.get(r["true_label"])
            pred_i = cls_idx.get(r["predicted_label"])
            if true_i is not None and pred_i is not None:
                cm[true_i][pred_i] += 1

        # Normalize by row (true labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        title = "Unconstrained VLM" if mode == "baseline" else "RAG-Grounded VLM"
        ax.set_title(title, fontsize=11)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy (row-normalized)")
    plt.suptitle("Confusion Matrices: Cross-Domain Evaluation", fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, "fig3_confusion_matrices.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_confusion_matrices.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def table1_latex(data: dict, out_dir: str) -> None:
    """Generate LaTeX table for the paper."""
    per_sample = data["per_sample"]

    domains = ["IFCB", "ZooScan"]
    modes = ["baseline", "rag"]

    rows = []
    for cls in sorted(set(r["true_label"] for r in per_sample)):
        row = [cls]
        for domain in domains:
            for mode in modes:
                results = [r for r in per_sample
                           if r["mode"] == mode and r["domain"] == domain and r["true_label"] == cls]
                if results:
                    correct = sum(1 for r in results if r["correct"])
                    acc = correct / len(results) * 100
                    row.append(f"{acc:.1f}")
                else:
                    row.append("---")
        rows.append(row)

    # Overall row
    overall = ["\\textbf{Overall}"]
    for domain in domains:
        for mode in modes:
            results = [r for r in per_sample if r["mode"] == mode and r["domain"] == domain]
            correct = sum(1 for r in results if r["correct"])
            acc = correct / len(results) * 100
            overall.append(f"\\textbf{{{acc:.1f}}}")
    rows.append(overall)

    latex = "\\begin{table}[t]\n\\centering\n\\caption{"
    latex += "Cross-domain classification accuracy (\\%) on the Planktonzilla-17M evaluation set. "
    latex += "IFCB: Imaging FlowCytobot (flow-through). ZooScan: flatbed scanner. "
    latex += "RAG-grounded VLM injects morphological rules into the prompt.}\n"
    latex += "\\label{tab:main_results}\n"
    latex += "\\resizebox{\\columnwidth}{!}{\n"
    latex += "\\begin{tabular}{l|cc|cc}\n"
    latex += "\\toprule\n"
    latex += " & \\multicolumn{2}{c|}{\\textbf{IFCB}} & \\multicolumn{2}{c}{\\textbf{ZooScan}} \\\\\n"
    latex += " & Baseline & RAG & Baseline & RAG \\\\\n"
    latex += "\\midrule\n"
    for row in rows:
        latex += " & ".join(row) + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}\n\\end{table}\n"

    path = os.path.join(out_dir, "table1_results.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/eval_6class.json")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_results(args.results)

    fig1_cross_domain_accuracy(data, args.out_dir)
    fig2_per_class_lift(data, args.out_dir)
    fig3_confusion_matrices(data, args.out_dir)
    table1_latex(data, args.out_dir)

    print("\nAll figures and tables generated.")


if __name__ == "__main__":
    main()
