"""
generate_final_figures.py
=========================
Generate final paper figures from 4-domain evaluation + ViT baselines.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.labelsize": 11, "axes.titlesize": 12})

CLASSES = ["Amphipoda", "Annelida", "Calanoida", "Ceratium", "Chaetognatha", "Conochilus", "Coscinodiscus", "Daphnia", "Fragilaria", "Keratella", "Oithonidae"]

def load_json(p):
    with open(p) as f: return json.load(f)

def fig1_dataset_shift(vit_data, out_dir):
    """ViT/ResNet/ConvNeXt catastrophic failure across domains."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ["ViT-B/16", "ResNet-50", "ConvNeXt-T"]
    ifcb = [vit_data[m]["ifcb"]["accuracy"]*100 for m in ["vit_b_16","resnet50","convnext_tiny"]]
    zs = [vit_data[m]["zooscan"]["accuracy"]*100 for m in ["vit_b_16","resnet50","convnext_tiny"]]
    
    x = np.arange(len(models)); w = 0.35
    b1 = ax1.bar(x-w/2, ifcb, w, label="IFCB (source)", color="#2196F3")
    b2 = ax1.bar(x+w/2, zs, w, label="ZooScan (target)", color="#FF9800")
    for b, v in zip(b1, ifcb): ax1.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(b2, zs): ax1.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)"); ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylim(0, 115); ax1.legend(); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.set_title("(a) Source vs target domain accuracy"); ax1.grid(axis="y", alpha=0.3)
    
    drops = [ifcb[i]-zs[i] for i in range(3)]
    colors = ["#d32f2f" for _ in drops]
    bars = ax2.barh(models, drops, color=colors)
    for b, v in zip(bars, drops): ax2.text(b.get_width()+1, b.get_y()+b.get_height()/2, f"{v:.1f}%", ha="left", va="center", fontweight="bold")
    ax2.set_xlabel("Accuracy Drop (%)"); ax2.set_title("(b) Dataset shift magnitude")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig1_dataset_shift.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig1_dataset_shift.pdf", bbox_inches="tight")
    plt.close(fig); print("  Saved fig1")

def fig2_rag_lift(vlm_data, out_dir):
    """RAG lift across 4 domains."""
    per_sample = vlm_data["per_sample"]
    domains = ["IFCB", "IFCB-NES", "ZooScan", "ZooLake"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Compute overall lift per domain
    lifts = {}
    for domain in domains:
        for mode in ["baseline", "rag"]:
            results = [r for r in per_sample if r["mode"] == mode and r["domain"] == domain]
            acc = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else 0
            if mode == "baseline": bl = acc
            else: rag = acc
        lifts[domain] = {"baseline": bl, "rag": rag, "lift": rag - bl}
    
    x = np.arange(len(domains)); w = 0.35
    b1 = ax.bar(x-w/2, [lifts[d]["baseline"] for d in domains], w, label="VLM baseline", color="#2196F3")
    b2 = ax.bar(x+w/2, [lifts[d]["rag"] for d in domains], w, label="VLM + RAG", color="#4caf50")
    for b, v in zip(b1, [lifts[d]["baseline"] for d in domains]): ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    for b, v in zip(b2, [lifts[d]["rag"] for d in domains]): ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    
    # Add lift annotations
    for i, d in enumerate(domains):
        lift = lifts[d]["lift"]
        color = "green" if lift > 0 else "red"
        ax.annotate(f"{lift:+.1f}%", xy=(i, max(lifts[d]["baseline"], lifts[d]["rag"]) + 4),
                   fontsize=9, fontweight="bold", ha="center", color=color)
    
    ax.set_ylabel("Overall Accuracy (%)"); ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylim(0, 105); ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("RAG Grounding Effect Across 4 Imaging Domains"); ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig2_rag_lift.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig2_rag_lift.pdf", bbox_inches="tight")
    plt.close(fig); print("  Saved fig2")

def fig3_per_class_heatmap(vlm_data, out_dir):
    """Heatmap of RAG lift per class per domain."""
    per_sample = vlm_data["per_sample"]
    domains = ["IFCB", "IFCB-NES", "ZooScan", "ZooLake"]
    
    # Get classes that appear in at least 1 domain
    classes_in_data = sorted(set(r["true_label"] for r in per_sample))
    
    # Compute lift matrix
    lift_matrix = []
    for cls in classes_in_data:
        row = []
        for domain in domains:
            for mode in ["baseline", "rag"]:
                results = [r for r in per_sample if r["mode"] == mode and r["true_label"] == cls and r["domain"] == domain]
                acc = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else float("nan")
                if mode == "baseline": bl = acc
                else: rag = acc
            if not results or np.isnan(bl):
                row.append(float("nan"))
            else:
                row.append(rag - bl)
        lift_matrix.append(row)
    
    lift_arr = np.array(lift_matrix)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(lift_arr, cmap="RdYlGn", vmin=-50, vmax=80, aspect="auto")
    
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains)
    ax.set_yticks(range(len(classes_in_data))); ax.set_yticklabels(classes_in_data)
    
    for i in range(len(classes_in_data)):
        for j in range(len(domains)):
            val = lift_arr[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 30 else "black"
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
            else:
                ax.text(j, i, "---", ha="center", va="center", fontsize=8, color="gray")
    
    fig.colorbar(im, label="RAG Accuracy Lift (%)", shrink=0.8)
    ax.set_title("Per-Class RAG Lift Across 4 Domains")
    
    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig3_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig3_heatmap.pdf", bbox_inches="tight")
    plt.close(fig); print("  Saved fig3")

def main():
    out_dir = "figures_final"
    os.makedirs(out_dir, exist_ok=True)
    
    vit_data = load_json("results/vit_baselines.json")
    vlm_data = load_json("results/eval_4domain.json")
    
    print("Generating final figures...")
    fig1_dataset_shift(vit_data, out_dir)
    fig2_rag_lift(vlm_data, out_dir)
    fig3_per_class_heatmap(vlm_data, out_dir)
    print("\nAll figures generated in figures_final/")

if __name__ == "__main__":
    main()
