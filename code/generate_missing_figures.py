"""Generate temporal OOD line plot and Pillow residual heatmap for the paper."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Figure 1: Temporal OOD Line Plot ──
def generate_temporal_ood_plot():
    with open(ROOT / "results" / "finetune_chen_saa" / "finetune_results_v4.json") as f:
        ft_data = json.load(f)
    with open(ROOT / "results" / "finetune_chen_saa" / "baseline_results.json") as f:
        bl_data = json.load(f)
    with open(ROOT / "results" / "sba_centrecrop_tta.json") as f:
        sba_tta_data = json.load(f)

    # Extract per-day accuracies
    bl_per_day = bl_data["ensemble_geometric"]["per_day"]
    ft_per_day = ft_data["finetune_ensemble_arithmetic"]["per_day"]
    sba_tta_per_day = sba_tta_data["per_day"]

    # Sort by OOD number
    ood_days = sorted(bl_per_day.keys(), key=lambda x: int(x.replace("OOD", "")))
    bl_accs = [bl_per_day[d] * 100 for d in ood_days]
    ft_accs = [ft_per_day[d] * 100 for d in ood_days]
    sba_tta_accs = [sba_tta_per_day[d] * 100 for d in ood_days]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(ood_days))
    ax.plot(x, bl_accs, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Baseline (no TTA)', zorder=3)
    ax.plot(x, ft_accs, 's-', color='#F44336', linewidth=2, markersize=7, label='SBA fine-tuned (no TTA)', zorder=3)
    ax.plot(x, sba_tta_accs, 'D-', color='#9C27B0', linewidth=2.5, markersize=8, label='SBA + TTA (best)', zorder=4)

    # Add Chen's BEsT reference line
    ax.axhline(y=83.05, color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.7, label="Chen's BEsT (83.05%)")

    # Fill between baseline and SBA+TTA to show improvement
    for i in range(len(ood_days)):
        if sba_tta_accs[i] > bl_accs[i]:
            ax.fill_between([x[i]-0.2, x[i]+0.2], [bl_accs[i]]*2, [sba_tta_accs[i]]*2,
                           alpha=0.2, color='#4CAF50', zorder=1)

    ax.set_xlabel('OOD Deployment Day', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ood_days, fontsize=10)
    ax.set_ylim(70, 98)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Temporal OOD Accuracy Across 10 Deployment Days', fontsize=14, fontweight='bold')

    # Add average annotations
    bl_avg = np.mean(bl_accs)
    ft_avg = np.mean(ft_accs)
    sba_tta_avg = np.mean(sba_tta_accs)
    ax.annotate(f'Avg: {bl_avg:.1f}%', xy=(9.4, bl_avg), fontsize=10, color='#2196F3', fontweight='bold')
    ax.annotate(f'Avg: {ft_avg:.1f}%', xy=(9.4, ft_avg - 1.0), fontsize=10, color='#F44336', fontweight='bold')
    ax.annotate(f'Avg: {sba_tta_avg:.1f}%', xy=(9.4, sba_tta_avg + 0.5), fontsize=10, color='#9C27B0', fontweight='bold')

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_temporal_ood_line.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Figure 2: Pillow Residual Heatmap ──
def generate_pillow_heatmap():
    from PIL import Image
    import cv2

    # Use a sample IFCB image
    img_dir = ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0" / "ceratium"
    sample_imgs = sorted(img_dir.glob("*.jpeg"))
    if not sample_imgs:
        print("No sample images found, skipping heatmap")
        return

    img_path = sample_imgs[0]

    # Load with Pillow 7.0 (current) - simulate Pillow 6.x by using NEAREST
    im = Image.open(img_path).convert("RGB")

    # Resize with bicubic (Pillow 7.0 default)
    im_bicubic = im.resize((224, 224), Image.BICUBIC)
    arr_bicubic = np.array(im_bicubic, dtype=np.float32) / 255.0

    # Resize with nearest-neighbor (Pillow 6.x default)
    im_nearest = im.resize((224, 224), Image.NEAREST)
    arr_nearest = np.array(im_nearest, dtype=np.float32) / 255.0

    # Compute residual
    residual = np.abs(arr_bicubic - arr_nearest)
    residual_gray = np.mean(residual, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].imshow(arr_nearest)
    axes[0].set_title('Pillow 6.x (Nearest-Neighbour)', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(arr_bicubic)
    axes[1].set_title('Pillow 7.0 (Bicubic)', fontsize=10)
    axes[1].axis('off')

    im3 = axes[2].imshow(residual_gray, cmap='hot', vmin=0, vmax=0.15)
    axes[2].set_title(f'|Residual| (mean={np.mean(residual_gray):.4f}, max={np.max(residual_gray):.3f})', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle('Pillow Library Version Impact: Nearest-Neighbour vs Bicubic Interpolation', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_pillow_residual_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_temporal_ood_plot()
    generate_pillow_heatmap()
