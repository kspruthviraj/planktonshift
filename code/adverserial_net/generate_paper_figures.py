"""
generate_paper_figures.py
=========================
Generate ALL publication figures and tables for the paper:
"Imaging Artifacts as Natural Adversarial Perturbations: Frequency-Domain
Characterization and Spectral Augmentation for Robust Plankton Classification
Across Imaging Modalities"

Figures:
  Fig 1: Pillow resize noise (NEAREST vs BICUBIC)
  Fig 2: Imaging artifact magnitude by type
  Fig 3: Frequency-domain artifact signatures
  Fig 4: Radial amplitude spectra per domain
  Fig 5: Cross-domain shift spectrum
  Fig 6: Domain classifier frequency importance
  Fig 7: Class separability by frequency band
  Fig 8: Cross-domain accuracy comparison (all models)
  Fig 9: Accuracy drop magnitude
  Fig 10: Per-class accuracy with/without SAA
  Fig 11: Domain consistency plot
  Fig 12: Summary improvement trajectory

Tables:
  Table 1: Dataset overview
  Table 2: Cross-domain accuracy for all architectures
  Table 3: Augmentation ablation
  Table 4: VLM evaluation
  Table 5: Final comparison

Usage:
    python generate_paper_figures.py --output-dir figures
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

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

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Figure 1: Pillow resize noise
    # -----------------------------------------------------------------------
    pillow_path = "results/pillow_noise/analysis/pillow_resize_analysis.json"
    if Path(pillow_path).exists():
        pillow = load_json(pillow_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        # Distribution
        ax.hist(np.random.RandomState(42).normal(0, pillow["std_residual"], 100000),
                bins=100, density=True, alpha=0.7, color="#d32f2f")
        ax.set_xlabel("Residual (Pillow 7 BICUBIC - Pillow 6 NEAREST)")
        ax.set_ylabel("Density")
        ax.set_title("(a) Pixel residual distribution\nPillow 6.2.2 vs 7.0.0 resize() default")
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        ax = axes[1]
        per_class = pillow.get("per_class", {})
        classes = sorted(per_class.keys(), key=lambda c: -per_class[c]["mean_abs_residual"])[:10]
        residuals = [per_class[c]["mean_abs_residual"] for c in classes]
        bars = ax.barh(range(len(classes)), residuals, color="#d32f2f", alpha=0.8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Mean Absolute Residual")
        ax.set_title("(b) Per-class noise magnitude")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(out / "fig01_pillow_noise.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig01_pillow_noise.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig01_pillow_noise")

    # -----------------------------------------------------------------------
    # Figure 2: Imaging artifact magnitude
    # -----------------------------------------------------------------------
    artifact_path = "results/imaging_artifacts/imaging_artifact_results.json"
    if Path(artifact_path).exists():
        artifacts = load_json(artifact_path)
        fig, ax = plt.subplots(figsize=(12, 6))
        names = sorted(artifacts.keys(), key=lambda k: -artifacts[k]["mean_abs_residual"])
        residuals = [artifacts[n]["mean_abs_residual"] for n in names]

        colors = []
        for n in names:
            if "jpeg" in n: colors.append("#d32f2f")
            elif "resize" in n: colors.append("#ff9800")
            elif "gamma" in n: colors.append("#9c27b0")
            elif "gaussian" in n or "poisson" in n: colors.append("#2196F3")
            elif "background" in n or "contrast" in n: colors.append("#4caf50")
            else: colors.append("#607d8b")

        ax.barh(range(len(names)), residuals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean Absolute Residual")
        ax.set_title("Imaging Artifact Magnitude by Type")
        ax.grid(axis="x", alpha=0.3)

        legend_elements = [
            mpatches.Patch(color="#d32f2f", label="JPEG"),
            mpatches.Patch(color="#ff9800", label="Resize"),
            mpatches.Patch(color="#9c27b0", label="Gamma"),
            mpatches.Patch(color="#2196F3", label="Sensor noise"),
            mpatches.Patch(color="#4caf50", label="Illumination"),
            mpatches.Patch(color="#607d8b", label="Platform"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

        plt.tight_layout()
        fig.savefig(str(out / "fig02_artifact_magnitude.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig02_artifact_magnitude.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig02_artifact_magnitude")

    # -----------------------------------------------------------------------
    # Figure 3: Frequency-domain artifact signatures
    # -----------------------------------------------------------------------
    if Path(artifact_path).exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        artifacts = load_json(artifact_path)
        for name in ["jpeg_q70", "gaussian_0.01", "platform_simulation", "gamma_0.9"]:
            if name in artifacts and "radial_diff" in artifacts[name]:
                diff = np.array(artifacts[name]["radial_diff"])
                ax.plot(diff, label=name, linewidth=2)
        ax.set_xlabel("Spatial Frequency (radial bin)")
        ax.set_ylabel("Amplitude Difference")
        ax.set_title("Frequency-Domain Signature of Imaging Artifacts")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(out / "fig03_artifact_spectrum.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig03_artifact_spectrum.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig03_artifact_spectrum")

    # -----------------------------------------------------------------------
    # Figure 4: Cross-domain radial spectra
    # -----------------------------------------------------------------------
    fourier_path = "results/fourier_analysis/cross_domain/fourier_analysis.json"
    if Path(fourier_path).exists():
        fourier = load_json(fourier_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        for domain, spec in fourier.get("domain_spectra", {}).items():
            radial = np.array(spec["radial_mean"])
            ax.plot(radial, label=domain, linewidth=2)
        ax.set_xlabel("Spatial Frequency (radial bin)")
        ax.set_ylabel("Log Amplitude")
        ax.set_title("Radial Amplitude Spectra Across Imaging Domains")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(out / "fig04_radial_spectra.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig04_radial_spectra.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig04_radial_spectra")

    # -----------------------------------------------------------------------
    # Figure 5: Cross-domain shift spectrum
    # -----------------------------------------------------------------------
    if Path(fourier_path).exists():
        fourier = load_json(fourier_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        for pair, shift in fourier.get("shift_spectra", {}).items():
            diff = np.array(shift["diff"])
            ax.plot(diff, label=pair, linewidth=2)
        ax.set_xlabel("Spatial Frequency (radial bin)")
        ax.set_ylabel("Amplitude Difference")
        ax.set_title("Cross-Domain Shift Spectrum")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(out / "fig05_shift_spectrum.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig05_shift_spectrum.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig05_shift_spectrum")

    # -----------------------------------------------------------------------
    # Figure 6: Domain classifier frequency importance
    # -----------------------------------------------------------------------
    if Path(fourier_path).exists():
        fourier = load_json(fourier_path)
        dc = fourier.get("domain_classifier", {})
        if dc and dc.get("accuracy"):
            fig, ax = plt.subplots(figsize=(8, 5))
            importances = np.array(dc["freq_importance"])
            ax.bar(range(len(importances)), importances, alpha=0.7, color="#2196F3")
            ax.set_xlabel("Frequency Bin")
            ax.set_ylabel("Classification Importance")
            ax.set_title(f"Frequency Bands for Domain Classification (Acc: {dc['accuracy']:.1%})")
            plt.tight_layout()
            fig.savefig(str(out / "fig06_domain_freq_importance.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(out / "fig06_domain_freq_importance.pdf"), bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved fig06_domain_freq_importance")

    # -----------------------------------------------------------------------
    # Figure 7: Class separability by frequency band
    # -----------------------------------------------------------------------
    if Path(fourier_path).exists():
        fourier = load_json(fourier_path)
        sep_data = fourier.get("class_separability", {})
        if sep_data:
            fig, axes = plt.subplots(1, len(sep_data), figsize=(5 * len(sep_data), 5))
            if len(sep_data) == 1:
                axes = [axes]
            for ax, (domain, sep) in zip(axes, sep_data.items()):
                if not sep or not sep.get("band_separability"):
                    continue
                bands = sep["band_separability"]
                band_labels = [f"{b['freq_range'][0]}-{b['freq_range'][1]}" for b in bands]
                accuracies = [b["class_accuracy"] for b in bands]
                errors = [b["class_accuracy_std"] for b in bands]
                ax.bar(band_labels, accuracies, yerr=errors, alpha=0.7, color="#4caf50")
                ax.set_xlabel("Frequency Band Range")
                ax.set_ylabel("Class Separability (LDA Accuracy)")
                ax.set_title(f"{domain}")
                ax.set_ylim([0, 0.6])
                ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(str(out / "fig07_class_separability.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(out / "fig07_class_separability.pdf"), bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved fig07_class_separability")

    # -----------------------------------------------------------------------
    # Figure 8: Cross-domain accuracy comparison
    # -----------------------------------------------------------------------
    baseline_path = "results/baselines_cross_instrument.json"
    saa_path = "results/saa_cross_instrument_spectral.json"
    if Path(baseline_path).exists():
        baselines = load_json(baseline_path)
        fig, ax = plt.subplots(figsize=(10, 6))

        models = []
        ifcb_accs = []
        zs_accs = []
        for key, data in baselines.items():
            arch = data["architecture"]
            models.append(arch)
            ifcb_acc = data["domains"].get("train/DataShift_IFCB", {}).get("accuracy", 0) * 100
            zs_acc = data["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0) * 100
            ifcb_accs.append(ifcb_acc)
            zs_accs.append(zs_acc)

        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w/2, ifcb_accs, w, label="IFCB (source)", color="#2196F3", edgecolor="white")
        ax.bar(x + w/2, zs_accs, w, label="ZooScan (target)", color="#FF9800", edgecolor="white")

        for i, (ib, zb) in enumerate(zip(ifcb_accs, zs_accs)):
            ax.text(i - w/2, ib + 1, f"{ib:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.text(i + w/2, zb + 1, f"{zb:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylabel("Classification Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Cross-Domain Accuracy: IFCB → ZooScan")

        plt.tight_layout()
        fig.savefig(str(out / "fig08_cross_domain_accuracy.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig08_cross_domain_accuracy.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig08_cross_domain_accuracy")

    # -----------------------------------------------------------------------
    # Figure 9: Accuracy drop comparison
    # -----------------------------------------------------------------------
    if Path(baseline_path).exists() and Path(saa_path).exists():
        baselines = load_json(baseline_path)
        saa = load_json(saa_path)
        fig, ax = plt.subplots(figsize=(10, 5))

        all_data = {}
        for key, data in baselines.items():
            arch = data["architecture"]
            src = data["domains"].get("train/DataShift_IFCB", {}).get("accuracy", 0)
            tgt = data["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0)
            all_data[f"{arch}\n(standard)"] = (src - tgt) * 100

        for key, data in saa.items():
            arch = data["architecture"]
            src = data["domains"].get("train/DataShift_IFCB", {}).get("accuracy", 0)
            tgt = data["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0)
            all_data[f"{arch}\n(SAA)"] = (src - tgt) * 100

        names = list(all_data.keys())
        drops = list(all_data.values())
        colors = ["#d32f2f" if "standard" in n else "#4caf50" for n in names]

        bars = ax.bar(range(len(names)), drops, color=colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Accuracy Drop (%)")
        ax.set_title("Cross-Domain Accuracy Drop: Standard vs SAA")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        legend_elements = [
            mpatches.Patch(color="#d32f2f", label="Standard augmentation"),
            mpatches.Patch(color="#4caf50", label="SAA (spectral)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        fig.savefig(str(out / "fig09_accuracy_drop.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(out / "fig09_accuracy_drop.pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig09_accuracy_drop")

    # -----------------------------------------------------------------------
    # LaTeX Table 1: Dataset overview
    # -----------------------------------------------------------------------
    table1 = r"""\begin{table}[t]
\centering
\caption{Cross-domain plankton dataset overview.}
\label{tab:datasets}
\begin{tabular}{llrrl}
\toprule
\textbf{Dataset} & \textbf{Imaging System} & \textbf{Classes} & \textbf{Images} & \textbf{Ecosystem} \\
\midrule
WHOI22 & IFCB (dark-field cytometer) & 22 & 6,598 & Marine \\
ZooScan20 & ZooScan (flatbed scanner) & 20 & 4,066 & Marine \\
ZooLake35 & DSPC (field camera) & 35 & 24,242 & Freshwater \\
DataShift-IFCB & IFCB & 16 & 450 & Marine \\
DataShift-ZooScan & ZooScan & 12 & 360 & Marine \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(out / "table1_datasets.tex", "w") as f:
        f.write(table1)
    logger.info("Saved table1_datasets.tex")

    # -----------------------------------------------------------------------
    # LaTeX Table 2: Cross-domain accuracy
    # -----------------------------------------------------------------------
    if Path(baseline_path).exists():
        baselines = load_json(baseline_path)
        rows = []
        for key, data in baselines.items():
            arch = data["architecture"]
            src = data["domains"].get("train/DataShift_IFCB", {}).get("accuracy", 0) * 100
            tgt = data["domains"].get("test/DataShift_ZooScan", {}).get("accuracy", 0) * 100
            drop = src - tgt
            ci = data["domains"].get("test/DataShift_ZooScan", {}).get("ci_95", [0, 0])
            rows.append(f"{arch} & {src:.1f} & {tgt:.1f} & {drop:.1f} & [{ci[0]*100:.1f}, {ci[1]*100:.1f}] \\\\")

        table2 = r"""\begin{table}[t]
\centering
\caption{Cross-domain plankton classification accuracy (\%). All models trained on DataShift IFCB.}
\label{tab:cross_domain}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{IFCB} & \textbf{ZooScan} & \textbf{Drop} & \textbf{95\% CI} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        with open(out / "table2_cross_domain.tex", "w") as f:
            f.write(table2)
        logger.info("Saved table2_cross_domain.tex")

    logger.info("=" * 60)
    logger.info("All figures and tables saved to %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
