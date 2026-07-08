"""
run_information_allocation_figure.py — The key takeaway figure (P1-3).

Creates the "information allocation" figure that visually encodes the core claim:
  - x-axis: radial frequency bins
  - left axis (bars): domain classifier accuracy per bin (instrument signature)
  - right axis (line): class separability per bin (biological morphology)

If the claim is correct, domain accuracy should peak at mid-frequencies while
class separability peaks at low frequencies — the visual IS the claim.

Also creates a per-band comparison showing the separation is consistent across
all three imaging domains (WHOI22, ZooScan20, ZooLake2).

Output: figures/fig_information_allocation.png
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RESULTS, FIGURES
from utils_pipeline import preprocess_image, compute_amplitude_spectrum, radial_average

OUT_FIG = FIGURES / "fig_information_allocation.png"
OUT_DATA = RESULTS / "tier1_corrected" / "information_allocation.json"
MAX_PER_CLASS = 40
N_BANDS = 10  # Finer bands for a smoother figure


def load_all_domains():
    domains = {}
    for name, path in [("WHOI22", DATA["whoi22"]), ("ZooScan20", DATA["zooscan20"]), ("ZooLake2", DATA["zoolake2"])]:
        root = Path(path)
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        imgs, labels = [], []
        for ci, cn in enumerate(class_names):
            n = 0
            for p in sorted((root / cn).iterdir()):
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    continue
                if n >= MAX_PER_CLASS:
                    break
                try:
                    im = Image.open(p).convert("RGB")
                    arr = preprocess_image(im)
                    imgs.append(arr)
                    labels.append(ci)
                    n += 1
                except Exception:
                    continue
        domains[name] = (imgs, np.array(labels), class_names)
        print(f"  {name}: {len(imgs)} images, {len(class_names)} classes")
    return domains


def extract_radial_features(imgs):
    feats = []
    for arr in imgs:
        gray = np.mean(arr, axis=2)
        amp = compute_amplitude_spectrum(gray)
        feats.append(radial_average(amp))
    maxlen = max(len(f) for f in feats)
    X = np.zeros((len(feats), maxlen))
    for i, f in enumerate(feats):
        X[i, :len(f)] = f
    return X


def per_band_accuracy(X, labels, n_bands, classifier="logreg"):
    """Compute classification accuracy for each frequency band."""
    maxlen = X.shape[1]
    band_size = maxlen // n_bands
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for bi in range(n_bands):
        s = bi * band_size
        e = (bi + 1) * band_size if bi < n_bands - 1 else maxlen
        X_band = X[:, s:e]
        if len(np.unique(labels)) < 2 or X_band.shape[1] < 1:
            accs.append(0.0)
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0) if classifier == "logreg" else LinearDiscriminantAnalysis()
        try:
            scores = cross_val_score(clf, X_band, labels, cv=cv, scoring="accuracy")
            accs.append(float(scores.mean()))
        except Exception:
            accs.append(0.0)
    return np.array(accs)


def main():
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Loading domains (Pipeline A)...")
    domains = load_all_domains()

    # Domain classification: 3-class (WHOI vs ZooScan vs ZooLake)
    print("\nComputing per-band domain classification accuracy...")
    X_all, y_all = [], []
    for di, (name, (imgs, _, _)) in enumerate(domains.items()):
        X = extract_radial_features(imgs)
        X_all.append(X)
        y_all.append(np.full(len(X), di))
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    domain_accs = per_band_accuracy(X_all, y_all, N_BANDS, "logreg")

    # Class separability per domain per band
    print("Computing per-band class separability per domain...")
    class_accs = {}
    for name, (imgs, labels, _) in domains.items():
        X = extract_radial_features(imgs)
        class_accs[name] = per_band_accuracy(X, labels, N_BANDS, "lda")

    # Save data
    data = {
        "n_bands": N_BANDS,
        "domain_accuracy_per_band": domain_accs.tolist(),
        "class_separability_per_band": {k: v.tolist() for k, v in class_accs.items()},
    }
    with open(OUT_DATA, "w") as f:
        json.dump(data, f, indent=2)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    band_centers = np.arange(N_BANDS)
    band_labels = [f"{int(i*112/N_BANDS)}" for i in range(N_BANDS)]

    # Left axis: domain classifier accuracy (bars)
    colors_domain = "#e74c3c"
    bars = ax1.bar(band_centers, domain_accs, width=0.6, alpha=0.7, color=colors_domain, label="Domain accuracy (instrument ID)")
    ax1.set_xlabel("Radial Frequency Bin", fontsize=12)
    ax1.set_ylabel("Domain Classification Accuracy", fontsize=12, color=colors_domain)
    ax1.tick_params(axis='y', labelcolor=colors_domain)
    ax1.set_xticks(band_centers)
    ax1.set_xticklabels(band_labels, fontsize=9)
    ax1.set_ylim(0.3, 1.0)

    # Right axis: class separability (lines)
    ax2 = ax1.twinx()
    colors_class = ["#2980b9", "#27ae60", "#f39c12"]
    for (name, accs), color in zip(class_accs.items(), colors_class):
        ax2.plot(band_centers, accs, 'o-', color=color, linewidth=2, markersize=6, label=f"Class separability ({name})")
    ax2.set_ylabel("Class Separability (LDA Accuracy)", fontsize=12, color="#2c3e50")
    ax2.set_ylim(0, 0.55)
    ax2.tick_params(axis='y', labelcolor="#2c3e50")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax1.set_title("Information Allocation Across Frequency Bands:\nInstrument Signatures vs Biological Morphology", fontsize=13, fontweight='bold')
    ax1.axvline(x=2, color='gray', linestyle='--', alpha=0.4, label='_nolegend_')
    ax1.text(1.0, 0.95, "Low freq:\nMorphology", fontsize=8, ha='center', transform=ax1.get_xaxis_transform())
    ax1.text(5.0, 0.95, "Mid freq:\nInstrument", fontsize=8, ha='center', transform=ax1.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(str(OUT_FIG), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved figure: {OUT_FIG}")
    print(f"Saved data: {OUT_DATA}")

    # Print summary
    print(f"\n{'='*60}\nINFORMATION ALLOCATION SUMMARY\n{'='*60}")
    print(f"{'Band':<6} {'Domain Acc':>12} {'WHOI22':>10} {'ZooScan':>10} {'ZooLake':>10}")
    for i in range(N_BANDS):
        print(f"{i:<6} {domain_accs[i]:>12.3f} {class_accs['WHOI22'][i]:>10.3f} {class_accs['ZooScan20'][i]:>10.3f} {class_accs['ZooLake2'][i]:>10.3f}")


if __name__ == "__main__":
    main()
