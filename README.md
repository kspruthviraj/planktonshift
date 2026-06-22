# Why Plankton Classifiers Fail Across Imaging Systems

## How This Project Started

This project began with a bug hunt. We noticed that plankton classifiers trained at Eawag were producing inconsistent results across seemingly identical runs. The culprit: a routine `pip install --upgrade Pillow` had changed the default image resampling filter from nearest-neighbour to bicubic interpolation. **49% of all pixels** in our plankton images were silently altered — not by any change in the biology, the water, or the camera, but by a software library update.

We quantified this "silent pipeline drift" and found pixel-level residuals up to 0.54 (out of 1.0), comparable in magnitude to adversarial attacks used to fool deep learning models. This raised a question: if a library update can act like an adversarial attack, what about the differences between entire imaging systems?

That question led us to decompose plankton images into frequency layers using Fourier analysis, and what we found changed how we think about cross-instrument plankton monitoring.

## What We Discovered

Plankton images carry two types of information in different "frequency bands" (think of it like separating the image into layers):
- **Low-frequency layers** carry the biological shape — the features taxonomists use to identify species (body outline, spines, shell shape)
- **Mid-frequency layers** carry the instrument signature — the lighting, contrast, and background artifacts specific to each camera

We proved this by training classifiers on images where we kept only certain frequency layers:
- **Low frequencies only**: classifier identifies species (43.8% accuracy) but not the camera
- **Mid frequencies only**: classifier identifies the camera (92.6% accuracy) but not the species

This separation is causal, not just correlational. And it lets us build better classifiers by augmenting training images in the frequency bands where instruments differ most, improving cross-instrument transfer by +5.9% to +11.6%.

## What You Need

- Python 3.10+
- A GPU (for training experiments)
- Plankton image datasets (see Data section below)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Check what experiments are available
python reproduce.py --status

# Run everything (requires data + GPU)
python reproduce.py

# Run just the Fourier analysis (CPU only, fast)
python reproduce.py --phase 1
```

## Project Layout

```
├── code/                           # All analysis scripts
│   ├── config.py                   # Paths, seeds, settings — edit this first
│   ├── fourier_shift_analysis.py   # Measure how cameras differ in frequency space
│   ├── frequency_masking_causality.py  # Prove that low=fishes, mid=cameras
│   ├── spectral_augmentation.py    # SBA: frequency-calibrated image augmentation
│   ├── train_cross_instrument_sba.py   # Train classifiers with SBA
│   ├── replicate_baseline_ood.py   # Reproduce Chen et al.'s 83% OOD result
│   ├── pillow_version_impact.py    # Does changing Pillow version affect accuracy?
│   ├── reverse_transfer_validation.py  # Does SBA work in both directions?
│   ├── representation_analysis.py  # UMAP plots of learned features
│   ├── ecological_impact_metrics.py    # Shannon diversity, Bray-Curtis
│   ├── bootstrap_confidence_intervals.py  # Statistical confidence
│   ├── confusion_matrices.py       # Which species are hardest to classify?
│   ├── segment_plankton.py         # Segment organisms from background
│   └── train_cross_instrument_sba.py   # Train classifiers with SBA
│
├── figures/                        # Pre-generated figures
├── results/                        # Pre-computed results (JSON files)
│   └── tier1/                      # Results from additional experiments
├── reproduce.py                    # Master script to run everything
└── requirements.txt                # Python packages needed
```

## The Experiments

### 1. Silent Pipeline Drift — The Pillow Library Problem

**Script:** `pillow_version_impact.py`

This is where the project started. We discovered that a routine Pillow library update (6.x → 7.0) changed the default resampling filter from nearest-neighbour to bicubic interpolation. This silently alters 49% of all pixels in plankton images, with residuals up to 0.54 — comparable to adversarial attacks.

We then tested whether this pixel-level change affects classification accuracy. Bicubic actually *improves* accuracy by +1.05%, but the change itself breaks reproducibility: identical code produces different results depending on the installed library version. For long-term monitoring, this is more damaging than a marginal accuracy change — it breaks temporal comparability.

### 2. Fourier Analysis — How Do Cameras Differ?

**Script:** `fourier_shift_analysis.py`

The Pillow discovery led us to ask: if a library update can act like an adversarial attack, what about differences between entire imaging systems? We take plankton images from three cameras (IFCB, ZooScan, DSPC) and decompose them into frequency layers using a 2D Fourier Transform. Then we measure:
- How much each frequency layer differs between cameras (the "shift spectrum")
- How well a simple classifier can tell which camera took an image from each layer
- How much species-identifying information each layer carries

**Key result:** A logistic regression on frequency features identifies the source camera with 83.1% accuracy.

### 3. Frequency Masking — Proving Causation

**Script:** `frequency_masking_causality.py`

To prove that low frequencies carry species info and mid frequencies carry camera info, we train separate classifiers on images with only certain frequency layers:
- **Low frequencies only** (shape, outline): 43.8% species accuracy
- **Mid frequencies only** (camera artifacts): 92.6% camera identification
- **All frequencies**: 46.0% species, 80.7% camera

This is the causal experiment that proves our theory.

### 4. SBA Augmentation — Making Classifiers Robust

**Script:** `spectral_augmentation.py`, `train_cross_instrument_sba.py`

Spectral Band Adversarial (SBA) augmentation adds noise to the frequency layers where cameras differ most during training. This teaches the classifier to ignore camera-specific artifacts and focus on the biological shape.

**Key result:** SBA improves transfer from IFCB to ZooScan by +5.9% (46.7% → 52.6%) and from ZooScan to IFCB by +11.6%.

### 5. Baseline OOD Reproduction

**Script:** `replicate_baseline_ood.py`

Reproduces Chen et al.'s 83% out-of-distribution accuracy on the ZooLake temporal benchmark (10 deployment days). Uses their exact preprocessing pipeline (proportional padding with black borders).

### 6. Ecological Impact

**Script:** `ecological_impact_metrics.py`

Computes Shannon diversity, Simpson diversity, species richness, and Bray-Curtis dissimilarity between true and predicted community compositions. This tells ecologists whether the classifier preserves the biological conclusions they care about.

**Key result:** SBA produces the lowest Bray-Curtis dissimilarity (0.096), meaning it best preserves the true community composition.

### 7. OOD Detection

**Script:** Results in `results/ood_detection_*.json`

An Isolation Forest trained on frequency features can detect when an image comes from a new, unseen camera (ROC-AUC: 0.72–0.92). This lets monitoring systems flag uncertain predictions.

## Data

You need these datasets to reproduce the experiments:

| Dataset | What It Is | Where to Get It |
|---------|-----------|-----------------|
| **WHOI22** | 22 marine phytoplankton species from IFCB | Publicly available from WHOI |
| **ZooScan20** | 20 marine zooplankton species from ZooScan | Publicly available from Villefranche observatory |
| **ZooLake2.0** | 35 freshwater plankton species from Lake Greifensee | https://doi.org/10.25678/000C6M |
| **Chen OOD data** | 10 independent deployment days from ZooLake | https://opendata.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift |
| **DataShift** | Curated IFCB/ZooScan subsets for cross-instrument testing | Included in Adverserial_net repository |

Once you have the data, edit `code/config.py` to point to your local paths.

## Reproducing Specific Results

**Domain classifier accuracy (83.1%):**
```bash
python code/fourier_shift_analysis.py
# Results in results/fourier_analysis.json
```

**Frequency masking:**
```bash
python code/frequency_masking_causality.py --epochs 15
# Results in results/tier1/frequency_masking.json
```

**Cross-instrument SBA:**
```bash
python code/train_cross_instrument_sba.py
# Results in results/sba_band_tta_cross_instrument.json
```

**Temporal OOD:**
```bash
python code/replicate_baseline_ood.py
# Results in results/baseline_chen_exact_pipeline.json
```

**Ecological metrics:**
```bash
python code/ecological_impact_metrics.py
# Results in results/ecological_metrics.json
```

## Software Versions

We pinned all dependencies in `requirements.txt`. Key versions:
- PyTorch 2.12.0
- timm 1.0.27
- Pillow 12.2.0
- scikit-learn 1.2+

**Important:** The Pillow version experiment specifically depends on Pillow behavior. If you want to reproduce the Pillow 6.x vs 7.0 comparison, you need to install both versions separately.

## Random Seeds

All experiments use fixed random seeds for reproducibility:
- Ensemble models: seeds 0, 1, 2
- Ablation experiments: seed 42
- Frequency masking: seed 42 (per band)

Seeds are set for `numpy`, `torch`, and Python's `random` module.
