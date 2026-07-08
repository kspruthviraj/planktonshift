# PlanktonShift — Why Plankton Classifiers Fail Across Imaging Systems

> **"Frequency-Domain Decomposition Reveals Domain-Specific and Biological Signals in Ecological Imaging"**

---

## What This Project Does

When you train a plankton classifier on images from one camera (say, an IFCB) and deploy it on a different camera (say, a ZooScan), accuracy crashes from ~100% to ~45%. This project explains **why** by decomposing the problem into frequency layers — like separating a song into bass, midrange, and treble — and shows that:

1. **Camera differences live in specific frequency layers** (a classifier identifies the source camera with 88% accuracy from frequency features alone)
2. **Species identity lives in the phase, not the amplitude** (scrambling phase drops accuracy to 24%; swapping amplitude from the target camera *improves* it to 53%)
3. **Low-frequency layers preserve species info** (48.9% accuracy with only the "bass" frequencies)
4. **Mid-frequency layers carry no species signal** (14.6% — below random chance for 6 classes)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up data (see Data section below)
#    Edit code/config.py to point to your local paths

# 3. Run everything (requires GPU for steps 02-06)
python code/run_master.py

# 4. Or run individual steps
python code/01_fourier_analysis.py        # CPU only, ~2 min
python code/02_amplitude_vs_phase.py      # GPU, ~30 min
python code/03_frequency_masking.py       # GPU, ~1 hour
python code/04_sba_cross_instrument.py    # GPU, ~6 hours
python code/05_da_baselines.py            # GPU, ~2 hours
python code/06_pillow_impact.py           # GPU, ~2 hours
python code/07_information_allocation.py  # CPU, ~1 min
python code/08_statistics.py              # CPU, ~1 min
python code/09_generality_test.py         # GPU, ~1 hour

# 5. Check what would run (dry run)
python code/run_master.py --dry-run
```

---

## Code Structure

### Utilities (shared by all steps)

| File | Purpose |
|------|---------|
| `code/config.py` | All paths, random seeds, and constants. **Edit this first** to set your data paths. |
| `code/utils_pipeline.py` | Shared functions: proportional padding preprocessing, bootstrap confidence intervals, McNemar statistical test, FFT utilities. |
| `code/run_master.py` | Orchestrates all steps in dependency order. Run this to reproduce everything. |

### Analysis Pipeline (numbered in execution order)

#### Step 01 — Fourier Analysis (`01_fourier_analysis.py`)
**Question:** How do different cameras "see" the same plankton species differently?

Decomposes images from three cameras (WHOI22, ZooScan20, ZooLake2) into frequency layers using 2D Fourier Transform. Measures how much each frequency layer differs between cameras (the "shift spectrum") and how much species information each layer carries. Finds that a simple logistic regression identifies the source camera with **88.0% accuracy** (grayscale) or **89.3%** (per-channel RGB) from frequency features alone. The +1.3% RGB gain confirms that colour information also carries domain cues.

**Output:** `results/tier1_corrected/fourier_analysis.json`

---

#### Step 02 — Amplitude vs Phase (`02_amplitude_vs_phase.py`)
**Question:** Does the biological shape of a plankton live in the "volume" or the "timing" of frequency components?

Every image has two frequency components: amplitude (how strong each frequency is) and phase (where each frequency appears). This experiment scrambles each component independently to test which one carries species-discriminative information. Finds that **phase scrambling drops accuracy to 24.1%** (phase = biology) while **amplitude swapping improves it to 52.6%** (amplitude = camera artifacts).

**Output:** `results/tier1_corrected/amplitude_vs_phase.json`

---

#### Step 03 — Frequency Masking (`03_frequency_masking.py`)
**Question:** Can a classifier identify plankton using only the "bass" frequencies?

Trains classifiers on images where only certain frequency layers are kept: low (bins 0-22, the "bass"), mid (bins 22-44, the "midrange"), or high (bins 44+, the "treble"). Finds that **low frequencies preserve 48.9% species accuracy** (comparable to full spectrum) while **mid frequencies achieve only 14.6%** (below random chance for 6 classes).

**Output:** `results/tier1_corrected/frequency_masking.json`

---

#### Step 04 — SBA Cross-Instrument (`04_sba_cross_instrument.py`)
**Question:** Does frequency-calibrated augmentation improve cross-camera transfer?

Trains ViT classifiers on the IFCB-to-ZooScan benchmark (6 classes, 384 training images) with three augmentation strategies: standard, SBA (noise added to camera-specific frequency bands), and phase-preserving. Runs 5 random seeds for honest confidence intervals. Finds **no statistically significant improvement** on this small benchmark (46.0% vs 45.8%, McNemar p=0.55).

**Output:** `results/tier1_corrected/sba_cross_instrument.json`

---

#### Step 05 — DA Baselines (`05_da_baselines.py`)
**Question:** How does SBA compare to generic domain adaptation methods?

Compares SBA against five alternatives: standard augmentation, RandAugment, heavy augmentation, FDA (Fourier Domain Adaptation), and CORAL (Correlation Alignment). All evaluated on the same benchmark with the same preprocessing.

**Output:** `results/tier1_corrected/da_baselines.json`

---

#### Step 06 — Pillow Impact (`06_pillow_impact.py`)
**Question:** Does a software library update silently change your images?

In 2020, the Pillow library changed its default resize filter, altering 49% of all pixels. This script measures whether those pixel changes affect classification accuracy on 10 out-of-distribution days. Finds **+0.11% accuracy difference** (not statistically significant, McNemar p=0.74).

**Output:** `results/tier1_corrected/pillow_impact.json`

---

#### Step 07 — Information Allocation Figure (`07_information_allocation.py`)
**Question:** Where in the frequency spectrum does each type of information live?

Generates the key figure showing, for each of 10 frequency bands, how much species information and how much domain/camera information the band carries. This is the figure that visually demonstrates the frequency-domain separation of biology from camera artifacts.

**Output:** `results/tier1_corrected/information_allocation.json`, `figures/fig_information_allocation.png`

---

#### Step 08 — Statistics Aggregation (`08_statistics.py`)
**Question:** What are all the key numbers, in one place?

Collects results from Steps 01-07 into a single JSON summary with bootstrap confidence intervals and McNemar p-values. Every number in the paper should be traceable to this file.

**Output:** `results/tier1_corrected/statistics_summary.json`

---

#### Step 09 — Generality Test (`09_generality_test.py`)
**Question:** Do these findings work beyond plankton?

Tests the frequency-domain framework on any imaging dataset. Can run on parallel source/target directories or simulate a second "instrument" from a single dataset. Confirms that the low-freq-species / mid-freq-nothing pattern holds beyond plankton.

**Output:** `results/generality/<tag>/frequency_decomposition.json`

---

### Supporting Code (`code/adverserial_net/`)

| File | Purpose |
|------|---------|
| `spectral_augmentation.py` | SBA augmentation implementation (used by Steps 04 and 05) |

### Legacy Code (`code/legacy/`)

Scripts from earlier iterations of the analysis. Not part of the corrected pipeline. Preserved for reference.

---

## Key Results

| Finding | Value | Step |
|---------|-------|------|
| Domain classifier (amplitude spectra) | **88.0%** (gray), **89.3%** (RGB) | 01 |
| Shift energy concentrated in | **Low frequencies** (bins 0-22) | 01 |
| Phase scrambling (species acc) | **24.1%** (from 46.0% baseline) | 02 |
| Amplitude swapping (species acc) | **52.6%** (from 46.0% baseline) | 02 |
| Low-freq masking (species acc) | **48.9%** | 03 |
| Mid-freq masking (species acc) | **14.6%** (below chance) | 03 |
| SBA temporal OOD (ZooLake) | **83.19%** (per-channel RGB SBA + TTA) | — |
| SBA temporal OOD grayscale | 81.87% (grayscale SBA, no TTA) | — |
| SBA cross-instrument | 46.0% vs 45.8% (not significant) | 04 |
| Pillow impact | +0.11% (not significant) | 06 |
| OOD detection (ROC-AUC) | **0.72-0.92** | 01 |

### Grayscale vs Per-Channel RGB

The analysis pipeline uses **both** grayscale and per-channel RGB, depending on the experiment:

**Per-channel RGB** (where colour carries domain information):
- Step 01 domain classifier: 89.3% (RGB) vs 88.0% (gray) — colour channels carry +1.3% camera identity
- Temporal OOD headline (83.19%): SBA is applied independently to R, G, B channels before averaging. This preserves colour-based domain cues that grayscale SBA discards.

**Grayscale** (for clean mechanistic decomposition):
- Steps 02-04 convert `I_gray = mean(R, G, B)` before FFT. This is intentional: the amplitude-vs-phase experiment (Step 02) needs a single 2D FFT to cleanly decompose into `F = A * exp(iφ)`. Doing this per-channel would give 3 separate decompositions with no principled way to combine them for the "does phase carry morphology?" question.
- The colour information is redundant for the mechanistic question (does phase or amplitude carry species info?) but matters for the deployment question (what's the best OOD accuracy?).

**Bottom line:** RGB gives +1.3% where it matters (domain classification, OOD accuracy). Grayscale is used for mechanistic experiments where a single 2D decomposition is needed.

### Temporal OOD Performance Progression (ZooLake, 10 days, 35 classes)

This table shows how each improvement accumulates. All use the same 3-model BEiT geometric ensemble and Chen's proportional-padding preprocessing.

| Configuration | TTA | Macro OOD Accuracy | vs Chen's BEsT (83.05%) |
|--------------|-----|--------------------|------------------------|
| Chen baseline (no SBA) | No | **81.65%** | -1.40% |
| Grayscale SBA finetuned | No | **82.11%** | -0.94% |
| Grayscale SBA finetuned | Yes (4 rotations) | **82.51%** | -0.54% |
| Per-channel RGB SBA finetuned | Yes (4 rotations) | **83.19%** | **+0.14%** |

Key observations:
- **Grayscale SBA** (no TTA) improves over baseline by +0.46% (81.65% -> 82.11%)
- **TTA** adds +0.40% (82.11% -> 82.51%)
- **Per-channel RGB** (vs grayscale) adds +0.68% (82.51% -> 83.19%)
- The per-channel RGB gain (+0.68%) is larger than the grayscale SBA gain (+0.46%), confirming that colour carries significant domain information

---

## Data Setup

Data is not included in the repository (too large). Download separately:

```bash
mkdir -p data

# WHOI22 (22 classes, 6,598 images) — publicly available from WHOI
# ZooScan20 (20 classes, 4,066 images) — publicly available from Villefranche
# DataShift IFCB/ZooScan subsets — from Planktonzilla-17M

# ZooLake2.0 (35 classes, 29,499 images):
wget https://doi.org/10.25678/000C6M -O data/zoolake2.zip
unzip data/zoolake2.zip -d data/chen_data/

# OOD data (10 deployment days):
# https://opendata.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift
```

Edit `code/config.py` to update paths for your local setup.

---

## Reproducing Specific Results

| Paper claim | Script | Output file |
|------------|--------|-------------|
| Domain classifier 88.0% | `01_fourier_analysis.py` | `fourier_analysis.json` |
| Phase-scrambled 24.1% | `02_amplitude_vs_phase.py` | `amplitude_vs_phase.json` |
| Low-freq masking 48.9% | `03_frequency_masking.py` | `frequency_masking.json` |
| SBA mean 46.0% | `04_sba_cross_instrument.py` | `sba_cross_instrument.json` |
| DA baselines comparison | `05_da_baselines.py` | `da_baselines.json` |
| Pillow +0.11% | `06_pillow_impact.py` | `pillow_impact.json` |
| Information allocation fig | `07_information_allocation.py` | `information_allocation.json` |
| All key numbers | `08_statistics.py` | `statistics_summary.json` |

---

## Random Seeds

| Purpose | Seeds |
|---------|-------|
| Ensemble members | 0, 1, 2 |
| SBA cross-instrument | 42, 999, 789, 123, 456 |
| Ablations | 42 |
| Bootstrap CI | 42 |

---

## GPU Requirements

- Minimum: 16 GB VRAM (for ViT-B/16 training)
- Recommended: 32 GB VRAM
- Steps 01, 07, 08: CPU only

---

## License

Apache 2.0
