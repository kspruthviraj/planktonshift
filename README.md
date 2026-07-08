# PlanktonShift — Why Plankton Classifiers Fail Across Imaging Systems

> **"Why Plankton Classifiers Fail Across Imaging Systems: Frequency-Domain Analysis, Spectral Augmentation, and Retrieval-Augmented Vision-Language Models for Robust Ecological Monitoring"**

**Authors:** Kyathanahally Sreenath, Chao Chen, Marc Reyes, Svenja Merkli, Ewa Merz, Francesco Pomati, Marco Baity-Jesi

---

## Key Findings

| Finding | Result |
|---------|--------|
| Domain classifier (amplitude spectra) | **88.0%** accuracy (gray), **89.3%** (RGB) |
| Shift energy concentrated in | **Low-frequency bands** (bins 0–22) |
| Phase scrambling (species acc) | **24.1%** (from 46.0% baseline) — phase carries morphology |
| Amplitude swapping (species acc) | **52.6%** (from 46.0% baseline) — amplitude = domain artifacts |
| Low-freq masking (species acc) | **48.9%** (comparable to full spectrum) |
| Mid-freq masking (species acc) | **14.6%** (below chance — no species info) |
| SBA temporal OOD (ZooLake) | **83.19%** (beats Chen's 83.05%) |
| SBA cross-instrument (IFCB→ZooScan) | 46.0% vs 45.8% baseline (not significant) |
| RAVL lift (IFCB-NES) | **+57.5%** |
| OOD detection (ROC-AUC) | **0.72–0.92** |

## Quick Start

```bash
pip install -r requirements.txt

# Run all corrected experiments
python code/run_master.py

# Or run individual experiments
python code/run_fourier_analysis_corrected.py
python code/run_frequency_masking_corrected.py
python code/run_sba_cross_instrument_corrected.py
python code/run_amplitude_vs_phase.py
python code/run_da_baselines.py
python code/run_pillow_impact_corrected.py
python code/run_information_allocation_figure.py
python code/run_statistics.py
```

## Project Structure

```
PlanktonShift/
├── README.md
├── requirements.txt
├── reproduce.py
├── code/                              # All source code
│   ├── config.py                      # Paths, seeds, settings
│   ├── utils_pipeline.py              # Shared preprocessing, bootstrap CI, McNemar
│   ├── run_master.py                  # Master orchestrator for corrected experiments
│   ├── run_fourier_analysis_corrected.py
│   ├── run_frequency_masking_corrected.py
│   ├── run_sba_cross_instrument_corrected.py
│   ├── run_amplitude_vs_phase.py
│   ├── run_da_baselines.py
│   ├── run_pillow_impact_corrected.py
│   ├── run_information_allocation_figure.py
│   ├── run_statistics.py
│   ├── experiment_frequency_decomposition_generality.py
│   ├── adverserial_net/               # Spectral augmentation, SBA implementation
│   └── datashift/                     # RAG, morphological catalog
├── results/                           # Experiment outputs (JSON)
│   ├── tier1_corrected/               # Corrected experiment results
│   ├── perchannel_sba/                # Temporal OOD results
│   └── generality/                    # Generality test results
├── figures/                           # Publication figures
├── data/                              # Vendored datasets
│   ├── whoi22/                        # WHOI22 (22 classes, 6,598 images)
│   ├── zooscan20/                     # ZooScan20 (20 classes, 4,066 images)
│   ├── cross_instrument/              # DataShift IFCB/ZooScan subsets
│   ├── chen_data/                     # ZooLake2.0 + OOD1-10
│   ├── datashift/                     # Original DataShift evaluation
│   └── zoolake_ood/                   # ZooLake OOD train/test
└── docs/                              # Audit and review documents
```

## Data

| Dataset | Source | Location |
|---------|--------|----------|
| **WHOI22** (22 classes, 6,598 images) | WHOI | `data/whoi22/` |
| **ZooScan20** (20 classes, 4,066 images) | Villefranche | `data/zooscan20/` |
| **DataShift IFCB/ZooScan** | Planktonzilla-17M | `data/cross_instrument/` |
| **ZooLake2.0** (35 classes, 29,499 images) | [DOI: 10.25678/000C6M](https://doi.org/10.25678/000C6M) | `data/chen_data/ZooLake2/` |
| **OOD1-10** (10 deployment days) | [Eawag portal](https://opendata.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift) | `data/chen_data/OOD_data/` |

## Data Setup

Data is not included in the repository (too large). Download separately and place in `data/`:

```bash
# Create data directory
mkdir -p data

# WHOI22 and ZooScan20 are publicly available from their respective repositories
# DataShift IFCB/ZooScan subsets are from the Planktonzilla-17M project

# ZooLake2.0 and OOD data:
wget https://doi.org/10.25678/000C6M -O data/zoolake2.zip
unzip data/zoolake2.zip -d data/chen_data/

# Or visit: https://opendata.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift
```

Edit `code/config.py` to update paths for your local setup.

## Corrected Experiments

All experiments use **Pipeline A** (Proportional Padding: resize to 128px keeping aspect ratio, black-pad to square, then resize to 224px). This matches Chen et al.'s preprocessing and was identified as the correct pipeline after an audit found that three incompatible pipelines had been used across experiments.

### Experiments and Scripts

| Experiment | Script | Output |
|-----------|--------|--------|
| Fourier shift analysis | `run_fourier_analysis_corrected.py` | `results/tier1_corrected/fourier_analysis.json` |
| Frequency masking (causal) | `run_frequency_masking_corrected.py` | `results/tier1_corrected/frequency_masking.json` |
| SBA cross-instrument | `run_sba_cross_instrument_corrected.py` | `results/tier1_corrected/sba_cross_instrument.json` |
| Amplitude vs Phase | `run_amplitude_vs_phase.py` | `results/tier1_corrected/amplitude_vs_phase.json` |
| DA baselines comparison | `run_da_baselines.py` | `results/tier1_corrected/da_baselines.json` |
| Pillow version impact | `run_pillow_impact_corrected.py` | `results/tier1_corrected/pillow_impact.json` |
| Information allocation | `run_information_allocation_figure.py` | `results/tier1_corrected/information_allocation.json` |
| Statistics summary | `run_statistics.py` | `results/tier1_corrected/statistics_summary.json` |
| Generality test | `experiment_frequency_decomposition_generality.py` | `results/generality/` |

### Key Corrected Findings

1. **Phase carries morphology, not amplitude.** Phase-scrambled images drop to 24.1% species accuracy (from 46.0%), while amplitude-swapped images improve to 52.6%. This inverts the paper's original claim.

2. **Shift energy is in low frequencies, not mid.** All three cross-domain pairs have maximal shift energy in band 0 (bins 0–22), not band 1 (bins 22–44).

3. **SBA provides zero mean improvement on the small cross-instrument benchmark.** With corrected preprocessing and all 5 seeds, SBA achieves 46.0% vs 45.8% baseline (McNemar p = 0.55). The previously reported +5.9% was a seed-cherry-picked artifact.

4. **Mid-frequency masking produces below-chance species accuracy.** 14.6% for 6 classes (chance = 16.7%), confirming mid frequencies carry no species information.

5. **Pillow impact is negligible.** +0.11% accuracy difference (McNemar p = 0.74), not the previously reported +1.05%.

## Random Seeds

| Purpose | Seeds |
|---------|-------|
| Ensemble members | 0, 1, 2 |
| SBA cross-instrument | 42, 999, 789, 123, 456 |
| Ablations | 42 |
| Bootstrap CI | 42 |

## GPU Requirements

- Minimum: 16 GB VRAM (for ViT-B/16 training)
- Recommended: 32 GB VRAM
- Fourier analysis and OOD detection: CPU only

## Citation

```bibtex
@article{kyathanahally2026plankton,
  title={Why Plankton Classifiers Fail Across Imaging Systems: Frequency-Domain Analysis, Spectral Augmentation, and Retrieval-Augmented Vision-Language Models for Robust Ecological Monitoring},
  author={Kyathanahally, Sreenath and Chen, Chao and Reyes, Marc and Merkli, Svenja and Merz, Ewa and Pomati, Francesco and Baity-Jesi, Marco},
  journal={Methods in Ecology and Evolution},
  year={2026}
}
```

## License

Apache 2.0
