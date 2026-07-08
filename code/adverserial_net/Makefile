# Adversarial Robustness + Dataset Shift in Plankton Classification
# =================================================================
# Complete experimental pipeline for the paper.
#
# Usage:
#   make all              # Run everything
#   make data             # Prepare all datasets
#   make analysis         # Run Fourier + artifact analysis
#   make train            # Train all models
#   make vlm              # Run VLM evaluation
#   make figures          # Generate paper figures
#   make clean            # Clean up

PYTHON ?= python3
DATA_DIR ?= data/cross_domain
RESULTS_DIR ?= results
FIGURES_DIR ?= figures

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
.PHONY: data
data:
	$(PYTHON) prepare_cross_domain_data.py --output-dir $(DATA_DIR) --all --max-per-class 300

# ---------------------------------------------------------------------------
# Phase 2: Analysis
# ---------------------------------------------------------------------------
.PHONY: analysis
analysis: fourier imaging_artifacts pillow_noise ood_detection

.PHONY: fourier
fourier:
	mkdir -p data/fourier_input
	ln -sf $(realpath $(DATA_DIR)/whoi22_full/train/WHOI22)/* data/fourier_input/WHOI22/ 2>/dev/null || true
	ln -sf $(realpath $(DATA_DIR)/whoi22_full/test/ZooScan20)/* data/fourier_input/ZooScan20/ 2>/dev/null || true
	ln -sf $(realpath $(DATA_DIR)/whoi22_full/test/ZooLake35)/* data/fourier_input/ZooLake35/ 2>/dev/null || true
	$(PYTHON) fourier_shift_analysis.py \
		--data-root data/fourier_input \
		--domains WHOI22 ZooScan20 ZooLake35 \
		--output-dir $(RESULTS_DIR)/fourier_analysis/cross_domain \
		--max-per-class 30

.PHONY: imaging_artifacts
imaging_artifacts:
	$(PYTHON) imaging_artifact_experiment.py \
		--data-dir $(DATA_DIR)/cross_instrument/train/DataShift_IFCB \
		--output-dir $(RESULTS_DIR)/imaging_artifacts \
		--max-images 50

.PHONY: pillow_noise
pillow_noise:
	$(PYTHON) -m venv .venv_pillow6
	.venv_pillow6/bin/pip install --quiet "Pillow==6.2.2" numpy
	$(PYTHON) -m venv .venv_pillow7
	.venv_pillow7/bin/pip install --quiet "Pillow==7.0.0" numpy
	.venv_pillow6/bin/python pillow_resize_experiment.py \
		--data-dir $(DATA_DIR)/cross_instrument/train/DataShift_IFCB \
		--output-dir $(RESULTS_DIR)/pillow_noise/v6.2.2 --max-images 100
	.venv_pillow7/bin/python pillow_resize_experiment.py \
		--data-dir $(DATA_DIR)/cross_instrument/train/DataShift_IFCB \
		--output-dir $(RESULTS_DIR)/pillow_noise/v7.0.0 --max-images 100
	$(PYTHON) pillow_resize_analyze.py \
		--v6-dir $(RESULTS_DIR)/pillow_noise/v6.2.2 \
		--v7-dir $(RESULTS_DIR)/pillow_noise/v7.0.0 \
		--output-dir $(RESULTS_DIR)/pillow_noise/analysis
	rm -rf .venv_pillow6 .venv_pillow7

.PHONY: ood_detection
ood_detection:
	$(PYTHON) spectral_ood_detection.py \
		--train-dir $(DATA_DIR)/cross_instrument/train/DataShift_IFCB \
		--test-dirs $(DATA_DIR)/cross_instrument/test/DataShift_ZooScan \
					$(DATA_DIR)/whoi22_full/test/ZooLake35 \
		--output-dir $(RESULTS_DIR)/ood_detection

# ---------------------------------------------------------------------------
# Phase 3: Training
# ---------------------------------------------------------------------------
.PHONY: train
train: train_baselines train_saa

.PHONY: train_baselines
train_baselines:
	$(PYTHON) train_with_saa.py \
		--data-dir $(DATA_DIR)/cross_instrument \
		--source-domain train/DataShift_IFCB \
		--target-domains test/DataShift_ZooScan \
		--architectures vit_b_16 resnet50 convnext_tiny \
		--augmentation standard --epochs 30 --batch-size 16 \
		--output $(RESULTS_DIR)/baselines_cross_instrument.json

.PHONY: train_saa
train_saa:
	$(PYTHON) train_with_saa.py \
		--data-dir $(DATA_DIR)/cross_instrument \
		--source-domain train/DataShift_IFCB \
		--target-domains test/DataShift_ZooScan \
		--architectures vit_b_16 \
		--augmentation saa_all --epochs 30 --batch-size 16 \
		--output $(RESULTS_DIR)/saa_cross_instrument.json

# ---------------------------------------------------------------------------
# Phase 4: VLM evaluation
# ---------------------------------------------------------------------------
.PHONY: vlm
vlm:
	$(PYTHON) artifact_aware_rag.py \
		--data-dir $(DATA_DIR)/cross_instrument \
		--output $(RESULTS_DIR)/aa_rag_results.json

# ---------------------------------------------------------------------------
# Phase 5: Figures
# ---------------------------------------------------------------------------
.PHONY: figures
figures:
	$(PYTHON) generate_paper_figures.py --output-dir $(FIGURES_DIR)

# ---------------------------------------------------------------------------
# All
# ---------------------------------------------------------------------------
.PHONY: all
all: data analysis train vlm figures

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf $(DATA_DIR) $(RESULTS_DIR) $(FIGURES_DIR) data/fourier_input
	rm -rf .venv_pillow6 .venv_pillow7
