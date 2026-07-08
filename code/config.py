"""
config.py — Central configuration for PlanktonShift reproduction.
All paths relative to this file's location. NO external project dependencies.
This project is self-sufficient: all data and models live under data/.
"""
from pathlib import Path
import os

# Project root (parent of code/ directory)
ROOT = Path(__file__).resolve().parent.parent

# Data paths — all vendored into this project's data/ directory
DATA = {
    "zoolake2": ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0",
    "ood": ROOT / "data" / "chen_data" / "OOD_data" / "OODs",
    "whoi22": ROOT / "data" / "whoi22",
    "zooscan20": ROOT / "data" / "zooscan20",
    "datashift_ifcb": ROOT / "data" / "datashift" / "eval" / "IFCB",
    "datashift_zooscan": ROOT / "data" / "datashift" / "eval" / "ZooScan",
    # Cross-instrument benchmark (aligned DataShift IFCB/ZooScan)
    "cross_ifcb": ROOT / "data" / "cross_instrument" / "train" / "DataShift_IFCB",
    "cross_zooscan": ROOT / "data" / "cross_instrument" / "test" / "DataShift_ZooScan",
    # DataShift eval_v2 (RAVL 4-domain: IFCB, IFCB-NES, ZooScan, ZooLake)
    "datashift_v2_ifcb": ROOT / "data" / "datashift" / "eval_v2" / "IFCB",
    "datashift_v2_ifcb_nes": ROOT / "data" / "datashift" / "eval_v2" / "IFCB-NES",
    "datashift_v2_zooscan": ROOT / "data" / "datashift" / "eval_v2" / "ZooScan",
    "datashift_v2_zoolake": ROOT / "data" / "datashift" / "eval_v2" / "ZooLake",
}

# Chen trained model paths (BEiT models from Chen et al. 2025)
CHEN_MODELS_DIR = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models"
MODELS = {
    "chen_beit_01": CHEN_MODELS_DIR / "trained_models" / "01" / "trained_model_tuned.pth",
    "chen_beit_02": CHEN_MODELS_DIR / "trained_models" / "02" / "trained_model_tuned.pth",
    "chen_beit_03": CHEN_MODELS_DIR / "trained_models" / "03" / "trained_model_tuned.pth",
    "chen_classes": CHEN_MODELS_DIR / "classes.npy",
}

# Result paths
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

# Reproducibility seeds
SEEDS = {
    "ensemble": [0, 1, 2],
    "ablation": 42,
    "train_val_split": 42,
    "numpy_default": 42,
    "sba_cross_instrument": [42, 999, 789, 123, 456],
}

# Architecture configs
ARCHITECTURES = {
    "beit": {
        "name": "beit_base_patch16_224.in22k_ft_in22k_in1k",
        "library": "timm",
        "normalization": "none",  # Raw [0,1] pixels
        "input_size": 224,
    },
    "vit_b_16": {
        "name": "vit_base_patch16_224",
        "library": "timm",
        "normalization": "imagenet",
        "input_size": 224,
    },
    "resnet50": {
        "name": "resnet50",
        "library": "torchvision",
        "normalization": "imagenet",
        "input_size": 224,
    },
}

# Training hyperparameters (Chen's recipe)
HYPER_CHEN = {
    "batch_size": 128,
    "weight_decay": 0.03,
    "lr_initial": 1e-4,
    "lr_finetune": 1e-5,
    "lr_final": 1e-6,
    "epochs_initial": 30,
    "epochs_finetune": 15,
    "epochs_final": 5,
}

# Chen et al. BEsT benchmark reference
CHEN_BEST_ACCURACY = 0.8305

# Classes (35 — Brachionus excluded in ZooLake2.0)
CLASSES_35 = [
    "aphanizomenon", "asplanchna", "asterionella", "bosmina", "ceratium",
    "chaoborus", "collotheca", "conochilus", "copepod_skins", "cyclops",
    "daphnia", "daphnia_skins", "diaphanosoma", "diatom_chain", "dinobryon",
    "dirt", "eudiaptomus", "filament", "fish", "fragilaria", "hydra",
    "kellicottia", "keratella_cochlearis", "keratella_quadrata", "leptodora",
    "maybe_cyano", "nauplius", "paradileptus", "polyarthra", "rotifers",
    "synchaeta", "trichocerca", "unknown", "unknown_plankton", "uroglena",
]

# Frequency band definitions — MATCHING THE PAPER (5 equal radial bins)
# For r_max=112 (224px image): band0=0-22, band1=22-44, band2=44-67, band3=67-89, band4=89-112
# The masking experiment uses 3 bands: low(0-22), mid(22-44), high(44+)
FREQ_BAND_FRACTIONS = {
    "low":  (0.00, 0.20),   # bins 0-22   -> species preserved
    "mid":  (0.20, 0.40),   # bins 22-44  -> instrument maximised
    "high": (0.40, 1.00),   # bins 44+    -> both degraded
    "all":  (0.00, 1.00),
}


def ensure_dirs():
    """Create all required directories."""
    for d in [RESULTS, FIGURES]:
        d.mkdir(parents=True, exist_ok=True)


def verify_data():
    """Check that required data exists."""
    issues = []
    for name, path in DATA.items():
        if not path.exists():
            issues.append(f"DATA MISSING: {name} at {path}")
    for name, path in MODELS.items():
        if not path.exists():
            issues.append(f"MODEL MISSING: {name} at {path}")
    if issues:
        print("WARNING: Some paths are missing:")
        for i in issues:
            print(f"  {i}")
        print("Download instructions are in README.md")
    else:
        print("All data and model paths verified.")
    return len(issues) == 0


if __name__ == "__main__":
    verify_data()
