"""
config.py — Central configuration for PlanktonShift reproduction.
All paths relative to this file's location. No hardcoded absolute paths.
"""
from pathlib import Path
import os

# Project root (parent of code/ directory)
ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA = {
    "zoolake2": ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0",
    "ood": ROOT / "data" / "chen_data" / "OOD_data" / "OODs",
    "whoi22": ROOT / "data" / "whoi22_full" / "train" / "WHOI22",
    "zooscan20": ROOT / "data" / "whoi22_full" / "test" / "ZooScan20",
    "datashift_ifcb": ROOT / "data" / "datashift" / "eval" / "IFCB",
    "datashift_zooscan": ROOT / "data" / "datashift" / "eval" / "ZooScan",
}

# Model paths
MODELS = {
    "chen_beit_01": ROOT / "results" / "finetune_chen_saa" / "chen_model_01_converted.pth",
    "chen_beit_02": ROOT / "results" / "finetune_chen_saa" / "chen_model_02_converted.pth",
    "chen_beit_03": ROOT / "results" / "finetune_chen_saa" / "chen_model_03_converted.pth",
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
        "name": "vit_b_16",
        "library": "torchvision",
        "normalization": "imagenet",  # [0.485,0.456,0.406]
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
    if issues:
        print("WARNING: Some data paths are missing:")
        for i in issues:
            print(f"  {i}")
        print("Download instructions are in README.md")
    else:
        print("All data paths verified.")
    return len(issues) == 0


if __name__ == "__main__":
    verify_data()
