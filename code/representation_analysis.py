"""
representation_analysis.py
===========================
Train baseline and SBA models, extract embeddings, and visualize
representation collapse vs. species clustering.

This is the CRITICAL experiment that proves the core thesis:
Baseline models learn instrument artifacts (cluster by domain).
SBA models learn biological morphology (cluster by species).

Usage:
    python representation_analysis.py \
        --data-dir /home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument \
        --output-dir figures/representation
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 224


class PlanktonDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        data_path = Path(data_dir)
        if not data_path.is_dir():
            return
        for cls_name, idx in class_to_idx.items():
            cls_dir = data_path / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), idx, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cls_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, cls_name


class SAATransform:
    def __init__(self, shift_spectrum=None):
        self.aug = SpectralAugmentation(
            shift_spectrum=shift_spectrum,
            strategies=["band_adversarial"],
            strength=0.5, p=0.8,
        )
        self.base = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        return self.base(Image.fromarray(arr_uint8, mode="L").convert("RGB"))


def train_model(model, train_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            logger.info("    Epoch %d/%d  Loss: %.4f  Acc: %.1f%%",
                        epoch + 1, epochs, total_loss / total, correct / total * 100)


def extract_embeddings(model, loader, device):
    """Extract CLS token embeddings from ViT."""
    model.eval()
    embeddings, labels, cls_names = [], [], []
    all_embeddings = []

    def hook_fn(module, input, output):
        # output shape: (batch, seq_len, hidden_dim) — CLS token is at position 0
        all_embeddings.append(output[:, 0].detach().cpu().numpy())

    handle = model.encoder.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, label_batch, cls_batch in loader:
            images = images.to(device)
            _ = model(images)
            labels.extend(label_batch.numpy())
            cls_names.extend(cls_batch)

    handle.remove()
    return np.vstack(all_embeddings), np.array(labels), cls_names


def compute_separability(embeddings, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(embeddings)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument")
    parser.add_argument("--output-dir", type=str, default="figures/representation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Discover classes
    train_dir = Path(args.data_dir) / "train" / "DataShift_IFCB"
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    logger.info("Classes: %s (%d)", classes, num_classes)

    # Load shift spectrum for SAA
    spectrum_path = "/home/sreenath/research-space/Adverserial_net/results/fourier_analysis/cross_domain/fourier_analysis.json"
    shift_spectrum = None
    if Path(spectrum_path).exists():
        with open(spectrum_path) as f:
            data = json.load(f)
        for pair, shift in data.get("shift_spectra", {}).items():
            shift_spectrum = np.array(shift["diff"])
            break
        logger.info("Loaded shift spectrum")

    # Transforms
    standard_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Training data (IFCB only)
    train_ds_standard = PlanktonDataset(str(train_dir), class_to_idx, standard_transform)
    train_ds_sba = PlanktonDataset(str(train_dir), class_to_idx, SAATransform(shift_spectrum))
    train_loader_standard = DataLoader(train_ds_standard, batch_size=16, shuffle=True, num_workers=4)
    train_loader_sba = DataLoader(train_ds_sba, batch_size=16, shuffle=True, num_workers=4)

    # Test data (both domains combined)
    test_samples = []
    for domain in ["train/DataShift_IFCB", "test/DataShift_ZooScan"]:
        domain_dir = Path(args.data_dir) / domain
        domain_name = "IFCB" if "IFCB" in domain else "ZooScan"
        if domain_dir.is_dir():
            for cls_name in classes:
                cls_dir = domain_dir / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_EXT:
                        test_samples.append((str(img_path), class_to_idx[cls_name], cls_name, domain_name))

    combined_test = PlanktonDataset.__new__(PlanktonDataset)
    combined_test.samples = [(s[0], s[1], s[2]) for s in test_samples]
    combined_test.transform = eval_transform
    test_loader = DataLoader(combined_test, batch_size=32, shuffle=False, num_workers=4)
    test_domains = [s[3] for s in test_samples]

    # Train baseline model
    logger.info("Training BASELINE model...")
    np.random.seed(42); torch.manual_seed(42)
    baseline_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    baseline_model.heads.head = nn.Linear(baseline_model.heads.head.in_features, num_classes)
    baseline_model = baseline_model.to(device)
    train_model(baseline_model, train_loader_standard, device, args.epochs)

    # Train SBA model
    logger.info("Training SBA model...")
    np.random.seed(42); torch.manual_seed(42)
    sba_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    sba_model.heads.head = nn.Linear(sba_model.heads.head.in_features, num_classes)
    sba_model = sba_model.to(device)
    train_model(sba_model, train_loader_sba, device, args.epochs)

    # Extract embeddings
    logger.info("Extracting BASELINE embeddings...")
    base_emb, base_labels, base_cls = extract_embeddings(baseline_model, test_loader, device)

    logger.info("Extracting SBA embeddings...")
    sba_emb, sba_labels, sba_cls = extract_embeddings(sba_model, test_loader, device)

    # Compute separability
    logger.info("Computing separability...")
    base_domain_sep, _ = compute_separability(base_emb, test_domains)
    sba_domain_sep, _ = compute_separability(sba_emb, test_domains)
    base_species_sep, _ = compute_separability(base_emb, base_cls)
    sba_species_sep, _ = compute_separability(sba_emb, sba_cls)

    results = {
        "baseline": {"domain_sep": base_domain_sep, "species_sep": base_species_sep},
        "sba": {"domain_sep": sba_domain_sep, "species_sep": sba_species_sep},
    }
    with open(out / "separability.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("  REPRESENTATION ANALYSIS")
    logger.info("=" * 60)
    logger.info("  Baseline: domain_sep=%.1f%%  species_sep=%.1f%%",
                base_domain_sep * 100, base_species_sep * 100)
    logger.info("  SBA:      domain_sep=%.1f%%  species_sep=%.1f%%",
                sba_domain_sep * 100, sba_species_sep * 100)
    logger.info("  Domain sep change: %+.1f%% (should decrease)",
                (sba_domain_sep - base_domain_sep) * 100)
    logger.info("  Species sep change: %+.1f%% (should be preserved)",
                (sba_species_sep - base_species_sep) * 100)
    logger.info("=" * 60)

    # Generate UMAP
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for emb, cls_list, name in [
            (base_emb, base_cls, "Baseline"),
            (sba_emb, sba_cls, "SBA")
        ]:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            emb_2d = reducer.fit_transform(emb)

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # By species
            ax = axes[0]
            unique_cls = sorted(set(cls_list))
            cmap = plt.cm.get_cmap("tab20", len(unique_cls))
            for i, cls in enumerate(unique_cls):
                mask = [c == cls for c in cls_list]
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[cmap(i)],
                           label=cls, alpha=0.6, s=15, edgecolors="none")
            ax.set_title(f"{name}: Colored by Species")
            ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.5)

            # By domain
            ax = axes[1]
            for domain, color, label in [("IFCB", "#d32f2f", "IFCB"), ("ZooScan", "#2196F3", "ZooScan")]:
                mask = [d == domain for d in test_domains]
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color,
                           label=label, alpha=0.6, s=15, edgecolors="none")
            ax.set_title(f"{name}: Colored by Instrument")
            ax.legend(fontsize=8)

            plt.tight_layout()
            fig.savefig(str(out / f"umap_{name.lower()}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(out / f"umap_{name.lower()}.pdf"), bbox_inches="tight")
            plt.close(fig)

        logger.info("UMAP figures saved to %s", out)

    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP visualization")


if __name__ == "__main__":
    main()
