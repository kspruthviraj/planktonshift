"""
chen_recipe_plus_saa.py
=======================
Replicate Chen's exact training recipe, then add SAA.

Phase 1: Chen's exact recipe (BEiT, medium aug, lr=1e-4, 50ep)
Phase 2: Chen's finetune (lr=1e-5, 50ep, same aug)
Phase 3: Add SAA on top (lr=1e-6, 10ep, SAA + Chen's aug)

Uses Chen's EXACT hyperparameters:
- Architecture: beit_base_patch16_224.in22k_ft_in22k_in1k
- Batch size: 128
- Weight decay: 0.03
- No normalization (raw [0,1] pixels)
- RandomResizedCrop(224, scale=(0.3, 1.0))

Usage:
    python chen_recipe_plus_saa.py
"""

import json, logging, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm

sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chen_recipe_saa.log", mode="a")],
)
logger = logging.getLogger(__name__)

IMG = 224
CLASSES = [
    "aphanizomenon","asplanchna","asterionella","bosmina","ceratium",
    "chaoborus","collotheca","conochilus","copepod_skins","cyclops",
    "daphnia","daphnia_skins","diaphanosoma","diatom_chain","dinobryon",
    "dirt","eudiaptomus","filament","fish","fragilaria","hydra",
    "kellicottia","keratella_cochlearis","keratella_quadrata","leptodora",
    "maybe_cyano","nauplius","paradileptus","polyarthra","rotifers",
    "synchaeta","trichocerca","unknown","unknown_plankton","uroglena",
]
C2I = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)

ZOOlAKE2 = "/home/sreenath/research-space/PlanktonShift/data/chen_data/ZooLake2/ZooLake2/ZooLake2.0"
OOD_DIR = "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs"
RESULTS = "/home/sreenath/research-space/PlanktonShift/results"


# ---------------------------------------------------------------------------
# Chen's EXACT augmentations (from for_plankton.py)
# ---------------------------------------------------------------------------
def chen_medium_aug():
    """Chen's 'medium' augmentation — EXACT copy from for_plankton.py."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.5, 1.5),
                                saturation=(0.5, 1.5), hue=(-0.03, 0.03)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 10.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                 scale=(0.8, 1.2), shear=(5, 5, 5, 5)),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        # NO normalization — Chen uses raw [0,1] pixels
    ])


class SAATransformWithChen:
    """SAA augmentation combined with Chen's medium augmentation."""
    def __init__(self, ss=None):
        self.aug = SpectralAugmentation(
            shift_spectrum=ss,
            strategies=["band_adversarial"],
            strength=0.5, p=0.8,
        )
        self.chen_aug = chen_medium_aug()

    def __call__(self, image):
        # Convert to numpy for SAA
        arr = np.array(image.convert("L"), dtype=np.float64) / 255.0
        # Apply SAA
        arr_aug = self.aug(arr)
        # Convert to uint8 for Chen's transforms
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        # Apply Chen's transforms
        return self.chen_aug(arr_uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZooLakeDataset(Dataset):
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
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        # Convert to numpy uint8 (Chen's pipeline)
        arr = np.array(image)
        if self.transform:
            return self.transform(arr), label
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0, label


# ---------------------------------------------------------------------------
# Shift spectrum
# ---------------------------------------------------------------------------
def compute_ss(d, cs, mx=30):
    sps = []
    for c in cs:
        cd = Path(d) / c
        if not cd.is_dir():
            continue
        n = 0
        for ip in sorted(cd.iterdir()):
            if ip.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                a = np.array(Image.open(ip).convert("L").resize((224, 224)), dtype=np.float64) / 255.0
                f = np.fft.fft2(a)
                amp = np.log1p(np.abs(np.fft.fftshift(f)))
                h, w = amp.shape
                cy, cx = h // 2, w // 2
                mr = min(cx, cy)
                Y, X = np.ogrid[:h, :w]
                R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
                r = np.zeros(mr)
                for rr in range(mr):
                    m = R == rr
                    if m.sum() > 0:
                        r[rr] = amp[m].mean()
                sps.append(r)
                n += 1
            except:
                continue
            if n >= mx:
                break
    if not sps:
        return None
    ml = max(len(s) for s in sps)
    m = np.zeros((len(sps), ml))
    for i, s in enumerate(sps):
        m[i, :len(s)] = s
    return m.std(axis=0)


# ---------------------------------------------------------------------------
# Evaluation with TTA
# ---------------------------------------------------------------------------
@torch.no_grad()
def rot_tta(model, imgs, dev):
    ps = []
    for k in range(4):
        r = torch.rot90(imgs, k, dims=[2, 3]).to(dev)
        o = model(r)
        l = o.logits if hasattr(o, "logits") else o
        ps.append(torch.softmax(l, dim=1))
        f = torch.flip(r, [3])
        o2 = model(f)
        l2 = o2.logits if hasattr(o2, "logits") else o2
        ps.append(torch.softmax(l2, dim=1))
    return torch.stack(ps).mean(0)


@torch.no_grad()
def eval_ood(model, ood_loaders, dev):
    model.eval()
    per_day = {}
    tc, tn = 0, 0
    for day, loader in ood_loaders.items():
        cr, n = 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            probs = rot_tta(model, imgs, dev)
            _, p = probs.max(1)
            cr += p.eq(lbls).sum().item()
            n += lbls.size(0)
        per_day[day] = cr / max(n, 1)
        tc += cr
        tn += n
    return tc / max(tn, 1), per_day


def train_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    tl, cr, tt = 0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(dev), lbls.to(dev)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tl += loss.item() * imgs.size(0)
        _, p = out.max(1)
        cr += p.eq(lbls).sum().item()
        tt += lbls.size(0)
    return tl / tt, cr / tt


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    dev = torch.device("cuda")
    os.makedirs(RESULTS, exist_ok=True)

    # OOD loaders (eval transform: Chen's resize + ToTensor, no normalization)
    eval_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
    ])
    ood_loaders = {}
    for od in sorted(Path(OOD_DIR).iterdir()):
        if od.is_dir():
            ds = ZooLakeDataset(str(od), C2I, eval_tf)
            if len(ds) > 0:
                ood_loaders[od.name] = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    logger.info("OOD cells: %s (%d total samples)",
                list(ood_loaders.keys()),
                sum(len(l.dataset) for l in ood_loaders.values()))

    # Shift spectrum for SAA
    ss = compute_ss(ZOOlAKE2, CLASSES)

    all_results = {}

    # ===================================================================
    # PHASE 1: Chen's exact recipe (BEiT, medium aug, lr=1e-4, 50ep)
    # ===================================================================
    logger.info("=" * 70)
    logger.info("  PHASE 1: Chen's exact recipe (BEiT, medium aug, lr=1e-4)")
    logger.info("=" * 70)

    train_ds = ZooLakeDataset(ZOOlAKE2, C2I, chen_medium_aug())
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    logger.info("Train: %d samples", len(train_ds))

    model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=True, num_classes=NC)
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, dev)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  Epoch %d/50  Loss:%.4f  Train:%.1f%%", epoch + 1, loss, acc * 100)

    p1_acc, p1_days = eval_ood(model, ood_loaders, dev)
    all_results["phase1_chen_recipe"] = {"overall": p1_acc, "per_day": p1_days}
    logger.info("  Phase 1 OOD: %.1f%%", p1_acc * 100)
    for day in sorted(p1_days.keys()):
        logger.info("    %s: %.1f%%", day, p1_days[day] * 100)

    torch.save(model.state_dict(), f"{RESULTS}/beit_chen_phase1.pth")

    # ===================================================================
    # PHASE 2: Chen's finetune (lr=1e-5, 50ep, same aug)
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 2: Chen's finetune (lr=1e-5, 50ep)")
    logger.info("=" * 70)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, dev)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  Epoch %d/50  Loss:%.4f  Train:%.1f%%", epoch + 1, loss, acc * 100)

    p2_acc, p2_days = eval_ood(model, ood_loaders, dev)
    all_results["phase2_chen_finetune"] = {"overall": p2_acc, "per_day": p2_days}
    logger.info("  Phase 2 OOD: %.1f%%  Improvement: %+.1f%%", p2_acc * 100, (p2_acc - p1_acc) * 100)
    for day in sorted(p2_days.keys()):
        logger.info("    %s: %.1f%% -> %.1f%% (%+.1f%%)",
                     day, p1_days.get(day, 0) * 100, p2_days[day] * 100,
                     (p2_days[day] - p1_days.get(day, 0)) * 100)

    torch.save(model.state_dict(), f"{RESULTS}/beit_chen_phase2.pth")

    # ===================================================================
    # PHASE 3: Add SAA on top (lr=1e-6, 10ep, SAA + Chen's aug)
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 3: Add SAA on top (lr=1e-6, 10ep)")
    logger.info("=" * 70)

    train_ds_saa = ZooLakeDataset(ZOOlAKE2, C2I, SAATransformWithChen(ss))
    train_loader_saa = DataLoader(train_ds_saa, batch_size=128, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        loss, acc = train_epoch(model, train_loader_saa, criterion, optimizer, dev)
        scheduler.step()
        logger.info("  Epoch %d/10  Loss:%.4f  Train:%.1f%%", epoch + 1, loss, acc * 100)

    p3_acc, p3_days = eval_ood(model, ood_loaders, dev)
    all_results["phase3_saa"] = {"overall": p3_acc, "per_day": p3_days}
    logger.info("  Phase 3 OOD: %.1f%%  vs Phase 1: %+.1f%%  vs Phase 2: %+.1f%%",
                p3_acc * 100, (p3_acc - p1_acc) * 100, (p3_acc - p2_acc) * 100)
    for day in sorted(p3_days.keys()):
        logger.info("    %s: %.1f%%", day, p3_days[day] * 100)

    torch.save(model.state_dict(), f"{RESULTS}/beit_chen_phase3_saa.pth")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SUMMARY: Chen's Recipe + SAA")
    logger.info("=" * 70)
    logger.info("  Phase 1 (Chen recipe, lr=1e-4, 50ep): %.1f%%", p1_acc * 100)
    logger.info("  Phase 2 (Chen finetune, lr=1e-5, 50ep): %.1f%%  (%+.1f%%)", p2_acc * 100, (p2_acc - p1_acc) * 100)
    logger.info("  Phase 3 (+SAA, lr=1e-6, 10ep):         %.1f%%  (%+.1f%%)", p3_acc * 100, (p3_acc - p2_acc) * 100)
    logger.info("  Chen BEsT (5 models × 3 versions):     83.0%%")
    logger.info("  vs Chen: %+.1f%%", p3_acc * 100 - 83)
    logger.info("=" * 70)

    with open(f"{RESULTS}/chen_recipe_saa.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s/chen_recipe_saa.json", RESULTS)


if __name__ == "__main__":
    main()
