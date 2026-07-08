"""
beit_saa_3phase.py
==================
3-phase BEiT training on v2a-segmented crops, following Chen's approach:

Phase 1: Initial training (lr=1e-4, 15 epochs) — learn features from crops
Phase 2: Fine-tune with SAA (lr=1e-5, 10 epochs) — add frequency-calibrated augmentation
Phase 3: Fine-tune again (lr=1e-6, 5 epochs) — final refinement

Evaluate on OOD with TTA after each phase.

Usage:
    python beit_saa_3phase.py
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
    handlers=[logging.StreamHandler(), logging.FileHandler("beit_saa_3phase.log", mode="a")],
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

ZOOlAKE2 = "/home/sreenath/research-space/PlanktonShift/data_segmentation/zoolake2"
OOD_DIR = "/home/sreenath/research-space/PlanktonShift/data_segmentation/ood"
RESULTS = "/home/sreenath/research-space/PlanktonShift/results"


class DS(Dataset):
    def __init__(self, d, c2i, tf):
        self.s, self.tf = [], tf
        for c, i in c2i.items():
            cd = Path(d) / c
            if not cd.is_dir():
                continue
            for ip in sorted(cd.glob("*_crop.png")):
                self.s.append((str(ip), i))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l = self.s[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), l, p


class SAAT:
    def __init__(self, ss=None):
        self.aug = SpectralAugmentation(
            shift_spectrum=ss,
            strategies=["band_adversarial"],
            strength=0.5, p=0.8,
        )
        self.base = transforms.Compose([
            transforms.Resize((IMG, IMG)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __call__(self, img):
        arr = np.array(img.convert("L"), dtype=np.float64) / 255.0
        arr_aug = self.aug(arr)
        arr_uint8 = (arr_aug * 255).clip(0, 255).astype(np.uint8)
        return self.base(Image.fromarray(arr_uint8, mode="L").convert("RGB"))


def compute_ss(d, cs, mx=30):
    sps = []
    for c in cs:
        cd = Path(d) / c
        if not cd.is_dir():
            continue
        n = 0
        for ip in sorted(cd.glob("*_crop.png")):
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
        for imgs, lbls, _ in loader:
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
    for imgs, lbls, _ in loader:
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


def main():
    dev = torch.device("cuda")
    os.makedirs(RESULTS, exist_ok=True)

    # Load OOD loaders
    eval_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ood_loaders = {}
    for od in sorted(Path(OOD_DIR).iterdir()):
        if od.is_dir():
            ds = DS(str(od), C2I, eval_tf)
            if len(ds) > 0:
                ood_loaders[od.name] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # Compute shift spectrum
    logger.info("Computing shift spectrum from crops...")
    ss = compute_ss(ZOOlAKE2, CLASSES)

    all_results = {}

    # === PHASE 1: Initial training (standard augmentation, lr=1e-4, 15 epochs) ===
    logger.info("=" * 70)
    logger.info("  PHASE 1: Initial BEiT training (standard aug, lr=1e-4, 15 epochs)")
    logger.info("=" * 70)

    train_tf_p1 = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_ds = DS(ZOOlAKE2, C2I, train_tf_p1)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    logger.info("Train: %d samples", len(train_ds))

    model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                               pretrained=True, num_classes=NC)
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    for epoch in range(15):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, dev)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Epoch %d/15  Loss:%.4f  Train:%.1f%%", epoch + 1, loss, acc * 100)

    # Evaluate Phase 1
    p1_acc, p1_days = eval_ood(model, ood_loaders, dev)
    all_results["phase1"] = {"overall": p1_acc, "per_day": p1_days}
    logger.info("  Phase 1 OOD: %.1f%%", p1_acc * 100)
    for day in sorted(p1_days.keys()):
        logger.info("    %s: %.1f%%", day, p1_days[day] * 100)

    torch.save(model.state_dict(), f"{RESULTS}/beit_crops_phase1.pth")
    logger.info("  Saved phase 1 model")

    # === PHASE 2: Fine-tune with SAA (lr=1e-5, 10 epochs) ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 2: Fine-tune with SAA (lr=1e-5, 10 epochs)")
    logger.info("=" * 70)

    train_tf_p2 = SAAT(ss)
    train_ds_saa = DS(ZOOlAKE2, C2I, train_tf_p2)
    train_loader_saa = DataLoader(train_ds_saa, batch_size=32, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        loss, acc = train_epoch(model, train_loader_saa, criterion, optimizer, dev)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Epoch %d/10  Loss:%.4f  Train:%.1f%%", epoch + 1, loss, acc * 100)

    # Evaluate Phase 2
    p2_acc, p2_days = eval_ood(model, ood_loaders, dev)
    all_results["phase2"] = {"overall": p2_acc, "per_day": p2_days}
    logger.info("  Phase 2 OOD: %.1f%%  Improvement: %+.1f%%", p2_acc * 100, (p2_acc - p1_acc) * 100)
    for day in sorted(p2_days.keys()):
        logger.info("    %s: %.1f%% -> %.1f%% (%+.1f%%)",
                     day, p1_days.get(day, 0) * 100, p2_days[day] * 100,
                     (p2_days[day] - p1_days.get(day, 0)) * 100)

    torch.save(model.state_dict(), f"{RESULTS}/beit_crops_phase2.pth")
    logger.info("  Saved phase 2 model")

    # === PHASE 3: Fine-tune again (lr=1e-6, 5 epochs) ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 3: Final fine-tune (lr=1e-6, 5 epochs)")
    logger.info("=" * 70)

    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    for epoch in range(5):
        loss, acc = train_epoch(model, train_loader_saa, criterion, optimizer, dev)
        scheduler.step()
        logger.info("  Epoch %d/5  Loss:%.4f  Train:%.1f%%", epoch + 1, loss, acc * 100)

    # Evaluate Phase 3
    p3_acc, p3_days = eval_ood(model, ood_loaders, dev)
    all_results["phase3"] = {"overall": p3_acc, "per_day": p3_days}
    logger.info("  Phase 3 OOD: %.1f%%  Improvement: %+.1f%%", p3_acc * 100, (p3_acc - p1_acc) * 100)
    for day in sorted(p3_days.keys()):
        logger.info("    %s: %.1f%%", day, p3_days[day] * 100)

    torch.save(model.state_dict(), f"{RESULTS}/beit_crops_phase3.pth")
    logger.info("  Saved phase 3 model")

    # === SUMMARY ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("  3-PHASE BEiT + SAA ON V2A CROPS — SUMMARY")
    logger.info("=" * 70)
    logger.info("  Phase 1 (standard, lr=1e-4, 15ep): %.1f%%", p1_acc * 100)
    logger.info("  Phase 2 (+SAA, lr=1e-5, 10ep):     %.1f%%  (%+.1f%%)", p2_acc * 100, (p2_acc - p1_acc) * 100)
    logger.info("  Phase 3 (+SAA, lr=1e-6, 5ep):      %.1f%%  (%+.1f%%)", p3_acc * 100, (p3_acc - p1_acc) * 100)
    logger.info("  Chen BEsT (full images, 5 models):  83.0%%")
    logger.info("  BEiT+SAA full images (3 models):    82.1%%")
    logger.info("  ViT on crops (no SAA):              73.6%%")
    logger.info("  vs Chen: %+.1f%%", p3_acc * 100 - 83)
    logger.info("=" * 70)

    # Save
    with open(f"{RESULTS}/beit_saa_3phase.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s/beit_saa_3phase.json", RESULTS)


if __name__ == "__main__":
    main()
