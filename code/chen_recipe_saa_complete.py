"""
chen_recipe_saa_complete.py
============================
Complete experiment: Chen's recipe + SAA, with both starting points.

Approach A: Train from ImageNet-22K with Chen's aug + SAA
  Phase A1: Chen's medium aug + SAA, lr=1e-4, 30 epochs
  Phase A2: Finetune with SAA, lr=1e-5, 15 epochs
  Phase A3: Finetune with SAA, lr=1e-6, 5 epochs

Approach B: Start from Chen's trained model, finetune with SAA
  Phase B1: Evaluate Chen's model (baseline)
  Phase B2: Finetune with SAA, lr=1e-6, 10 epochs
  Phase B3: Finetune with SAA, lr=1e-7, 5 epochs

Uses:
- Chen's EXACT data: ZooLake2.0 → OOD1-10
- Chen's EXACT hyperparameters: batch=128, weight_decay=0.03
- Chen's EXACT augmentation (medium) + SAA on top
- NO normalization (raw [0,1] pixels like Chen)

Usage:
    python chen_recipe_saa_complete.py
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
    handlers=[logging.StreamHandler(), logging.FileHandler("chen_saa_complete.log", mode="a")],
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
CHEN_MODEL = "/home/sreenath/research-space/PlanktonShift/results/finetune_chen_saa/chen_model_01_converted.pth"
RESULTS = "/home/sreenath/research-space/PlanktonShift/results"


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
def chen_medium_aug():
    """Chen's 'medium' augmentation — EXACT copy, adapted for PIL input."""
    return transforms.Compose([
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
    ])


def chen_aug_plus_saa(ss=None):
    """Chen's medium augmentation + SAA band adversarial."""
    saa = SpectralAugmentation(
        shift_spectrum=ss, strategies=["band_adversarial"],
        strength=0.5, p=0.8,
    )

    class CombinedAug:
        def __init__(self):
            self.saa = saa
            self.chen = chen_medium_aug()

        def __call__(self, image):
            # image is PIL Image
            arr = np.array(image)
            # SAA on grayscale
            gray = np.mean(arr[:, :, :3], axis=2).astype(np.float64) / 255.0
            aug_gray = self.saa(gray)
            aug_uint8 = (aug_gray * 255).clip(0, 255).astype(np.uint8)
            # Convert to 3-channel PIL
            aug_rgb = np.stack([aug_uint8] * 3, axis=-1)
            aug_pil = Image.fromarray(aug_rgb)
            return self.chen(aug_pil)

    return CombinedAug()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZooLakeDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        for cls_name, idx in class_to_idx.items():
            cls_dir = Path(data_dir) / cls_name
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
        if self.transform:
            return self.transform(image), label
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0, label


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
# Evaluation
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


def train_epochs(model, loader, criterion, optimizer, scheduler, dev, n_epochs, phase_name, ckpt_path=None):
    for epoch in range(n_epochs):
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
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_epochs - 1:
            logger.info("    %s Epoch %d/%d  Loss:%.4f  Train:%.1f%%",
                        phase_name, epoch + 1, n_epochs, tl / tt, cr / tt * 100)
        # Save checkpoint after every epoch
        if ckpt_path:
            torch.save(model.state_dict(), ckpt_path)
    # Final save
    if ckpt_path:
        torch.save(model.state_dict(), ckpt_path)
        logger.info("    Saved checkpoint: %s", ckpt_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    dev = torch.device("cuda")
    os.makedirs(RESULTS, exist_ok=True)

    # OOD loaders
    eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])
    ood_loaders = {}
    for od in sorted(Path(OOD_DIR).iterdir()):
        if od.is_dir():
            ds = ZooLakeDataset(str(od), C2I, eval_tf)
            if len(ds) > 0:
                ood_loaders[od.name] = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # Shift spectrum
    ss = compute_ss(ZOOlAKE2, CLASSES)
    logger.info("Shift spectrum: %s", "computed" if ss is not None else "None")

    all_results = {}
    criterion = nn.CrossEntropyLoss()

    # ===================================================================
    # APPROACH A: Train from ImageNet-22K with Chen's aug + SAA
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("  APPROACH A: ImageNet-22K + Chen's aug + SAA")
    logger.info("=" * 70)

    train_ds_a = ZooLakeDataset(ZOOlAKE2, C2I, chen_aug_plus_saa(ss))
    train_loader_a = DataLoader(train_ds_a, batch_size=128, shuffle=True, num_workers=0)
    logger.info("Train: %d samples", len(train_ds_a))

    model_a = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                                 pretrained=True, num_classes=NC).to(dev)

    # Phase A1: lr=1e-4, 30 epochs
    a1_ckpt = f"{RESULTS}/beit_saa_phaseA1.pth"
    logger.info("  Phase A1: lr=1e-4, 30 epochs")
    opt = optim.AdamW(model_a.parameters(), lr=1e-4, weight_decay=0.03)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    train_epochs(model_a, train_loader_a, criterion, opt, sched, dev, 30, "A1", a1_ckpt)
    a1_acc, a1_days = eval_ood(model_a, ood_loaders, dev)
    all_results["A1"] = {"overall": a1_acc, "per_day": a1_days}
    logger.info("  A1 OOD: %.1f%%", a1_acc * 100)

    # Phase A2: lr=1e-5, 15 epochs
    a2_ckpt = f"{RESULTS}/beit_saa_phaseA2.pth"
    logger.info("  Phase A2: lr=1e-5, 15 epochs")
    opt = optim.AdamW(model_a.parameters(), lr=1e-5, weight_decay=0.03)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    train_epochs(model_a, train_loader_a, criterion, opt, sched, dev, 15, "A2", a2_ckpt)
    a2_acc, a2_days = eval_ood(model_a, ood_loaders, dev)
    all_results["A2"] = {"overall": a2_acc, "per_day": a2_days}
    logger.info("  A2 OOD: %.1f%%  (%+.1f%%)", a2_acc * 100, (a2_acc - a1_acc) * 100)

    # Phase A3: lr=1e-6, 5 epochs
    a3_ckpt = f"{RESULTS}/beit_saa_phaseA3.pth"
    if os.path.exists(a3_ckpt):
        logger.info("  Phase A3: Loading checkpoint")
        model_a.load_state_dict(torch.load(a3_ckpt, weights_only=True))
    else:
        logger.info("  Phase A3: lr=1e-6, 5 epochs")
        opt = optim.AdamW(model_a.parameters(), lr=1e-6, weight_decay=0.03)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
        train_epochs(model_a, train_loader_a, criterion, opt, sched, dev, 5, "A3")
        torch.save(model_a.state_dict(), a3_ckpt)
    a3_acc, a3_days = eval_ood(model_a, ood_loaders, dev)
    all_results["A3"] = {"overall": a3_acc, "per_day": a3_days}
    logger.info("  A3 OOD: %.1f%%  (%+.1f%%)", a3_acc * 100, (a3_acc - a1_acc) * 100)

    torch.save(model_a.state_dict(), f"{RESULTS}/beit_saa_approachA.pth")
    del model_a
    torch.cuda.empty_cache()

    # ===================================================================
    # APPROACH B: Start from Chen's trained model, finetune with SAA
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("  APPROACH B: Chen's model + SAA finetune")
    logger.info("=" * 70)

    model_b = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k",
                                 pretrained=False, num_classes=NC)
    chen_sd = torch.load(CHEN_MODEL, weights_only=True)
    model_b.load_state_dict(chen_sd, strict=True)
    model_b = model_b.to(dev)
    logger.info("  Loaded Chen's converted model")

    # Phase B1: Evaluate baseline
    b1_acc, b1_days = eval_ood(model_b, ood_loaders, dev)
    all_results["B1"] = {"overall": b1_acc, "per_day": b1_days}
    logger.info("  B1 (Chen baseline): %.1f%%", b1_acc * 100)

    train_ds_b = ZooLakeDataset(ZOOlAKE2, C2I, chen_aug_plus_saa(ss))
    train_loader_b = DataLoader(train_ds_b, batch_size=128, shuffle=True, num_workers=0)

    # Phase B2: lr=1e-6, 10 epochs
    logger.info("  Phase B2: lr=1e-6, 10 epochs")
    opt = optim.AdamW(model_b.parameters(), lr=1e-6, weight_decay=0.03)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    train_epochs(model_b, train_loader_b, criterion, opt, sched, dev, 10, "B2")
    b2_acc, b2_days = eval_ood(model_b, ood_loaders, dev)
    all_results["B2"] = {"overall": b2_acc, "per_day": b2_days}
    logger.info("  B2 OOD: %.1f%%  (%+.1f%%)", b2_acc * 100, (b2_acc - b1_acc) * 100)

    # Phase B3: lr=1e-7, 5 epochs
    logger.info("  Phase B3: lr=1e-7, 5 epochs")
    opt = optim.AdamW(model_b.parameters(), lr=1e-7, weight_decay=0.03)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    train_epochs(model_b, train_loader_b, criterion, opt, sched, dev, 5, "B3")
    b3_acc, b3_days = eval_ood(model_b, ood_loaders, dev)
    all_results["B3"] = {"overall": b3_acc, "per_day": b3_days}
    logger.info("  B3 OOD: %.1f%%  (%+.1f%%)", b3_acc * 100, (b3_acc - b1_acc) * 100)

    torch.save(model_b.state_dict(), f"{RESULTS}/beit_saa_approachB.pth")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("  COMPLETE RESULTS")
    logger.info("=" * 70)
    logger.info("  Approach A (ImageNet-22K + SAA):")
    logger.info("    A1 (lr=1e-4, 30ep): %.1f%%", a1_acc * 100)
    logger.info("    A2 (lr=1e-5, 15ep): %.1f%%", a2_acc * 100)
    logger.info("    A3 (lr=1e-6, 5ep):  %.1f%%", a3_acc * 100)
    logger.info("  Approach B (Chen's model + SAA):")
    logger.info("    B1 (Chen baseline): %.1f%%", b1_acc * 100)
    logger.info("    B2 (lr=1e-6, 10ep): %.1f%%", b2_acc * 100)
    logger.info("    B3 (lr=1e-7, 5ep):  %.1f%%", b3_acc * 100)
    logger.info("  Chen BEsT: 83.0%%")
    best = max(a3_acc, b3_acc)
    logger.info("  Best: %.1f%%  vs Chen: %+.1f%%", best * 100, best * 100 - 83)
    logger.info("=" * 70)

    with open(f"{RESULTS}/chen_saa_complete.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Done!")


if __name__ == "__main__":
    main()
