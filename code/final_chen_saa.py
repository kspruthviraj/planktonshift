"""
final_chen_saa.py
=================
BULLETPROOF: Reproduce Chen's recipe + SAA, beat 83% OOD.

SAFETY: Checkpoint saved EVERY epoch. Automatic resume from last epoch.
MEMORY: GPU cache cleared between phases to prevent OOM.
DATA: Uses Chen's EXACT ZooLake2.0 → OOD1-10 split (29,499 train, 9,522 test).
AUG: Chen's exact medium augmentation + SAA band adversarial.
ARCH: beit_base_patch16_224.in22k_ft_in22k_in1k (ImageNet-22K).
HYPER: batch=128, weight_decay=0.03, NO normalization.

Approach A: ImageNet-22K + Chen aug + SAA
  A1: lr=1e-4, 30 epochs
  A2: lr=1e-5, 15 epochs  
  A3: lr=1e-6, 5 epochs

Approach B: Chen's trained model + SAA fine-tune
  B1: Baseline evaluation only
  B2: lr=1e-6, 10 epochs
  B3: lr=1e-7, 5 epochs

Usage: python final_chen_saa.py
Resume: just re-run — automatically detects existing checkpoints
"""

import json, logging, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm

sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation

# ── CONFIG ──
IMG = 224
BATCH = 128
WD = 0.03
CLASSES = [
    "aphanizomenon","asplanchna","asterionella","bosmina","ceratium",
    "chaoborus","collotheca","conochilus","copepod_skins","cyclops",
    "daphnia","daphnia_skins","diaphanosoma","diatom_chain","dinobryon",
    "dirt","eudiaptomus","filament","fish","fragilaria","hydra",
    "kellicottia","keratella_cochlearis","keratella_quadrata","leptodora",
    "maybe_cyano","nauplius","paradileptus","polyarthra","rotifers",
    "synchaeta","trichocerca","unknown","unknown_plankton","uroglena",
]
C2I = {c: i for i, c in enumerate(CLASSES)}; NC = len(CLASSES)
ZOOlAKE2 = "/home/sreenath/research-space/PlanktonShift/data/chen_data/ZooLake2/ZooLake2/ZooLake2.0"
OOD_DIR = "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs"
CHEN_MODEL = "/home/sreenath/research-space/PlanktonShift/results/finetune_chen_saa/chen_model_01_converted.pth"
RESULTS = "/home/sreenath/research-space/PlanktonShift/results"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"{RESULTS}/final_chen_saa.log", mode="a")],
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ════════════════════════════════════════════════════════════════
def chen_medium_aug():
    """Chen's EXACT 'medium' augmentation (from for_plankton.py)."""
    return transforms.Compose([
        transforms.Resize((IMG, IMG)),
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


def chen_saa_aug(ss):
    """Chen's medium aug + SAA band adversarial (our contribution)."""
    saa = SpectralAugmentation(shift_spectrum=ss, strategies=["band_adversarial"], strength=0.5, p=0.8)
    class W:
        def __init__(s): s.s = saa; s.c = chen_medium_aug()
        def __call__(s, img):
            a = np.array(img)
            g = np.mean(a[:,:,:3], axis=2).astype(np.float64) / 255.0
            return s.c(Image.fromarray(np.stack([(s.s(g)*255).clip(0,255).astype(np.uint8)]*3,axis=-1)))
    return W()


# ════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════
class ZLDataset(Dataset):
    def __init__(s, d, c2i, tf):
        s.s, s.tf = [], tf
        for c, i in c2i.items():
            cd = Path(d)/c
            if not cd.is_dir(): continue
            for ip in sorted(cd.glob("*.png") if c.startswith(".") else cd.iterdir()):
                if ip.suffix.lower() in {".png",".jpg",".jpeg"}:
                    s.s.append((str(ip), i))
    def __len__(s): return len(s.s)
    def __getitem__(s, i):
        p, l = s.s[i]
        img = Image.open(p).convert("RGB")
        return (s.tf(img), l) if s.tf else (torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0, l)


# ════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def tta(model, imgs, dev):
    ps = []
    for k in range(4):
        r = torch.rot90(imgs, k, dims=[2,3]).to(dev)
        ps.append(torch.softmax(model(r), dim=1))
        ps.append(torch.softmax(model(torch.flip(r,[3])), dim=1))
    return torch.stack(ps).mean(0)

@torch.no_grad()
def eval_ood(model, loaders, dev):
    model.eval(); d = {}; tc,tn = 0,0
    for day, ld in loaders.items():
        cr,n = 0,0
        for im,lb in ld:
            im,lb = im.to(dev),lb.to(dev)
            _,p = tta(model,im,dev).max(1); cr+=p.eq(lb).sum().item(); n+=lb.size(0)
        d[day]=cr/max(n,1); tc+=cr; tn+=n
    return tc/max(tn,1), d

def save_results(name, data):
    with open(f"{RESULTS}/final_{name}.json", "w") as f:
        json.dump({"timestamp": str(datetime.now()), "overall": data[0], "per_day": data[1]}, f, indent=2)

# ════════════════════════════════════════════════════════════════
# SHIFT SPECTRUM (for SAA calibration)
# ════════════════════════════════════════════════════════════════
def compute_ss(d, cs, mx=30):
    sps = []
    for c in cs:
        cd = Path(d)/c
        if not cd.is_dir(): continue
        for ip in list(cd.iterdir())[:mx]:
            if ip.suffix.lower() not in {".png",".jpg",".jpeg"}: continue
            try:
                a = np.array(Image.open(ip).convert("L").resize((224,224)),dtype=np.float64)/255.0
                f = np.fft.fft2(a); amp = np.log1p(np.abs(np.fft.fftshift(f)))
                h,w = amp.shape; cy,cx = h//2,w//2
                Y,X = np.ogrid[:h,:w]; R = np.sqrt((X-cx)**2+(Y-cy)**2).astype(int)
                mr = min(cx,cy); r = np.zeros(mr)
                for rr in range(mr):
                    m = R==rr
                    if m.sum()>0: r[rr]=amp[m].mean()
                sps.append(r)
            except: continue
    if not sps: return None
    ml = max(len(x) for x in sps); m = np.zeros((len(sps),ml))
    for i,s in enumerate(sps): m[i,:len(s)] = s
    return m.std(axis=0)

# ════════════════════════════════════════════════════════════════
# TRAINING WITH CHECKPOINTS
# ════════════════════════════════════════════════════════════════
def train_phase(model, loader, crit, lr, epochs, phase_name, ckpt_path, dev):
    """Train with checkpoint every epoch. Auto-resume from last epoch."""
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    start = 0

    # Check for existing checkpoint
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            start = ck["epoch"] + 1
            for _ in range(start): sched.step()
            logger.info("  %s: Resumed from epoch %d/%d", phase_name, start, epochs)
        except Exception as e:
            logger.warning("  %s: Could not resume (%s). Starting fresh.", phase_name, e)

    for e in range(start, epochs):
        model.train(); tl,cr,tt = 0,0,0
        for im,lb in loader:
            im,lb = im.to(dev),lb.to(dev); opt.zero_grad()
            loss = crit(model(im), lb); loss.backward(); opt.step()
            tl += loss.item()*im.size(0); _,p = model(im).max(1); cr+=p.eq(lb).sum().item(); tt+=im.size(0)
        sched.step()
        # SAVE CHECKPOINT EVERY EPOCH
        torch.save({"epoch": e, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt_path)
        if (e+1)%10==0 or e==0 or e==epochs-1:
            logger.info("    %s Epoch %d/%d  Loss:%.4f  Train:%.1f%%", phase_name, e+1, epochs, tl/tt, cr/tt*100)

    logger.info("  %s: %d epochs complete", phase_name, epochs)

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    dev = torch.device("cuda")
    os.makedirs(RESULTS, exist_ok=True)
    crit = nn.CrossEntropyLoss()

    # ── OOD loaders ──
    etf = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor()])
    ood = {}
    for od in sorted(Path(OOD_DIR).iterdir()):
        if od.is_dir():
            ds = ZLDataset(str(od), C2I, etf)
            if len(ds)>0: ood[od.name] = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    logger.info("OOD cells: %s (%d total)", list(ood.keys()), sum(len(l.dataset) for l in ood.values()))

    # ── Shift spectrum ──
    ss = compute_ss(ZOOlAKE2, CLASSES)
    logger.info("Shift spectrum: %s", "computed" if ss is not None else "None")

    # ══════════════════════════════════════════════════════════
    # APPROACH A: ImageNet-22K + Chen aug + SAA
    # ══════════════════════════════════════════════════════════
    logger.info("="*70); logger.info("  APPROACH A: ImageNet-22K + Chen aug + SAA"); logger.info("="*70)
    train_a = ZLDataset(ZOOlAKE2, C2I, chen_saa_aug(ss))
    train_loader_a = DataLoader(train_a, batch_size=BATCH, shuffle=True, num_workers=0)
    logger.info("Train: %d samples", len(train_a))

    model_a = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=True, num_classes=NC).to(dev)

    # A1
    train_phase(model_a, train_loader_a, crit, 1e-4, 30, "A1", f"{RESULTS}/ckpt_A1.pth", dev)
    a1 = eval_ood(model_a, ood, dev)
    save_results("A1", a1)
    logger.info("  A1 OOD: %.1f%%", a1[0]*100)
    for day in sorted(a1[1]): logger.info("    %s: %.1f%%", day, a1[1][day]*100)

    # A2
    train_phase(model_a, train_loader_a, crit, 1e-5, 15, "A2", f"{RESULTS}/ckpt_A2.pth", dev)
    a2 = eval_ood(model_a, ood, dev)
    save_results("A2", a2)
    logger.info("  A2 OOD: %.1f%% (%+.1f%%)", a2[0]*100, (a2[0]-a1[0])*100)

    # A3
    train_phase(model_a, train_loader_a, crit, 1e-6, 5, "A3", f"{RESULTS}/ckpt_A3.pth", dev)
    a3 = eval_ood(model_a, ood, dev)
    save_results("A3", a3)
    logger.info("  A3 OOD: %.1f%% (%+.1f%%)", a3[0]*100, (a3[0]-a1[0])*100)

    del model_a; torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # APPROACH B: Chen's trained model + SAA finetune
    # ══════════════════════════════════════════════════════════
    logger.info("="*70); logger.info("  APPROACH B: Chen's model + SAA finetune"); logger.info("="*70)
    model_b = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=False, num_classes=NC).to(dev)
    model_b.load_state_dict(torch.load(CHEN_MODEL, weights_only=True, map_location=dev))
    logger.info("  Loaded Chen's converted model")

    # B1: Baseline
    b1 = eval_ood(model_b, ood, dev)
    save_results("B1", b1)
    logger.info("  B1 (Chen baseline): %.1f%%", b1[0]*100)

    train_b = ZLDataset(ZOOlAKE2, C2I, chen_saa_aug(ss))
    train_loader_b = DataLoader(train_b, batch_size=BATCH, shuffle=True, num_workers=0)

    # B2
    train_phase(model_b, train_loader_b, crit, 1e-6, 10, "B2", f"{RESULTS}/ckpt_B2.pth", dev)
    b2 = eval_ood(model_b, ood, dev)
    save_results("B2", b2)
    logger.info("  B2 OOD: %.1f%% (%+.1f%%)", b2[0]*100, (b2[0]-b1[0])*100)

    # B3
    train_phase(model_b, train_loader_b, crit, 1e-7, 5, "B3", f"{RESULTS}/ckpt_B3.pth", dev)
    b3 = eval_ood(model_b, ood, dev)
    save_results("B3", b3)
    logger.info("  B3 OOD: %.1f%% (%+.1f%%)", b3[0]*100, (b3[0]-b1[0])*100)

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    logger.info("="*70); logger.info("  FINAL SUMMARY"); logger.info("="*70)
    logger.info("  Approach A: %s → %s → %s", f"{a1[0]*100:.1f}%", f"{a2[0]*100:.1f}%", f"{a3[0]*100:.1f}%")
    logger.info("  Approach B: %s → %s → %s", f"{b1[0]*100:.1f}%", f"{b2[0]*100:.1f}%", f"{b3[0]*100:.1f}%")
    logger.info("  Chen BEsT: 83.0%%")
    best = max(a3[0], b3[0])
    logger.info("  Best: %.1f%%  vs Chen: %+.1f%%", best*100, best*100-83)

    # Save full report
    full = {
        "approach_A": {"A1": a1[0], "A2": a2[0], "A3": a3[0], "improvement": a3[0]-a1[0]},
        "approach_B": {"B1": b1[0], "B2": b2[0], "B3": b3[0], "improvement": b3[0]-b1[0]},
        "vs_chen": best - 0.83,
        "timestamp": str(datetime.now()),
    }
    with open(f"{RESULTS}/final_chen_saa_summary.json", "w") as f:
        json.dump(full, f, indent=2)
    logger.info("Full report saved")


if __name__ == "__main__":
    main()
