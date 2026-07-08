"""approach_B_only.py — Fine-tune Chen's model with Chen aug + SAA."""
import json, logging, os, sys, numpy as np, torch
from pathlib import Path
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm

sys.path.insert(0, "/home/sreenath/research-space/Adverserial_net")
from spectral_augmentation import SpectralAugmentation
from transformers import BeitForImageClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMG, BATCH, WD = 224, 128, 0.03
CLASSES = [c.strip() for c in "aphanizomenon,asplanchna,asterionella,bosmina,ceratium,chaoborus,collotheca,conochilus,copepod_skins,cyclops,daphnia,daphnia_skins,diaphanosoma,diatom_chain,dinobryon,dirt,eudiaptomus,filament,fish,fragilaria,hydra,kellicottia,keratella_cochlearis,keratella_quadrata,leptodora,maybe_cyano,nauplius,paradileptus,polyarthra,rotifers,synchaeta,trichocerca,unknown,unknown_plankton,uroglena".split(",")]
C2I = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)
ZOO = "/home/sreenath/research-space/PlanktonShift/data/chen_data/ZooLake2/ZooLake2/ZooLake2.0"
OOD = "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs"
CM = "/home/sreenath/research-space/PlanktonShift/results/finetune_chen_saa/chen_model_01_converted.pth"
RESULTS = "/home/sreenath/research-space/PlanktonShift/results"


def chen_aug():
    return transforms.Compose([
        transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 360)), transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.03, 0.03)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 10.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(5, 5, 5, 5)),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
    ])


def compute_ss(d, cs, mx=30):
    sps = []
    for c in cs:
        cd = Path(d) / c
        if not cd.is_dir(): continue
        for ip in list(cd.iterdir())[:mx]:
            if ip.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
            try:
                a = np.array(Image.open(ip).convert("L").resize((224, 224)), dtype=np.float64) / 255.0
                f = np.fft.fft2(a); amp = np.log1p(np.abs(np.fft.fftshift(f)))
                h, w = amp.shape; cy, cx = h // 2, w // 2
                Y, X = np.ogrid[:h, :w]; R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
                mr = min(cx, cy); r = np.zeros(mr)
                for rr in range(mr):
                    m = R == rr
                    if m.sum() > 0: r[rr] = amp[m].mean()
                sps.append(r)
            except: continue
    if not sps: return None
    ml = max(len(x) for x in sps); m = np.zeros((len(sps), ml))
    for i, s in enumerate(sps): m[i, :len(s)] = s
    return m.std(axis=0)


class ZD(Dataset):
    def __init__(s, d, c2i, tf):
        s.s, s.tf = [], tf
        for c, i in c2i.items():
            cd = Path(d) / c
            if not cd.is_dir(): continue
            for ip in cd.iterdir():
                if ip.suffix.lower() in {".png", ".jpg", ".jpeg"}: s.s.append((str(ip), i))

    def __len__(s): return len(s.s)

    def __getitem__(s, i):
        p, l = s.s[i]; img = Image.open(p).convert("RGB")
        return (s.tf(img), l) if s.tf else (torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0, l)


@torch.no_grad()
def tta(model, imgs, dev):
    ps = []
    for k in range(4):
        r = torch.rot90(imgs, k, dims=[2, 3]).to(dev)
        out = model(pixel_values=r)
        l = out.logits if hasattr(out, "logits") else out
        ps.append(torch.softmax(l, dim=1))
        f = torch.flip(r, [3])
        out2 = model(pixel_values=f)
        l2 = out2.logits if hasattr(out2, "logits") else out2
        ps.append(torch.softmax(l2, dim=1))
    return torch.stack(ps).mean(0)


@torch.no_grad()
def eval_ood(model, loaders, dev):
    model.eval(); d = {}; tc, tn = 0, 0
    for day, ld in loaders.items():
        cr, n = 0, 0
        for im, lb in ld:
            im, lb = im.to(dev), lb.to(dev)
            _, p = tta(model, im, dev).max(1); cr += p.eq(lb).sum().item(); n += lb.size(0)
        d[day] = cr / max(n, 1); tc += cr; tn += n
    return tc / max(tn, 1), d


def train_phase(model, loader, lr, epochs, name, ckpt_path, dev):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    start = 0
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = ck["epoch"] + 1
            for _ in range(start): sched.step()
            logger.info("  %s: Resumed from epoch %d/%d", name, start, epochs)
        except: pass
    for e in range(start, epochs):
        model.train(); tl, cr, tt = 0, 0, 0
        for im, lb in loader:
            im, lb = im.to(dev), lb.to(dev); opt.zero_grad()
            out = model(pixel_values=im, labels=lb)
            out.loss.backward(); opt.step()
            tl += out.loss.item() * im.size(0)
            _, p = out.logits.max(1); cr += p.eq(lb).sum().item(); tt += im.size(0)
        sched.step()
        torch.save({"epoch": e, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt_path)
        if (e + 1) % 5 == 0 or e == 0 or e == epochs - 1:
            logger.info("    %s Epoch %d/%d  Loss:%.4f  Train:%.1f%%", name, e + 1, epochs, tl / tt, cr / tt * 100)


def main():
    dev = torch.device("cuda")
    etf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])
    ood = {}
    for od in sorted(Path(OOD).iterdir()):
        if od.is_dir():
            ds = ZD(str(od), C2I, etf)
            if len(ds) > 0: ood[od.name] = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    logger.info("OOD: %d cells", len(ood))
    ss = compute_ss(ZOO, CLASSES)

    logger.info("Loading Chen model (HuggingFace BEiT)...")
    model = BeitForImageClassification.from_pretrained(
        "microsoft/beit-base-patch16-224", num_labels=36, ignore_mismatched_sizes=True
    )
    sd = torch.load(CM, weights_only=True, map_location=dev)
    for k in list(sd.keys()):
        if 'k_proj.bias' in k:
            del sd[k]
    model.load_state_dict(sd, strict=False)
    model.config.num_labels = 36
    model.num_labels = 36
    model = model.to(dev)
    logger.info("Loaded!")

    # B1
    b1 = eval_ood(model, ood, dev)
    logger.info("B1 (Chen baseline): %.1f%%", b1[0] * 100)
    json.dump({"overall": b1[0], "per_day": b1[1]}, open(f"{RESULTS}/final_B1.json", "w"))

    # SAA + Chen aug
    saa = SpectralAugmentation(shift_spectrum=ss, strategies=["band_adversarial"], strength=0.5, p=0.8)

    class W:
        def __init__(s): s.s = saa; s.c = chen_aug()

        def __call__(s, img):
            a = np.array(img); g = np.mean(a[:, :, :3], axis=2).astype(np.float64) / 255.0
            return s.c(Image.fromarray(np.stack([(s.s(g) * 255).clip(0, 255).astype(np.uint8)] * 3, axis=-1)))

    train_b = ZD(ZOO, C2I, W())
    train_loader_b = DataLoader(train_b, batch_size=BATCH, shuffle=True, num_workers=0)

    # B2
    train_phase(model, train_loader_b, 1e-6, 10, "B2", f"{RESULTS}/ckpt_B2.pth", dev)
    b2 = eval_ood(model, ood, dev)
    logger.info("B2 OOD: %.1f%% (%+.1f%%)", b2[0] * 100, (b2[0] - b1[0]) * 100)
    json.dump({"overall": b2[0], "per_day": b2[1]}, open(f"{RESULTS}/final_B2.json", "w"))

    # B3
    train_phase(model, train_loader_b, 1e-7, 5, "B3", f"{RESULTS}/ckpt_B3.pth", dev)
    b3 = eval_ood(model, ood, dev)
    logger.info("B3 OOD: %.1f%% (%+.1f%%)", b3[0] * 100, (b3[0] - b1[0]) * 100)
    json.dump({"overall": b3[0], "per_day": b3[1]}, open(f"{RESULTS}/final_B3.json", "w"))

    logger.info("APPROACH B COMPLETE. Best: %.1f%%  vs Chen: %+.1f%%", max(b1[0], b2[0], b3[0]) * 100,
                max(b1[0], b2[0], b3[0]) * 100 - 83)


if __name__ == "__main__":
    main()
