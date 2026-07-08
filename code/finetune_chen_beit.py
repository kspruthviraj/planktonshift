"""finetune_chen_beit.py — Fine-tune Chen's BEiT models with SAA."""
import json, logging, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import timm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-5
WEIGHT_DECAY = 0.03
BATCH = 128
RESULTS = "/home/sreenath/research-space/PlanktonShift/results/finetune_chen_saa"
MODELS = "/home/sreenath/research-space/PlanktonShift/data/chen_models/beit_models/trained_BEiT_models/trained_models"
ZOOLAKE = "/home/sreenath/research-space/PlanktonShift/data/chen_data/ZooLake2/ZooLake2/ZooLake2.0"
OOD = "/home/sreenath/research-space/PlanktonShift/data/chen_data/OOD_data/OODs"

CLASSES = [
    "aphanizomenon","asplanchna","asterionella","bosmina",
    "ceratium","chaoborus","collotheca","conochilus","copepod_skins",
    "cyclops","daphnia","daphnia_skins","diaphanosoma","diatom_chain",
    "dinobryon","dirt","eudiaptomus","filament","fish","fragilaria",
    "hydra","kellicottia","keratella_cochlearis","keratella_quadrata",
    "leptodora","maybe_cyano","nauplius","paradileptus","polyarthra",
    "rotifers","synchaeta","trichocerca","unknown","unknown_plankton","uroglena",
]


class DS(Dataset):
    def __init__(self, d, c2i, tf):
        self.s, self.tf = [], tf
        for c, i in c2i.items():
            cd = Path(d) / c
            if not cd.is_dir():
                continue
            for ip in sorted(cd.iterdir()):
                if ip.suffix.lower() in SUPPORTED_EXT:
                    self.s.append((str(ip), i))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        p, l = self.s[i]
        img = Image.open(p).convert("RGB")
        if self.tf:
            img = self.tf(img)
        return img, l


class SAAT:
    """Fast SAA: pre-compute noise envelope from shift spectrum, apply after resize."""
    def __init__(self, ss=None):
        self.noise_env = None
        if ss is not None:
            h, w = IMG_SIZE, IMG_SIZE
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            R_int = np.clip(R.astype(int), 0, len(ss) - 1)
            self.noise_env = ss[R_int]
        self.resize = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
        ])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        img = self.resize(img)
        arr = np.array(img, dtype=np.float64) / 255.0
        if self.noise_env is not None and np.random.random() < 0.8:
            noise = np.random.randn(IMG_SIZE, IMG_SIZE) * self.noise_env * 0.5
            for c in range(3):
                arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 1)
        return self.aug(Image.fromarray((arr * 255).astype(np.uint8)))


def compute_ss(d, cs, mx=30):
    sps = []
    for c in cs:
        cd = Path(d) / c
        if not cd.is_dir():
            continue
        n = 0
        for ip in sorted(cd.iterdir()):
            if ip.suffix.lower() not in SUPPORTED_EXT:
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
            except Exception:
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
        ps.append(torch.softmax(model(r), dim=1))
        ps.append(torch.softmax(model(torch.flip(r, [3])), dim=1))
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


def main():
    os.makedirs(RESULTS, exist_ok=True)
    dev = torch.device("cuda")
    c2i = {c: i for i, c in enumerate(CLASSES)}
    nc = len(CLASSES)

    logger.info("Computing shift spectrum...")
    ss = compute_ss(ZOOLAKE, CLASSES)

    eval_tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()])

    ood_loaders = {}
    for od in sorted(Path(OOD).iterdir()):
        if od.is_dir():
            ds = DS(str(od), c2i, eval_tf)
            if len(ds) > 0:
                ood_loaders[od.name] = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    train_ds = DS(ZOOLAKE, c2i, SAAT(ss))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    logger.info("Data: %d train, %d OOD cells", len(train_ds), len(ood_loaders))

    # Load baseline
    baseline = json.load(open(f"{RESULTS}/baseline_results.json"))

    all_results = {}

    for mi in [1, 2, 3]:
        logger.info("=" * 60)
        logger.info("Fine-tuning model 0%d with SAA", mi)
        logger.info("=" * 60)

        ckpt = torch.load(f"{MODELS}/0{mi}/trained_model_tuned.pth", map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        for k in list(sd.keys()):
            if "relative_position_index" in k:
                del sd[k]

        model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=False, num_classes=nc)
        model.load_state_dict(sd, strict=True)
        model = model.to(dev)
        logger.info("  Loaded model 0%d (epoch=%d)", mi, ckpt.get("epoch", 0))

        before_acc = baseline["per_model"][f"model_0{mi}"]["accuracy"]
        before_days = baseline["per_model"][f"model_0{mi}"]["per_day"]
        logger.info("  BEFORE: %.1f%%", before_acc * 100)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            model.train()
            tl, cr, tt = 0, 0, 0
            for imgs, lbls in train_loader:
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
            logger.info("    Epoch %d/%d  Loss:%.4f  Train:%.1f%%", epoch + 1, EPOCHS, tl / tt, cr / tt * 100)

        after_acc, after_days = eval_ood(model, ood_loaders, dev)
        logger.info("  AFTER: %.1f%%  Improvement: %+.1f%%", after_acc * 100, (after_acc - before_acc) * 100)
        for day in sorted(after_days.keys()):
            b = before_days.get(day, 0)
            a = after_days[day]
            logger.info("    %s: %.1f%% -> %.1f%% (%+.1f%%)", day, b * 100, a * 100, (a - b) * 100)

        all_results[f"model_0{mi}"] = {
            "before": {"overall": before_acc, "per_day": before_days},
            "after": {"overall": after_acc, "per_day": after_days},
            "improvement": after_acc - before_acc,
        }

        torch.save(model.state_dict(), f"{RESULTS}/model_0{mi}_finetuned_v4.pth")
        logger.info("  Saved")
        del model
        torch.cuda.empty_cache()

    # Ensemble
    logger.info("=" * 60)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 60)

    all_days = sorted(set().union(*[set(r["after"]["per_day"].keys()) for r in all_results.values()]))
    before_geom = baseline["ensemble_geometric"]["overall"]

    ft_arith = {}
    ft_geom = {}
    for day in all_days:
        vals = [r["after"]["per_day"].get(day, 0) for r in all_results.values()]
        ft_arith[day] = float(np.mean(vals))
        ft_geom[day] = float(np.exp(np.mean([np.log(v + 1e-8) for v in vals])))

    ft_a = np.mean(list(ft_arith.values()))
    ft_g = np.mean(list(ft_geom.values()))

    logger.info("  Baseline geometric: %.1f%%", before_geom * 100)
    logger.info("  Fine-tuned arithmetic: %.1f%%  (%+.1f%%)", ft_a * 100, (ft_a - before_geom) * 100)
    logger.info("  Fine-tuned geometric:  %.1f%%  (%+.1f%%)", ft_g * 100, (ft_g - before_geom) * 100)
    logger.info("  Chen BEsT: 83.0%%")
    logger.info("  vs Chen: %+.1f%%", ft_g * 100 - 83)

    final = {
        "baseline_geometric": before_geom,
        "finetune_per_model": all_results,
        "finetune_ensemble_arithmetic": {"overall": ft_a, "per_day": ft_arith},
        "finetune_ensemble_geometric": {"overall": ft_g, "per_day": ft_geom},
        "vs_chen": ft_g - 0.83,
    }
    with open(f"{RESULTS}/finetune_results_v4.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    logger.info("Results saved. DONE.")


if __name__ == "__main__":
    main()
