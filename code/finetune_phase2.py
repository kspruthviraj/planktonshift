"""
finetune_phase2.py
==================
Phase 2: Continue fine-tuning each model with lower LR (1e-6) for 5 more epochs.
Also saves per-sample softmax probabilities for flexible ensembling.

Usage:
    python finetune_phase2.py
"""
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
PHASE2_EPOCHS = 5
PHASE2_LR = 1e-6
WEIGHT_DECAY = 0.03
BATCH = 256  # Larger batch size
RESULTS = "/home/sreenath/research-space/PlanktonShift/results/finetune_chen_saa"
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
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
def eval_ood(model, ood_loaders, dev, save_probs=False):
    """Evaluate on OOD data. Optionally save per-sample probabilities."""
    model.eval()
    per_day = {}
    tc, tn = 0, 0
    all_probs = {}  # day -> numpy array of probabilities
    all_labels = {}  # day -> numpy array of labels

    for day, loader in ood_loaders.items():
        cr, n = 0, 0
        day_probs = []
        day_labels = []
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            probs = rot_tta(model, imgs, dev)
            day_probs.append(probs.cpu().numpy())
            day_labels.append(lbls.cpu().numpy())
            _, p = probs.max(1)
            cr += p.eq(lbls).sum().item()
            n += lbls.size(0)
        per_day[day] = cr / max(n, 1)
        tc += cr
        tn += n
        if save_probs:
            all_probs[day] = np.concatenate(day_probs)
            all_labels[day] = np.concatenate(day_labels)

    return tc / max(tn, 1), per_day, all_probs, all_labels


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

    # Load phase 1 results
    phase1 = json.load(open(f"{RESULTS}/finetune_results_v4.json"))

    all_results = {}
    all_model_probs = {}  # model_name -> {day -> probs}

    for mi in [1, 2, 3]:
        # Load phase 1 fine-tuned model
        model_path = f"{RESULTS}/model_0{mi}_finetuned_v4.pth"
        if not os.path.exists(model_path):
            logger.warning("Phase 1 model not found: %s", model_path)
            continue

        logger.info("=" * 60)
        logger.info("Phase 2: Model 0%d (lr=%.0e, %d epochs)", mi, PHASE2_LR, PHASE2_EPOCHS)
        logger.info("=" * 60)

        model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=False, num_classes=nc)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(dev)

        # Evaluate before phase 2
        before_acc, before_days, _, _ = eval_ood(model, ood_loaders, dev)
        logger.info("  BEFORE phase 2: %.1f%%", before_acc * 100)

        # Phase 2: fine-tune with lower LR
        optimizer = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better calibration

        for epoch in range(PHASE2_EPOCHS):
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
            logger.info("    Epoch %d/%d  Loss:%.4f  Train:%.1f%%", epoch + 1, PHASE2_EPOCHS, tl / tt, cr / tt * 100)

        # Evaluate after phase 2
        after_acc, after_days, probs, labels = eval_ood(model, ood_loaders, dev, save_probs=True)
        logger.info("  AFTER phase 2: %.1f%%  Improvement: %+.1f%%", after_acc * 100, (after_acc - before_acc) * 100)
        for day in sorted(after_days.keys()):
            b = before_days.get(day, 0)
            a = after_days[day]
            logger.info("    %s: %.1f%% -> %.1f%% (%+.1f%%)", day, b * 100, a * 100, (a - b) * 100)

        all_results[f"model_0{mi}"] = {
            "before_phase2": {"overall": before_acc, "per_day": before_days},
            "after_phase2": {"overall": after_acc, "per_day": after_days},
            "improvement": after_acc - before_acc,
        }

        # Save per-sample probabilities
        all_model_probs[f"model_0{mi}"] = {day: probs[day].tolist() for day in probs}
        np.savez(f"{RESULTS}/model_0{mi}_phase2_probs.npz", **{day: probs[day] for day in probs})
        logger.info("  Saved probabilities")

        torch.save(model.state_dict(), f"{RESULTS}/model_0{mi}_phase2.pth")
        logger.info("  Saved model")
        del model
        torch.cuda.empty_cache()

    # Ensemble from saved probabilities
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ENSEMBLE (all 3 models, phase 2)")
    logger.info("=" * 60)

    # Load all probabilities
    model_probs = {}
    for mi in [1, 2, 3]:
        prob_path = f"{RESULTS}/model_0{mi}_phase2_probs.npz"
        if os.path.exists(prob_path):
            data = np.load(prob_path)
            model_probs[f"model_0{mi}"] = {day: data[day] for day in data.files}

    if len(model_probs) >= 2:
        all_days = sorted(model_probs[list(model_probs.keys())[0]].keys())

        # Arithmetic ensemble
        arith_per_day = {}
        for day in all_days:
            probs = [model_probs[m][day] for m in model_probs]
            mean_probs = np.mean(probs, axis=0)
            preds = mean_probs.argmax(axis=1)
            # Get labels
            ds = DS(str(Path(OOD) / day), c2i, eval_tf)
            labels = np.array([s[1] for s in ds.s[: len(preds)]])
            arith_per_day[day] = float((preds == labels).mean())

        # Geometric ensemble
        geom_per_day = {}
        for day in all_days:
            probs = [model_probs[m][day] for m in model_probs]
            log_probs = [np.log(p + 1e-8) for p in probs]
            mean_log = np.mean(log_probs, axis=0)
            geom = np.exp(mean_log)
            geom = geom / geom.sum(axis=1, keepdims=True)
            preds = geom.argmax(axis=1)
            ds = DS(str(Path(OOD) / day), c2i, eval_tf)
            labels = np.array([s[1] for s in ds.s[: len(preds)]])
            geom_per_day[day] = float((preds == labels).mean())

        arith_overall = np.mean(list(arith_per_day.values()))
        geom_overall = np.mean(list(geom_per_day.values()))

        logger.info("  Arithmetic ensemble: %.1f%%", arith_overall * 100)
        logger.info("  Geometric ensemble:  %.1f%%", geom_overall * 100)
        logger.info("  Chen BEsT: 83.0%%")
        logger.info("  vs Chen: %+.1f%%", geom_overall * 100 - 83)
        logger.info("")
        logger.info("  Per-day (geometric):")
        for day in sorted(geom_per_day.keys()):
            logger.info("    %s: %.1f%%", day, geom_per_day[day] * 100)

        all_results["ensemble_arithmetic"] = {"overall": arith_overall, "per_day": arith_per_day}
        all_results["ensemble_geometric"] = {"overall": geom_overall, "per_day": geom_per_day}

    with open(f"{RESULTS}/phase2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Results saved. DONE.")


if __name__ == "__main__":
    main()
