"""phase2_simple.py — Phase 2: Continue fine-tuning with lower LR."""
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

IMG_SIZE = 224
EPOCHS = 5
LR = 1e-6
BATCH = 128
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
SUPPORTED_EXT = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}


class DS(Dataset):
    def __init__(self, d, c2i, tf):
        self.s, self.tf = [], tf
        for c, i in c2i.items():
            cd = Path(d) / c
            if not cd.is_dir(): continue
            for ip in sorted(cd.iterdir()):
                if ip.suffix.lower() in SUPPORTED_EXT:
                    self.s.append((str(ip), i))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l = self.s[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), l


eval_tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()])


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
    per_day, tc, tn = {}, 0, 0
    for day, loader in ood_loaders.items():
        cr, n = 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            probs = rot_tta(model, imgs, dev)
            _, p = probs.max(1)
            cr += p.eq(lbls).sum().item()
            n += lbls.size(0)
        per_day[day] = cr / max(n, 1)
        tc += cr; tn += n
    return tc / max(tn, 1), per_day


@torch.no_grad()
def collect_probs(model, ood_loaders, dev):
    """Collect per-sample softmax probabilities."""
    model.eval()
    all_probs = {}
    for day, loader in ood_loaders.items():
        day_probs = []
        for imgs, _ in loader:
            imgs = imgs.to(dev)
            probs = rot_tta(model, imgs, dev)
            day_probs.append(probs.cpu().numpy())
        all_probs[day] = np.concatenate(day_probs)
    return all_probs


def main():
    os.makedirs(RESULTS, exist_ok=True)
    dev = torch.device("cuda")
    c2i = {c: i for i, c in enumerate(CLASSES)}
    nc = len(CLASSES)

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])

    ood_loaders = {}
    for od in sorted(Path(OOD).iterdir()):
        if od.is_dir():
            ds = DS(str(od), c2i, eval_tf)
            if len(ds) > 0:
                ood_loaders[od.name] = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    train_ds = DS(ZOOLAKE, c2i, train_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    logger.info("Data: %d train, %d OOD cells", len(train_ds), len(ood_loaders))

    # Load phase 1 results for BEFORE scores
    phase1 = json.load(open(f"{RESULTS}/finetune_results_v4.json"))

    all_results = {}

    for mi in [1, 2, 3]:
        model_path = f"{RESULTS}/model_0{mi}_finetuned_v4.pth"
        if not os.path.exists(model_path):
            continue

        logger.info("=" * 60)
        logger.info("Phase 2: Model 0%d (lr=%.0e, %d epochs, batch=%d)", mi, LR, EPOCHS, BATCH)
        logger.info("=" * 60)

        model = timm.create_model("beit_base_patch16_224.in22k_ft_in22k_in1k", pretrained=False, num_classes=nc)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(dev)

        before_acc = phase1["finetune_per_model"][f"model_0{mi}"]["after"]["overall"]
        before_days = phase1["finetune_per_model"][f"model_0{mi}"]["after"]["per_day"]
        logger.info("  BEFORE phase 2: %.1f%%", before_acc * 100)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.03)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        for epoch in range(EPOCHS):
            model.train(); tl, cr, tt = 0, 0, 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(dev), lbls.to(dev)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward(); optimizer.step()
                tl += loss.item() * imgs.size(0)
                _, p = out.max(1); cr += p.eq(lbls).sum().item(); tt += lbls.size(0)
            scheduler.step()
            logger.info("    Epoch %d/%d  Loss:%.4f  Train:%.1f%%", epoch + 1, EPOCHS, tl / tt, cr / tt * 100)

        after_acc, after_days = eval_ood(model, ood_loaders, dev)
        logger.info("  AFTER: %.1f%%  Improvement: %+.1f%%", after_acc * 100, (after_acc - before_acc) * 100)
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
        probs = collect_probs(model, ood_loaders, dev)
        np.savez(f"{RESULTS}/model_0{mi}_phase2_probs.npz", **probs)
        torch.save(model.state_dict(), f"{RESULTS}/model_0{mi}_phase2.pth")
        logger.info("  Saved model + probabilities")
        del model; torch.cuda.empty_cache()

    # Ensemble
    logger.info("=" * 60)
    logger.info("ENSEMBLE (phase 2)")
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

        arith_per_day, geom_per_day = {}, {}
        for day in all_days:
            probs = [model_probs[m][day] for m in model_probs]
            # Arithmetic
            a = np.mean(probs, axis=0).argmax(axis=1)
            # Geometric
            g = np.exp(np.mean([np.log(p + 1e-8) for p in probs], axis=0))
            g = g / g.sum(axis=1, keepdims=True)
            g_pred = g.argmax(axis=1)
            # Labels
            ds = DS(str(Path(OOD) / day), c2i, eval_tf)
            labels = np.array([s[1] for s in ds.s[: len(a)]])
            arith_per_day[day] = float((a == labels).mean())
            geom_per_day[day] = float((g_pred == labels).mean())

        arith_o = np.mean(list(arith_per_day.values()))
        geom_o = np.mean(list(geom_per_day.values()))

        logger.info("  Arithmetic: %.1f%%", arith_o * 100)
        logger.info("  Geometric:  %.1f%%", geom_o * 100)
        logger.info("  Chen BEsT: 83.0%%")
        logger.info("  vs Chen: %+.1f%%", geom_o * 100 - 83)
        for day in sorted(geom_per_day.keys()):
            logger.info("    %s: %.1f%%", day, geom_per_day[day] * 100)

        all_results["ensemble_arithmetic"] = {"overall": arith_o, "per_day": arith_per_day}
        all_results["ensemble_geometric"] = {"overall": geom_o, "per_day": geom_per_day}

    with open(f"{RESULTS}/phase2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("DONE. Results: %s/phase2_results.json", RESULTS)


if __name__ == "__main__":
    main()
