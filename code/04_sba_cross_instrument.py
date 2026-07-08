"""
04_sba_cross_instrument.py — Does frequency-calibrated augmentation help classifiers?

STEP 4: SPECTRAL BAND ADVERSARIAL (SBA) AUGMENTATION
=====================================================

Now that we know which frequency bands carry camera artifacts (Step 01) and
which carry species info (Step 03), we can build a smarter augmentation:
during training, we add noise ONLY to the camera-specific frequency bands,
teaching the model to ignore camera differences while preserving species
recognition.

MATHEMATICAL PIPELINE:
======================

Input: grayscale image g(x, y) (after Pipeline A preprocessing).

Step 1 — Compute FFT:
    F(u, v) = FFT2{ g }
    A(u, v) = |F(u, v)|           (amplitude)
    phi(u, v) = angle(F(u, v))    (phase)

Step 2 — Load shift spectrum from Step 01:
    Delta_A(r) = A_bar_source(r) - A_bar_target(r)

    This tells us WHICH frequency bins differ most between cameras.

Step 3 — SBA spectral noise augmentation:
    For each frequency bin (u, v) at radial distance r:
        noise(u, v) = alpha * Delta_A(r) * N(0, 1)
        A_aug(u, v) = A(u, v) + noise(u, v)

    where alpha = strength parameter (default 0.5), and N(0,1) is
    standard Gaussian noise. The noise is PROPORTIONAL to the observed
    shift — bins where cameras differ more get more noise.

Step 4 — SBA band adversarial augmentation:
    Target only mid-frequency bins (where camera artifacts concentrate):
        M_mid(u, v) = 1  if  0.20*r_max <= r <= 0.40*r_max
                      0  otherwise
        A_aug(u, v) = A(u, v) + alpha * M_mid(u, v) * |N(0,1)|

    Only mid-frequency amplitudes are perturbed; low (species) and high
    (noise) frequencies are left untouched.

Step 5 — Reconstruct augmented image:
    F_aug(u, v) = A_aug(u, v) * exp(i * phi(u, v))
    g_aug = real( IFFT2{ F_aug } )

    Phase is PRESERVED — the biological shape (in phase) is unchanged.
    Only amplitude (camera artifacts) is perturbed.

Step 6 — Phase-preserving SBA (control):
    Same as Step 3 but without shift calibration:
        A_aug(u, v) = A(u, v) + alpha * 0.1 * N(0, 1)
    Applied to ALL frequency bins uniformly.

Step 7 — Train ViT-B/16 on augmented source, evaluate on unmodified target.
    Repeat with 5 random seeds for honest confidence intervals.
    Compare using McNemar's paired test.

WHY GRAYSCALE FOR CROSS-INSTRUMENT, BUT RGB FOR TEMPORAL OOD?
  This script (cross-instrument benchmark) uses grayscale SBA for
  consistency with Steps 02-03 and because the 6-class IFCB/ZooScan
  benchmark has limited data (384 images).

  The headline 83.19% result (temporal OOD, ZooLake) uses PER-CHANNEL
  RGB SBA: each R, G, B channel gets independent frequency-domain noise.
  This gives +1.3% over grayscale SBA (81.87%) because colour information
  also carries domain cues (e.g., ZooLake field camera has different colour
  balance than IFCB).  The per-channel pipeline is in
  `code/perchannel_sba_finetune.py` (legacy).

WHY IT MATTERS:
  SBA exploits the frequency-domain separation found in Steps 01-03:
  perturb the bands that carry camera artifacts (amplitude), preserve
  the bands that carry species info (phase). If this works, it validates
  the entire frequency-domain framework.

Output: results/tier1_corrected/sba_cross_instrument.json
"""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "adverserial_net"))
from config import DATA, RESULTS, SEEDS, FREQ_BAND_FRACTIONS
from utils_pipeline import preprocess_image, bootstrap_ci, mcnemar_test, discover_classes, SUPPORTED_EXT
from spectral_augmentation import SpectralAugmentation

OUT = RESULTS / "tier1_corrected" / "sba_cross_instrument.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossInstrumentDataset(Dataset):
    def __init__(self, root, classes_dict, sba=None, augment=False):
        self.sba = sba
        self.augment = augment
        self.samples = []
        for cn, idx in classes_dict.items():
            cd = Path(root) / cn
            if not cd.is_dir():
                continue
            for p in sorted(cd.iterdir()):
                if p.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(p), idx))
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        im = Image.open(path).convert("RGB")
        arr = preprocess_image(im)  # Pipeline A
        if self.augment:
            if self.sba:
                gray = np.mean(arr, axis=2)
                gray_aug = self.sba(gray)
                arr = np.stack([gray_aug] * 3, axis=2)
            pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            arr = np.array(self.aug_tf(pil), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


class PhasePreserveSBA:
    """Phase-preserving perturbation: perturb amplitude, keep phase.
    This is the controlled version that the original paper found 'hurts'."""
    def __init__(self, strength=0.5, p=0.8):
        self.strength = strength
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        f = np.fft.fft2(image)
        amp = np.abs(f)
        phase = np.angle(f)
        perturbation = np.random.randn(*image.shape) * self.strength * 0.1
        amp_new = np.maximum(amp + perturbation, 0)
        f_new = amp_new * np.exp(1j * phase)
        return np.real(np.fft.ifft2(f_new)).clip(0, 1)


def load_shift_spectrum():
    """Load the IFCB->ZooScan shift spectrum from the corrected Fourier analysis."""
    path = RESULTS / "tier1_corrected" / "fourier_analysis.json"
    if path.exists():
        with open(path) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "WHOI" in key and "ZooScan" in key:
                return np.array(val.get("diff", []))
    # Fallback to original
    path2 = RESULTS / "adverserial_net" / "fourier_analysis_zoolake2" / "fourier_analysis.json"
    if path2.exists():
        with open(path2) as f:
            fa = json.load(f)
        for key, val in fa.get("shift_spectra", {}).items():
            if "WHOI" in key and "ZooScan" in key:
                return np.array(val.get("diff", []))
    return None


def train_and_eval(train_ds, test_ds, num_classes, seed, epochs=30, lr=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in tqdm(train_loader, desc=f"    seed{seed} ep{ep+1}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
    # Eval
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            preds.append(p)
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    acc = float((preds == labels).mean())
    del model
    torch.cuda.empty_cache()
    return acc, preds, labels


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = DATA["cross_ifcb"]
    target = DATA["cross_zooscan"]
    print(f"Device: {DEVICE}")
    print(f"Source: {source}\nTarget: {target}")
    print(f"Pipeline: Proportional Padding (A)")
    print(f"Seeds: {SEEDS['sba_cross_instrument']}")

    classes_dict, names = discover_classes(source, target)
    print(f"Common classes ({len(names)}): {names}")

    shift_spectrum = load_shift_spectrum()
    print(f"Shift spectrum: {len(shift_spectrum) if shift_spectrum is not None else 0} bins")

    sba = SpectralAugmentation(
        shift_spectrum=shift_spectrum, strength=0.5,
        strategies=["spectral_noise", "band_adversarial"], p=0.8,
    ) if shift_spectrum is not None else None

    phase_sba = PhasePreserveSBA(strength=0.5, p=0.8)

    configs = {
        "standard": {"sba": None},
        "sba_band": {"sba": sba},
        "phase_preserve": {"sba": phase_sba},
    }

    results = {"_meta": {
        "source": str(source), "target": str(target),
        "preprocessing": "ProportionalPadding(128)->Resize(224) [Pipeline A]",
        "seeds": SEEDS["sba_cross_instrument"],
        "n_classes": len(names),
    }}

    for aug_name, cfg in configs.items():
        print(f"\n{'='*60}\nAugmentation: {aug_name}\n{'='*60}")
        seed_results = []
        all_preds_by_seed = {}

        for seed in SEEDS["sba_cross_instrument"]:
            print(f"\n  Seed {seed}:")
            train_ds = CrossInstrumentDataset(source, classes_dict, sba=cfg["sba"], augment=True)
            test_ds = CrossInstrumentDataset(target, classes_dict, augment=False)
            acc, preds, labels = train_and_eval(train_ds, test_ds, len(names), seed, epochs=30)
            mean_ci, lo, hi = bootstrap_ci((preds == labels).astype(float))
            seed_results.append({"seed": seed, "accuracy": acc, "ci_95": [lo, hi]})
            all_preds_by_seed[seed] = preds.tolist()
            print(f"    acc={acc:.4f} CI[{lo:.4f},{hi:.4f}]")

        accs = [r["accuracy"] for r in seed_results]
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        # CI across seeds
        seed_mean, seed_lo, seed_hi = bootstrap_ci(accs, n_boot=5000)

        results[aug_name] = {
            "per_seed": seed_results,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "seed_ci_95": [seed_lo, seed_hi],
            "best_seed": max(accs),
            "worst_seed": min(accs),
            "n_seeds": len(accs),
            "labels": labels.tolist(),
            "preds_by_seed": all_preds_by_seed,
        }
        print(f"\n  {aug_name}: mean={mean_acc:.4f}+/-{std_acc:.4f} CI[{seed_lo:.4f},{seed_hi:.4f}]")

    # McNemar: best SBA seed vs best baseline seed (paired on same test set)
    if "standard" in results and "sba_band" in results:
        base_preds = np.array(results["standard"]["preds_by_seed"][42])
        sba_preds = np.array(results["sba_band"]["preds_by_seed"][42])
        labels = np.array(results["standard"]["labels"])
        stat, pval, (n01, n10) = mcnemar_test(base_preds, sba_preds, labels)
        results["mcnemar_standard_vs_sba_seed42"] = {
            "statistic": stat, "p_value": pval,
            "discordant": {"a_right_b_wrong": n01, "a_wrong_b_right": n10},
        }
        print(f"\nMcNemar (standard vs SBA band, seed 42): stat={stat:.3f} p={pval:.4f} discordant=({n01},{n10})")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
