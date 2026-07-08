"""
02_amplitude_vs_phase.py -- Does the shape of a plankton live in the "bass" or the "treble"?

STEP 2: CONTROLLED AMPLITUDE-VS-PHASE EXPERIMENT
=================================================

Every image can be decomposed into two components in frequency space:
  - AMPLITUDE: how strong each frequency is (like the volume of each note)
  - PHASE: where each frequency appears in the image (like the timing of each note)

This script tests which component carries the species-discriminative information:
  (a) Phase-scrambled: keep amplitude, randomise phase -> if species info is in
      phase, accuracy should drop
  (b) Amplitude-swapped: swap amplitude with a target-domain image, keep phase
      -> if amplitude carries camera artifacts, accuracy should improve
  (c) Both-scrambled: randomise both -> negative control

WHY IT MATTERS:
  If phase carries the biological shape (body outline, spines, horns), then
  augmentation methods should perturb amplitude (camera artifacts) while
  preserving phase (biology). This is exactly what SBA does.

Output: results/tier1_corrected/amplitude_vs_phase.json
"""
import sys, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RESULTS, SEEDS, FREQ_BAND_FRACTIONS
from utils_pipeline import preprocess_image, bootstrap_ci, discover_classes, SUPPORTED_EXT

OUT = RESULTS / "tier1_corrected" / "amplitude_vs_phase.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scramble_phase(arr01):
    """Replace phase with random phase, keep amplitude."""
    gray = np.mean(arr01, axis=2)
    f = np.fft.fft2(gray)
    amp = np.abs(f)
    random_phase = np.random.uniform(-np.pi, np.pi, gray.shape)
    f_new = amp * np.exp(1j * random_phase)
    img_back = np.real(np.fft.ifft2(f_new))
    img_back = np.clip((img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8), 0, 1)
    return np.stack([img_back] * 3, axis=2).astype(np.float32)


def swap_amplitude(source_arr, target_arr):
    """Swap amplitude between source and target, keep phase of source."""
    gray_s = np.mean(source_arr, axis=2)
    gray_t = np.mean(target_arr, axis=2)
    f_s = np.fft.fft2(gray_s)
    f_t = np.fft.fft2(gray_t)
    amp_t = np.abs(f_t)
    phase_s = np.angle(f_s)
    f_new = amp_t * np.exp(1j * phase_s)
    img_back = np.real(np.fft.ifft2(f_new))
    img_back = np.clip(img_back, 0, 1)
    return np.stack([img_back] * 3, axis=2).astype(np.float32)


def scramble_both(arr01):
    """Scramble both amplitude and phase (negative control)."""
    gray = np.mean(arr01, axis=2)
    random_phase = np.random.uniform(-np.pi, np.pi, gray.shape)
    random_amp = np.abs(np.fft.fft2(np.random.randn(*gray.shape)))
    f_new = random_amp * np.exp(1j * random_phase)
    img_back = np.real(np.fft.ifft2(f_new))
    img_back = np.clip((img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8), 0, 1)
    return np.stack([img_back] * 3, axis=2).astype(np.float32)


class AmpPhaseDataset(Dataset):
    """Dataset with amplitude/phase manipulation.
    condition: 'normal', 'phase_scrambled', 'amp_swapped', 'both_scrambled'
    For amp_swapped, target_pool provides amplitude donors.
    """
    def __init__(self, root, classes_dict, target_root=None, condition="normal", augment=False):
        self.condition = condition
        self.augment = augment
        self.target_root = target_root
        self.samples = []
        for cn, idx in classes_dict.items():
            cd = Path(root) / cn
            if not cd.is_dir():
                continue
            for p in sorted(cd_dir_iter(cd)):
                self.samples.append((str(p), idx))
        # Preload target images for amplitude swap
        self.target_images = []
        if condition == "amp_swapped" and target_root:
            self._load_target_images(target_root, classes_dict)
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ])

    def _load_target_images(self, root, classes_dict, max_total=100):
        for cn in classes_dict:
            cd = Path(root) / cn
            if not cd.is_dir():
                continue
            for p in sorted(cd_dir_iter(cd)):
                if len(self.target_images) >= max_total:
                    return
                try:
                    im = Image.open(p).convert("RGB")
                    self.target_images.append(preprocess_image(im))
                except Exception:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        im = Image.open(path).convert("RGB")
        arr = preprocess_image(im)
        if self.condition == "phase_scrambled":
            arr = scramble_phase(arr)
        elif self.condition == "amp_swapped" and self.target_images:
            donor = self.target_images[np.random.randint(len(self.target_images))]
            arr = swap_amplitude(arr, donor)
        elif self.condition == "both_scrambled":
            arr = scramble_both(arr)
        if self.augment:
            pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            arr = np.array(self.aug_tf(pil), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


def cd_dir_iter(cd):
    """Iterate image files in a class directory."""
    for p in sorted(cd.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXT:
            yield p


def train_and_eval(train_ds, test_ds, num_classes, seed, epochs=15, lr=1e-4):
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
        for x, y in tqdm(train_loader, desc=f"      ep{ep+1}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            preds.append(p)
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    acc = float((preds == labels).mean())
    mean, lo, hi = bootstrap_ci((preds == labels).astype(float))
    del model
    torch.cuda.empty_cache()
    return acc, lo, hi, len(labels)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = DATA["cross_ifcb"]
    target = DATA["cross_zooscan"]
    print(f"Device: {DEVICE}")
    print(f"Amplitude vs Phase controlled experiment (Pipeline A)")
    classes_dict, names = discover_classes(source, target)
    print(f"Common classes ({len(names)}): {names}")

    seed = SEEDS["ablation"]
    conditions = ["normal", "phase_scrambled", "amp_swapped", "both_scrambled"]
    results = {"_meta": {
        "source": str(source), "target": str(target),
        "preprocessing": "ProportionalPadding(128)->Resize(224) [Pipeline A]",
        "seed": seed, "n_classes": len(names),
        "conditions": conditions,
    }}

    for cond in conditions:
        print(f"\n{'='*60}\nCondition: {cond}\n{'='*60}")
        train_ds = AmpPhaseDataset(source, classes_dict, target_root=target,
                                    condition=cond, augment=True)
        test_ds = AmpPhaseDataset(target, classes_dict, condition="normal", augment=False)
        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"  skip {cond}: empty")
            continue
        acc, lo, hi, n = train_and_eval(train_ds, test_ds, len(names), seed)
        results[cond] = {"species_accuracy": acc, "ci_95": [lo, hi], "n_test": n}
        print(f"  Species acc: {acc:.4f} CI[{lo:.4f},{hi:.4f}] n={n}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nAMPLITUDE vs PHASE RESULTS\n{'='*60}")
    print(f"{'Condition':<18} {'Species Acc':>12} {'95% CI':>18}")
    for c in conditions:
        if c in results:
            r = results[c]
            ci = f"[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]"
            print(f"{c:<18} {r['species_accuracy']:>12.3f} {ci:>18}")
    print(f"\nInterpretation:")
    print(f"  If phase_scrambled preserves species acc -> amplitude carries morphology")
    print(f"  If amp_swapped hurts species acc -> amplitude carries diagnostic info")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
