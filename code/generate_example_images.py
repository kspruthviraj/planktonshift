"""Generate example images for the draft paper."""
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "adverserial_net"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_augmentation import SpectralAugmentation
import json

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Find a sample plankton image
sample_dir = ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0" / "ceratium"
sample_imgs = sorted(sample_dir.glob("*.jpeg"))
if not sample_imgs:
    sample_imgs = sorted(sample_dir.glob("*.jpg"))
img_path = sample_imgs[0]
print(f"Using: {img_path.name}")


def resize_with_proportions(im, desired_size=128):
    old_size = im.size
    if max(old_size) > desired_size:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (desired_size, desired_size), color=0)
    offset = ((desired_size - im.size[0]) // 2, (desired_size - im.size[1]) // 2)
    new_im.paste(im, offset)
    return new_im


def bandpass_filter(image_np, band):
    gray = np.mean(image_np, axis=2) if len(image_np.shape) == 3 else image_np
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    r_max = min(crow, ccol)

    mask = np.zeros((rows, cols), dtype=np.float32)
    if band == 'low':
        r_inner, r_outer = 0, int(r_max * 0.25)
    elif band == 'mid':
        r_inner, r_outer = int(r_max * 0.25), int(r_max * 0.75)
    elif band == 'high':
        r_inner, r_outer = int(r_max * 0.75), r_max
    else:
        r_inner, r_outer = 0, r_max

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if r_inner <= dist <= r_outer:
                mask[i, j] = 1.0

    Fshift_filtered = Fshift * mask
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    img_back = np.fft.ifft2(F_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
    return img_back


# ── 1. Frequency Masking Example ──
print("Generating frequency masking example...")
im = Image.open(img_path).convert("RGB")
im = resize_with_proportions(im, 224)
arr = np.array(im, dtype=np.float32) / 255.0

low = bandpass_filter(arr, 'low')
mid = bandpass_filter(arr, 'mid')
high = bandpass_filter(arr, 'high')

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(arr)
axes[0].set_title('Original Image\n(All frequencies)', fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(low, cmap='gray')
axes[1].set_title('Low Frequencies Only\n(Body shape, outline)', fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(mid, cmap='gray')
axes[2].set_title('Mid Frequencies Only\n(Lighting, contrast)', fontsize=11, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(high, cmap='gray')
axes[3].set_title('High Frequencies Only\n(Noise, fine texture)', fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.suptitle('Frequency Masking: What Each Band Contains', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_frequency_masking_example.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIGURES_DIR / 'fig_frequency_masking_example.png'}")


# ── 2. SBA Augmentation Example ──
print("Generating SBA augmentation example...")
shift_path = ROOT / "results" / "adverserial_net" / "fourier_analysis" / "cross_domain" / "fourier_analysis.json"
shift_spectrum = None
if shift_path.exists():
    with open(shift_path) as f:
        fa = json.load(f)
    for key, val in fa.get("shift_spectra", {}).items():
        if "ZooScan" in key and "WHOI" in key:
            shift_spectrum = np.array(val.get("diff", []))
            break

sba = SpectralAugmentation(
    shift_spectrum=shift_spectrum,
    strength=0.5,
    strategies=["spectral_noise", "band_adversarial"],
    p=1.0,
)

# Apply SBA to the image
gray = np.array(im.convert("L"), dtype=np.float64) / 255.0
gray_aug_spectral = sba._spectral_noise(gray.copy())
gray_aug_band = sba._band_adversarial(gray.copy())

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original Image\n(Grayscale)', fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(gray_aug_spectral, cmap='gray')
axes[1].set_title('After SBA Spectral Noise\n(Noise in mid-frequency bands)', fontsize=11, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(np.abs(gray_aug_spectral - gray), cmap='hot')
axes[2].set_title('Difference (what SBA changed)\n(Mid-frequency bands altered)', fontsize=11, fontweight='bold')
axes[2].axis('off')

plt.suptitle('SBA Augmentation: Adding Calibrated Noise to Camera-Specific Frequencies', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_sba_augmentation_example.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIGURES_DIR / 'fig_sba_augmentation_example.png'}")


# ── 3. CenterCrop vs Proportional Padding ──
print("Generating CenterCrop vs proportional padding example...")
im_orig = Image.open(img_path).convert("RGB")

# Proportional padding (Chen's method)
im_prop = resize_with_proportions(im_orig, 128)
im_prop = im_prop.resize((224, 224), Image.BILINEAR)

# CenterCrop (our method)
from torchvision import transforms
cc_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])
im_cc = cc_transform(im_orig)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(im_prop)
axes[0].set_title('Proportional Padding (Chen)\nFull organism visible, black borders', fontsize=10, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(im_cc)
axes[1].set_title('CenterCrop (Our method)\nMay clip organisms at edges', fontsize=10, fontweight='bold')
axes[1].axis('off')

plt.suptitle('Two Preprocessing Methods', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_preprocessing_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIGURES_DIR / 'fig_preprocessing_comparison.png'}")

print("\nAll example images generated.")
