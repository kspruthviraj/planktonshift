"""
spectral_augmentation.py
========================
Spectral Band Adversarial augmentation (SBA) for cross-domain plankton classification.

UPDATED: Incorporates research findings on what actually works:
1. FDA-style low-frequency amplitude swap (Yang & Soatto 2020)
2. Spectral noise proportional to observed cross-domain shift
3. Band-selective adversarial perturbation
REMOVED: phase_preserve (hurts performance — destroys morphological signal)

Usage:
    from spectral_augmentation import SpectralAugmentation, FDAAugmentation
"""

import numpy as np
from PIL import Image
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class SpectralAugmentation:
    """Frequency-domain augmentation for domain-robust training.
    
    Strategies:
    - amplitude_mix: Mix amplitude spectrum with synthetic target-domain amplitudes
    - spectral_noise: Add noise proportional to cross-domain shift spectrum
    - band_adversarial: Target domain-discriminative frequency bands
    - fda_swap: FDA-style low-frequency amplitude swap with target images
    """

    STRATEGIES = ["amplitude_mix", "spectral_noise", "band_adversarial", "fda_swap"]

    def __init__(
        self,
        shift_spectrum: Optional[np.ndarray] = None,
        target_images: Optional[List[np.ndarray]] = None,
        strength: float = 0.5,
        strategies: Optional[list] = None,
        p: float = 0.5,
    ):
        self.shift_spectrum = shift_spectrum
        self.target_images = target_images  # For FDA-style swap
        self.strength = strength
        self.strategies = strategies or ["spectral_noise", "band_adversarial"]
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image
        strategy = np.random.choice(self.strategies)
        if strategy == "amplitude_mix":
            return self._amplitude_mix(image)
        elif strategy == "spectral_noise":
            return self._spectral_noise(image)
        elif strategy == "band_adversarial":
            return self._band_adversarial(image)
        elif strategy == "fda_swap":
            return self._fda_swap(image)
        return image

    def _to_freq(self, image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        return fshift, np.abs(fshift), np.angle(fshift)

    def _from_freq(self, amplitude, phase):
        fshift = amplitude * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(fshift)
        return np.real(np.fft.ifft2(f_ishift)).clip(0, 1)

    def _fda_swap(self, image: np.ndarray) -> np.ndarray:
        """FDA-style: swap low-frequency amplitude with a random target image.
        
        Following Yang & Soatto (CVPR 2020): decompose into amplitude and phase,
        swap the low-frequency amplitude components, reconstruct.
        """
        fft_src = np.fft.fft2(image)
        amp_src = np.abs(fft_src)
        pha_src = np.angle(fft_src)
        h, w = image.shape

        if self.target_images and len(self.target_images) > 0:
            target = self.target_images[np.random.randint(len(self.target_images))]
            amp_trg = np.abs(np.fft.fft2(target))
        else:
            amp_trg = self._generate_synthetic_target_amplitude(h, w)

        # beta controls the size of the low-frequency swap window
        beta = np.random.uniform(0.01, 0.1)
        h_shift = max(1, int(h * beta / 2))
        w_shift = max(1, int(w * beta / 2))

        # Swap low-frequency amplitude in all 4 corners of the spectrum
        # (fft2 output: low frequencies are at corners, not center)
        amp_new = amp_src.copy()
        amp_new[:h_shift, :w_shift] = amp_trg[:h_shift, :w_shift]
        amp_new[:h_shift, -w_shift:] = amp_trg[:h_shift, -w_shift:]
        amp_new[-h_shift:, :w_shift] = amp_trg[-h_shift:, :w_shift]
        amp_new[-h_shift:, -w_shift:] = amp_trg[-h_shift:, -w_shift:]

        # Reconstruct
        fft_modified = amp_new * np.exp(1j * pha_src)
        return np.real(np.fft.ifft2(fft_modified)).clip(0, 1)

    def _spectral_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise proportional to the cross-domain shift spectrum."""
        _, amplitude, phase = self._to_freq(image)
        h, w = amplitude.shape

        if self.shift_spectrum is not None:
            shift_2d = self._radial_to_2d(self.shift_spectrum, h, w)
            noise = np.random.randn(h, w) * shift_2d * self.strength * 0.1
        else:
            # Default: low-frequency noise
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            max_r = np.sqrt(cx ** 2 + cy ** 2)
            envelope = np.exp(-R / (max_r * 0.3))
            noise = np.random.randn(h, w) * envelope * self.strength * 0.05

        amplitude_noisy = np.maximum(amplitude + noise, 0)
        return self._from_freq(amplitude_noisy, phase)

    def _band_adversarial(self, image: np.ndarray) -> np.ndarray:
        """Target adversarial perturbation to mid-frequency bands."""
        _, amplitude, phase = self._to_freq(image)
        h, w = amplitude.shape

        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        normalized = R / max_r

        # Target mid-frequency bands (where domain artifacts live)
        mask = ((normalized >= 0.1) & (normalized <= 0.4)).astype(float)
        perturbation = np.random.randn(h, w) * mask * self.strength * 0.15

        amplitude_perturbed = np.maximum(amplitude + amplitude * perturbation, 0)
        return self._from_freq(amplitude_perturbed, phase)

    def _amplitude_mix(self, image: np.ndarray) -> np.ndarray:
        """Mix amplitude with synthetic target-domain perturbation."""
        _, amplitude, phase = self._to_freq(image)
        h, w = amplitude.shape

        if self.shift_spectrum is not None:
            perturbation = self._radial_to_2d(self.shift_spectrum, h, w) * np.random.randn(h, w)
        else:
            perturbation = self._random_low_freq_perturbation(h, w)

        alpha = np.random.uniform(0.1, self.strength)
        mixed = amplitude * (1 - alpha) + np.maximum(amplitude + perturbation, 0) * alpha
        return self._from_freq(mixed, phase)

    def _radial_to_2d(self, radial, h, w):
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        R_int = np.clip(R.astype(int), 0, len(radial) - 1)
        return radial[R_int]

    def _random_low_freq_perturbation(self, h, w):
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        envelope = (R / max_r < 0.3).astype(float)
        return np.random.randn(h, w) * envelope

    def _generate_synthetic_target_amplitude(self, h, w):
        """Generate synthetic target-domain amplitude from shift spectrum."""
        if self.shift_spectrum is not None:
            shift_2d = self._radial_to_2d(self.shift_spectrum, h, w)
            return np.abs(shift_2d) * (1 + 0.1 * np.random.randn(h, w))
        else:
            # Random low-frequency amplitude pattern
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            max_r = np.sqrt(cx ** 2 + cy ** 2)
            return np.exp(-R / (max_r * 0.2)) * np.random.uniform(0.5, 2.0)


class FDAAugmentation:
    """Fourier Domain Adaptation augmentation (Yang & Soatto 2020).
    
    Preloads target-domain images for efficient FDA-style augmentation
    during training. This is the most effective single augmentation
    for cross-domain robustness.
    """

    def __init__(self, target_image_paths: list, beta_range=(0.01, 0.05),
                 max_images: int = 50):
        self.beta_range = beta_range
        self.target_images = []
        for path in target_image_paths[:max_images]:
            try:
                img = Image.open(path).convert("L")
                img = img.resize((224, 224))
                self.target_images.append(np.array(img, dtype=np.float64) / 255.0)
            except Exception:
                continue
        logger.info("FDA: loaded %d target-domain images", len(self.target_images))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply FDA: swap low-frequency amplitude with random target image."""
        if not self.target_images:
            return image

        target = self.target_images[np.random.randint(len(self.target_images))]
        beta = np.random.uniform(*self.beta_range)

        # FFT
        fft_src = np.fft.fft2(image)
        fft_trg = np.fft.fft2(target)

        amp_src = np.abs(np.fft.fftshift(fft_src))
        pha_src = np.angle(np.fft.fftshift(fft_src))
        amp_trg = np.abs(np.fft.fftshift(fft_trg))

        h, w = image.shape
        cy, cx = h // 2, w // 2
        h_shift = max(1, int(h * beta / 2))
        w_shift = max(1, int(w * beta / 2))

        # Swap low-frequency amplitude
        amp_src[cy-h_shift:cy+h_shift, cx-w_shift:cx+w_shift] = \
            amp_trg[cy-h_shift:cy+h_shift, cx-w_shift:cx+w_shift]

        # Reconstruct
        fft_modified = np.fft.ifftshift(amp_src * np.exp(1j * pha_src))
        result = np.real(np.fft.ifft2(fft_modified)).clip(0, 1)
        return result


def load_shift_spectrum(path: str) -> Optional[np.ndarray]:
    """Load shift spectrum from fourier_shift_analysis.py output."""
    import json
    from pathlib import Path
    if not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    for pair, shift in data.get("shift_spectra", {}).items():
        return np.array(shift["diff"])
    if "diff" in data:
        return np.array(data["diff"])
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python spectral_augmentation.py <image_path>")
        sys.exit(1)

    img = Image.open(sys.argv[1]).convert("L").resize((224, 224))
    arr = np.array(img, dtype=np.float64) / 255.0

    ss = load_shift_spectrum("results/fourier_analysis/cross_domain/fourier_analysis.json")

    for strategy in ["spectral_noise", "band_adversarial", "fda_swap", "amplitude_mix"]:
        aug = SpectralAugmentation(
            shift_spectrum=ss, strength=0.5,
            strategies=[strategy], p=1.0
        )
        result = aug(arr)
        Image.fromarray((result * 255).astype(np.uint8)).save(f"augmented_{strategy}.png")
        print(f"Saved augmented_{strategy}.png  residual={np.abs(result - arr).mean():.6f}")
