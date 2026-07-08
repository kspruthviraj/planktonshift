"""Generate graphical abstract for the paper."""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

# Colors
c_blue = '#2196F3'
c_red = '#F44336'
c_green = '#4CAF50'
c_orange = '#FF9800'
c_purple = '#9C27B0'
c_gray = '#607D8B'

# ── Box 1: Instrument Images ──
box1 = FancyBboxPatch((0.3, 1.5), 2.2, 2.5, boxstyle="round,pad=0.15",
                       facecolor='#E3F2FD', edgecolor=c_blue, linewidth=2)
ax.add_patch(box1)
ax.text(1.4, 3.6, 'Imaging Systems', ha='center', fontsize=10, fontweight='bold', color=c_blue)
ax.text(1.4, 3.1, 'IFCB (dark-field)', ha='center', fontsize=8, color=c_gray)
ax.text(1.4, 2.7, 'ZooScan (scanner)', ha='center', fontsize=8, color=c_gray)
ax.text(1.4, 2.3, 'DSPC (field camera)', ha='center', fontsize=8, color=c_gray)
ax.text(1.4, 1.8, 'Different optics', ha='center', fontsize=7, style='italic', color=c_gray)

# Arrow 1
ax.annotate('', xy=(3.0, 2.75), xytext=(2.5, 2.75),
            arrowprops=dict(arrowstyle='->', color=c_gray, lw=2))

# ── Box 2: Fourier Analysis ──
box2 = FancyBboxPatch((3.0, 1.5), 2.5, 2.5, boxstyle="round,pad=0.15",
                       facecolor='#FFF3E0', edgecolor=c_orange, linewidth=2)
ax.add_patch(box2)
ax.text(4.25, 3.6, 'Fourier Decomposition', ha='center', fontsize=10, fontweight='bold', color=c_orange)

# Draw mini spectrum
freqs = np.linspace(0, 1, 50)
low = np.exp(-freqs * 3) * 0.8
mid = np.exp(-(freqs - 0.4)**2 / 0.02) * 0.6
high = np.exp(-(freqs - 0.8)**2 / 0.05) * 0.3

ax.fill_between(freqs + 3.2, 1.6, 1.6 + low * 1.2, alpha=0.5, color=c_green, label='Species')
ax.fill_between(freqs + 3.2, 1.6, 1.6 + mid * 1.2, alpha=0.4, color=c_red, label='Domain')
ax.text(4.25, 2.9, 'Low freq = species', ha='center', fontsize=7, color=c_green, fontweight='bold')
ax.text(4.25, 2.5, 'Mid freq = domain', ha='center', fontsize=7, color=c_red, fontweight='bold')
ax.text(4.25, 2.0, '83.1% domain acc', ha='center', fontsize=7, color=c_orange)

# Arrow 2
ax.annotate('', xy=(6.0, 2.75), xytext=(5.5, 2.75),
            arrowprops=dict(arrowstyle='->', color=c_gray, lw=2))

# ── Box 3: Causal Masking ──
box3 = FancyBboxPatch((6.0, 1.5), 2.5, 2.5, boxstyle="round,pad=0.15",
                       facecolor='#E8F5E9', edgecolor=c_green, linewidth=2)
ax.add_patch(box3)
ax.text(7.25, 3.6, 'Causal Validation', ha='center', fontsize=10, fontweight='bold', color=c_green)
ax.text(7.25, 3.1, 'Mask frequencies', ha='center', fontsize=8, color=c_gray)
ax.text(7.25, 2.7, 'Low only: 43.8%', ha='center', fontsize=9, color=c_green, fontweight='bold')
ax.text(7.25, 2.3, 'Mid only: 92.6%', ha='center', fontsize=9, color=c_red, fontweight='bold')
ax.text(7.25, 1.9, 'Causal proof', ha='center', fontsize=7, style='italic', color=c_gray)

# Arrow 3
ax.annotate('', xy=(9.0, 2.75), xytext=(8.5, 2.75),
            arrowprops=dict(arrowstyle='->', color=c_gray, lw=2))

# ── Box 4: Mitigation Tools ──
box4 = FancyBboxPatch((9.0, 1.5), 2.5, 2.5, boxstyle="round,pad=0.15",
                       facecolor='#F3E5F5', edgecolor=c_purple, linewidth=2)
ax.add_patch(box4)
ax.text(10.25, 3.6, 'Calibrated Tools', ha='center', fontsize=10, fontweight='bold', color=c_purple)
ax.text(10.25, 3.1, 'SBA: +5.9%', ha='center', fontsize=9, color=c_purple, fontweight='bold')
ax.text(10.25, 2.7, 'RAVL: +57.5%', ha='center', fontsize=9, color=c_purple, fontweight='bold')
ax.text(10.25, 2.3, 'OOD router', ha='center', fontsize=8, color=c_gray)
ax.text(10.25, 1.9, 'Bray-Curtis: 0.096', ha='center', fontsize=7, color=c_gray)

# Arrow 4
ax.annotate('', xy=(12.0, 2.75), xytext=(11.5, 2.75),
            arrowprops=dict(arrowstyle='->', color=c_gray, lw=2))

# ── Box 5: Deployment ──
box5 = FancyBboxPatch((11.5, 1.5), 2.2, 2.5, boxstyle="round,pad=0.15",
                       facecolor='#ECEFF1', edgecolor=c_gray, linewidth=2)
ax.add_patch(box5)
ax.text(12.6, 3.6, 'Deployment', ha='center', fontsize=10, fontweight='bold', color=c_gray)
ax.text(12.6, 3.1, 'Multi-instrument', ha='center', fontsize=8, color=c_gray)
ax.text(12.6, 2.7, 'monitoring', ha='center', fontsize=8, color=c_gray)
ax.text(12.6, 2.3, 'networks', ha='center', fontsize=8, color=c_gray)
ax.text(12.6, 1.8, 'Ecological metrics', ha='center', fontsize=7, style='italic', color=c_gray)

# Title
ax.text(7, 4.5, 'Frequency-Domain Decomposition Reveals Domain-Specific and Biological Signals in Ecological Imaging',
        ha='center', fontsize=12, fontweight='bold', color='#212121')

plt.tight_layout()
out_path = FIGURES_DIR / "fig_graphical_abstract.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
