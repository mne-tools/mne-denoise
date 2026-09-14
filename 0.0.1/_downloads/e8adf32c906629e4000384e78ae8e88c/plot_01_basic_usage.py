r"""
ZapLine: Line Noise Removal Fundamentals.
==========================================

This tutorial demonstrates **ZapLine** for removing power line noise (50/60 Hz).

ZapLine is a spatial filtering technique that:
1. Splits data into smooth (low-freq) and residual (high-freq + line noise)
2. Uses DSS to find components that maximize line noise power
3. Projects out the line noise components from the residual
4. Reconstructs clean data as smooth + cleaned residual

Key advantages over notch filters:
- Preserves neural signals AT the line frequency
- Removes harmonics automatically
- Adapts to spatially complex line noise sources

Reference: de Cheveigné, A. (2020). ZapLine: A simple and effective method
to remove power line artifacts. NeuroImage, 207, 116356.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mne_denoise.viz.zapline import (
    plot_cleaning_summary,
    plot_component_scores,
    plot_psd_comparison,
)
from mne_denoise.zapline import ZapLine

# %%
# Part 1: Synthetic Data
# ----------------------
# We create synthetic data with controllable line noise to demonstrate
# ZapLine's effectiveness.
#
# The data contains:
# - A 10 Hz "neural" signal (what we want to preserve)
# - A 50 Hz "line noise" signal (what we want to remove)
# - White noise

print("Generating synthetic data with 50 Hz line noise...")

# Parameters
sfreq = 1000  # Sampling rate
duration = 5  # seconds
n_channels = 8
n_times = int(sfreq * duration)
t = np.arange(n_times) / sfreq

# Random state for reproducibility
rng = np.random.RandomState(42)

# Create mixing matrices for spatial structure
# Neural signal has one spatial pattern
neural_pattern = rng.randn(n_channels)
neural_pattern /= np.linalg.norm(neural_pattern)

# Line noise has a different spatial pattern
line_pattern = rng.randn(n_channels)
line_pattern /= np.linalg.norm(line_pattern)

# Generate source signals
neural_source = np.sin(2 * np.pi * 10 * t)  # 10 Hz neural
line_source = 2.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz line noise (strong)

# Mix to channels
data = np.zeros((n_channels, n_times))
for i in range(n_channels):
    data[i] = (
        neural_pattern[i] * neural_source
        + line_pattern[i] * line_source
        + rng.randn(n_times) * 0.5  # White noise
    )

print(f"Data shape: {data.shape}")
print("Signal: 10 Hz neural + 50 Hz line noise + white noise")

# %%
# Visualize Original Data
# -----------------------
# Let's look at the power spectral density (PSD) before cleaning.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Time domain
ax = axes[0]
ax.plot(t[:500], data[0, :500], "b-", alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Channel 0 - Time Domain (First 500ms)")

# Frequency domain
ax = axes[1]
freqs, psd = signal.welch(data, sfreq, nperseg=sfreq)
ax.semilogy(freqs, psd.T, alpha=0.5)
ax.axvline(50, color="r", linestyle="--", label="50 Hz (line noise)")
ax.axvline(10, color="g", linestyle="--", label="10 Hz (neural)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("Power Spectral Density - All Channels")
ax.legend()
ax.set_xlim(0, 100)

plt.tight_layout()
plt.show()

# %%
# Apply ZapLine
# -------------
# We use the `ZapLine` class to remove 50 Hz and its harmonics.


print("\nApplying ZapLine...")
# Instantiate ZapLine
est = ZapLine(
    line_freq=50,
    sfreq=sfreq,
    n_remove="auto",  # Automatically detect number of components
    threshold=2.5,  # Z-score threshold for auto-detection
)
est.fit(data)
cleaned = est.transform(data)

print(f"Components removed: {est.n_removed_}")
print(f"Component scores (eigenvalues): {est.eigenvalues_}")
print(f"Harmonics processed: {est.n_harmonics_}")

# %%
# Compare Before/After
# --------------------
# Let's visualize the cleaning effect using the reusable viz function.

# Use plot_psd_comparison for a clean comparison
plot_psd_comparison(data, cleaned, sfreq, line_freq=50, show=True)

# %%
# Component Scores
# ----------------
# View the DSS component scores to understand what was removed.

plot_component_scores(est, show=True)

# %%
# Cleaning Summary
# ----------------
# A comprehensive summary combining PSD, scores, and statistics.

plot_cleaning_summary(data, cleaned, est, sfreq, line_freq=50, show=True)

# %%
# Quantify Reduction
# ------------------
# Measure the power reduction at 50 Hz.

idx_50 = np.argmin(np.abs(freqs - 50))
idx_50 = np.argmin(np.abs(freqs - 50))
idx_10 = np.argmin(np.abs(freqs - 10))

_, psd_orig = signal.welch(data, sfreq, nperseg=sfreq)
_, psd_clean = signal.welch(cleaned, sfreq, nperseg=sfreq)

power_50_orig = np.mean(psd_orig[:, idx_50])
power_50_clean = np.mean(psd_clean[:, idx_50])
power_10_orig = np.mean(psd_orig[:, idx_10])
power_10_clean = np.mean(psd_clean[:, idx_10])

reduction_50_db = 10 * np.log10(power_50_orig / power_50_clean)
preservation_10 = power_10_clean / power_10_orig * 100

print("\n=== Results ===")
print(f"50 Hz power reduction: {reduction_50_db:.1f} dB")
print(f"10 Hz power preserved: {preservation_10:.1f}%")

# %%
# Part 2: Multi-Harmonic Removal
# ------------------------------
# ZapLine can remove multiple harmonics of the line frequency.
# Let's add harmonics to our synthetic data.

print("\n\nPart 2: Multi-Harmonic Line Noise")

# Add harmonics
line_source_harmonics = (
    2.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz
    + 1.0 * np.sin(2 * np.pi * 100 * t)  # 100 Hz (2nd harmonic)
    + 0.5 * np.sin(2 * np.pi * 150 * t)  # 150 Hz (3rd harmonic)
)

data_harmonics = np.zeros((n_channels, n_times))
for i in range(n_channels):
    data_harmonics[i] = (
        neural_pattern[i] * neural_source
        + line_pattern[i] * line_source_harmonics
        + rng.randn(n_times) * 0.5
    )

# Apply ZapLine
# Apply ZapLine
est_harmonics = ZapLine(
    line_freq=50,
    sfreq=sfreq,
    n_remove=1,
    n_harmonics=3,  # Explicitly request 3 harmonics
)
est_harmonics.fit(data_harmonics)
cleaned_harmonics = est_harmonics.transform(data_harmonics)

print(f"Harmonics processed: {est_harmonics.n_harmonics_}")

# Compare PSDs
fig, ax = plt.subplots(figsize=(10, 5))
freqs, psd_orig = signal.welch(data_harmonics, sfreq, nperseg=sfreq)
freqs, psd_clean = signal.welch(cleaned_harmonics, sfreq, nperseg=sfreq)

ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", label="Original", alpha=0.7)
ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")

for h in [50, 100, 150]:
    ax.axvline(h, color="r", linestyle="--", alpha=0.5)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("ZapLine Removes All Harmonics")
ax.legend()
ax.set_xlim(0, 200)
plt.show()


# %%
# Conclusion
# ----------
# ZapLine is a powerful tool for removing line noise because:
#
# 1. **Spatial Selectivity**: Only removes components that carry line noise
# 2. **Harmonic Coverage**: Automatically handles harmonics
# 3. **Neural Preservation**: Preserves neural signals at line frequency
# 4. **Automatic Selection**: Can auto-detect number of components to remove
#
# Key parameters:
# - `line_freq`: 50 Hz (Europe) or 60 Hz (Americas)
# - `n_remove`: Number of components or "auto"
# - `n_harmonics`: How many harmonics to include
# - `nkeep`: PCA dimensionality reduction (for high-channel data)
