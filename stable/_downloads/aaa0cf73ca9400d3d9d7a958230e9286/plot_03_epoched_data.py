r"""
ZapLine: Epoched Data and Real Data Examples.
==============================================

This tutorial demonstrates:
1. ZapLine with epoched MEG data (NoiseTools data3.mat)
2. High-channel MEG data (NoiseTools example_data.mat)
3. The sklearn-style Transformer API

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

# %%
# Imports
# -------
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat

from mne_denoise.viz.zapline import (
    plot_cleaning_summary,
    plot_psd_comparison,
)
from mne_denoise.zapline import ZapLine

# %%
# Part 1: Synthetic Epoched Data
# ------------------------------
# Create epoched data with line noise to demonstrate ZapLine on trials.

print("Part 1: Synthetic Epoched Data")

# Parameters
sfreq = 500  # Sampling rate
n_epochs = 30
n_channels = 16
n_times = 250  # 0.5 seconds per epoch

rng = np.random.RandomState(42)

# Create spatial patterns
neural_pattern = rng.randn(n_channels)
neural_pattern /= np.linalg.norm(neural_pattern)

line_pattern = rng.randn(n_channels)
line_pattern /= np.linalg.norm(line_pattern)

# Generate epoched data
t = np.arange(n_times) / sfreq
epochs_data = np.zeros((n_epochs, n_channels, n_times))

for i in range(n_epochs):
    # Neural signal (evoked-like)
    neural_source = np.sin(2 * np.pi * 10 * t) * np.exp(-t / 0.2)

    # Line noise (constant across epochs, different phase)
    phase = rng.uniform(0, 2 * np.pi)
    line_source = 1.5 * np.sin(2 * np.pi * 50 * t + phase)

    for ch in range(n_channels):
        epochs_data[i, ch] = (
            neural_pattern[ch] * neural_source
            + line_pattern[ch] * line_source
            + rng.randn(n_times) * 0.3
        )

print(f"Epochs data shape: {epochs_data.shape}")  # (n_epochs, n_channels, n_times)

# %%
# Apply ZapLine to Epoched Data
# -----------------------------
# ZapLine expects 2D data, so we concatenate epochs for fitting.

print("\nApplying ZapLine to epoched data...")

# Concatenate epochs for ZapLine
data_concat = epochs_data.reshape(n_channels, -1)  # (channels, epochs*times)
print(f"Concatenated shape: {data_concat.shape}")

# Apply ZapLine
est = ZapLine(line_freq=50, sfreq=sfreq, n_remove=1)
est.fit(data_concat)
cleaned = est.transform(data_concat)

# Reshape back to epochs
cleaned_epochs = cleaned.reshape(n_epochs, n_channels, n_times)
print(f"Cleaned epochs shape: {cleaned_epochs.shape}")

# %%
# Compare Before/After
# ^^^^^^^^^^^^^^^^^^^^

# Use the reusable viz function for PSD comparison
plot_psd_comparison(data_concat, cleaned, sfreq, line_freq=50, show=True)

# Show comprehensive cleaning summary
plot_cleaning_summary(data_concat, cleaned, est, sfreq, line_freq=50, show=True)

# %%
# Part 2: Real MEG Epoched Data (NoiseTools data3.mat)
# ----------------------------------------------------
# MEG epoched data from NoiseTools.
# Shape: (900 times, 151 channels, 30 epochs), sr=300 Hz

print("\nPart 2: Real MEG Epoched Data")

try:
    script_dir = Path(__file__).parent
except NameError:
    script_dir = Path.cwd()
data_dir = script_dir / "data"

# Load data3.mat (MEG epoched)
data3_path = data_dir / "data3.mat"
if data3_path.exists():
    mat = loadmat(str(data3_path))
    meg_epochs = mat["data"]  # (times, channels, epochs) = (900, 151, 30)
    sfreq_meg = float(mat["sr"].flatten()[0])

    # Use first 10 epochs as in MATLAB example
    meg_epochs = meg_epochs[:, :, :10]  # (900, 151, 10)

    # Transpose to (channels, times*epochs) for ZapLine
    n_times_meg, n_ch_meg, n_ep_meg = meg_epochs.shape
    meg_concat = meg_epochs.transpose(1, 0, 2).reshape(n_ch_meg, -1)  # (151, 9000)

    # Demean
    meg_concat = meg_concat - np.mean(meg_concat, axis=1, keepdims=True)

    # Scale to reasonable units (MEG data is in Tesla, very small values)
    scale_factor = 1e12  # Convert to pT
    meg_concat = meg_concat * scale_factor

    print(
        f"Loaded data3.mat: {n_ep_meg} epochs, {n_ch_meg} channels, {n_times_meg} times"
    )
    print(f"Concatenated shape: {meg_concat.shape}")
    print(f"Sampling rate: {sfreq_meg} Hz")

    # Apply ZapLine (50 Hz)
    est_meg = ZapLine(
        line_freq=50,
        sfreq=sfreq_meg,
        n_remove=2,  # As in MATLAB example
    )
    est_meg.fit(meg_concat)
    cleaned_meg = est_meg.transform(meg_concat)

    print(f"Components removed: {est_meg.n_removed_}")

    # Use the reusable viz functions
    plot_psd_comparison(
        meg_concat, cleaned_meg, sfreq_meg, line_freq=50, fmax=150, show=True
    )

    # Measure reduction
    nperseg = min(meg_concat.shape[1], int(sfreq_meg * 2))
    freqs, psd_orig = signal.welch(meg_concat, sfreq_meg, nperseg=nperseg)
    _, psd_clean = signal.welch(cleaned_meg, sfreq_meg, nperseg=nperseg)
    idx_50 = np.argmin(np.abs(freqs - 50))
    reduction_db = 10 * np.log10(
        np.mean(psd_orig[:, idx_50]) / np.mean(psd_clean[:, idx_50])
    )
    print(f"50 Hz power reduction: {reduction_db:.1f} dB")

else:
    print(f"Data not found: {data3_path}")
    print("Download from: http://audition.ens.fr/adc/NoiseTools/DATA/data3.mat")

# %%
# Part 3: High-Channel MEG Data (NoiseTools example_data.mat)
# -----------------------------------------------------------
# MEG data with many channels (275), demonstrating nkeep parameter.
# Shape: (3000 times, 275 channels, 30 epochs), sr=600 Hz

print("\nPart 3: High-Channel MEG Data")

example_data_path = data_dir / "example_data.mat"
if example_data_path.exists():
    mat = loadmat(str(example_data_path))
    meg_high = mat["meg"]  # (times, channels, epochs) = (3000, 275, 30)
    sfreq_high = float(mat["sr"].flatten()[0])

    # Use first epoch as in MATLAB example
    meg_high = meg_high[:, :, 0].T  # (275, 3000)

    # Demean
    meg_high = meg_high - np.mean(meg_high, axis=1, keepdims=True)

    # Scale to reasonable units (MEG data is in Tesla, very small values)
    scale_factor = 1e12  # Convert to pT
    meg_high = meg_high * scale_factor

    print(f"Loaded example_data.mat: {meg_high.shape}")
    print(f"Sampling rate: {sfreq_high} Hz")

    # Apply ZapLine with nkeep
    est_high = ZapLine(
        line_freq=50,
        sfreq=sfreq_high,
        n_remove=6,  # As in MATLAB example
        nkeep=50,  # Reduce dimensionality
    )
    est_high.fit(meg_high)
    cleaned_high = est_high.transform(meg_high)

    print(f"Components removed: {est_high.n_removed_}")

    # Use the reusable viz functions
    plot_psd_comparison(
        meg_high, cleaned_high, sfreq_high, line_freq=50, fmax=150, show=True
    )
    plot_cleaning_summary(
        meg_high, cleaned_high, est_high, sfreq_high, line_freq=50, show=True
    )

    # Measure reduction
    nperseg = min(meg_high.shape[1], int(sfreq_high * 2))
    freqs, psd_orig = signal.welch(meg_high, sfreq_high, nperseg=nperseg)
    _, psd_clean = signal.welch(cleaned_high, sfreq_high, nperseg=nperseg)
    idx_50 = np.argmin(np.abs(freqs - 50))
    reduction_db = 10 * np.log10(
        np.mean(psd_orig[:, idx_50]) / np.mean(psd_clean[:, idx_50])
    )
    print(f"50 Hz power reduction: {reduction_db:.1f} dB")

else:
    print(f"Data not found: {example_data_path}")
    print("Download from: http://audition.ens.fr/adc/NoiseTools/DATA/example_data.mat")

# %%
# Conclusion
# ----------
# ZapLine handles various data types:
#
# 1. **Epoched Data**: Concatenate epochs, apply ZapLine, reshape back
# 2. **High-Channel Data**: Use nkeep parameter to reduce dimensionality
# 3. **Transformer API**: sklearn-style fit/transform workflow
#
# Key observations from real data:
# - Real MEG data often needs 2-6 components removed
# - nkeep=50 works well for high-channel (>100) data
# - 50 Hz reduction typically >30 dB
