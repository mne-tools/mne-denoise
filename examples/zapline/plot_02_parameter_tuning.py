r"""
ZapLine: Parameter Tuning and Real Data.
=========================================

This example shows how the main ZapLine tuning parameters change the cleaning
behavior on synthetic data, then applies the same workflow to the real MNE
Sample MEG recording.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal

from mne_denoise.viz import (
    plot_component_cleaning_summary,
    plot_component_patterns,
    plot_component_score_curve,
    plot_psd_comparison,
)
from mne_denoise.zapline import ZapLine

# %%
# Part 1: n_remove Parameter
# --------------------------
# The `n_remove` parameter controls how many spatial components are removed.
# Too few: line noise remains. Too many: neural signal lost.

print("Part 1: Exploring n_remove parameter")

# Generate synthetic data with multiple line noise sources
sfreq = 1000
duration = 5
n_channels = 16
n_times = int(sfreq * duration)
t = np.arange(n_times) / sfreq

rng = np.random.RandomState(42)

# Create 3 distinct line noise sources (simulating multiple power sources)
n_line_sources = 3
line_patterns = rng.randn(n_channels, n_line_sources)
for i in range(n_line_sources):
    line_patterns[:, i] /= np.linalg.norm(line_patterns[:, i])

# Neural pattern
neural_pattern = rng.randn(n_channels)
neural_pattern /= np.linalg.norm(neural_pattern)

# Generate data
neural_source = np.sin(2 * np.pi * 10 * t)
line_sources = np.zeros((n_line_sources, n_times))
for i in range(n_line_sources):
    phase = rng.uniform(0, 2 * np.pi)
    amp = 2.0 - i * 0.5  # Decreasing amplitude
    line_sources[i] = amp * np.sin(2 * np.pi * 50 * t + phase)

data = np.zeros((n_channels, n_times))
for i in range(n_channels):
    data[i] = neural_pattern[i] * neural_source
    for j in range(n_line_sources):
        data[i] += line_patterns[i, j] * line_sources[j]
    data[i] += rng.randn(n_times) * 0.3

print(f"Data with {n_line_sources} line noise sources")

# %%
# Compare Different n_remove Values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Small values tend to leave narrowband noise behind. Larger values clean more
# aggressively, but eventually start removing structure that is no longer just
# line noise.

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

n_remove_values = [1, 2, 3, 4, 5, "auto"]

for idx, n_remove in enumerate(n_remove_values):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    est = ZapLine(line_freq=50, sfreq=sfreq, n_select=n_remove)
    est.fit(data)
    cleaned = est.transform(data)

    freqs, psd_orig = signal.welch(data, sfreq, nperseg=sfreq)
    freqs, psd_clean = signal.welch(cleaned, sfreq, nperseg=sfreq)

    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(50, color="r", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_title(f"n_select={n_remove} (actual: {est.n_removed_})")
    ax.set_xlabel("Frequency (Hz)")

    if col == 0:
        ax.set_ylabel("PSD")
    if idx == 0:
        ax.legend()

plt.suptitle("Effect of n_remove Parameter", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Part 2: nkeep Parameter (High Channel Count)
# ---------------------------------------------
# For data with many channels, the optional `nkeep` parameter reduces
# dimensionality before DSS to avoid overfitting.

print("\nPart 2: nkeep parameter for high-channel data")

# Create high-channel-count data
n_channels_high = 128
data_high = np.zeros((n_channels_high, n_times))

# Multiple line sources with different spatial patterns
n_line_sources = 2
line_patterns_high = rng.randn(n_channels_high, n_line_sources)
for i in range(n_line_sources):
    line_patterns_high[:, i] /= np.linalg.norm(line_patterns_high[:, i])

neural_pattern_high = rng.randn(n_channels_high)
neural_pattern_high /= np.linalg.norm(neural_pattern_high)

for i in range(n_channels_high):
    data_high[i] = neural_pattern_high[i] * neural_source
    for j in range(n_line_sources):
        data_high[i] += line_patterns_high[i, j] * line_sources[j]
    data_high[i] += rng.randn(n_times) * 0.3

print(f"High-channel data: {data_high.shape}")

# %%
# Compare Different nkeep Values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The `nkeep` parameter matters most when the channel count is high. It trades
# some spatial richness for a more stable low-rank DSS problem.

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

nkeep_values = [None, 64, 32, 16]

for idx, nkeep in enumerate(nkeep_values):
    ax = axes[idx]

    est_high = ZapLine(line_freq=50, sfreq=sfreq, n_select=2, nkeep=nkeep)
    est_high.fit(data_high)
    cleaned_high = est_high.transform(data_high)

    freqs, psd_orig = signal.welch(data_high, sfreq, nperseg=sfreq)
    freqs, psd_clean = signal.welch(cleaned_high, sfreq, nperseg=sfreq)

    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(50, color="r", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_title(f"nkeep={nkeep if nkeep else 'All (128)'}")
    ax.set_xlabel("Frequency (Hz)")

    if idx == 0:
        ax.set_ylabel("PSD")
        ax.legend()

plt.suptitle("Effect of nkeep Parameter (128 channels)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Part 3: Component Scores
# ------------------------
# The eigenvalues (scores) indicate how much each component
# carries line noise. High scores = strong line noise.
# The selected components should be the ones that combine a strong narrowband
# signature with a plausible spatial pattern.

print("\nPart 3: Understanding component scores")

# Use a fixed removal count to guarantee selected components for pattern plots.
est_scores = ZapLine(line_freq=50, sfreq=sfreq, n_select=3)
est_scores.fit(data)
plot_montage = mne.channels.make_standard_montage("standard_1020")
plot_info = mne.create_info(plot_montage.ch_names[:n_channels], sfreq, "eeg")
plot_info.set_montage(plot_montage)
plot_picks = np.arange(len(plot_info.ch_names))

# Use the reusable viz functions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_component_score_curve(est_scores, ax=axes[0], show=False)
plot_component_patterns(
    est_scores,
    info=plot_info,
    picks=plot_picks,
    n_components=1,
    ax=axes[1],
    show=False,
)
plt.tight_layout()
plt.show()

# %%
# Part 4: Real MEG Data (MNE Sample)
# ----------------------------------
# Apply ZapLine to a real MEG recording managed by MNE's dataset fetcher. MNE
# verifies the dataset archive and reuses its local copy across examples.

print("\nPart 4: Real MEG Data")

# Load 30 seconds of gradiometer data and resample to keep the gallery build
# compact. The MNE Sample recording was acquired in a 60 Hz mains environment.
sample_path = mne.datasets.sample.data_path()
raw_path = sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"
raw_meg = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
grad_picks = mne.pick_types(raw_meg.info, meg="grad", exclude="bads")
raw_meg.pick(grad_picks).crop(0, 30).load_data().resample(300, verbose="ERROR")
meg_data = raw_meg.get_data() * 1e12  # Constant scale for readable values
sfreq_meg = raw_meg.info["sfreq"]

# Demean
meg_data = meg_data - np.mean(meg_data, axis=1, keepdims=True)

print(f"Loaded MNE Sample data: {meg_data.shape}, sfreq={sfreq_meg} Hz")
print("Real gradiometer data with 60 Hz line interference")

# %%
# Apply ZapLine to MEG Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^

# Apply ZapLine (60 Hz for this dataset)
est_meg = ZapLine(
    line_freq=60,
    sfreq=sfreq_meg,
    n_select=2,  # As in MATLAB example
)
est_meg.fit(meg_data)
cleaned_meg = est_meg.transform(meg_data)

print(f"Components removed: {est_meg.n_removed_}")

# %%
# Compare Before/After for MEG
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# On the real recording, the relevant check is whether the 60 Hz peak drops
# sharply without flattening the rest of the broadband spectrum.

# Use reusable viz function for PSD comparison
plot_psd_comparison(
    meg_data,
    cleaned_meg,
    sfreq=sfreq_meg,
    line_freq=60,
    fmax=150,
    show=True,
)

# %%
# Component Cleaning Summary
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

plot_component_cleaning_summary(
    scores=getattr(est_meg, "scores_", getattr(est_meg, "eigenvalues_", None)),
    selected_count=getattr(est_meg, "n_removed_", 0),
    patterns=getattr(est_meg, "patterns_", None),
    removed=meg_data - cleaned_meg,
    sources=getattr(est_meg, "sources_", None),
    sfreq=sfreq_meg,
    line_freq=60,
    title="Component Cleaning Summary (ZapLine)",
    show=True,
)

# %%
# Measure Reduction
# ^^^^^^^^^^^^^^^^^

freqs_meg, psd_orig_meg = signal.welch(meg_data, sfreq_meg, nperseg=int(sfreq_meg * 2))
_, psd_clean_meg = signal.welch(cleaned_meg, sfreq_meg, nperseg=int(sfreq_meg * 2))
idx_60 = np.argmin(np.abs(freqs_meg - 60))
power_60_orig = np.mean(psd_orig_meg[:, idx_60])
power_60_clean = np.mean(psd_clean_meg[:, idx_60])
reduction_db = 10 * np.log10(power_60_orig / power_60_clean)

print("\n=== MEG Results ===")
print(f"60 Hz power reduction: {reduction_db:.1f} dB")
print(f"Components removed: {est_meg.n_removed_}")

# %%
# Conclusion
# ----------
# A practical starting point is to use ``n_select="auto"`` or a small manual
# value such as 1 to 3, then increase it only if clear line-noise structure
# remains. ``nkeep`` becomes useful for high-channel recordings, ``threshold``
# controls how aggressive automatic removal is, and ``n_harmonics`` matters
# most when the sampling rate is high enough for several harmonics to be
# present.
