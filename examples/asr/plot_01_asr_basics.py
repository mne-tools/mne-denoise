r"""
Artifact Subspace Reconstruction: Basic Usage.
==============================================

This example demonstrates standard ASR on a synthetic multichannel EEG signal
with short spatial burst artifacts.
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR
from mne_denoise.qa import variance_removed
from mne_denoise.viz import plot_signal_overlay

# %%
# Synthetic EEG with Bursts
# -------------------------
sfreq = 250.0
duration = 12.0
n_channels = 8
n_times = int(sfreq * duration)
times = np.arange(n_times) / sfreq
rng = np.random.default_rng(42)

brain = np.zeros((n_channels, n_times))
for ch_idx in range(n_channels):
    phase = rng.uniform(0, 2 * np.pi)
    brain[ch_idx] = (
        0.5 * np.sin(2 * np.pi * 10 * times + phase)
        + 0.2 * np.sin(2 * np.pi * 6 * times + 0.5 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

data = brain.copy()
burst_mask = np.zeros(n_times, dtype=bool)
spatial = rng.standard_normal((n_channels, 2))
spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)

for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
    start = int(onset * sfreq)
    end = int(stop * sfreq)
    burst_mask[start:end] = True
    data[:, start:end] += spatial @ (8.0 * rng.standard_normal((2, end - start)))

# %%
# Fit and Apply ASR
# -----------------
# Conservative cutoffs around 20 are a typical starting point for adult EEG and
# should still be validated with QC plots and downstream analyses.
asr = ASR(
    sfreq=sfreq,
    cutoff=20.0,
    calibration="auto",
    filter_kind="none",
    max_dims=0.5,
)
clean = asr.fit_transform(data)

print(f"Calibration windows kept: {asr.clean_window_mask_.sum()}")
print(f"Repaired windows: {(asr.n_components_reconstructed_ > 0).sum()}")
print(f"Repaired sample fraction: {asr.fraction_reconstructed_samples_:.2%}")
print(f"Variance removed: {variance_removed(data, clean):.2f}%")

# %%
# Plot One Channel
# ----------------
# ``plot_signal_overlay`` draws the before/after pair and, via its optional
# ``reference`` and ``highlight_mask`` arguments, overlays the ground-truth
# signal and shades the burst windows directly.
plot_signal_overlay(
    data,
    clean,
    times,
    pick=0,
    scale_after=False,
    before_label="Noisy",
    after_label="ASR cleaned",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="ASR burst repair",
    reference=brain[0],
    reference_label="Reference signal",
    highlight_mask=burst_mask,
    highlight_label="Burst artifact",
    show=False,
)

plt.show()
