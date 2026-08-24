"""
Basic Sensor Noise Suppression
==============================

This example adds independent sensor noise to a spatially redundant signal,
fits SNS, and applies the same workflow to an MNE Raw object.
"""

import mne
import numpy as np

from mne_denoise.sns import SNS
from mne_denoise.viz import plot_signal_overlay

rng = np.random.default_rng(12)
n_channels, n_sources, n_times = 24, 4, 2500
sources = rng.standard_normal((n_sources, n_times))
shared = rng.standard_normal((n_channels, n_sources)) @ sources
observed = shared + 0.45 * rng.standard_normal(shared.shape)

model = SNS(n_neighbors=12, outlier_threshold=8.0)
cleaned = model.fit_transform(observed)

error_before = np.linalg.norm(observed - shared) / np.linalg.norm(shared)
error_after = np.linalg.norm(cleaned - shared) / np.linalg.norm(shared)
print(f"Relative error before SNS: {error_before:.3f}")
print(f"Relative error after SNS:  {error_after:.3f}")
print(f"Effective neighbors: {model.n_neighbors_}")
print(f"Samples rejected while fitting: {model.rejected_sample_count_}")

plot_signal_overlay(
    observed,
    cleaned,
    np.arange(n_times) / 250.0,
    pick=0,
    start=0.0,
    stop=1.6,
    scale_after=False,
    before_label="Observed",
    after_label="SNS",
    reference=shared[0],
    reference_label="Shared signal",
    x_label="Time (s)",
    title="Sensor Noise Suppression — channel 0",
    show=False,
)

# MNE containers use the same estimator. The returned object is a copy with its
# metadata and channel layout preserved.
info = mne.create_info(
    [f"EEG {index:03d}" for index in range(n_channels)], 250.0, "eeg"
)
raw = mne.io.RawArray(observed, info, verbose=False)
raw_clean = SNS(n_neighbors=12).fit_transform(raw)
print(type(raw_clean).__name__, raw_clean.get_data().shape)
