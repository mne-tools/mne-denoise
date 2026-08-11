"""
Basic reference-free BSS-CCA
============================

This example mixes narrow-band neural sources with broadband muscle sources,
attenuates the muscle activity with BSS-CCA, and then runs the same workflow on
an MNE Raw object.

BSS-CCA needs no reference channel: it separates sources by how correlated they
are with a delayed copy of the recording, and muscle activity — being close to
temporally white — lands in the lowest-correlation components.
"""

import mne
import numpy as np

from mne_denoise.bss_cca import BSSCCA, compute_bss_cca
from mne_denoise.viz import plot_psd_comparison, plot_signal_overlay

sfreq = 250.0
n_channels, n_times = 21, int(250.0 * 10)
rng = np.random.default_rng(2006)
times = np.arange(n_times) / sfreq

# Narrow-band "brain" sources: delta, alpha, and low beta.
brain_sources = np.vstack(
    [
        np.sin(2 * np.pi * 2.0 * times),
        np.sin(2 * np.pi * 10.0 * times + 0.4),
        np.sin(2 * np.pi * 21.0 * times + 1.1),
    ]
)
# Broadband "muscle" sources.
muscle_sources = rng.standard_normal((3, n_times))

clean = rng.standard_normal((n_channels, 3)) @ brain_sources
observed = clean + 0.8 * (rng.standard_normal((n_channels, 3)) @ muscle_sources)

# %%
# Removing the three lowest-correlation components recovers the neural signal.
# ``n_remove`` is the operating knob used in the original paper; sweeping it is
# how you choose an operating point.

cleaned, info = compute_bss_cca(observed, n_remove=3, sfreq=sfreq, preserve_mean=False)

clean_centered = clean - clean.mean(axis=1, keepdims=True)
observed_centered = observed - observed.mean(axis=1, keepdims=True)


def relative_error(estimate):
    """Relative RMS error against the known clean signal."""
    return float(
        np.linalg.norm(estimate - clean_centered) / np.linalg.norm(clean_centered)
    )


print(f"Canonical correlations: {np.round(info['correlations'], 3)}")
print(f"Components removed:     {info['n_removed']} of {info['input_rank']}")
print(f"Relative error before:  {relative_error(observed_centered):.3f}")
print(f"Relative error after:   {relative_error(cleaned):.3f}")

# %%
# The signed autocorrelation is the honest diagnostic. Canonical correlations
# cannot be negative, so a component carrying energy near the Nyquist frequency
# would rank high despite being anti-correlated at the lag. What matters is
# whether such a component is *retained*: a slightly negative value among the
# components you already discarded is just noise.

kept = info["kept_mask"]
retained_aliased = kept & (info["autocorrelations"] < -0.1)
print(f"Signed lag-1 autocorrelation: {np.round(info['autocorrelations'], 3)}")
print(f"Retained but anti-correlated: {np.flatnonzero(retained_aliased).tolist()}")

# %%
# One channel, before and after, against the known clean waveform.

plot_signal_overlay(
    observed,
    cleaned,
    times,
    pick=0,
    start=0.0,
    stop=2.0,
    scale_after=False,
    before_label="Observed",
    after_label="BSS-CCA",
    reference=clean_centered[0],
    reference_label="Clean signal",
    x_label="Time (s)",
    title="Reference-free BSS-CCA - channel 0",
    show=False,
)

# %%
# Broadband power falls while the narrow-band neural peaks survive.

plot_psd_comparison(
    observed,
    cleaned,
    sfreq=sfreq,
    fmin=1.0,
    fmax=120.0,
    show=False,
)

# %%
# MNE containers use the same estimator and return a copy of the same object,
# with timing, annotations, bad channels, and unselected channels preserved.

info_mne = mne.create_info(
    [f"EEG {index:03d}" for index in range(n_channels)], sfreq, "eeg"
)
raw = mne.io.RawArray(observed, info_mne, verbose=False)
raw_clean = BSSCCA(n_remove=3).fit_transform(raw)
print(type(raw_clean).__name__, raw_clean.get_data().shape)

# %%
# Muscle artifacts are non-stationary, so a single operator across a long
# recording is often the wrong model. ``segment_len`` reproduces the contiguous
# 10-second block scheme used clinically.

raw_blocked = BSSCCA(segment_len=10.0, n_remove=3).fit_transform(raw)
print(f"Block-wise output: {raw_blocked.get_data().shape}")
