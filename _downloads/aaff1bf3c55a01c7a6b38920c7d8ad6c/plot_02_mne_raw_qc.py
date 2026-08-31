r"""
Artifact Subspace Reconstruction: Raw QC and Annotations.
=========================================================

This example shows ASR on an ``mne.io.RawArray`` with optional clean_windows-
style final window rejection. The repaired spans and rejected spans are kept
as annotations for downstream QC instead of deleting samples immediately.
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_denoise.asr import ASR
from mne_denoise.viz import plot_signal_overlay

# %%
# Synthetic Raw
# -------------
sfreq = 250.0
duration = 15.0
n_eeg = 8
n_times = int(sfreq * duration)
times = np.arange(n_times) / sfreq
rng = np.random.default_rng(7)

brain = np.zeros((n_eeg, n_times))
for ch_idx in range(n_eeg):
    phase = rng.uniform(0, 2 * np.pi)
    brain[ch_idx] = (
        0.4 * np.sin(2 * np.pi * 10 * times + phase)
        + 0.15 * np.sin(2 * np.pi * 6 * times + 0.5 * phase)
        + 0.04 * rng.standard_normal(n_times)
    )

eog = 0.25 * np.sin(2 * np.pi * 1.2 * times)[np.newaxis, :]
data = brain.copy()
spatial = rng.standard_normal((n_eeg, 2))
spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)

for onset, stop in ((4.5, 5.3), (9.0, 9.7), (12.0, 12.6)):
    start = int(onset * sfreq)
    end = int(stop * sfreq)
    data[:, start:end] += spatial @ (7.0 * rng.standard_normal((2, end - start)))

raw_data = np.vstack([data, eog])
ch_names = [f"EEG {idx:02d}" for idx in range(n_eeg)] + ["EOG 01"]
ch_types = ["eeg"] * n_eeg + ["eog"]
info = mne.create_info(ch_names, sfreq, ch_types)
raw = mne.io.RawArray(raw_data, info, verbose=False)

# %%
# Fit and Apply ASR
# -----------------
asr = ASR(
    cutoff=5.0,
    calibration="auto",
    picks="eeg",
    filter_kind="none",
    window_criterion=0.25,
    window_criterion_tolerances=(-np.inf, 2.5),
    verbose=False,
)
raw_clean = asr.fit_transform(raw)
repair_annotations = asr.to_annotations()
reject_annotations = asr.to_annotations("rejection")

print(f"Repaired sample fraction: {asr.fraction_reconstructed_samples_:.2%}")
print(
    "Retained after final window rejection: "
    f"{asr.fraction_retained_after_window_rejection_:.2%}"
)

# %%
# Plot One EEG Channel
# --------------------
# The repair and rejection annotations are passed straight to
# ``plot_signal_overlay`` as ``highlight_spans`` instead of shading the axis by
# hand; the reference trace uses the ``reference`` argument.
channel = "EEG 00"
noisy = raw.get_data(picks=[channel])[0]
clean = raw_clean.get_data(picks=[channel])[0]

spans = [
    {
        "onset": onset,
        "duration": dur,
        "color": "C3",
        "alpha": 0.12,
        "label": "ASR repair",
    }
    for onset, dur in zip(repair_annotations.onset, repair_annotations.duration)
] + [
    {
        "onset": onset,
        "duration": dur,
        "color": "0.2",
        "alpha": 0.08,
        "label": "Window reject mask",
    }
    for onset, dur in zip(reject_annotations.onset, reject_annotations.duration)
]

plot_signal_overlay(
    noisy,
    clean,
    times,
    scale_after=False,
    before_label="Noisy EEG",
    after_label="ASR cleaned EEG",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="ASR repairs plus optional final window rejection mask",
    reference=brain[0],
    reference_label="Reference EEG",
    highlight_spans=spans,
    show=False,
)

plt.show()
