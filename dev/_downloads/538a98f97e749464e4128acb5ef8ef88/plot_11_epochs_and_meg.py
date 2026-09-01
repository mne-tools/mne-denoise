r"""
ASR on Epochs and on MEG.
=========================

The same :class:`~mne_denoise.asr.ASR` estimator works across MNE container
types and channel types. This example shows two cases the other examples do not:

1. **Epochs** --- calibrate on continuous ``Raw`` and clean segmented
   ``mne.Epochs`` with the fitted model.
2. **MEG** --- ASR is unit/scale agnostic, so ``picks="mag"`` cleans
   magnetometers exactly as ``picks="eeg"`` cleans EEG.
"""

# %%
# Part 1 - Epochs: fit on Raw, transform Epochs
# ---------------------------------------------
import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_denoise.asr import ASR
from mne_denoise.viz import plot_signal_overlay

rng = np.random.default_rng(5)
sfreq = 250.0
n_channels, n_times = 8, 12000  # 48 s
t = np.arange(n_times) / sfreq
brain = np.vstack(
    [
        0.6 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))
        + 0.05 * rng.standard_normal(n_times)
        for _ in range(n_channels)
    ]
)
contaminated = brain.copy()
for start in np.linspace(500, n_times - 500, 9).astype(int):
    spatial = rng.standard_normal(n_channels)
    spatial /= np.linalg.norm(spatial)
    contaminated[:, start : start + 200] += 10.0 * np.outer(
        spatial, rng.standard_normal(200)
    )

info = mne.create_info([f"EEG{i:02d}" for i in range(n_channels)], sfreq, "eeg")
raw = mne.io.RawArray(contaminated, info, verbose="ERROR")
events = mne.make_fixed_length_events(raw, duration=2.0)
epochs = mne.Epochs(
    raw, events, tmin=0.0, tmax=2.0, baseline=None, preload=True, verbose="ERROR"
)

# Calibrate on the continuous data, then clean the epochs.
asr = ASR(cutoff=20.0, picks="eeg", verbose=False).fit(raw)
epochs_clean = asr.transform(epochs)
print(
    f"Epochs in: {epochs.get_data().shape} -> cleaned: {epochs_clean.get_data().shape}"
)

before = epochs.get_data()[:, 0].ravel()
after = epochs_clean.get_data()[:, 0].ravel()
plot_signal_overlay(
    before,
    after,
    np.arange(before.size) / sfreq,
    scale_after=False,
    before_label="epochs (raw)",
    after_label="epochs (ASR)",
    x_label="concatenated epoch time (s)",
    y_label="Amplitude (a.u.)",
    title="ASR on mne.Epochs (channel 0)",
    show=False,
)

# %%
# Part 2 - MEG: clean magnetometers
# ---------------------------------
# ASR is scale-agnostic, so MEG (Tesla-scale) is handled like EEG.
sample = mne.datasets.sample.data_path()
raw_meg = mne.io.read_raw_fif(
    sample / "MEG" / "sample" / "sample_audvis_raw.fif", preload=True, verbose="ERROR"
)
raw_meg.pick("mag").crop(0, 40).resample(150, verbose="ERROR")
raw_meg.filter(1.0, None, verbose="ERROR")

asr_meg = ASR(cutoff=20.0, picks="mag", verbose=False)
meg_clean = asr_meg.fit_transform(raw_meg.copy())

var_before = float(np.var(raw_meg.get_data()))
var_after = float(np.var(meg_clean.get_data()))
print(f"MEG variance reduced: {100.0 * (1 - var_after / var_before):.1f}%")

noisiest = int(np.argmax(np.var(raw_meg.get_data(), axis=1)))
plot_signal_overlay(
    raw_meg,
    meg_clean,
    raw_meg.times,
    pick=raw_meg.ch_names[noisiest],
    scale_after=False,
    before_label="raw",
    after_label="ASR",
    x_label="Time (s)",
    y_label="Amplitude (T)",
    title=f"ASR on MEG magnetometer {raw_meg.ch_names[noisiest]}",
    show=False,
)

plt.show()
