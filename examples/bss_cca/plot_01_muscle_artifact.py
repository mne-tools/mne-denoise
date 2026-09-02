r"""
Attenuating broadband muscle artifact with BSS-CCA
==================================================

Can lagged CCA separate a deliberately added broadband muscle-like process from
a cleaner EEG substrate while preserving data outside the artifact periods?
This example uses a real MNE Sample EEG recording with two controlled,
spatially structured muscle-like processes active in known time windows.

BSS-CCA orders components by their lagged temporal correlation. Broadband
muscle-like activity can occupy the low-correlation end, but low correlation is
a selection heuristic rather than an artifact label. The unmodified
average-referenced recording is a reference substrate for this controlled
comparison, not noise-free neural ground truth.

Because this controlled construction injects two independent muscle-like
processes, we use ``n_remove=2`` as a transparent operating choice. This does
not imply that CCA components correspond one-to-one with the injected sources,
and real recordings require independent validation of the removal rule.

The use case is motivated by
:footcite:p:`declercq2006_bss_cca,vergult2007_bss_cca`.

References
----------
.. footbibliography::
"""

# %%
# Load a band-limited EEG substrate
# ----------------------------------
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from mne_denoise.bss_cca import BSSCCA
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_psd_comparison, plot_signal_overlay

sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=False,
    verbose="ERROR",
)
raw.pick("eeg", exclude="bads").crop(0.0, 20.0).load_data()
raw.resample(250.0, verbose="ERROR")
raw.filter(1.0, 45.0, verbose="ERROR")
raw.set_eeg_reference("average", projection=False, verbose="ERROR")

# The unmodified average-referenced recording is the reference substrate, not
# perfect neural ground truth.
reference_data = raw.get_data()
sfreq = raw.info["sfreq"]
n_channels = reference_data.shape[0]
n_times = reference_data.shape[1]

# %%
# Add two known broadband muscle-like processes
# ----------------------------------------------
rng = np.random.default_rng(2006)
muscle_filter = butter(
    4,
    (25.0, 80.0),
    btype="bandpass",
    fs=sfreq,
    output="sos",
)
muscle_sources = sosfiltfilt(
    muscle_filter,
    rng.standard_normal((2, n_times)),
    axis=-1,
)
muscle_sources /= np.std(muscle_sources, axis=1, keepdims=True)

artifact_mask = np.zeros(n_times, dtype=bool)
artifact_envelope = np.zeros(n_times)
for start_seconds, stop_seconds in ((4.0, 6.0), (12.0, 14.0)):
    start = int(round(start_seconds * sfreq))
    stop = int(round(stop_seconds * sfreq))
    artifact_mask[start:stop] = True
    artifact_envelope[start:stop] = np.hanning(stop - start)
muscle_sources *= artifact_envelope

artifact_spatial = rng.standard_normal((n_channels, 2))
artifact_spatial -= artifact_spatial.mean(axis=0, keepdims=True)
artifact_spatial /= np.linalg.norm(artifact_spatial, axis=0, keepdims=True)
channel_scale = np.median(np.std(reference_data, axis=1))
artifact_multiplier = 4.0
artifact_data = (
    artifact_multiplier * channel_scale * (artifact_spatial @ muscle_sources)
)
corrupted_data = reference_data + artifact_data

corrupted = mne.io.RawArray(
    corrupted_data,
    raw.info.copy(),
    first_samp=raw.first_samp,
    verbose=False,
)
corrupted.set_annotations(raw.annotations.copy())

# %%
# Fit BSS-CCA with the known controlled component count
# -------------------------------------------------------
model = BSSCCA(
    lag_samples=1,
    n_remove=2,
    preserve_mean=True,
    verbose=False,
)
cleaned = model.fit_transform(corrupted)
cleaned_data = cleaned.get_data()

# %%
# Evaluate artifact attenuation and quiet-signal preservation
# -----------------------------------------------------------
artifact_before = rms_change(
    corrupted_data[:, artifact_mask],
    reference_data[:, artifact_mask],
)
artifact_after = rms_change(
    cleaned_data[:, artifact_mask],
    reference_data[:, artifact_mask],
)
artifact_residual_ratio = artifact_after / artifact_before

quiet_mask = ~artifact_mask
quiet_error = rms_change(
    cleaned_data[:, quiet_mask],
    reference_data[:, quiet_mask],
)
quiet_scale = np.sqrt(np.mean(reference_data[:, quiet_mask] ** 2))
quiet_relative_error = quiet_error / quiet_scale

removed_mask = ~model.kept_mask_
removed_correlations = model.correlations_[removed_mask]
removed_autocorrelations = model.autocorrelations_[removed_mask]
print(f"Artifact residual ratio: {artifact_residual_ratio:.3f}")
print(f"Quiet-period relative error: {quiet_relative_error:.3f}")
print(f"Number of components removed: {model.n_removed_}")
print(
    f"Canonical correlations of removed components: {np.round(removed_correlations, 3)}"
)
print(
    "Signed lagged autocorrelations of removed components: "
    f"{np.round(removed_autocorrelations, 3)}"
)

# %%
# Inspect a muscle-dominated channel
# -----------------------------------
channel_index = int(np.argmax(np.linalg.norm(artifact_data[:, artifact_mask], axis=1)))
channel_name = raw.ch_names[channel_index]
plot_signal_overlay(
    corrupted,
    cleaned,
    raw.times,
    pick=channel_name,
    start=3.0,
    stop=7.0,
    scale_after=False,
    before_label="corrupted",
    after_label="BSS-CCA",
    reference=reference_data[channel_index],
    reference_label="unmodified recording",
    highlight_mask=artifact_mask,
    highlight_label="muscle-like artifact window",
    x_label="Time (s)",
    y_label="Amplitude (V)",
    title=f"Broadband muscle-like artifact at {channel_name}",
    show=False,
)

# %%
# Compare broadband power before and after
# -----------------------------------------
# PSD attenuation is a useful diagnostic for the broadband use case, but the
# controlled reference comparison above is the preservation endpoint.
plot_psd_comparison(
    corrupted,
    cleaned,
    sfreq=sfreq,
    fmin=1.0,
    fmax=95.0,
    picks="eeg",
    show=False,
)
