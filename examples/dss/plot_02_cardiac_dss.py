r"""
Removing a planted cardiac-locked artifact with DSS
====================================================

Can a cardiac-locked DSS component learned from one interval attenuate a known
planted cardiac artifact in independent held-out data while preserving the
unmodified MNE Sample EEG substrate?

The MNE Sample recording supplies realistic EEG data and R-peak timing. A
fixed rank-one QRS-like artifact is then planted at those detected peaks. The
cardiac-removal operator is fitted only on the contaminated training interval,
and the same fitted operator is applied to held-out contaminated data and to
the corresponding held-out reference. Cardiac locking is a bias criterion, not
an artifact identity label: real recordings can contain neural or other
activity correlated with the cardiac cycle
:footcite:p:`sarela2005_dss,decheveigne_simon2008_spatial`.

References
----------
.. footbibliography::
"""

# %%
# Load the MNE Sample substrate and detect real R peaks
# -----------------------------------------------------
import mne
import numpy as np
from mne.datasets import sample

from mne_denoise.dss import DSS, CycleAverageBias
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_evoked_gfp_comparison, plot_signal_overlay

sample_path = sample.data_path(update_path=False)
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=False,
    verbose="ERROR",
)
raw.crop(0.0, 60.0).load_data()

# Detect the cardiac timing while MEG and the recording's ECG channel, when
# available, are still present. No artificial ECG channel is constructed.
ecg_events, _, _ = mne.preprocessing.find_ecg_events(
    raw,
    ch_name=None,
    verbose=False,
)
detected_r_peak_count = len(ecg_events)

reference = raw.copy().pick("eeg", exclude="bads")
reference.filter(1.0, 40.0, verbose="ERROR")
reference.set_eeg_reference("average", projection=False, verbose="ERROR")
reference_data = reference.get_data()
sfreq = float(reference.info["sfreq"])
n_channels = len(reference.ch_names)

# %%
# Plant one rank-one QRS-like artifact at the detected peaks
# -----------------------------------------------------------
qrs_window = (-0.10, 0.20)
qrs_start = int(round(qrs_window[0] * sfreq))
qrs_stop = int(round(qrs_window[1] * sfreq))
qrs_offsets = np.arange(qrs_start, qrs_stop)
qrs_times = qrs_offsets / sfreq
qrs_template = np.exp(-0.5 * ((qrs_times + 0.025) / 0.014) ** 2)
qrs_template -= 0.65 * np.exp(-0.5 * ((qrs_times - 0.040) / 0.024) ** 2)
qrs_template -= qrs_template.mean()
qrs_template /= np.sqrt(np.mean(qrs_template**2))

rng = np.random.default_rng(18)
cardiac_topography = rng.standard_normal(n_channels)
cardiac_topography -= cardiac_topography.mean()
cardiac_topography /= np.linalg.norm(cardiac_topography)

cardiac_waveform = np.zeros(reference.n_times)
r_peak_samples = ecg_events[:, 0].astype(int)
r_peak_indices = r_peak_samples - int(reference.first_samp)
complete_peak_mask = (r_peak_indices + qrs_start >= 0) & (
    r_peak_indices + qrs_stop <= reference.n_times
)
r_peak_samples = r_peak_samples[complete_peak_mask]
r_peak_indices = r_peak_indices[complete_peak_mask]
if len(r_peak_samples) < 8:
    raise RuntimeError(
        "The cropped Sample recording must provide at least eight complete "
        "R-peak windows for temporal training and held-out evaluation."
    )
for peak_index in r_peak_indices:
    cardiac_waveform[peak_index + qrs_start : peak_index + qrs_stop] += qrs_template
unscaled_artifact = cardiac_topography[:, np.newaxis] * cardiac_waveform

reference_centered = reference_data - np.median(
    reference_data,
    axis=1,
    keepdims=True,
)
robust_reference_scale = 1.4826 * np.median(np.abs(reference_centered))
if not np.isfinite(robust_reference_scale) or robust_reference_scale <= 0.0:
    raise RuntimeError("The Sample EEG robust reference scale must be positive.")
requested_artifact_to_reference_ratio = 8.0
artifact_scale_factor = (
    requested_artifact_to_reference_ratio
    * robust_reference_scale
    / np.sqrt(np.mean(qrs_template**2))
)
artifact_data = unscaled_artifact * artifact_scale_factor
contaminated_data = reference_data + artifact_data

contaminated = mne.io.RawArray(
    contaminated_data,
    reference.info.copy(),
    first_samp=reference.first_samp,
    verbose=False,
)
contaminated.set_annotations(reference.annotations.copy())

# %%
# Split the recording in time before fitting
# -------------------------------------------
# The first 30 seconds are training data and the second 30 seconds are held
# out. A one-sample adjustment makes the two public Raw crops half-open.
split_time = 30.0
train_contaminated = contaminated.copy().crop(
    tmin=0.0,
    tmax=split_time - 1.0 / sfreq,
)
held_out_contaminated = contaminated.copy().crop(
    tmin=split_time,
    tmax=60.0 - 1.0 / sfreq,
)
held_out_reference = reference.copy().crop(
    tmin=split_time,
    tmax=60.0 - 1.0 / sfreq,
)

split_sample = int(raw.first_samp + round(split_time * sfreq))
train_events = r_peak_samples[r_peak_samples < split_sample]
held_out_events = r_peak_samples[r_peak_samples >= split_sample]
if min(len(train_events), len(held_out_events)) < 4:
    raise RuntimeError(
        "The Sample crop must provide at least four complete R peaks in both "
        "the training and held-out intervals."
    )

train_contaminated = train_contaminated.copy().pick("eeg", exclude="bads")
held_out_contaminated = held_out_contaminated.copy().pick("eeg", exclude="bads")
held_out_reference = held_out_reference.copy().pick("eeg", exclude="bads")

bias = CycleAverageBias(
    event_samples=train_events,
    window=qrs_window,
    window_unit="seconds",
    sfreq=sfreq,
    event_origin="raw",
    first_samp=train_contaminated.first_samp,
)
n_components = 6
n_select = 1
model = DSS(
    bias=bias,
    n_components=n_components,
    n_select=n_select,
    component_action="subtract",
    normalize_input=False,
    verbose=False,
)
model.fit(train_contaminated)

# One fit is used for both held-out substrates. The reference transformation
# makes the planted artifact difference identifiable after the same operator.
cleaned_held_out = model.transform(held_out_contaminated)
cleaned_held_out_reference = model.transform(held_out_reference)

# %%
# Evaluate the isolated planted artifact and clean-input preservation
# -------------------------------------------------------------------
artifact_before_data = held_out_contaminated.get_data() - held_out_reference.get_data()
artifact_after_data = (
    cleaned_held_out.get_data() - cleaned_held_out_reference.get_data()
)

artifact_before = mne.io.RawArray(
    artifact_before_data,
    held_out_contaminated.info.copy(),
    first_samp=held_out_contaminated.first_samp,
    verbose=False,
)
artifact_after = mne.io.RawArray(
    artifact_after_data,
    held_out_contaminated.info.copy(),
    first_samp=held_out_contaminated.first_samp,
    verbose=False,
)
artifact_before.set_annotations(held_out_contaminated.annotations.copy())
artifact_after.set_annotations(held_out_contaminated.annotations.copy())

held_out_epoch_events = np.column_stack(
    [
        held_out_events,
        np.zeros(len(held_out_events), dtype=int),
        np.ones(len(held_out_events), dtype=int),
    ]
)
artifact_before_epochs = mne.Epochs(
    artifact_before,
    held_out_epoch_events,
    event_id={"R peak": 1},
    tmin=qrs_window[0],
    tmax=qrs_window[1],
    baseline=None,
    preload=True,
    reject=None,
    verbose=False,
)
artifact_after_epochs = mne.Epochs(
    artifact_after,
    held_out_epoch_events,
    event_id={"R peak": 1},
    tmin=qrs_window[0],
    tmax=qrs_window[1],
    baseline=None,
    preload=True,
    reject=None,
    verbose=False,
)
artifact_before_evoked = artifact_before_epochs.average()
artifact_after_evoked = artifact_after_epochs.average()
artifact_before_rms = np.sqrt(np.mean(artifact_before_evoked.get_data() ** 2))
artifact_after_rms = np.sqrt(np.mean(artifact_after_evoked.get_data() ** 2))
artifact_residual_ratio = artifact_after_rms / artifact_before_rms
artifact_attenuation_db = 20.0 * np.log10(artifact_before_rms / artifact_after_rms)

reference_data = held_out_reference.get_data()
cleaned_reference_data = cleaned_held_out_reference.get_data()
reference_scale = np.sqrt(np.mean(reference_data**2))
clean_input_relative_rms_change = (
    rms_change(reference_data, cleaned_reference_data) / reference_scale
)
clean_input_waveform_correlation = float(
    np.corrcoef(reference_data.ravel(), cleaned_reference_data.ravel())[0, 1]
)
clean_input_retained_power = np.sum(cleaned_reference_data**2) / np.sum(
    reference_data**2
)

# This channel is predeclared from the planted topography, not selected from
# the held-out attenuation result.
cardiac_channel_index = int(np.argmax(np.abs(cardiac_topography)))
cardiac_channel = held_out_contaminated.ch_names[cardiac_channel_index]
representative_event_time = (
    int(held_out_events[0]) - int(held_out_contaminated.first_samp)
) / sfreq

print("Held-out cardiac DSS")
print("Sample crop: 0.0-60.0 s")
print(f"Detected R-peak count: {detected_r_peak_count}")
print(f"Usable R-peak count: {len(r_peak_samples)}")
print(f"Training R-peak count: {len(train_events)}")
print(f"Held-out R-peak count: {len(held_out_events)}")
print(f"EEG channel count: {n_channels}")
print("Planted artifact rank: 1")
print(
    "Artifact amplitude scaling: "
    f"{requested_artifact_to_reference_ratio:.1f} x robust EEG scale "
    "(1.4826 x global median absolute deviation)"
)
print(f"n_components: {n_components}")
print(f"n_select: {n_select} (rank-one planted artifact)")
print(f"QRS-locked planted-artifact RMS before: {artifact_before_rms:.6e}")
print(f"QRS-locked planted-artifact RMS after:  {artifact_after_rms:.6e}")
print(f"Planted-artifact residual ratio: {artifact_residual_ratio:.4f}")
print(f"Planted-artifact attenuation: {artifact_attenuation_db:.2f} dB")
print(f"Clean-input relative RMS change: {clean_input_relative_rms_change:.4f}")
print(f"Clean-input waveform correlation: {clean_input_waveform_correlation:.4f}")
print(f"Clean-input retained-power ratio: {clean_input_retained_power:.4f}")
print(
    "Artifact endpoint: (contaminated - reference) versus "
    "(cleaned contaminated - cleaned reference)."
)
print(
    "Preservation endpoint: the operator's change to held-out unmodified "
    "Sample EEG, not clean-neural ground truth."
)
print(
    "Cardiac locking is a bias criterion, not an artifact identity label; real "
    "recordings can contain neural or other cardiac-correlated activity."
)

# %%
# Inspect the isolated artifact and a cardiac-dominated EEG channel
# -----------------------------------------------------------------
plot_evoked_gfp_comparison(
    artifact_before_evoked,
    artifact_after_evoked,
    times=artifact_before_evoked.times,
    ci=None,
    labels=("planted cardiac artifact", "residual after DSS"),
    x_label="Time from R peak (s)",
    y_label="Sensor RMS (a.u.)",
    title="Held-out planted cardiac artifact around real R peaks",
    show=False,
)

plot_signal_overlay(
    held_out_contaminated,
    cleaned_held_out,
    held_out_contaminated.times,
    pick=cardiac_channel,
    start=max(0.0, representative_event_time - 0.35),
    stop=min(held_out_contaminated.times[-1], representative_event_time + 0.50),
    scale_after=False,
    before_label="held-out contaminated",
    after_label="DSS cleaned",
    reference=reference_data[cardiac_channel_index],
    reference_label="held-out Sample EEG reference",
    highlight_spans=[
        {
            "onset": representative_event_time + qrs_window[0],
            "duration": qrs_window[1] - qrs_window[0],
            "label": "QRS-locked evaluation window",
        }
    ],
    x_label="Time in held-out interval (s)",
    y_label="EEG amplitude (V)",
    title=f"Cardiac-locked subtraction at {cardiac_channel}",
    show=False,
)

# %%
# Interpretation
# --------------
# R-peak timing comes from the real MNE Sample recording, while the EEG
# substrate is real but the added rank-one cardiac artifact is controlled. The
# difference signals isolate that planted artifact before and after the same
# fitted operator. Cardiac locking is only the DSS bias criterion, so a real
# cardiac-locked component can include neural or other activity correlated with
# the cardiac cycle. The preservation values quantify how the learned operator
# changes the unmodified held-out Sample EEG substrate; they do not establish
# preservation of a noise-free neural ground truth.
