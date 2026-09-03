r"""
Suppressing sensor-specific noise with SNS
==========================================

Can SNS recover channels contaminated by independent sensor noise while making
relatively small changes to channels that were not deliberately corrupted?
This example uses an MNE Sample EEG recording to make the spatial covariance
realistic, then adds known broadband noise to two sensors.

SNS relies on the signal of interest being spatially redundant across sensors
while the targeted noise is specific to individual sensors. It is therefore
not a generic artifact-removal method: a source or artifact shared across the
array does not satisfy the assumption. The unmodified filtered recording is a
reference substrate for this controlled comparison, not noise-free neural
ground truth.

The use case is motivated by :footcite:p:`decheveigne_simon2008_sensor`.

References
----------
.. footbibliography::
"""

# %%
# Load a realistic EEG substrate
# ------------------------------
import mne
import numpy as np

from mne_denoise.progress import ProgressEvent
from mne_denoise.qa import rms_change
from mne_denoise.sns import SNS
from mne_denoise.viz import plot_signal_overlay

sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=False,
    verbose="ERROR",
)
raw.pick("eeg", exclude="bads").crop(0.0, 20.0).load_data()
raw.resample(200.0, verbose="ERROR")
raw.filter(1.0, 45.0, verbose="ERROR")

# The untouched recording is the reference substrate for both endpoints below.
reference_data = raw.get_data()

# %%
# Plant independent sensor-specific noise
# ----------------------------------------
rng = np.random.default_rng(2008)
corrupted_data = reference_data.copy()
channel_scale = np.median(np.std(reference_data, axis=1))
corrupted_indices = np.array([5, reference_data.shape[0] - 6])
noise_multiplier = 3.0

for index in corrupted_indices:
    corrupted_data[index] += (
        noise_multiplier * channel_scale * rng.standard_normal(raw.n_times)
    )

corrupted = mne.io.RawArray(
    corrupted_data,
    raw.info.copy(),
    first_samp=raw.first_samp,
    verbose=False,
)
corrupted.set_annotations(raw.annotations.copy())
corrupted_channel_names = [raw.ch_names[index] for index in corrupted_indices]
untouched_indices = np.setdiff1d(np.arange(reference_data.shape[0]), corrupted_indices)

# %%
# Fit SNS and evaluate recovery and preservation
# -----------------------------------------------
sns = SNS(
    preserve_mean=True,
    verbose=False,
)
# In interactive use, this callback slot can receive
# mne_denoise.progress.TqdmProgress when the optional ``progress`` extra is
# installed.
progress_events: list[ProgressEvent] = []
cleaned = sns.fit_transform(corrupted, callback=progress_events.append)
cleaned_data = cleaned.get_data()

first_event = progress_events[0]
last_event = progress_events[-1]
print(
    "Progress callback: "
    f"{len(progress_events)} events, "
    f"{first_event.method}/{first_event.stage}, "
    f"final={last_event.current}/{last_event.total}"
)

corrupted_error = rms_change(
    corrupted_data[corrupted_indices],
    reference_data[corrupted_indices],
)
cleaned_error = rms_change(
    cleaned_data[corrupted_indices],
    reference_data[corrupted_indices],
)
reference_scale = np.sqrt(np.mean(reference_data[corrupted_indices] ** 2))
corrupted_relative_error = corrupted_error / reference_scale
cleaned_relative_error = cleaned_error / reference_scale

untouched_change = rms_change(
    cleaned_data[untouched_indices],
    reference_data[untouched_indices],
)
untouched_scale = np.sqrt(np.mean(reference_data[untouched_indices] ** 2))
untouched_relative_change = untouched_change / untouched_scale

print(f"Deliberately corrupted channels: {corrupted_channel_names}")
print(f"Corrupted-channel error before SNS: {corrupted_relative_error:.3f}")
print(f"Corrupted-channel error after SNS:  {cleaned_relative_error:.3f}")
print(f"Untouched-channel relative change:  {untouched_relative_change:.3f}")
print(f"Effective neighbor count:           {sns.n_neighbors_}")

# %%
# Inspect one reconstructed sensor
# ---------------------------------
channel_index = int(corrupted_indices[0])
plot_signal_overlay(
    corrupted,
    cleaned,
    raw.times,
    pick=corrupted_channel_names[0],
    start=2.0,
    stop=4.0,
    scale_after=False,
    before_label="corrupted",
    after_label="SNS",
    reference=reference_data[channel_index],
    reference_label="unmodified recording",
    x_label="Time (s)",
    y_label="Amplitude (V)",
    title=f"Sensor-specific corruption at {corrupted_channel_names[0]}",
    show=False,
)
