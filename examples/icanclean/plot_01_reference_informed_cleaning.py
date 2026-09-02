r"""
Reference-informed cleaning with iCanClean
==========================================

Can a reference channel help remove an artifact shared with the primary
recording, and what happens when the reference also contains signal of
interest? This controlled MNE Raw construction contains a known 10-Hz target
signal, a separate broadband artifact, and two otherwise comparable reference
conditions.

iCanClean uses canonical correlation to identify variance shared between the
primary and reference blocks. Shared variance is not automatically artifact:
the leaky-reference condition deliberately places some target signal in the
reference so that the CCA assumption can be evaluated directly. This controlled
demonstration is not a claim that the cited papers performed this exact
experiment.

The use case is motivated by
:footcite:p:`downey_ferris2022_icanclean,downey_ferris2023_icanclean_phantom`.

References
----------
.. footbibliography::
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Construct a controlled primary recording and two references
# ------------------------------------------------------------
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from mne_denoise.icanclean import ICanClean
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_signal_overlay

rng = np.random.default_rng(2023)
sfreq = 250.0
duration = 20.0
n_primary = 8
n_times = int(round(duration * sfreq))
times = np.arange(n_times) / sfreq

primary_names = [f"EEG {index:03d}" for index in range(1, n_primary + 1)]
ref_names = ["REF 001"]
info = mne.create_info(
    primary_names + ref_names,
    sfreq,
    ch_types=["eeg"] * n_primary + ["misc"],
)

# A known 10-Hz spatial component is the signal of interest.
target_topography = np.array([1.0, 0.8, 0.4, -0.2, -0.6, -0.9, -0.7, -0.3])
target_topography /= np.linalg.norm(target_topography)
target_timecourse = np.sin(2.0 * np.pi * 10.0 * times)
target_template = target_topography[:, None] * target_timecourse[None, :]

# Add two other neural-like sources whose spatial patterns are orthogonal to
# the target pattern, so target projection changes remain interpretable.
other_topographies = rng.standard_normal((n_primary, 2))
other_topographies -= target_topography[:, None] * (
    target_topography @ other_topographies
)
other_topographies, _ = np.linalg.qr(other_topographies)
other_timecourses = np.vstack(
    [
        0.7 * np.sin(2.0 * np.pi * 6.0 * times + 0.35),
        0.4 * np.sin(2.0 * np.pi * 18.0 * times - 0.4),
    ]
)
clean_primary = target_template + other_topographies @ other_timecourses
clean_primary += 0.03 * rng.standard_normal(clean_primary.shape)

# The broadband artifact occupies a spatial direction separate from the known
# clean sources and is active only in two tapered time windows.
artifact_topography = np.array([0.9, 0.6, 0.3, -0.2, -0.5, -0.8, -0.7, -0.4])
artifact_topography -= target_topography * (target_topography @ artifact_topography)
artifact_topography -= other_topographies @ (other_topographies.T @ artifact_topography)
artifact_topography /= np.linalg.norm(artifact_topography)

artifact_filter = butter(
    4,
    (20.0, 80.0),
    btype="bandpass",
    fs=sfreq,
    output="sos",
)
artifact_timecourse = sosfiltfilt(
    artifact_filter,
    rng.standard_normal(n_times),
)
artifact_timecourse /= np.std(artifact_timecourse)

artifact_mask = np.zeros(n_times, dtype=bool)
artifact_envelope = np.zeros(n_times)
for start_seconds, stop_seconds in ((3.0, 7.0), (12.0, 16.0)):
    start = int(round(start_seconds * sfreq))
    stop = int(round(stop_seconds * sfreq))
    artifact_mask[start:stop] = True
    artifact_envelope[start:stop] = np.hanning(stop - start)
artifact_timecourse *= artifact_envelope

artifact_amplitude = 2.0
artifact = artifact_amplitude * artifact_topography[:, None] * artifact_timecourse
contaminated_primary = clean_primary + artifact

# The good reference carries the artifact but no intentional target signal.
# The leaky reference differs only by a controlled amount of the target.
reference_artifact = artifact_amplitude * artifact_timecourse
reference_noise = 0.05 * rng.standard_normal(n_times)
target_leak_amplitude = 0.6
good_reference = reference_artifact + reference_noise
leaky_reference = (
    reference_artifact + target_leak_amplitude * target_timecourse + reference_noise
)

raw_good = mne.io.RawArray(
    np.vstack([contaminated_primary, good_reference]),
    info.copy(),
    verbose=False,
)
raw_leaky = mne.io.RawArray(
    np.vstack([contaminated_primary, leaky_reference]),
    info.copy(),
    verbose=False,
)

# %%
# Clean both reference conditions with the same public estimator settings
# ------------------------------------------------------------------------
# ``mode="global"`` keeps the example focused on reference information rather
# than windowing behavior. Sliding, calibrated, and hybrid modes are available
# in the API documentation but are intentionally not compared here.
good_model = ICanClean(
    sfreq=sfreq,
    ref_channels=ref_names,
    primary_channels=primary_names,
    mode="global",
    threshold=0.7,
    clean_with="X",
    verbose=False,
)
good_clean = good_model.fit_transform(raw_good)

leaky_model = ICanClean(
    sfreq=sfreq,
    ref_channels=ref_names,
    primary_channels=primary_names,
    mode="global",
    threshold=0.7,
    clean_with="X",
    verbose=False,
)
leaky_clean = leaky_model.fit_transform(raw_leaky)

good_clean_primary = good_clean.get_data(picks=primary_names)
leaky_clean_primary = leaky_clean.get_data(picks=primary_names)

# %%
# Evaluate reconstruction and target-signal preservation
# -------------------------------------------------------
artifact_before = rms_change(
    contaminated_primary,
    clean_primary,
)
good_error = rms_change(
    good_clean_primary,
    clean_primary,
)
leaky_error = rms_change(
    leaky_clean_primary,
    clean_primary,
)
good_residual_ratio = good_error / artifact_before
leaky_residual_ratio = leaky_error / artifact_before

target_denominator = np.sum(target_template**2)
target_clean = np.sum(clean_primary * target_template) / target_denominator
target_good = np.sum(good_clean_primary * target_template) / target_denominator
target_leaky = np.sum(leaky_clean_primary * target_template) / target_denominator
good_target_retention = target_good / target_clean
leaky_target_retention = target_leaky / target_clean

print(f"Good-reference components removed: {int(np.sum(good_model.n_removed_))}")
print(f"Leaky-reference components removed: {int(np.sum(leaky_model.n_removed_))}")
print(f"Good-reference residual/reconstruction ratio: {good_residual_ratio:.3f}")
print(f"Leaky-reference residual/reconstruction ratio: {leaky_residual_ratio:.3f}")
print(f"Good-reference target retention: {good_target_retention:.3f}")
print(f"Leaky-reference target retention: {leaky_target_retention:.3f}")
print(
    "Good-reference squared canonical correlations: "
    f"{np.round(good_model.correlations_.ravel(), 3)}"
)
print(
    "Leaky-reference squared canonical correlations: "
    f"{np.round(leaky_model.correlations_.ravel(), 3)}"
)

# %%
# Inspect the good-reference reconstruction
# ------------------------------------------
artifact_channel_index = int(
    np.argmax(np.linalg.norm(artifact[:, artifact_mask], axis=1))
)
artifact_channel = primary_names[artifact_channel_index]
plot_signal_overlay(
    raw_good,
    good_clean,
    times,
    pick=artifact_channel,
    start=2.5,
    stop=7.5,
    scale_after=False,
    before_label="contaminated primary",
    after_label="good-reference cleaned",
    reference=clean_primary[artifact_channel_index],
    reference_label="known clean primary",
    highlight_mask=artifact_mask,
    highlight_label="artifact window",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title=f"Good reference removes shared artifact at {artifact_channel}",
    show=False,
)

# %%
# Inspect the reference-leakage control
# -------------------------------------
target_channel_index = int(np.argmax(np.abs(target_topography)))
target_channel = primary_names[target_channel_index]
plot_signal_overlay(
    good_clean,
    leaky_clean,
    times,
    pick=target_channel,
    start=8.0,
    stop=10.0,
    scale_after=False,
    before_label="good-reference cleaned",
    after_label="leaky-reference cleaned",
    reference=clean_primary[target_channel_index],
    reference_label="known clean primary",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="Reference leakage can remove shared signal of interest",
    show=False,
)
