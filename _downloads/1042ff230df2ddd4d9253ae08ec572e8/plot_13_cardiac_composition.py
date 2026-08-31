"""
Cardiac DSS as an Explicit Composition
======================================

This example keeps cardiac cleaning visible as three public operations:

1. detect QRS events with MNE;
2. define a fixed-window :class:`~mne_denoise.dss.CycleAverageBias`; and
3. fit :class:`~mne_denoise.dss.DSS` with ``component_action="subtract"``.

The synthetic recording supplies isolated neural ground truth so attenuation
and preservation can both be checked on held-out data. A reproducible
cardiac-locked component can contain neural signal as well as artifact, so the
choice to subtract one component below is a scientific decision, not an
automatic guarantee of safe cleaning.
"""

# Authors: mne-denoise developers

from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import find_ecg_events

from mne_denoise.dss import DSS, CycleAverageBias
from mne_denoise.viz import (
    plot_channel_time_course_comparison,
    plot_evoked_gfp_comparison,
)


def _event_locked_average(data, event_samples, first_samp, window):
    """Return the sensor average in one half-open QRS window."""
    events = np.asarray(event_samples, dtype=int) - int(first_samp)
    start, stop = window
    return np.mean(
        np.stack([data[:, event + start : event + stop] for event in events]),
        axis=0,
    )


# %%
# Make a deterministic MNE recording
# ----------------------------------
# The neural ground truth includes alpha activity and a slow oscillation at the
# 1.25 Hz heart rate. The latter deliberately overlaps the cardiac spectrum.

rng = np.random.default_rng(18)
sfreq = 200.0
duration = 60.0
n_times = int(sfreq * duration)
times = np.arange(n_times) / sfreq
first_samp = 1_000
n_eeg = 6

neural_sources = np.vstack(
    [
        np.sin(2 * np.pi * 10.0 * times),
        0.7 * np.sin(2 * np.pi * 1.25 * times + 0.4),
        0.4 * np.sin(2 * np.pi * 18.0 * times + 1.1),
    ]
)
neural_topographies = rng.normal(size=(n_eeg, len(neural_sources)))
neural = neural_topographies @ neural_sources
neural /= np.std(neural)

qrs_samples = np.arange(int(sfreq), n_times - int(sfreq), int(0.8 * sfreq))
qrs_times = np.arange(-20, 31)
qrs_shape = np.exp(-0.5 * (qrs_times / 3.0) ** 2)
qrs_shape -= 0.35 * np.exp(-0.5 * ((qrs_times - 8) / 5.0) ** 2)
ecg = np.zeros(n_times)
for sample in qrs_samples:
    ecg[sample - 20 : sample + 31] += qrs_shape

cardiac_topography = np.array([20.0, 14.0, -10.0, 7.0, -5.0, 3.0])
cardiac = np.outer(cardiac_topography, ecg)
background = 0.08 * rng.standard_normal((n_eeg, n_times))
mixture = neural + cardiac + background

ch_names = [f"EEG {index:03d}" for index in range(1, n_eeg + 1)] + ["ECG"]
info = mne.create_info(ch_names, sfreq, ["eeg"] * n_eeg + ["ecg"])
raw = mne.io.RawArray(
    np.vstack([mixture, ecg + 0.01 * rng.standard_normal(n_times)]),
    info,
    first_samp=first_samp,
    verbose=False,
)

# %%
# Detect events, split, and fit only on training data
# ---------------------------------------------------
# ``find_ecg_events`` returns acquisition-numbered event samples. Cropping an
# MNE Raw changes ``first_samp``, so each bias declares that origin explicitly.

train = raw.copy().crop(tmin=0.0, tmax=29.995)
held_out = raw.copy().crop(tmin=30.0, tmax=59.995)
train_events, _, _ = find_ecg_events(train, ch_name="ECG", verbose=False)
held_out_events, _, _ = find_ecg_events(held_out, ch_name="ECG", verbose=False)
train_eeg = train.copy().pick("eeg")
held_out_eeg = held_out.copy().pick("eeg")

window_seconds = (-0.15, 0.25)
window_samples = tuple(round(value * sfreq) for value in window_seconds)
bias = CycleAverageBias(
    event_samples=train_events[:, 0],
    window=window_seconds,
    window_unit="seconds",
    sfreq=sfreq,
    event_origin="raw",
    first_samp=train_eeg.first_samp,
)
cardiac_dss = DSS(
    bias=bias,
    rank=n_eeg,
    n_components=4,
    n_select=1,
    component_action="subtract",
    normalize_input=False,
)
cardiac_dss.fit(train_eeg)
cleaned = cardiac_dss.transform(held_out_eeg)

# %%
# Require attenuation and preservation together
# ---------------------------------------------
# The event bias is used only during ``fit``. ``transform`` is a frozen spatial
# operator and does not inspect the held-out event times. Those events are used
# below only to evaluate QRS-locked attenuation.

before_locked = _event_locked_average(
    held_out_eeg.get_data(),
    held_out_events[:, 0],
    held_out_eeg.first_samp,
    window_samples,
)
after_locked = _event_locked_average(
    cleaned.get_data(),
    held_out_events[:, 0],
    cleaned.first_samp,
    window_samples,
)
attenuation_db = 20 * np.log10(
    np.sqrt(np.mean(before_locked**2)) / np.sqrt(np.mean(after_locked**2))
)

held_slice = slice(int(30 * sfreq), n_times)
neural_info = mne.create_info(ch_names[:-1], sfreq, ["eeg"] * n_eeg)
neural_held_out = mne.io.RawArray(
    neural[:, held_slice],
    neural_info,
    first_samp=held_out_eeg.first_samp,
    verbose=False,
)
neural_after = cardiac_dss.transform(neural_held_out).get_data()
neural_before = neural_held_out.get_data()
retained_power = np.sum(neural_after**2) / np.sum(neural_before**2)
waveform_correlation = np.corrcoef(neural_before.ravel(), neural_after.ravel())[0, 1]

print(f"Held-out QRS-locked RMS attenuation: {attenuation_db:.2f} dB")
print(f"Held-out neural retained power: {100 * retained_power:.1f}%")
print(f"Held-out neural waveform correlation: {waveform_correlation:.4f}")

# These illustrative gates require artifact attenuation and neural preservation
# simultaneously. They are not universal acceptance criteria for real studies.
passes_example_gates = (
    attenuation_db >= 6.0 and retained_power >= 0.8 and waveform_correlation >= 0.95
)
print(f"Passes the predeclared illustrative gates: {passes_example_gates}")

# %%
# Inspect the held-out event-locked result
# ----------------------------------------

locked_times = np.arange(window_samples[0], window_samples[1]) / sfreq
plot_evoked_gfp_comparison(
    before_locked,
    after_locked,
    times=locked_times,
    ci=None,
    labels=("Unchanged", "Frozen DSS subtraction"),
    x_label="Time from QRS (s)",
    y_label="Sensor RMS (a.u.)",
    title="Held-out QRS-locked global field power",
    show=True,
)

# %%
# Individual sensor traces make it possible to see whether attenuation is
# spatially concentrated or indiscriminate. This uses the same package-level
# before/after visualization API as the other denoising examples.

plot_channel_time_course_comparison(
    before_locked,
    after_locked,
    picks=range(min(3, n_eeg)),
    times=locked_times,
    before_label="Unchanged",
    after_label="Frozen DSS subtraction",
    x_label="Time from QRS (s)",
    show=True,
)
