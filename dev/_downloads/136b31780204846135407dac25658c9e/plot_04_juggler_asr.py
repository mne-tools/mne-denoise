r"""
Juggler ASR when clean windows are scarce
=========================================

How does reference selection behave when frequent short artifacts leave few
fully clean calibration windows? This controlled recording stresses the
window-based calibration assumption behind standard ASR and compares it with
the public ``JugglerASR(strategy="dbscan")`` estimator, which selects
reference samples point by point.

The known burst mask provides the artifact endpoint. Recovery toward the clean
substrate outside those bursts provides the preservation endpoint. The
reference fractions make the calibration decision visible. This substrate
illustrates the regime that motivates Juggler ASR; it is not a benchmark of
all ASR variants.

The use case is motivated by Juggler's ASR
:footcite:p:`kim2025_juggler_asr` and the standard ASR calibration
evaluation of :footcite:p:`chang2020_asr`.

References
----------
.. footbibliography::
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Construct a densely contaminated recording
# -------------------------------------------
import numpy as np

from mne_denoise.asr import ASR, JugglerASR
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_asr_calibration_fraction, plot_signal_overlay

rng = np.random.default_rng(21)
sfreq = 250.0
duration = 14.0
n_channels = 8
n_times = int(round(duration * sfreq))
times = np.arange(n_times) / sfreq

clean = np.empty((n_channels, n_times), dtype=float)
for channel in range(n_channels):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    clean[channel] = (
        0.45 * np.sin(2.0 * np.pi * 10.0 * times + phase)
        + 0.18 * np.sin(2.0 * np.pi * 6.0 * times + 0.4 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

contaminated = clean.copy()
burst_mask = np.zeros(n_times, dtype=bool)
artifact_spatial = rng.standard_normal((n_channels, 2))
artifact_spatial /= np.linalg.norm(artifact_spatial, axis=0, keepdims=True)
for onset in np.arange(3.0, 11.5, 0.28):
    start = int(round(onset * sfreq))
    stop = min(n_times, start + int(round(0.10 * sfreq)))
    burst_mask[start:stop] = True
    burst_source = 7.0 * rng.standard_normal((2, stop - start))
    contaminated[:, start:stop] += artifact_spatial @ burst_source

# %%
# Compare standard and Juggler reference selection
# ------------------------------------------------
standard = ASR(
    sfreq=sfreq,
    cutoff=20.0,
    picks=None,
    verbose=False,
)
juggler = JugglerASR(
    sfreq=sfreq,
    cutoff=20.0,
    strategy="dbscan",
    picks=None,
    verbose=False,
)

cleaned_standard = np.asarray(standard.fit_transform(contaminated))
cleaned_juggler = np.asarray(juggler.fit_transform(contaminated))

# The public mask accessor returns each estimator's native representation:
# clean windows for standard ASR and reference samples for Juggler ASR.
standard_clean_window_fraction = standard.get_calibration_mask().mean()
juggler_reference_sample_fraction = juggler.get_calibration_mask().mean()
# These percentages summarize each method in its native calibration unit;
# they describe retained calibration support rather than directly comparable
# sample fractions.

artifact_before = rms_change(
    contaminated[:, burst_mask],
    clean[:, burst_mask],
)
standard_artifact_after = rms_change(
    cleaned_standard[:, burst_mask],
    clean[:, burst_mask],
)
juggler_artifact_after = rms_change(
    cleaned_juggler[:, burst_mask],
    clean[:, burst_mask],
)
standard_artifact_residual_ratio = standard_artifact_after / artifact_before
juggler_artifact_residual_ratio = juggler_artifact_after / artifact_before

quiet_mask = ~burst_mask
quiet_scale = np.sqrt(np.mean(clean[:, quiet_mask] ** 2))
standard_quiet_relative_error = (
    rms_change(
        cleaned_standard[:, quiet_mask],
        clean[:, quiet_mask],
    )
    / quiet_scale
)
juggler_quiet_relative_error = (
    rms_change(
        cleaned_juggler[:, quiet_mask],
        clean[:, quiet_mask],
    )
    / quiet_scale
)

print(
    "Standard ASR clean-window fraction: "
    f"{standard_clean_window_fraction:.1%}, "
    f"artifact residual ratio={standard_artifact_residual_ratio:.3f}, "
    f"quiet relative error={standard_quiet_relative_error:.3f}"
)
print(
    "Juggler ASR reference-sample fraction: "
    f"{juggler_reference_sample_fraction:.1%}, "
    f"artifact residual ratio={juggler_artifact_residual_ratio:.3f}, "
    f"quiet relative error={juggler_quiet_relative_error:.3f}"
)

# %%
# Make the reference-fraction comparison visible
# -----------------------------------------------
plot_asr_calibration_fraction(
    [standard, juggler],
    labels=[
        "Standard ASR\n(clean windows)",
        "Juggler ASR\n(reference samples)",
    ],
    title="Calibration support retained by each method",
    show=False,
)

# %%
# Inspect reconstruction of the dense bursts
# --------------------------------------------
channel = int(np.argmax(np.max(np.abs(artifact_spatial), axis=1)))
plot_signal_overlay(
    cleaned_standard,
    cleaned_juggler,
    times,
    pick=channel,
    scale_after=False,
    before_label="standard ASR",
    after_label="Juggler ASR",
    reference=clean[channel],
    reference_label="clean substrate",
    highlight_mask=burst_mask,
    highlight_label="artifact",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="Dense short-burst reconstruction",
    show=False,
)
