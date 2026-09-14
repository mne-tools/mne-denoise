r"""
Repairing transient EEG bursts with ASR
=======================================

Can standard Artifact Subspace Reconstruction (ASR) reduce short,
high-amplitude multichannel bursts while preserving the underlying clean
signal? A controlled recording answers both parts of that question because
the clean substrate and artifact mask are known.

The clean calibration segment is supplied separately from the contaminated
recording. We evaluate artifact attenuation and quiet-signal preservation as
separate endpoints. A cutoff of 20 is one representative setting here; it is
not a universal optimum, and suitable values depend on the dataset and the
scientific endpoint.

This use case is motivated by standard ASR and its parameter evaluation
:footcite:p:`kothe_jung2016_asr,chang2018_asr,chang2020_asr`.

References
----------
.. footbibliography::
"""

# %%
# Construct a controlled multichannel recording
# ----------------------------------------------
import numpy as np

from mne_denoise.asr import ASR
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_asr_repair_timeline, plot_signal_overlay

rng = np.random.default_rng(42)
sfreq = 250.0
duration = 12.0
n_channels = 8
n_times = int(round(sfreq * duration))
times = np.arange(n_times) / sfreq

clean = np.empty((n_channels, n_times), dtype=float)
for channel in range(n_channels):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    clean[channel] = (
        0.5 * np.sin(2.0 * np.pi * 10.0 * times + phase)
        + 0.2 * np.sin(2.0 * np.pi * 6.0 * times + 0.5 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

artifact_mask = np.zeros(n_times, dtype=bool)
spatial = rng.standard_normal((n_channels, 2))
spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
contaminated = clean.copy()
for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
    start = int(round(onset * sfreq))
    stop_sample = int(round(stop * sfreq))
    artifact_mask[start:stop_sample] = True
    burst = 8.0 * rng.standard_normal((2, stop_sample - start))
    contaminated[:, start:stop_sample] += spatial @ burst

# The first three seconds are clean calibration data, separate from the
# contaminated recording that will be transformed.
calibration = clean[:, : int(round(3.0 * sfreq))]

# %%
# Fit standard ASR and evaluate the two scientific endpoints
# -----------------------------------------------------------
asr = ASR(
    sfreq=sfreq,
    cutoff=20.0,
    calibration="manual",
    picks=None,
    verbose=False,
)
cleaned = np.asarray(asr.fit_transform(contaminated, calibration=calibration))

artifact_before = rms_change(
    contaminated[:, artifact_mask],
    clean[:, artifact_mask],
)
artifact_after = rms_change(
    cleaned[:, artifact_mask],
    clean[:, artifact_mask],
)
artifact_residual_ratio = artifact_after / artifact_before

quiet_mask = ~artifact_mask
quiet_error = rms_change(
    cleaned[:, quiet_mask],
    clean[:, quiet_mask],
)
quiet_scale = np.sqrt(np.mean(clean[:, quiet_mask] ** 2))
quiet_relative_error = quiet_error / quiet_scale

diagnostics = asr.get_diagnostics()
print(f"Artifact residual ratio: {artifact_residual_ratio:.3f}")
print(f"Quiet relative error: {quiet_relative_error:.3f}")
print(
    f"Reconstructed samples/windows: "
    f"{diagnostics['fraction_reconstructed_samples']:.1%} / "
    f"{diagnostics['fraction_reconstructed_windows']:.1%}"
)

# %%
# Inspect a representative repaired burst
# -----------------------------------------
channel = int(np.argmax(np.linalg.norm(spatial, axis=1)))
plot_signal_overlay(
    contaminated,
    cleaned,
    times,
    pick=channel,
    start=3.4,
    stop=5.4,
    scale_after=False,
    before_label="contaminated",
    after_label="ASR cleaned",
    reference=clean[channel],
    reference_label="clean substrate",
    highlight_mask=artifact_mask,
    highlight_label="artifact",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="Transient burst reconstruction",
    show=False,
)
plot_asr_repair_timeline(
    asr,
    title="Standard ASR repair timeline",
    show=False,
)
