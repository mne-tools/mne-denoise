r"""
Adaptive ASR for changing recording statistics
==============================================

Can adaptive calibration follow a change in the clean recording statistics
while continuing to suppress transient artifacts? This controlled example
contains a baseline calibration chunk, a changed-regime adaptation chunk, and
an independent changed-regime test chunk. Frozen ASR is compared with
``AdaptiveASR`` after one adaptation chunk.

The quiet changed-regime test data are the preservation endpoint. Error in the
independent test burst windows is the artifact endpoint. The comparison
illustrates the streaming ``fit`` / ``partial_fit`` / ``transform`` lifecycle
on one controlled substrate; it is not a benchmark of adaptive variants.

The use case is motivated by adaptive ASR research
:footcite:p:`tsai2024_adaptive_asr` and the standard ASR evaluation
:footcite:p:`chang2020_asr`.

References
----------
.. footbibliography::
"""

# %%
# Create independent baseline, adaptation, and test chunks
# ---------------------------------------------------------
import numpy as np

from mne_denoise.asr import ASR, AdaptiveASR
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_signal_overlay

rng = np.random.default_rng(17)
sfreq = 200.0
n_channels = 8
segment_seconds = 12.0
segment_samples = int(round(segment_seconds * sfreq))
test_times = np.arange(segment_samples) / sfreq

mixing = rng.standard_normal((n_channels, 3))
mixing /= np.linalg.norm(mixing, axis=0, keepdims=True)


def make_regime(rng, mixing, time, amplitudes, noise_scale):
    """Generate one independent realization of a covariance regime."""
    frequencies = (10.0, 6.0, 0.8)
    sources = np.asarray(
        [
            amplitude
            * np.sin(2.0 * np.pi * frequency * time + rng.uniform(0.0, 2.0 * np.pi))
            + 0.05 * rng.standard_normal(time.size)
            for frequency, amplitude in zip(frequencies, amplitudes)
        ]
    )
    sensor_noise = noise_scale * rng.standard_normal((mixing.shape[0], time.size))
    return mixing @ sources + sensor_noise


baseline = make_regime(
    rng,
    mixing,
    test_times,
    amplitudes=(0.35, 0.20, 0.08),
    noise_scale=0.05,
)
adaptation = make_regime(
    rng,
    mixing,
    test_times,
    amplitudes=(0.80, 0.45, 0.20),
    noise_scale=0.12,
)
test_clean = make_regime(
    rng,
    mixing,
    test_times,
    amplitudes=(0.80, 0.45, 0.20),
    noise_scale=0.12,
)

# Artifacts are injected only into the independent test chunk.
test_contaminated = test_clean.copy()
artifact_mask = np.zeros(segment_samples, dtype=bool)
artifact_spatial = rng.standard_normal(n_channels)
artifact_spatial /= np.linalg.norm(artifact_spatial)
for onset in (2.0, 6.0, 10.0):
    start = int(round(onset * sfreq))
    stop = min(segment_samples, start + int(round(0.6 * sfreq)))
    artifact_mask[start:stop] = True
    artifact_source = rng.standard_normal(stop - start)
    test_contaminated[:, start:stop] += 5.0 * np.outer(
        artifact_spatial, artifact_source
    )

# The adaptation and test chunks are separate calls to ``make_regime`` with
# the same changed-regime parameters, so they have the same regime but
# independent phases, noise, and time-series realizations.

# %%
# Calibrate once, adapt once, and evaluate only on the independent test chunk
# ---------------------------------------------------------------------------
# AdaptiveASR uses unfiltered statistics internally. Match that setting in
# the frozen comparator so the comparison isolates the calibration update.
frozen = ASR(
    sfreq=sfreq,
    cutoff=20.0,
    calibration="manual",
    filter_kind="none",
    picks=None,
    verbose=False,
)
frozen.fit(baseline)
frozen_clean = np.asarray(frozen.transform(test_contaminated))

adaptive = AdaptiveASR(
    sfreq=sfreq,
    cutoff=20.0,
    variant="psw",
    picks=None,
    verbose=False,
)
adaptive.fit(baseline)
adaptive.partial_fit(adaptation)
adaptive_clean = np.asarray(adaptive.transform(test_contaminated))

artifact_before = rms_change(
    test_contaminated[:, artifact_mask],
    test_clean[:, artifact_mask],
)
frozen_artifact_after = rms_change(
    frozen_clean[:, artifact_mask],
    test_clean[:, artifact_mask],
)
adaptive_artifact_after = rms_change(
    adaptive_clean[:, artifact_mask],
    test_clean[:, artifact_mask],
)
frozen_artifact_residual_ratio = frozen_artifact_after / artifact_before
adaptive_artifact_residual_ratio = adaptive_artifact_after / artifact_before

quiet_mask = ~artifact_mask
quiet_scale = np.sqrt(np.mean(test_clean[:, quiet_mask] ** 2))
frozen_quiet_error = (
    rms_change(
        frozen_clean[:, quiet_mask],
        test_clean[:, quiet_mask],
    )
    / quiet_scale
)
adaptive_quiet_error = (
    rms_change(
        adaptive_clean[:, quiet_mask],
        test_clean[:, quiet_mask],
    )
    / quiet_scale
)

print(
    "Frozen ASR: "
    f"artifact residual ratio={frozen_artifact_residual_ratio:.3f}, "
    f"quiet test error={frozen_quiet_error:.3f}"
)
print(
    "Adaptive ASR: "
    f"artifact residual ratio={adaptive_artifact_residual_ratio:.3f}, "
    f"quiet test error={adaptive_quiet_error:.3f}"
)
print("Adaptation chunks supplied: 1")

# %%
# Compare frozen and adaptive outputs on the independent test chunk
# -----------------------------------------------------------------
channel = int(np.argmax(np.abs(artifact_spatial)))
plot_signal_overlay(
    frozen_clean,
    adaptive_clean,
    test_times,
    pick=channel,
    scale_after=False,
    before_label="frozen ASR",
    after_label="adaptive ASR",
    reference=test_clean[channel],
    reference_label="clean substrate",
    highlight_mask=artifact_mask,
    highlight_label="artifact",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="Independent changed-regime test chunk",
    show=False,
)
