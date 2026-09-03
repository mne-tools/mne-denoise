r"""
Removing a high-amplitude single-channel artifact with local SSA
=================================================================

Can local SSA remove a large EOG-like waveform from a single-channel EEG-like
signal while preserving a known neural transient? The transient deliberately
overlaps the artifact so that recovery and preservation are evaluated as
separate endpoints.

Local SSA targets high-amplitude artifacts in single-channel EEG by clustering
delay-coordinate vectors and reconstructing local PCA subspaces
:footcite:p:`teixeira2006_local_ssa`. This is a controlled methodological
illustration, not a replication of the paper's clinical EEG, and the synthetic
transient is not a clinical epileptiform model.

Local SSA treats high-energy local subspace structure as artifact-like. Genuine
neural activity that occupies the same structure can also be removed, which is
the central limitation evaluated here.

References
----------
.. footbibliography::
"""

# %%
# Construct a single-channel EEG-like signal with an overlapping blink waveform
# ------------------------------------------------------------------------------
import numpy as np

from mne_denoise.qa import rms_change
from mne_denoise.ssa import LocalSingularSpectrumAnalysis
from mne_denoise.viz import plot_signal_overlay

sfreq = 128.0
times = np.arange(int(8.0 * sfreq)) / sfreq
rng = np.random.default_rng(20260902)

background = 0.18 * rng.standard_normal(times.size) + 0.12 * np.sin(
    2.0 * np.pi * 6.0 * times
)
neural_transient = (
    1.0
    * np.exp(-0.5 * ((times - 3.15) / 0.09) ** 2)
    * np.sin(2.0 * np.pi * 12.0 * (times - 3.15))
)
clean_reference = background + neural_transient

# A smooth asymmetric biphasic waveform is a compact EOG/blink-like artifact.
eog_artifact = 5.0 * (
    np.exp(-0.5 * ((times - 2.8) / 0.22) ** 2)
    - 0.55 * np.exp(-0.5 * ((times - 3.25) / 0.34) ** 2)
)
observed = clean_reference + eog_artifact
artifact_mask = (times >= 2.0) & (times <= 4.2)
transient_mask = (times >= 2.9) & (times <= 3.4)

# %%
# Apply local SSA and evaluate artifact recovery and transient preservation
# -------------------------------------------------------------------------
cleaner = LocalSingularSpectrumAnalysis(sfreq=sfreq, verbose=False)
cleaned = cleaner.fit_transform(observed[np.newaxis])[0]

artifact_error_before = rms_change(
    observed[artifact_mask], clean_reference[artifact_mask]
)
artifact_error_after = rms_change(
    cleaned[artifact_mask], clean_reference[artifact_mask]
)
artifact_residual_ratio = artifact_error_after / artifact_error_before

template = neural_transient[transient_mask]
template_energy = np.dot(template, template)
reference_transient_gain = (
    np.dot(clean_reference[transient_mask], template) / template_energy
)
cleaned_transient_gain = np.dot(cleaned[transient_mask], template) / template_energy
transient_gain = cleaned_transient_gain / reference_transient_gain
transient_correlation = np.corrcoef(
    cleaned[transient_mask] - background[transient_mask],
    template,
)[0, 1]

effective_cluster_count = int(np.asarray(cleaner.n_clusters_).ravel()[0])
selected_dimensions = np.asarray(cleaner.subspace_dimensions_[0], dtype=int)
print(f"Artifact residual ratio: {artifact_residual_ratio:.3f}")
print(f"Neural-transient gain: {transient_gain:.3f}")
print(f"Neural-transient correlation: {transient_correlation:.3f}")
print(f"Effective local cluster count: {effective_cluster_count}")
print(f"Selected local subspace dimensions: {selected_dimensions}")

# %%
# Inspect the artifact/transient interval
# ----------------------------------------
plot_signal_overlay(
    observed,
    cleaned,
    times,
    start=2.0,
    stop=4.2,
    reference=clean_reference,
    reference_label="known clean reference",
    highlight_mask=artifact_mask,
    highlight_label="EOG-like artifact window",
    highlight_spans=[
        {
            "onset": 2.9,
            "duration": 0.5,
            "color": "C4",
            "alpha": 0.12,
            "label": "neural transient window",
        }
    ],
    scale_after=False,
    before_label="observed",
    after_label="Local SSA residual",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="Local SSA around an EOG-like artifact and neural transient",
    show=False,
)

# %%
# Interpret the overlap stress test
# ----------------------------------
# In this deliberately difficult overlapping case, Local SSA strongly reduces
# the high-amplitude EOG-like artifact but also substantially attenuates the
# known neural transient. This is the intended preservation stress test:
# high-amplitude neural structure that overlaps the learned local artifact
# subspaces is not guaranteed to survive.
