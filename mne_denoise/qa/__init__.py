"""Quality-assurance metrics for attenuation, preservation, and deployment.

This module provides spectral QA metrics that quantify how well a
line-noise removal method cleans EEG data without introducing artefacts.
All PSD computations use the **geometric-mean PSD** (mean of log-PSDs
across channels via ``scipy.signal.welch``), matching the Zapline-plus
methodology (de Cheveigné, 2020).

Functions
---------
geometric_mean_psd
    Geometric-mean PSD across channels (scipy Welch).
noise_surr_ratio
    R(f₀) — ratio of peak power to surrounding floor.
peak_attenuation_db
    Attenuation (dB) at a target frequency.
below_noise_distortion
    Broadband spectral distortion outside line-noise bands.
overclean_proportion
    Fraction of surrounding spectrum that got over-suppressed.
underclean_proportion
    Whether the line-noise peak is still prominent.
compute_all_qa_metrics
    Compute all QA metrics per harmonic and return a summary dict.
"""

import importlib.util
import pathlib

from .metrics import (
    below_noise_distortion,
    geometric_mean_psd,
    noise_surr_ratio,
)

# ``mne_denoise.qa`` started as a module and later became a package containing
# endpoint-specific submodules.  Load the original API under a private name so
# established imports remain valid while new code can use ``qa.metrics``,
# ``qa.preservation``, ``qa.ground_truth``, and the other focused modules.
_legacy_path = pathlib.Path(__file__).resolve().parent.parent / "qa.py"
_legacy_spec = importlib.util.spec_from_file_location(
    "mne_denoise._qa_legacy", _legacy_path
)
if _legacy_spec is None or _legacy_spec.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load legacy QA API from {_legacy_path}")
_legacy = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(_legacy)

below_noise_distortion_db = _legacy.below_noise_distortion_db
channel_variance_ratio = _legacy.channel_variance_ratio
geometric_mean_psd_ratio = _legacy.geometric_mean_psd_ratio
max_abs_change = _legacy.max_abs_change
noise_surround_ratio = _legacy.noise_surround_ratio
peak_attenuation_db = _legacy.peak_attenuation_db
rms_change = _legacy.rms_change
spectral_distortion = _legacy.spectral_distortion
suppression_ratio = _legacy.suppression_ratio
variance_removed = _legacy.variance_removed
compute_all_qa_metrics = _legacy.compute_all_qa_metrics
overclean_proportion = _legacy.overclean_proportion
underclean_proportion = _legacy.underclean_proportion

__all__ = [
    "geometric_mean_psd",
    "noise_surr_ratio",
    "peak_attenuation_db",
    "below_noise_distortion",
    "overclean_proportion",
    "underclean_proportion",
    "compute_all_qa_metrics",
    "below_noise_distortion_db",
    "channel_variance_ratio",
    "geometric_mean_psd_ratio",
    "max_abs_change",
    "noise_surround_ratio",
    "rms_change",
    "spectral_distortion",
    "suppression_ratio",
    "variance_removed",
]
