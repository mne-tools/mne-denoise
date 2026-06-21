"""Quality-assurance metrics for line-noise removal benchmarks.

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

from .metrics import (
    below_noise_distortion,
    compute_all_qa_metrics,
    geometric_mean_psd,
    noise_surr_ratio,
    overclean_proportion,
    peak_attenuation_db,
    underclean_proportion,
)

__all__ = [
    "geometric_mean_psd",
    "noise_surr_ratio",
    "peak_attenuation_db",
    "below_noise_distortion",
    "overclean_proportion",
    "underclean_proportion",
    "compute_all_qa_metrics",
]
