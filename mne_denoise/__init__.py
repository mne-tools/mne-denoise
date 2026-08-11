"""MNE-Denoise: Advanced spatial and spectral denoising tools for MNE-Python.

MNE-Denoise provides a suite of advanced artifact removal and signal processing
techniques built on top of MNE-Python. It is designed to be fully compatible with
the scikit-learn estimator API, allowing these algorithms to be seamlessly integrated
into robust, reproducible machine learning pipelines.

The package includes implementations for standard, adaptive, and nonlinear algorithms
tailored for EEG and MEG continuous/epoched data.

Available Modules
-----------------
asr : Artifact Subspace Reconstruction
    Automated, data-driven spatial filtering for transient artifact rejection.
    Includes Standard ASR, Adaptive ASR (streaming non-stationarity correction),
    and Juggler-style sliding window ASR, plus experimental Riemannian geometry hooks.

dss : Denoising Source Separation
    Components analysis that maximizes reproducibility across trials or maximizes
    power in targeted frequency bands. Includes linear and iterative-nonlinear variants.

zapline : ZapLine Power Line Noise Removal
    High-fidelity line noise (50/60Hz) removal via spatial filtering that preserves
    narrowband physiological power.

icanclean : Reference-Based Artifact Removal
    Uses Canonical Correlation Analysis (CCA) to remove artifacts by projecting them
    out based on dedicated reference channels (e.g., EOG or EMG).

spectrum_interpolation : Spectrum Interpolation
    Removes power-line noise and its harmonics by interpolating spectral amplitudes
    while preserving phase.

bss_cca : Reference-Free BSS-CCA
    Blind source separation by canonical correlation between the data and a
    lagged copy of itself, for broadband muscle-artifact attenuation.

sns : Sensor Noise Suppression
    Removes channel-specific noise by reconstructing each sensor from correlated
    neighboring sensors.

"""

from . import asr, bss_cca, dss, icanclean, sns, spectrum_interpolation, zapline
from ._covariance import compute_covariance

__version__ = "0.0.1"

__all__ = [
    "asr",
    "bss_cca",
    "compute_covariance",
    "dss",
    "icanclean",
    "spectrum_interpolation",
    "sns",
    "zapline",
]
