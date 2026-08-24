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

ssa : Singular Spectrum Analysis
    Temporal decomposition for trend, oscillation, and residual reconstruction.

bss_cca : Reference-Free BSS-CCA
    Blind source separation by canonical correlation between the data and a
    lagged copy of itself, for broadband muscle-artifact attenuation.

sns : Sensor Noise Suppression
    Removes channel-specific noise by reconstructing each sensor from correlated
    neighboring sensors.

sound : SOUND Automatic Noise Suppression
    Forward-model-based Wiener estimation that suppresses per-channel noise by
    reconstructing each sensor from all others (Mutanen et al., 2018).

sspsir : SSP-SIR Muscle Artifact Suppression
    Signal-space projection with source-informed reconstruction, targeting
    TMS-evoked muscle artifacts (Mutanen et al., 2016).

overcorrection : Forward-Model Overcorrection Metrics
    Quantifies how much a linear spatial filter attenuates or distorts
    hypothetical cortical sources through a forward model (Mutanen et al., 2022).

"""

from . import (
    asr,
    bss_cca,
    dss,
    icanclean,
    overcorrection,
    sns,
    sound,
    spectrum_interpolation,
    ssa,
    sspsir,
    zapline,
)
from ._covariance import compute_covariance
from .overcorrection import quantify_overcorrection

__version__ = "0.0.1"

__all__ = [
    "asr",
    "bss_cca",
    "compute_covariance",
    "dss",
    "icanclean",
    "overcorrection",
    "quantify_overcorrection",
    "sound",
    "spectrum_interpolation",
    "ssa",
    "sns",
    "sspsir",
    "zapline",
]
