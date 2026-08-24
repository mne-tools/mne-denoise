"""Denoising Source Separation (DSS).

This module contains:
- Core DSS algorithms (linear and nonlinear)
- Variants and applications (Time-shift DSS, SSVEP, Narrowband)

For ZapLine, see `mne_denoise.zapline`.
"""

# Core
# Denoisers & Biases (Flat API)
from .denoisers import (
    AverageBias,
    BandpassBias,
    CombFilterBias,
    CycleAverageBias,
    DCTDenoiser,
    GaussDenoiser,
    KurtosisDenoiser,
    LagAverageBias,
    LinearDenoiser,
    LineNoiseBias,
    NonlinearDenoiser,
    PeakFilterBias,
    QuasiPeriodicDenoiser,
    RobustTanhDenoiser,
    SkewDenoiser,
    SmoothingBias,
    SmoothTanhDenoiser,
    SpectrogramBias,
    SpectrogramDenoiser,
    TanhMaskDenoiser,
    WienerMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)
from .linear import DSS, compute_dss
from .nonlinear import IterativeDSS, iterative_dss, iterative_dss_one

# Utils (exposed for convenience if needed)
from .utils import convergence, whitening
from .utils.segmentation import CovarianceSegmenter, FixedWindowSegmenter
from .utils.selection import (
    auto_select_components,
    auto_select_components_robust,
    detect_eigenvalue_knee,
    iterative_outlier_removal,
)

# Variants (Modules)
from .variants import narrowband, ssvep, tsr
from .variants.narrowband import narrowband_dss, narrowband_scan
from .variants.ssvep import ssvep_dss

# Variants (Direct Access)
from .variants.tsr import TimeShiftDSS, smooth_dss

__all__ = [
    # Core
    "compute_dss",
    "DSS",
    "TimeShiftDSS",
    "iterative_dss",
    "iterative_dss_one",
    "IterativeDSS",
    # Variants modules
    "tsr",
    "ssvep",
    "narrowband",
    # Variants functions
    "smooth_dss",
    "ssvep_dss",
    "narrowband_scan",
    "narrowband_dss",
    # Utils
    "whitening",
    "convergence",
    # Denoisers (from .denoisers)
    "LinearDenoiser",
    "NonlinearDenoiser",
    "AverageBias",
    "BandpassBias",
    "LineNoiseBias",
    "PeakFilterBias",
    "CombFilterBias",
    "CycleAverageBias",
    "WienerMaskDenoiser",
    "TanhMaskDenoiser",
    "RobustTanhDenoiser",
    "GaussDenoiser",
    "SkewDenoiser",
    "DCTDenoiser",
    "SpectrogramBias",
    "SpectrogramDenoiser",
    "QuasiPeriodicDenoiser",
    "KurtosisDenoiser",
    "SmoothTanhDenoiser",
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
    "LagAverageBias",
    "SmoothingBias",
]
