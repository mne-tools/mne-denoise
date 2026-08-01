"""Denoising Source Separation (DSS).

This module contains:
- Core DSS algorithms (linear and nonlinear)
- Variants and applications (TSR, SSVEP, Narrowband)

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
    LagAveragingBias,
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
    TimeShiftBias,
    WienerMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)
from .linear import DSS, compute_dss
from .nonlinear import IterativeDSS, iterative_dss, iterative_dss_one
from .time_shift import (
    TimeShiftDSS,
    TimeShiftDSSDiagnostics,
    compute_time_shift_dss,
)

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
from .variants.tsr import lag_averaging_dss, smooth_dss, time_shift_dss

__all__ = [
    # Core
    "compute_dss",
    "DSS",
    "iterative_dss",
    "iterative_dss_one",
    "IterativeDSS",
    "compute_time_shift_dss",
    "TimeShiftDSS",
    "TimeShiftDSSDiagnostics",
    # Variants modules
    "tsr",
    "ssvep",
    "narrowband",
    # Variants functions
    "lag_averaging_dss",
    "time_shift_dss",
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
    "LagAveragingBias",
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
    "TimeShiftBias",
    "SmoothingBias",
]
