"""Denoising Source Separation methods."""

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
    VarianceMaskDenoiser,
    WienerMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)
from .linear import DSS, compute_dss
from .nonlinear import IterativeDSS, iterative_dss, iterative_dss_one

# Segmentation strategies (exposed for convenience)
from .segmentation import CovarianceSegmenter, FixedWindowSegmenter

# Variants (exposed for convenience)
from .variants import (
    TimeShiftDSS,
    narrowband_dss,
    narrowband_scan,
    smooth_dss,
    ssvep_dss,
)

__all__ = [
    # Core
    "compute_dss",
    "DSS",
    "IterativeDSS",
    "iterative_dss",
    "iterative_dss_one",
    # Variants
    "TimeShiftDSS",
    "smooth_dss",
    "ssvep_dss",
    "narrowband_scan",
    "narrowband_dss",
    # Segmentation
    "CovarianceSegmenter",
    "FixedWindowSegmenter",
    # Denoisers (from .denoisers)
    "LinearDenoiser",
    "NonlinearDenoiser",
    "AverageBias",
    "CycleAverageBias",
    "BandpassBias",
    "LineNoiseBias",
    "PeakFilterBias",
    "CombFilterBias",
    "LagAverageBias",
    "SmoothingBias",
    "SpectrogramBias",
    "VarianceMaskDenoiser",
    "WienerMaskDenoiser",
    "TanhMaskDenoiser",
    "RobustTanhDenoiser",
    "GaussDenoiser",
    "SkewDenoiser",
    "DCTDenoiser",
    "SpectrogramDenoiser",
    "QuasiPeriodicDenoiser",
    "KurtosisDenoiser",
    "SmoothTanhDenoiser",
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
]
