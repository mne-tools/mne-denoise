"""Pluggable denoiser functions for DSS."""

from __future__ import annotations

from .artifact import CycleAverageBias
from .averaging import AverageBias
from .base import LinearDenoiser, NonlinearDenoiser
from .ica import (
    GaussDenoiser,
    KurtosisDenoiser,
    RobustTanhDenoiser,
    SkewDenoiser,
    SmoothTanhDenoiser,
    TanhMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)
from .masking import (
    VarianceMaskDenoiser,
    WienerMaskDenoiser,
)
from .periodic import CombFilterBias, PeakFilterBias, QuasiPeriodicDenoiser
from .spectral import (
    BandpassBias,
    LineNoiseBias,
)
from .spectrogram import SpectrogramBias, SpectrogramDenoiser
from .temporal import (
    DCTDenoiser,
    LagAverageBias,
    SmoothingBias,
)

__all__ = [
    # Base classes
    "LinearDenoiser",
    "NonlinearDenoiser",
    # Linear biases
    "AverageBias",
    "CycleAverageBias",
    "BandpassBias",
    "LineNoiseBias",
    "PeakFilterBias",
    "CombFilterBias",
    "LagAverageBias",
    "SmoothingBias",
    "SpectrogramBias",
    # Nonlinear denoisers
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
    # Beta helpers (FastICA Newton step)
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
]
