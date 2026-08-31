"""Singular Spectrum Analysis methods."""

from .basic import (
    SingularSpectrumAnalysis,
    compute_basic_ssa,
    ssa_clean_channel,
    ssa_decompose,
    ssa_w_correlation,
)
from .local import (
    LocalSingularSpectrumAnalysis,
    compute_local_ssa,
    local_ssa_clean_channel,
)

__all__ = [
    "LocalSingularSpectrumAnalysis",
    "SingularSpectrumAnalysis",
    "compute_basic_ssa",
    "compute_local_ssa",
    "local_ssa_clean_channel",
    "ssa_clean_channel",
    "ssa_decompose",
    "ssa_w_correlation",
]
