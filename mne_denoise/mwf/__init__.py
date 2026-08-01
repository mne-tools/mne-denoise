"""Semi-supervised multi-channel Wiener filtering.

``MultichannelWienerFilter`` is the canonical estimator name. ``MWF`` is a
documented short alias. The implementation uses the zero-delay GEVD formulation
from Somers, Francart, and Bertrand (2018); explicit temporal delay embedding is
not yet implemented.
"""

from .core import (
    MWF,
    MultichannelWienerFilter,
    compute_mwf,
    hf_power_mask,
    mwf_filter,
)

__all__ = [
    "MWF",
    "MultichannelWienerFilter",
    "compute_mwf",
    "hf_power_mask",
    "mwf_filter",
]
