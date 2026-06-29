"""Empirical Mode Decomposition (EMD/EEMD) artifact removal."""
from .core import EMDDenoiser, emd_clean_channel

__all__ = ["EMDDenoiser", "emd_clean_channel"]
