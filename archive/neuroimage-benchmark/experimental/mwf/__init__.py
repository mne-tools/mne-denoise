"""Multi-channel Wiener filter (MWF) artifact removal (Somers et al. 2018)."""

from .core import MWF, hf_power_mask, mwf_filter

__all__ = ["MWF", "mwf_filter", "hf_power_mask"]
