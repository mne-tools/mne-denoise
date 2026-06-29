"""Reference-free BSS-CCA (auto-CCA) muscle/EMG artifact removal."""
from .core import AutoCCA, _autocca_filter

__all__ = ["AutoCCA", "_autocca_filter"]
