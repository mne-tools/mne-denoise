"""DSS Variants and Applications.

Contains specialized implementations of DSS for specific tasks:
- Time-shift DSS: lag-augmented repeated-trial DSS and temporal smoothing
- SSVEP: SSVEP enhancement using comb filters
- Narrowband: Frequency scanning and band-specific extraction
"""

from .narrowband import narrowband_dss, narrowband_scan
from .ssvep import ssvep_dss
from .tsr import TimeShiftDSS, smooth_dss

__all__ = [
    "TimeShiftDSS",
    "smooth_dss",
    "ssvep_dss",
    "narrowband_scan",
    "narrowband_dss",
]
