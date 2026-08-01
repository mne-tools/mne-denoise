"""DSS Variants and Applications.

Contains specialized implementations of DSS for specific tasks:
- Lag-averaging DSS for temporal structure
- SSVEP: SSVEP enhancement using comb filters
- Narrowband: Frequency scanning and band-specific extraction
"""

from .narrowband import narrowband_dss, narrowband_scan
from .ssvep import ssvep_dss
from .tsr import lag_averaging_dss, smooth_dss, time_shift_dss

__all__ = [
    "lag_averaging_dss",
    "time_shift_dss",
    "smooth_dss",
    "ssvep_dss",
    "narrowband_scan",
    "narrowband_dss",
]
