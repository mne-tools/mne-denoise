"""MNE-Denoise: Denoising tools for MNE-Python.

Modules
-------
- `dss`: Denoising Source Separation (Linear, Nonlinear, Variants).
- `zapline`: ZapLine line noise removal.
- `qa`: Quality assurance metrics.
"""

from . import dss, qa, zapline

__version__ = "0.0.1"

__all__ = [
    "dss",
    "qa",
    "zapline",
]
