"""Spectrum interpolation for power-line noise removal.

This module implements the spectrum-interpolation method of Leske & Dalal
(2019), a frequency-domain line-noise remover translated from FieldTrip's
``ft_preproc_dftfilter`` spectrum-interpolation mode.

References
----------
.. [1] Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG and
       MEG data via spectrum interpolation. NeuroImage, 189, 763-776.
"""

from .core import SpectrumInterpolation, interpolate_spectrum

__all__ = [
    "SpectrumInterpolation",
    "interpolate_spectrum",
]

