"""Singular Spectrum Analysis (SSA) artifact removal.

This module contains:

- ``ssa_clean_channel``: SSA cleaning of a single 1-D channel.
- ``compute_ssa``: SSA cleaning of a multichannel array + a diagnostics dict.
- ``SingularSpectrumAnalysis``: the scikit-learn estimator, compatible with
  MNE-Python objects or NumPy arrays. ``SSA`` is a convenience alias.

SSA embeds each channel into a trajectory matrix, decomposes it by SVD, and
drops eigentriples whose dominant frequency falls in an artifact band (slow
drift / ocular by default) while preserving oscillatory neural activity. It is
per-recording and unsupervised.

References
----------
.. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for
       Time Series. Springer. https://doi.org/10.1007/978-3-642-34913-3
"""

from .core import SingularSpectrumAnalysis, compute_ssa, ssa_clean_channel

SSA = SingularSpectrumAnalysis

__all__ = [
    "SSA",
    "SingularSpectrumAnalysis",
    "compute_ssa",
    "ssa_clean_channel",
]
