"""Basic and local Singular Spectrum Analysis (SSA).

This module contains:

- ``ssa_clean_channel``: SSA cleaning of a single 1-D channel.
- ``ssa_decompose``: additive Basic SSA decomposition of one series.
- ``compute_basic_ssa``: frequency-guided Basic SSA cleaning of a multichannel
  array.
- ``compute_local_ssa``: Teixeira local SSA artifact removal.
- ``SingularSpectrumAnalysis``: the scikit-learn estimator, compatible with
  MNE-Python objects or NumPy arrays.
- ``LocalSingularSpectrumAnalysis``: the local SSA estimator.

SSA methods are per-recording and unsupervised. Their decompositions are
transductive: changing record or epoch boundaries can change the result.

References
----------
.. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for
       Time Series. Springer. https://doi.org/10.1007/978-3-642-34913-3
"""

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
