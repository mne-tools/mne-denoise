"""Multi-channel Wiener filter (MWF) artifact removal.

This module contains:

- ``hf_power_mask``: broadband high-frequency artifact-segment detector.
- ``mwf_filter`` / ``compute_mwf``: the array-based Wiener filter.
- ``MWF``: the scikit-learn estimator, compatible with MNE-Python objects or
  NumPy arrays.

The MWF (Somers, Francart & Bertrand 2018) is a generic, reference-free spatial
artifact cleaner and the spatial-filter core of the RELAX pipeline. It recovers
the clean signal via ``R_clean @ R_artifact^{-1} @ X`` from artifact-free and
artifact-present segment covariances. It is a general cleaner rather than an
artifact-specific method; validate preservation of the band of interest.

References
----------
.. [1] Somers, B., Francart, T., & Bertrand, A. (2018). A generic EEG artifact
       removal algorithm based on the multi-channel Wiener filter. Journal of
       Neural Engineering, 15(3), 036007.
       https://doi.org/10.1088/1741-2552/aaac92
"""

from .core import MWF, compute_mwf, hf_power_mask, mwf_filter

__all__ = [
    "MWF",
    "compute_mwf",
    "hf_power_mask",
    "mwf_filter",
]
