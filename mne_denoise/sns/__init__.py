"""Sensor Noise Suppression (SNS) artifact removal.

This module contains:

- ``compute_sns_weights``: compute the SNS operator from a covariance.
- ``compute_sns``: clean a multichannel array with SNS.
- ``SNS``: the scikit-learn estimator, compatible with MNE-Python objects or
  NumPy arrays.

SNS regenerates each channel from a projection onto its most-correlated
neighbours [1]_. It assumes that signals of interest are spatially correlated
and sensor-specific noise is not; those assumptions require regime-specific
validation.

References
----------
.. [1] de Cheveigné, A., & Simon, J. Z. (2008). Sensor noise suppression.
       Journal of Neuroscience Methods, 168(1), 195-202.
       https://doi.org/10.1016/j.jneumeth.2007.09.012
"""

from .core import SNS, compute_sns, compute_sns_weights

__all__ = [
    "SNS",
    "compute_sns",
    "compute_sns_weights",
]
