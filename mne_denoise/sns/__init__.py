"""Sensor Noise Suppression (SNS) artifact removal.

This module contains:

- ``compute_sns_weights``: build the SNS spatial operator from a covariance.
- ``compute_sns``: one-shot SNS cleaning of a multichannel array.
- ``SNS``: the scikit-learn estimator, compatible with MNE-Python objects or
  NumPy arrays.

SNS (de Cheveigne & Simon 2008) suppresses sensor-specific noise by regenerating
each channel from a projection onto its most-correlated neighbour channels:
spatially correlated brain signal is retained, uncorrelated sensor noise is
removed.

References
----------
.. [1] de Cheveigne, A., & Simon, J. Z. (2008). Sensor noise suppression.
       Journal of Neuroscience Methods, 168(1), 195-202.
       https://doi.org/10.1016/j.jneumeth.2007.09.012
"""

from .core import SNS, compute_sns, compute_sns_weights

__all__ = [
    "SNS",
    "compute_sns",
    "compute_sns_weights",
]
