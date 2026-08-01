"""Reference-free lagged-CCA artifact attenuation.

This module contains:

- ``compute_lagged_cca``: the one-shot array-based BSS-CCA algorithm.
- ``LaggedCCA``: the scikit-learn estimator (leakage-safe ``fit``/``transform``),
  compatible with MNE-Python objects or NumPy arrays.

Canonical-Correlation-Analysis blind source separation (BSS-CCA) removes muscle
/ EMG artifacts without a reference channel by ranking components on temporal
autocorrelation and dropping the broadband (low-autocorrelation) ones. It is the
reference-free counterpart to :class:`~mne_denoise.icanclean.ICanClean`.

References
----------
.. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
       Van Huffel, S. (2006). Canonical correlation analysis applied to remove
       muscle artifacts from the electroencephalogram. IEEE Transactions on
       Biomedical Engineering, 53(12), 2583-2587.
       https://doi.org/10.1109/TBME.2006.879459
"""

from .core import LaggedCCA, compute_lagged_cca

__all__ = [
    "LaggedCCA",
    "compute_lagged_cca",
]
