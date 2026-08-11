"""Reference-free BSS-CCA artifact attenuation.

This module contains:

- ``compute_bss_cca``: the one-shot array implementation of BSS-CCA [1]_.
- ``BSSCCA``: the scikit-learn estimator with leakage-safe ``fit``/``transform``,
  compatible with MNE-Python objects or NumPy arrays.

Blind source separation by canonical correlation analysis (BSS-CCA) removes
muscle/EMG artifacts without a reference channel. It solves CCA between the
recording and a lagged copy of itself, ranks the resulting components by
lagged correlation, and drops the broadband low-correlation components in
which muscle activity concentrates [1]_. Clinical validation of the method on
ictal EEG, together with its contiguous 10 s block scheme, is reported in
[2]_. It is the reference-free counterpart to
:class:`~mne_denoise.icanclean.ICanClean`.

References
----------
.. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
       Van Huffel, S. (2006). Canonical correlation analysis applied to remove
       muscle artifacts from the electroencephalogram. IEEE Transactions on
       Biomedical Engineering, 53(12), 2583-2587.
       https://doi.org/10.1109/TBME.2006.879459
.. [2] Vergult, A., De Clercq, W., Palmini, A., Vanrumste, B., Dupont, P.,
       Van Huffel, S., & Van Paesschen, W. (2007). Improving the interpretation
       of ictal scalp EEG: BSS-CCA algorithm for muscle artifact removal.
       Epilepsia, 48(5), 950-958.
       https://doi.org/10.1111/j.1528-1167.2007.01031.x
"""

from .core import BSSCCA, compute_bss_cca

__all__ = [
    "BSSCCA",
    "compute_bss_cca",
]
