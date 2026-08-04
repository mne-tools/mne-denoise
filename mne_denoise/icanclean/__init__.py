"""ICanClean.

This module contains:
- ``compute_icanclean``: The core array-based iCanClean algorithm.
- ``ICanClean``: The Scikit-learn estimator compatible with MNE-Python
  objects or NumPy arrays.
- ``RecursiveICanClean``: An explicitly experimental stateful composition of
  recursive covariance CCA and the iCanClean subtraction rule.

iCanClean removes artifact subspaces shared by primary channels and
reference channels using canonical correlation analysis (CCA) [1]_ [2]_ [3]_.

References
----------
.. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.
.. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
       Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
       23(19), 8214. https://doi.org/10.3390/s23198214
.. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
       Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
       https://doi.org/10.3390/s23020928
.. [4] Nordin, A. D., Hairston, W. D., & Ferris, D. P. (2018). Dual-electrode
       motion artifact cancellation for mobile electroencephalography.
       Journal of Neural Engineering, 15(5), 056024.
       https://doi.org/10.1088/1741-2552/aad7d7
.. [5] Hotelling, H. (1936). Relations between two sets of variates.
       Biometrika, 28(3/4), 321-377.
"""

from .core import ICanClean, compute_icanclean
from .recursive import RecursiveICanClean

__all__ = [
    "ICanClean",
    "RecursiveICanClean",
    "compute_icanclean",
]
