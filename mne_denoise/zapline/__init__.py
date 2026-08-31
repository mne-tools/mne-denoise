"""ZapLine module for line noise removal.

This module implements the ZapLine algorithm (de Cheveigné, 2020) and
its adaptive variant ZapLine-plus (Klug & Kloosterman, 2022).

References
----------
.. [1] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 207, 116356.
.. [2] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension
       for automatic and adaptive removal of frequency-specific noise artifacts
       in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
       https://doi.org/10.1002/hbm.25832
"""

from .core import ZapLine

__all__ = [
    "ZapLine",
]
