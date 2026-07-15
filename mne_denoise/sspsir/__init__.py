"""SSP-SIR: suppress TMS-evoked muscle artifacts (Mutanen et al. 2016)."""

from .core import SSPSIR, compute_sspsir_operator

__all__ = ["SSPSIR", "compute_sspsir_operator"]

