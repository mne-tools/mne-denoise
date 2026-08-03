"""SSP-SIR: suppress TMS-evoked muscle artifacts (Mutanen et al. 2016)."""

from .core import SSPSIR, compute_sir, compute_sspsir

__all__ = ["SSPSIR", "compute_sir", "compute_sspsir"]
