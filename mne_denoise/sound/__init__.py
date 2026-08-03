"""SOUND: automatic, robust noise suppression for EEG/MEG (Mutanen et al. 2018)."""

from .core import SOUND, compute_sound, compute_sound_ref_best

__all__ = ["SOUND", "compute_sound", "compute_sound_ref_best"]
