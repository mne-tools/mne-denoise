"""Centralized logging utility for the mne-denoise package.

This module provides a unified `set_log_level_from_verbose` function that maps
MNE-style `verbose` arguments (e.g., `True`, `False`, `"INFO"`, `"DEBUG"`) directly
onto the root `mne_denoise` Python logger. This ensures consistent log filtering
across all package modules (ASR, DSS, ZapLine, etc.) while allowing individual
submodules to inherit the global logging level gracefully.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("mne_denoise")


def set_log_level_from_verbose(verbose: bool | str | int | None) -> None:
    """Set the mne-denoise logger level from an MNE-style ``verbose`` value.

    Parameters
    ----------
    verbose : bool | str | int | None
        ``True`` maps to ``INFO`` and ``False`` to ``WARNING``; a level name
        (e.g. ``"DEBUG"``) or integer is used directly; ``None`` leaves the
        current level unchanged so external logging configuration is respected.
    """
    if verbose is None:
        return
    if isinstance(verbose, bool):
        level: int | str = logging.INFO if verbose else logging.WARNING
    elif isinstance(verbose, str):
        level = verbose.upper()
    else:
        level = int(verbose)
    logger.setLevel(level)
