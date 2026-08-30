"""Artifact removal and signal denoising for EEG and MEG."""

from importlib import import_module

from . import (
    asr,
    bss_cca,
    dss,
    icanclean,
    overcorrection,
    progress,
    qa,
    sns,
    sound,
    spectrum_interpolation,
    ssa,
    sspsir,
    zapline,
)
from ._covariance import compute_covariance
from .overcorrection import quantify_overcorrection

__version__ = "0.0.1"


def __getattr__(name: str):
    """Load the optional visualization namespace when it is requested."""
    if name == "viz":
        module = import_module(".viz", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "asr",
    "bss_cca",
    "compute_covariance",
    "dss",
    "icanclean",
    "overcorrection",
    "quantify_overcorrection",
    "progress",
    "qa",
    "sound",
    "spectrum_interpolation",
    "ssa",
    "sns",
    "sspsir",
    "viz",
    "zapline",
]
