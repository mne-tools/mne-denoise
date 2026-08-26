"""Wavelet thresholding and wavelet-enhanced ICA (wICA) artifact removal."""
from .core import (
    WaveletICA,
    wavelet_denoise,
    wavelet_denoise_multichannel,
    wavelet_extract_artifact,
)

__all__ = [
    "WaveletICA",
    "wavelet_denoise",
    "wavelet_denoise_multichannel",
    "wavelet_extract_artifact",
]
