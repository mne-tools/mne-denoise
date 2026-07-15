"""DSS Internal Utilities."""

from .convergence import Gamma179, GammaPredictive
from .covariance import compute_covariance, compute_evoked_covariance
from .segmentation import CovarianceSegmenter, FixedWindowSegmenter
from .selection import (
    auto_select_components,
    auto_select_components_robust,
    detect_eigenvalue_knee,
    eigenvalue_ratio_selection,
    iterative_outlier_removal,
    max_gap_selection,
)
from .whitening import compute_whitener, whiten_data

__all__ = [
    "whiten_data",
    "compute_whitener",
    "compute_covariance",
    "compute_evoked_covariance",
    "iterative_outlier_removal",
    "auto_select_components",
    "auto_select_components_robust",
    "detect_eigenvalue_knee",
    "eigenvalue_ratio_selection",
    "max_gap_selection",
    "CovarianceSegmenter",
    "FixedWindowSegmenter",
    "Gamma179",
    "GammaPredictive",
]
