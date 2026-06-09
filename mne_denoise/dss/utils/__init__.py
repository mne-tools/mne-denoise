"""DSS Internal Utilities."""

from .convergence import Gamma179, GammaPredictive
from .covariance import compute_covariance, compute_evoked_covariance
from .segmentation import CovarianceSegmenter, FixedWindowSegmenter
from .selection import (
    auto_select_components,
    auto_select_components_robust,
    detect_eigenvalue_knee,
    iterative_outlier_removal,
)
from .whitening import (
    compute_data_covariance_whitener,
    whiten_from_data_covariance,
)

__all__ = [
    "whiten_from_data_covariance",
    "compute_data_covariance_whitener",
    "compute_covariance",
    "compute_evoked_covariance",
    "iterative_outlier_removal",
    "auto_select_components",
    "auto_select_components_robust",
    "detect_eigenvalue_knee",
    "Gamma179",
    "GammaPredictive",
    "CovarianceSegmenter",
    "FixedWindowSegmenter",
]
