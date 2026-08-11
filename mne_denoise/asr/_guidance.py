"""Guidance-model helpers for the experimental GuidedASR estimator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._covariance import compute_covariance
from ._validation import _validate_covariance_matrix

_EPS = float(np.finfo(np.float64).eps)


def _compute_guidance_covariance(
    data: np.ndarray,
    biases: Sequence[Any] | None,
    *,
    name: str,
) -> np.ndarray | None:
    """Compute the equally weighted spatial covariance of a bias bank.

    Each bias covariance is divided by its trace before averaging. This makes
    bias operators equal structural votes and intentionally discards their
    output-amplitude scale. Equal weighting is an experimental GuidedASR model
    choice, not a property of ASR or DSS.
    """
    if not biases:
        return None

    n_channels = data.shape[0]
    covariance = np.zeros((n_channels, n_channels), dtype=np.float64)
    for index, bias in enumerate(biases):
        biased = bias.apply(data) if hasattr(bias, "apply") else bias(data)
        biased = np.asarray(biased, dtype=np.float64)
        if biased.shape != data.shape:
            raise ValueError(
                f"{name}[{index}] must return shape {data.shape}, got {biased.shape}"
            )
        if not np.all(np.isfinite(biased)):
            raise ValueError(f"{name}[{index}] must return only finite values")

        bias_covariance = _validate_covariance_matrix(
            compute_covariance(biased),
            n_channels=n_channels,
            name=f"{name}[{index}] covariance",
        )
        trace = float(np.trace(bias_covariance))
        if trace <= _EPS:
            raise ValueError(f"{name}[{index}] covariance must have positive trace")
        covariance += bias_covariance / trace

    return covariance / len(biases)


def _guided_component_weights(
    variances: np.ndarray,
    eigenvectors: np.ndarray,
    thresholds: np.ndarray,
    *,
    forced_keep: np.ndarray,
    artifact_covariance: np.ndarray | None,
    preserve_covariance: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Return keep weights for components that standard ASR flags.

    The baseline soft-ASR weight is ``threshold / variance`` for flagged
    components and one otherwise. A covariance contributes evidence only when
    a component's trace-normalized quadratic score exceeds the isotropic score
    ``1 / n_channels``. Guidance is the normalized preserve-minus-artifact
    evidence in ``[-1, 1]``. Positive guidance lifts a flagged component toward
    one; negative guidance lowers it toward zero. ``strength`` linearly
    interpolates between the baseline and guided weights.
    """
    flagged = variances >= thresholds
    asr_weights = np.ones_like(variances, dtype=np.float64)
    asr_weights[flagged] = thresholds[flagged] / np.maximum(variances[flagged], _EPS)

    if artifact_covariance is None and preserve_covariance is None:
        weights = asr_weights
    else:
        isotropic_score = 1.0 / eigenvectors.shape[0]
        preserve_scores = np.zeros(eigenvectors.shape[1])
        if preserve_covariance is not None:
            trace = float(np.trace(preserve_covariance))
            if trace > _EPS:
                preserve_scores = np.maximum(
                    np.sum(eigenvectors * (preserve_covariance @ eigenvectors), axis=0)
                    / trace
                    - isotropic_score,
                    0.0,
                )
        artifact_scores = np.zeros(eigenvectors.shape[1])
        if artifact_covariance is not None:
            trace = float(np.trace(artifact_covariance))
            if trace > _EPS:
                artifact_scores = np.maximum(
                    np.sum(eigenvectors * (artifact_covariance @ eigenvectors), axis=0)
                    / trace
                    - isotropic_score,
                    0.0,
                )
        total_scores = preserve_scores + artifact_scores
        guidance = np.divide(
            preserve_scores - artifact_scores,
            total_scores,
            out=np.zeros_like(total_scores),
            where=total_scores > _EPS,
        )
        guided_weights = np.where(
            guidance >= 0.0,
            asr_weights + guidance * (1.0 - asr_weights),
            asr_weights * (1.0 + guidance),
        )
        weights = asr_weights + strength * (guided_weights - asr_weights)

    weights = np.where(flagged, weights, 1.0)
    weights = np.where(forced_keep, 1.0, weights)
    return np.clip(weights, 0.0, 1.0)
