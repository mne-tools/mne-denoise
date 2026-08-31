"""DSS component-selection helpers."""

from __future__ import annotations

import numpy as np

__all__ = [
    "auto_select_components",
    "auto_select_components_robust",
    "detect_eigenvalue_knee",
    "iterative_outlier_removal",
]


def iterative_outlier_removal(scores: np.ndarray, sigma: float = 3.0) -> int:
    """Count values removed by iterative mean-plus-sigma thresholding.

    At each iteration, values above ``mean + sigma * std`` are removed from the
    remaining scores; the process stops when no value qualifies and returns the
    number removed.

    Parameters
    ----------
    scores : array-like
        Component scores.
    sigma : float, default=3.0
        Threshold multiplier.

    Returns
    -------
    int
        Number of removed scores.
    """
    scores = np.asarray(scores)
    n_outliers = 0
    remaining = scores.copy()

    while len(remaining) > 2:
        mean_val = np.mean(remaining)
        std_val = np.std(remaining)

        if std_val < 1e-12:
            break

        threshold = mean_val + sigma * std_val
        outliers = remaining > threshold

        if not np.any(outliers):
            break

        n_outliers += np.sum(outliers)
        remaining = remaining[~outliers]

    return n_outliers


def auto_select_components(eigenvalues: np.ndarray, threshold: float = 3.0) -> int:
    """Select a component count with :func:`iterative_outlier_removal`.

    Parameters
    ----------
    eigenvalues : array-like
        DSS component scores.
    threshold : float, default=3.0
        Sigma threshold.

    Returns
    -------
    int
        Selected component count.
    """
    return iterative_outlier_removal(eigenvalues, threshold)


def detect_eigenvalue_knee(
    scores: np.ndarray,
    rel_floor: float = 0.01,
    min_ratio: float = 3.0,
) -> int:
    """Select components above the largest qualifying score drop.

    Parameters
    ----------
    scores : array-like
        Component scores in descending order.
    rel_floor : float, default=0.01
        Relative floor for valid drop anchors.
    min_ratio : float, default=3.0
        Minimum adjacent-score ratio.

    Returns
    -------
    int
        Number of scores above the knee, or zero if no qualifying knee exists.

    Notes
    -----
    The drop is evaluated in log space with the relative-floor and minimum-ratio
    gates. This is a package heuristic.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n < 2:
        return int(n)

    max_score = scores[0]
    if max_score <= 0:
        return 0

    # log-space drops between consecutive eigenvalues (drops[i] = log10(s[i] / s[i+1]))
    floor = max_score * 1e-12
    log_scores = np.log10(np.clip(scores, floor, None))
    drops = -np.diff(log_scores)

    # Only consider drop positions anchored on a "still meaningful" eigenvalue
    # (avoids picking the very last gap deep in the numerical noise tail).
    anchor_threshold = max_score * rel_floor
    valid = scores[:-1] > anchor_threshold
    if not np.any(valid):
        return 0

    valid_drops = np.where(valid, drops, -np.inf)
    knee_idx = int(np.argmax(valid_drops))

    # Sanity gate: require a meaningful gap to avoid false positives on
    # smoothly-decaying spectra.
    if valid_drops[knee_idx] < np.log10(min_ratio):
        return 0

    return knee_idx + 1


def auto_select_components_robust(
    eigenvalues: np.ndarray,
    sigma: float = 3.0,
    knee_rel_floor: float = 0.01,
    knee_min_ratio: float = 3.0,
) -> int:
    """Combine outlier and knee component counts and return the larger count.

    Parameters
    ----------
    eigenvalues : array-like
        DSS component scores in descending order.
    sigma : float, default=3.0
        Outlier threshold multiplier.
    knee_rel_floor : float, default=0.01
        Relative knee floor.
    knee_min_ratio : float, default=3.0
        Minimum knee ratio.

    Returns
    -------
    int
        Larger of the two proposed counts.
    """
    n_outlier = iterative_outlier_removal(eigenvalues, sigma=sigma)
    n_knee = detect_eigenvalue_knee(
        eigenvalues, rel_floor=knee_rel_floor, min_ratio=knee_min_ratio
    )
    return int(max(n_outlier, n_knee))
