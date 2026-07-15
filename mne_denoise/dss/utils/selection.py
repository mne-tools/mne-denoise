"""Component selection utilities for DSS.

Provides automatic component selection using outlier detection.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np


def iterative_outlier_removal(scores: np.ndarray, sigma: float = 3.0) -> int:
    """Detect outliers iteratively using mean + sigma threshold.

    This algorithm iteratively identifies values that exceed `mean + sigma * std`,
    removes them from consideration, and repeats until no more outliers are found.
    This is equivalent to MATLAB's `iterative_outlier_removal` from NoiseTools.

    Useful for automatic component selection in DSS applications, such as:
    - ZapLine: Selecting how many line-noise components to remove
    - Narrowband scan: Identifying significant frequency peaks
    - Any DSS where components need automatic thresholding

    Parameters
    ----------
    scores : ndarray
        Component scores (e.g., eigenvalues, power ratios).
        Higher values are considered more significant.
    sigma : float
        Sigma threshold for outlier detection. Default 3.0.
        Components with `score > mean + sigma * std` are outliers.

    Returns
    -------
    n_outliers : int
        Number of outliers (significant components) detected.

    Examples
    --------
    >>> from mne_denoise.dss.utils import iterative_outlier_removal
    >>> scores = np.array([0.9, 0.8, 0.2, 0.15, 0.1, 0.08])
    >>> n_significant = iterative_outlier_removal(scores, sigma=2.0)
    >>> print(f"Found {n_significant} significant components")

    References
    ----------
    NoiseTools: http://audition.ens.fr/adc/NoiseTools/
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
    """Select number of components using outlier detection.

    Convenience wrapper around `iterative_outlier_removal` for DSS eigenvalues.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues (component scores).
    threshold : float
        Sigma threshold for outlier detection. Default 3.0.

    Returns
    -------
    n_components : int
        Number of significant components to keep/remove.
    """
    return iterative_outlier_removal(eigenvalues, threshold)


def eigenvalue_ratio_selection(
    eigenvalues: np.ndarray, ratio_threshold: float = 2.0
) -> int:
    """Return the count before the first consecutive eigenvalue-ratio drop."""
    values = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    if values.size < 2:
        return int(values.size)
    for index in range(values.size - 1):
        if values[index + 1] < 1e-15:
            return index + 1
        if values[index] / values[index + 1] >= float(ratio_threshold):
            return index + 1
    return 0


def max_gap_selection(eigenvalues: np.ndarray, min_ratio: float = 1.2) -> int:
    """Return the count at the largest meaningful consecutive eigenvalue gap."""
    values = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    if values.size < 2:
        return int(values.size)
    ratios = values[:-1] / np.maximum(values[1:], 1e-15)
    index = int(np.argmax(ratios))
    return index + 1 if ratios[index] >= float(min_ratio) else 0


def detect_eigenvalue_knee(
    scores: np.ndarray,
    rel_floor: float = 0.01,
    min_ratio: float = 3.0,
) -> int:
    """Locate the knee/elbow in a sorted-descending eigenvalue spectrum.

    Finds the index ``k`` such that the largest log-space drop between
    consecutive eigenvalues occurs between ``scores[k-1]`` and ``scores[k]``,
    interpreting that drop as the boundary between a cluster of significant
    components and the noise floor. Returns ``k`` (the count of components
    above the knee).

    The drop must (a) occur above a relative floor — to avoid picking knees
    that sit entirely in the noise tail — and (b) exceed ``log10(min_ratio)``
    decades, so a smoothly decaying spectrum without a clear bimodal split
    correctly returns 0.

    Parameters
    ----------
    scores : ndarray
        Component scores in descending order (e.g., DSS eigenvalues).
    rel_floor : float, default=0.01
        Relative cutoff for valid knee anchors. Only drop positions ``i`` with
        ``scores[i] > scores[0] * rel_floor`` are considered. Default 0.01
        means anchors must be at least 1% of the maximum eigenvalue.
    min_ratio : float, default=3.0
        Minimum drop ratio (linear scale) between consecutive eigenvalues
        required to qualify as a knee. A drop of ``min_ratio=3.0`` corresponds
        to a 0.477-decade gap in log10 space.

    Returns
    -------
    n_above_knee : int
        Number of eigenvalues above the detected knee. Returns 0 when no
        qualifying knee exists (monotonic decay or all-noise spectra).

    Notes
    -----
    Complementary to :func:`iterative_outlier_removal`, which is robust when
    a small number of components stand out as statistical outliers but fails
    when many co-equal strong components are present (e.g., high-channel-count
    MEG with coherent line noise spread across many sensors). Use
    :func:`auto_select_components_robust` to combine both.

    Examples
    --------
    >>> import numpy as np
    >>> # User MEG case from Issue #34: 7 strong + 8 near-zero
    >>> evs = np.array(
    ...     [
    ...         0.989,
    ...         0.969,
    ...         0.715,
    ...         0.677,
    ...         0.584,
    ...         0.422,
    ...         0.164,
    ...         0.005,
    ...         0.0012,
    ...         1.5e-4,
    ...         8.6e-5,
    ...         7.6e-5,
    ...         4.7e-5,
    ...         3.8e-5,
    ...         3.2e-7,
    ...     ]
    ... )
    >>> detect_eigenvalue_knee(evs)
    7
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
    """Auto-select component count using outlier + knee detection.

    Layers :func:`iterative_outlier_removal` and :func:`detect_eigenvalue_knee`
    and returns the maximum of the two counts. This handles both regimes:

    - **Few outliers** (typical EEG): :func:`iterative_outlier_removal` finds
      the 1-2 dominant components.
    - **Many co-equal strong components** (typical high-channel-count MEG with
      coherent line noise): the outlier path returns 0 because the strong
      components don't stand out from each other; :func:`detect_eigenvalue_knee`
      catches the boundary to the noise floor.
    - **Clean spectrum** (monotonic decay, no clear bimodal split): both
      return 0 — no false removals.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues in descending order.
    sigma : float, default=3.0
        Sigma threshold for :func:`iterative_outlier_removal`.
    knee_rel_floor : float, default=0.01
        ``rel_floor`` for :func:`detect_eigenvalue_knee`.
    knee_min_ratio : float, default=3.0
        ``min_ratio`` for :func:`detect_eigenvalue_knee`.

    Returns
    -------
    n_components : int
        Suggested number of significant components.
    """
    n_outlier = iterative_outlier_removal(eigenvalues, sigma=sigma)
    n_knee = detect_eigenvalue_knee(
        eigenvalues, rel_floor=knee_rel_floor, min_ratio=knee_min_ratio
    )
    return int(max(n_outlier, n_knee))
