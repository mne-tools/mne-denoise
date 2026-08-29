"""Tests for component-selection helpers and behavioral contracts."""

from __future__ import annotations

import numpy as np

from mne_denoise.dss.selection import (
    auto_select_components,
    auto_select_components_robust,
    detect_eigenvalue_knee,
    iterative_outlier_removal,
)

# Realistic MEG-like spectrum: 7 strong components followed by a near-zero
# noise tail.
REALISTIC_MEG_EIGENVALUES = np.array(
    [
        9.88999359e-01,
        9.68951301e-01,
        7.14728232e-01,
        6.76753765e-01,
        5.83699080e-01,
        4.22202798e-01,
        1.63730711e-01,
        5.00206326e-03,
        1.18624482e-03,
        1.48571576e-04,
        8.64564508e-05,
        7.59793074e-05,
        4.66650479e-05,
        3.74553067e-05,
        3.18800906e-07,
    ]
)


def test_detect_eigenvalue_knee_contract():
    """Knee detection handles ordinary spectra and its documented gates."""
    cases = [
        ("single dominant", [0.95, 0.05, 0.04, 0.03, 0.02, 0.01], {}, 1),
        ("two dominant", [0.9, 0.8, 0.1, 0.08, 0.05, 0.02], {}, 2),
        ("smooth decay", [0.5, 0.47, 0.44, 0.40, 0.37, 0.33, 0.30], {}, 0),
        ("empty", [], {}, 0),
        ("single value", [0.5], {}, 1),
        ("all zero", np.zeros(5), {}, 0),
        ("minimum ratio rejects", [0.9, 0.45, 0.40, 0.35, 0.30], {"min_ratio": 3.0}, 0),
        ("minimum ratio accepts", [0.9, 0.45, 0.40, 0.35, 0.30], {"min_ratio": 1.5}, 1),
        ("relative floor", [0.9, 0.8, 0.7, 1e-5, 1e-10], {}, 3),
        ("relative floor excludes all", [0.9, 0.8, 0.1], {"rel_floor": 2.0}, 0),
    ]
    for label, eigenvalues, kwargs, expected in cases:
        actual = detect_eigenvalue_knee(np.asarray(eigenvalues), **kwargs)
        assert actual == expected, label


def test_robust_component_selection_contract():
    """Robust selection combines its methods and forwards their controls."""
    realistic_outlier = iterative_outlier_removal(REALISTIC_MEG_EIGENVALUES, sigma=3.0)
    realistic_knee = detect_eigenvalue_knee(REALISTIC_MEG_EIGENVALUES)
    realistic_robust = auto_select_components_robust(REALISTIC_MEG_EIGENVALUES)
    assert realistic_outlier == 0
    assert realistic_knee == 7
    assert realistic_robust == 7

    evs = np.array([100.0, 0.1, 0.09, 0.08, 0.07, 0.06])
    n_outlier = iterative_outlier_removal(evs, sigma=3.0)
    n_knee = detect_eigenvalue_knee(evs)
    assert auto_select_components_robust(evs) == max(n_outlier, n_knee)

    smooth = np.array([0.5, 0.47, 0.44, 0.40, 0.37, 0.33, 0.30])
    assert auto_select_components_robust(smooth) == 0

    gated = np.array([0.9, 0.45, 0.40, 0.35, 0.30])
    assert auto_select_components_robust(gated, knee_min_ratio=3.0) == 0
    assert auto_select_components_robust(gated, knee_min_ratio=1.5) >= 1


def test_iterative_outlier_removal_contract():
    """Outlier removal is conservative, monotonic in sigma, and total-safe."""
    extreme = np.array([1000.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert iterative_outlier_removal(extreme, sigma=3.0) >= 1

    obvious = np.array([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert iterative_outlier_removal(obvious, sigma=2.0) >= 1
    assert iterative_outlier_removal(obvious, sigma=10.0) <= iterative_outlier_removal(
        obvious, sigma=1.0
    )
    assert iterative_outlier_removal(np.full(5, 0.5), sigma=3.0) == 0
    assert iterative_outlier_removal(np.array([1.0, 0.1]), sigma=2.0) == 0
    assert iterative_outlier_removal(np.array([]), sigma=2.0) == 0


def test_auto_select_components_alias_unchanged():
    """The backwards-compatible alias remains the outlier selector."""
    scores = np.array([10.0, 0.5, 0.4, 0.3, 0.2])
    assert auto_select_components(scores, threshold=3.0) == iterative_outlier_removal(
        scores, sigma=3.0
    )
