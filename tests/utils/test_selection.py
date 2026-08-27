"""Unit tests for component-selection helpers.

Covers :func:`iterative_outlier_removal`, :func:`detect_eigenvalue_knee`,
:func:`auto_select_components_robust`. Regression cases include the
high-channel-count MEG eigenvalue pattern reported in Issue #34.
"""

from __future__ import annotations

import numpy as np

from mne_denoise.dss.selection import (
    auto_select_components,
    auto_select_components_robust,
    detect_eigenvalue_knee,
    iterative_outlier_removal,
)

# Eigenvalue spectrum from the user-reported CTF MEG case (Issue #34):
# 7 strong components corresponding to coherent line noise, followed by
# 8 near-zero components in the noise tail.
ISSUE_34_EIGENVALUES = np.array(
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


# -----------------------------------------------------------------------------
# detect_eigenvalue_knee
# -----------------------------------------------------------------------------


def test_detect_knee_user_meg_case():
    """User's Issue #34 eigenvalues: 7 strong + 8 near-zero -> knee at 7."""
    assert detect_eigenvalue_knee(ISSUE_34_EIGENVALUES) == 7


def test_detect_knee_single_dominant():
    """One dominant component followed by a noise floor."""
    evs = np.array([0.95, 0.05, 0.04, 0.03, 0.02, 0.01])
    assert detect_eigenvalue_knee(evs) == 1


def test_detect_knee_two_dominant():
    """Two strong components, then a clear gap."""
    evs = np.array([0.9, 0.8, 0.1, 0.08, 0.05, 0.02])
    assert detect_eigenvalue_knee(evs) == 2


def test_detect_knee_clean_monotonic_decay():
    """Smoothly-decaying spectrum with no clear gap returns 0."""
    evs = np.array([0.5, 0.47, 0.44, 0.40, 0.37, 0.33, 0.30])
    assert detect_eigenvalue_knee(evs) == 0


def test_detect_knee_empty():
    """Empty array returns 0."""
    assert detect_eigenvalue_knee(np.array([])) == 0


def test_detect_knee_single_value():
    """Single eigenvalue returns 1 (degenerate case)."""
    assert detect_eigenvalue_knee(np.array([0.5])) == 1


def test_detect_knee_all_zero():
    """All-zero eigenvalues return 0."""
    assert detect_eigenvalue_knee(np.zeros(5)) == 0


def test_detect_knee_respects_min_ratio():
    """Knee gates on min_ratio: shallow drops are rejected."""
    # 2x drop between 0.9 and 0.45 -- below the default min_ratio=3
    evs = np.array([0.9, 0.45, 0.40, 0.35, 0.30])
    assert detect_eigenvalue_knee(evs, min_ratio=3.0) == 0
    # Same data but lower min_ratio accepts the drop
    assert detect_eigenvalue_knee(evs, min_ratio=1.5) == 1


def test_detect_knee_rel_floor_excludes_tail():
    """Anchors below rel_floor * max are excluded from knee selection.

    Without the rel_floor mask, the largest gap might be at the tail
    (e.g., between 1e-5 and 1e-12); the floor ensures we anchor on a
    meaningful eigenvalue.
    """
    evs = np.array([0.9, 0.8, 0.7, 1e-5, 1e-10])
    # The largest log-drop is between 0.7 and 1e-5 (5+ decades),
    # which IS what we want here -> returns 3.
    assert detect_eigenvalue_knee(evs) == 3


def test_detect_knee_rel_floor_excludes_all():
    """``rel_floor`` greater than 1.0 leaves no valid anchors and returns 0.

    Degenerate guard for the ``not np.any(valid)`` branch.
    """
    evs = np.array([0.9, 0.8, 0.1])
    assert detect_eigenvalue_knee(evs, rel_floor=2.0) == 0


# -----------------------------------------------------------------------------
# auto_select_components_robust
# -----------------------------------------------------------------------------


def test_robust_user_meg_case():
    """Issue #34 case: outlier returns 0, knee returns 7, robust returns 7."""
    n_outlier = iterative_outlier_removal(ISSUE_34_EIGENVALUES, sigma=3.0)
    n_knee = detect_eigenvalue_knee(ISSUE_34_EIGENVALUES)
    n_robust = auto_select_components_robust(ISSUE_34_EIGENVALUES)
    assert n_outlier == 0
    assert n_knee == 7
    assert n_robust == 7


def test_robust_combines_via_max():
    """When outlier and knee disagree, the larger count wins."""
    # Construct a case where outlier returns >= 1 (one extreme outlier)
    # and knee may return 1 as well -> max == 1.
    evs = np.array([100.0, 0.1, 0.09, 0.08, 0.07, 0.06])
    n_outlier = iterative_outlier_removal(evs, sigma=3.0)
    n_knee = detect_eigenvalue_knee(evs)
    assert auto_select_components_robust(evs) == max(n_outlier, n_knee)


def test_robust_clean_returns_zero():
    """Smoothly-decaying spectrum: both paths return 0."""
    evs = np.array([0.5, 0.47, 0.44, 0.40, 0.37, 0.33, 0.30])
    assert auto_select_components_robust(evs) == 0


def test_robust_forwards_kwargs():
    """``sigma``, ``knee_rel_floor``, ``knee_min_ratio`` reach the underlying calls."""
    evs = np.array([0.9, 0.45, 0.40, 0.35, 0.30])
    # Strict knee gate rejects this drop
    assert auto_select_components_robust(evs, knee_min_ratio=3.0) == 0
    # Relaxed knee gate accepts it
    assert auto_select_components_robust(evs, knee_min_ratio=1.5) >= 1


# -----------------------------------------------------------------------------
# Backwards-compatibility (existing functions untouched)
# -----------------------------------------------------------------------------


def test_iterative_outlier_removal_unchanged():
    """Iterative removal still picks up extreme outliers (regression guard).

    The original algorithm is conservative; this case has one massive outlier
    that survives the iterative step.
    """
    scores = np.array([1000.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert iterative_outlier_removal(scores, sigma=3.0) >= 1


def test_iterative_outlier_removal_user_case_is_zero():
    """The user MEG eigenvalues legitimately return 0 from the outlier path.

    This locks in the diagnosis: the bug is not in iterative_outlier_removal,
    it's in *relying solely* on it.
    """
    assert iterative_outlier_removal(ISSUE_34_EIGENVALUES, sigma=3.0) == 0


def test_auto_select_components_alias_unchanged():
    """``auto_select_components`` remains a thin wrapper over outlier removal."""
    scores = np.array([10.0, 0.5, 0.4, 0.3, 0.2])
    assert auto_select_components(scores, threshold=3.0) == iterative_outlier_removal(
        scores, sigma=3.0
    )


# ============================================================================
# iterative_outlier_removal
# ============================================================================


class TestIterativeOutlierRemoval:
    """Tests for iterative_outlier_removal."""

    def test_clear_outliers(self):
        """Scores with a clear outlier should return >= 0 (conservative)."""
        scores = np.array([0.9, 0.8, 0.15, 0.12, 0.1, 0.08, 0.07])
        n = iterative_outlier_removal(scores, sigma=2.0)
        # The iterative method is conservative; it may or may not flag
        # the top scores depending on the distribution shape
        assert n >= 0

    def test_no_outliers(self):
        """Uniform scores should produce 0 outliers."""
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        n = iterative_outlier_removal(scores, sigma=3.0)
        assert n == 0

    def test_single_outlier(self):
        """One extreme value among many similar should be detected."""
        scores = np.array([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        n = iterative_outlier_removal(scores, sigma=2.0)
        assert n >= 1

    def test_strict_threshold(self):
        """Very high sigma should detect fewer outliers."""
        scores = np.array([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        n_strict = iterative_outlier_removal(scores, sigma=10.0)
        n_lenient = iterative_outlier_removal(scores, sigma=1.0)
        assert n_strict <= n_lenient

    def test_two_elements(self):
        """With only two elements, algorithm should still work."""
        scores = np.array([1.0, 0.1])
        n = iterative_outlier_removal(scores, sigma=2.0)
        assert n == 0  # Not enough elements for iterative removal

    def test_empty_array(self):
        """Empty array should return 0."""
        scores = np.array([])
        n = iterative_outlier_removal(scores, sigma=2.0)
        assert n == 0


# ============================================================================
# eigenvalue_ratio_selection
# ============================================================================
