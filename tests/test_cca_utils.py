"""Tests for the shared canonical correlation solver in mne_denoise._cca."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._cca import canonical_correlation


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_cca_basic_shapes(rng):
    """CCA returns correct shapes."""
    n, px, py = 200, 8, 4
    X = rng.standard_normal((n, px))
    Y = rng.standard_normal((n, py))
    A, B, R, U, V = canonical_correlation(X, Y)

    d = min(px, py)
    assert A.shape == (px, d)
    assert B.shape == (py, d)
    assert R.shape == (d,)
    assert U.shape == (n, d)
    assert V.shape == (n, d)


def test_cca_correlations_descending(rng):
    """Canonical correlations are sorted descending."""
    X = rng.standard_normal((300, 10))
    Y = rng.standard_normal((300, 6))
    _, _, R, _, _ = canonical_correlation(X, Y)
    assert np.all(np.diff(R) <= 1e-10)


def test_cca_correlations_bounded(rng):
    """Canonical correlations are in [0, 1]."""
    X = rng.standard_normal((200, 8))
    Y = rng.standard_normal((200, 5))
    _, _, R, _, _ = canonical_correlation(X, Y)
    assert np.all(R >= -1e-10)
    assert np.all(R <= 1.0 + 1e-10)


def test_cca_perfect_correlation():
    """Perfectly correlated signals give R ~= 1."""
    t = np.linspace(0, 1, 500)
    X = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    Y = X @ np.array([[0.5, 0.3], [-0.2, 0.8]])
    _, _, R, _, _ = canonical_correlation(X, Y)
    np.testing.assert_allclose(R[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(R[1], 1.0, atol=1e-6)


def test_cca_uncorrelated(rng):
    """Independent signals give low correlations."""
    X = rng.standard_normal((5000, 8))
    Y = rng.standard_normal((5000, 4))
    _, _, R, _, _ = canonical_correlation(X, Y)
    assert R.max() < 0.15


def test_cca_unit_variance(rng):
    """Canonical variates have unit variance (ddof=1)."""
    X = rng.standard_normal((300, 6))
    Y = rng.standard_normal((300, 4))
    _, _, _, U, V = canonical_correlation(X, Y)
    np.testing.assert_allclose(U.std(axis=0, ddof=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(V.std(axis=0, ddof=1), 1.0, atol=1e-10)


def test_weighted_cca_matches_repeated_observations(rng):
    """Integer observation weights match explicit row replication."""
    X = rng.standard_normal((80, 5))
    Y = X[:, :3] @ rng.standard_normal((3, 4)) + 0.2 * rng.standard_normal((80, 4))
    weights = rng.integers(1, 4, size=80)

    _, _, weighted, U, V = canonical_correlation(X, Y, sample_weight=weights)
    _, _, repeated, _, _ = canonical_correlation(
        np.repeat(X, weights, axis=0), np.repeat(Y, weights, axis=0)
    )

    np.testing.assert_allclose(weighted, repeated, atol=1e-12)
    np.testing.assert_allclose(
        np.sqrt(weights @ (U**2) / weights.sum()), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sqrt(weights @ (V**2) / weights.sum()), 1.0, atol=1e-12
    )


@pytest.mark.parametrize(
    "weights, match",
    [
        (np.ones(9), "shape"),
        (np.r_[np.ones(9), np.nan], "finite"),
        (np.r_[np.ones(9), -1.0], "non-negative"),
        (np.zeros(10), "positive sum"),
    ],
)
def test_weighted_cca_rejects_invalid_weights(weights, match):
    """Weighted CCA validates its observation measure."""
    X = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match=match):
        canonical_correlation(X, X, sample_weight=weights)


def test_cca_mismatched_samples_raises(rng):
    """Different n_samples raises ValueError."""
    X = rng.standard_normal((100, 5))
    Y = rng.standard_normal((80, 3))
    with pytest.raises(ValueError, match="same number of samples"):
        canonical_correlation(X, Y)


def test_cca_rank_deficient(rng):
    """Handles rank-deficient input gracefully."""
    X = rng.standard_normal((100, 4))
    base = rng.standard_normal((100, 2))
    Y = np.column_stack([base, base @ rng.standard_normal((2, 3))])
    _A, _B, R, _U, _V = canonical_correlation(X, Y)
    assert R.shape[0] <= 2


def test_cca_zero_rank_returns_empty(rng):
    """Constant input has zero rank and yields width-zero outputs."""
    X = np.ones((100, 4))
    Y = rng.standard_normal((100, 3))
    A, B, R, U, V = canonical_correlation(X, Y)
    assert A.shape == (4, 0)
    assert B.shape == (3, 0)
    assert R.shape == (0,)
    assert U.shape == (100, 0)
    assert V.shape == (100, 0)


# ---------------------------------------------------------------------------
# Documented hazards
#
# Both behaviours below are intentional properties of this solver. They are
# pinned here because callers must design around them; see the module warning.
# ---------------------------------------------------------------------------


def test_cca_coefficients_use_a_pivoted_coordinate_basis(rng):
    """Rank-deficient input puts literal zero rows in the coefficient matrix.

    The rank reduction selects columns of the identity by QR pivot order rather
    than spanning the data's row space, so ``A`` cannot be inverted to map
    component space back to feature space without annihilating whole features.
    """
    sources = rng.standard_normal((300, 3))
    X = sources @ rng.standard_normal((3, 6))  # rank 3 in 6 features
    Y = np.roll(X, 1, axis=0)
    A, _B, _R, _U, _V = canonical_correlation(X, Y)

    assert A.shape == (6, 3)
    zero_rows = np.flatnonzero(np.all(np.abs(A) < 1e-12, axis=1))
    assert zero_rows.size == 3, "expected one zero row per pivoted-out feature"

    # Inverting the coefficient matrix zeroes exactly those features.
    Xc = X - X.mean(axis=0, keepdims=True)
    round_trip = (A @ np.linalg.pinv(A)).T @ Xc.T
    assert np.allclose(round_trip[zero_rows], 0.0)

    # Least-squares back-projection against the data has no such failure.
    Z = A.T @ Xc.T
    mixing = np.linalg.lstsq(Z.T, Xc, rcond=None)[0].T
    np.testing.assert_allclose(mixing @ Z, Xc.T, atol=1e-8)


def test_cca_correlations_are_non_negative_for_anticorrelated_pairs():
    """A perfectly anti-correlated pair yields R = 1, not R = -1.

    ``R`` comes from singular values; the sign is absorbed into ``B``. Callers
    that read ``R`` as a signed correlation must account for this.
    """
    t = np.linspace(0, 20 * np.pi, 1000)
    X = np.column_stack([np.sin(t), np.cos(t)])
    Y = -X
    _A, B, R, U, V = canonical_correlation(X, Y)

    np.testing.assert_allclose(R, 1.0, atol=1e-8)
    assert np.all(B[np.abs(B) > 1e-8] * 1.0 != 0)
    # The anti-correlation survives in the variates themselves.
    signed = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(R.size)])
    np.testing.assert_allclose(np.abs(signed), 1.0, atol=1e-8)
