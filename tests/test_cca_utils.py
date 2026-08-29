"""Tests for the shared canonical correlation solver in mne_denoise._cca."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._cca import canonical_correlation


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(42)


def _centered_orthonormal(rng, n_samples, n_columns):
    """Return mean-free columns with unit sample covariance."""
    raw = rng.standard_normal((n_samples, n_columns + 1))
    raw -= raw.mean(axis=0, keepdims=True)
    basis, _ = np.linalg.qr(raw)
    return basis[:, :n_columns] * np.sqrt(n_samples - 1)


# ---------------------------------------------------------------------------
# Scientific contracts
# ---------------------------------------------------------------------------


def test_cca_has_the_analytical_spectrum_and_orthonormal_variates():
    """CCA recovers known correlations and orthogonal unit-variance pairs."""
    n_samples = 400
    latent = _centered_orthonormal(np.random.default_rng(0), n_samples, 4)
    expected = np.array([0.8, 0.35])
    X = latent[:, :2]
    Y = np.column_stack(
        (
            expected[0] * latent[:, 0] + np.sqrt(1.0 - expected[0] ** 2) * latent[:, 2],
            -expected[1] * latent[:, 1]
            + np.sqrt(1.0 - expected[1] ** 2) * latent[:, 3],
        )
    )

    _A, _B, correlations, U, V = canonical_correlation(X, Y)

    np.testing.assert_allclose(correlations, expected, atol=1e-12)
    assert np.all(correlations >= 0.0)
    np.testing.assert_allclose(
        U.T @ U / (n_samples - 1), np.eye(expected.size), atol=1e-12
    )
    np.testing.assert_allclose(
        V.T @ V / (n_samples - 1), np.eye(expected.size), atol=1e-12
    )
    pair_correlations = U.T @ V / (n_samples - 1)
    np.testing.assert_allclose(np.abs(np.diag(pair_correlations)), expected, atol=1e-12)
    np.testing.assert_allclose(
        pair_correlations - np.diag(np.diag(pair_correlations)), 0.0, atol=1e-12
    )


def test_cca_rank_deficient_recovers_the_nonzero_singular_geometry():
    """Rank-revealing CCA keeps the shared dimensions of singular inputs."""
    n_samples = 500
    latent = _centered_orthonormal(np.random.default_rng(1), n_samples, 3)
    X = np.column_stack((latent[:, 0], latent[:, 1], latent[:, 0] + 2 * latent[:, 1]))
    mixed = 0.6 * latent[:, 1] + 0.8 * latent[:, 2]
    Y = np.column_stack((latent[:, 0], mixed, 2 * latent[:, 0]))

    A, B, correlations, U, V = canonical_correlation(X, Y)

    assert A.shape[1] == B.shape[1] == correlations.size == 2
    np.testing.assert_allclose(correlations, [1.0, 0.6], atol=1e-10)
    np.testing.assert_allclose(U.T @ U / (n_samples - 1), np.eye(2), atol=1e-10)
    np.testing.assert_allclose(V.T @ V / (n_samples - 1), np.eye(2), atol=1e-10)


def test_cca_zero_rank_returns_empty(rng):
    """A constant block has no canonical dimensions."""
    X = np.ones((100, 4))
    Y = rng.standard_normal((100, 3))
    A, B, R, U, V = canonical_correlation(X, Y)

    assert A.shape[1] == B.shape[1] == R.size == U.shape[1] == V.shape[1] == 0


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


def test_weighted_cca_rejects_invalid_weights():
    """Weighted CCA validates its observation measure."""
    X = np.arange(20.0).reshape(10, 2)
    cases = [
        ("wrong shape", np.ones(9), "shape"),
        ("nonfinite", np.r_[np.ones(9), np.nan], "finite"),
        ("negative", np.r_[np.ones(9), -1.0], "non-negative"),
        ("zero sum", np.zeros(10), "positive sum"),
    ]
    for _label, weights, match in cases:
        with pytest.raises(ValueError, match=match):
            canonical_correlation(X, X, sample_weight=weights)


# ---------------------------------------------------------------------------
# Public validation
# ---------------------------------------------------------------------------


def test_cca_mismatched_samples_raises(rng):
    """Different sample counts are rejected at the solver boundary."""
    X = rng.standard_normal((100, 5))
    Y = rng.standard_normal((80, 3))
    with pytest.raises(ValueError, match="same number of samples"):
        canonical_correlation(X, Y)
