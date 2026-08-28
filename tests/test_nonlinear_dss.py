"""Unit tests for nonlinear (iterative) DSS."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss import iterative_dss
from mne_denoise.dss.denoisers import KurtosisDenoiser, VarianceMaskDenoiser
from mne_denoise.dss.nonlinear import iterative_dss_one

# =============================================================================
# iterative_dss_one - Core Single Component Algorithm
# =============================================================================


def test_iterative_dss_one_basic():
    """iterative_dss_one should extract a single component."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 8, 1000
    X_whitened = rng.standard_normal((n_whitened, n_times))

    denoiser = KurtosisDenoiser()
    w, source, n_iter, converged = iterative_dss_one(X_whitened, denoiser)

    assert w.shape == (n_whitened,)
    assert source.shape == (n_times,)
    assert n_iter > 0
    assert isinstance(converged, bool)
    # Weight should be normalized
    assert_allclose(np.linalg.norm(w), 1.0, atol=1e-10)


def test_iterative_dss_one_uses_alpha_and_beta_controls():
    """Scalar and callable update controls produce valid normalized weights."""
    rng = np.random.default_rng(42)
    X_whitened = rng.standard_normal((5, 500))

    def beta_func(source):
        return -np.mean(1 - np.tanh(source) ** 2)

    for alpha, beta in [(0.5, -0.5), (lambda source: 1 / np.std(source), beta_func)]:
        weight, source, _, _ = iterative_dss_one(
            X_whitened, KurtosisDenoiser(), alpha=alpha, beta=beta
        )
        assert weight.shape == (X_whitened.shape[0],)
        assert source.shape == (X_whitened.shape[1],)
        assert_allclose(np.linalg.norm(weight), 1.0, atol=1e-10)


def test_iterative_dss_one_early_convergence_is_reported():
    """A loose tolerance reports convergence after the completed update."""
    rng = np.random.default_rng(102)
    X_whitened = rng.standard_normal((5, 500))

    _, _, n_iter, converged = iterative_dss_one(
        X_whitened, np.tanh, max_iter=8, tol=1e6, random_state=0
    )

    assert converged
    assert n_iter == 1


def test_iterative_dss_one_nonconvergence_has_meaningful_diagnostics():
    """A signal-killing denoiser reaches the iteration budget without lying."""
    rng = np.random.default_rng(103)
    X_whitened = rng.standard_normal((5, 500))

    _, _, n_iter, converged = iterative_dss_one(
        X_whitened,
        lambda source: np.zeros_like(source),
        max_iter=3,
        random_state=0,
    )

    assert not converged
    assert n_iter == 3


# =============================================================================
# iterative_dss - Multi-Component Extraction
# =============================================================================


def test_iterative_dss_methods_return_component_contract():
    """Deflation and symmetric solvers return the same component layout."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 2000))

    for method in ("deflation", "symmetric"):
        filters, sources, patterns, conv = iterative_dss(
            data, KurtosisDenoiser(), n_components=3, method=method, max_iter=10
        )

        assert filters.shape == (3, 8)
        assert sources.shape == (3, 2000)
        assert patterns.shape == (8, 3)
        assert conv.shape == (3, 2)
        assert np.all((conv[:, 0] >= 1) & (conv[:, 0] <= 10))
        assert np.all(np.isin(conv[:, 1], [0.0, 1.0]))


def test_iterative_dss_invalid_method():
    """iterative_dss should raise error for invalid method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 500))

    denoiser = KurtosisDenoiser()

    with pytest.raises(ValueError, match="Unknown method"):
        iterative_dss(data, denoiser, n_components=2, method="invalid")


def test_iterative_dss_with_rank():
    """iterative_dss should respect rank parameter."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 1000))

    denoiser = KurtosisDenoiser()
    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, rank=5, max_iter=10
    )

    assert filters.shape == (3, 10)


def test_iterative_dss_with_w_init():
    """iterative_dss should accept initial weight matrix."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, 1000))

    denoiser = KurtosisDenoiser()

    # Create initial weights (n_components, n_whitened)
    w_init = rng.standard_normal((3, 6))

    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, w_init=w_init, max_iter=10
    )

    assert filters.shape == (3, 6)


# =============================================================================
# Functional Tests with Known Expected Outputs
# =============================================================================


def test_iterative_dss_extracts_and_orders_known_sources():
    """The nonlinear bias extracts the high-kurtosis source first."""
    rng = np.random.default_rng(42)
    n_samples = 2_000

    square = np.sign(np.sin(np.linspace(0, 10, n_samples)))
    square = (square - square.mean()) / square.std()
    gaussian = rng.standard_normal(n_samples)
    gaussian = (gaussian - gaussian.mean()) / gaussian.std()
    expected_sources = np.vstack([square, gaussian])

    angle = 0.37
    mixing = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    data = mixing @ expected_sources

    filters, extracted, _, conv = iterative_dss(
        data,
        np.tanh,
        n_components=2,
        method="deflation",
        max_iter=100,
        tol=1e-8,
        random_state=42,
    )

    repeat_filters, repeat_extracted, _, _ = iterative_dss(
        data,
        np.tanh,
        n_components=2,
        method="deflation",
        max_iter=100,
        tol=1e-8,
        random_state=42,
    )
    assert_allclose(np.abs(repeat_filters), np.abs(filters))
    assert_allclose(np.abs(repeat_extracted), np.abs(extracted))

    correlations = np.abs(np.corrcoef(extracted, expected_sources)[:2, 2:])
    assert correlations[0, 0] > 0.99
    assert correlations[1, 1] > 0.99
    scores = np.mean(extracted * np.tanh(extracted), axis=1)
    assert scores[0] > scores[1]
    assert_allclose(conv[:, 1], 1.0)


# =============================================================================
# Edge Cases
# =============================================================================


def test_iterative_dss_one_tiny_init():
    """iterative_dss_one should handle tiny initial weights."""
    rng = np.random.default_rng(42)
    n_whitened, n_times = 5, 500
    X_whitened = rng.standard_normal((n_whitened, n_times))

    # Very small initial weight
    w_init = np.ones(n_whitened) * 1e-20

    denoiser = KurtosisDenoiser()
    w, source, n_iter, converged = iterative_dss_one(
        X_whitened, denoiser, w_init=w_init
    )

    # Should still produce valid output
    assert w.shape == (n_whitened,)
    assert_allclose(np.linalg.norm(w), 1.0, atol=1e-10)


def test_iterative_dss_symmetric_w_init():
    """iterative_dss symmetric should accept w_init."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, 1000))

    denoiser = KurtosisDenoiser()

    w_init = rng.standard_normal((3, 6))

    filters, sources, patterns, conv = iterative_dss(
        data, denoiser, n_components=3, method="symmetric", w_init=w_init, max_iter=10
    )

    assert filters.shape == (3, 6)


def test_iterative_dss_orthogonality_check():
    """Iterative DSS components should be orthogonal (decorrelated) in whitened space."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 5, 500
    data = rng.standard_normal((n_ch, n_times))

    # Make channels correlated
    data = np.dot(rng.standard_normal((n_ch, n_ch)), data)

    denoiser = VarianceMaskDenoiser()  # Any denoiser

    # Use deflation method
    filters_def, sources_def, _, _ = iterative_dss(
        data, denoiser, n_components=3, method="deflation", random_state=42
    )

    # Sources should be decorrelated
    corr_def = np.corrcoef(sources_def)
    off_diag_def = corr_def - np.diag(np.diag(corr_def))
    assert np.max(np.abs(off_diag_def)) < 1e-10, "Deflation sources not decorrelated"

    # Use symmetric method
    filters_sym, sources_sym, _, _ = iterative_dss(
        data, denoiser, n_components=3, method="symmetric", random_state=42
    )

    corr_sym = np.corrcoef(sources_sym)
    off_diag_sym = corr_sym - np.diag(np.diag(corr_sym))
    assert np.max(np.abs(off_diag_sym)) < 1e-10, "Symmetric sources not decorrelated"


def test_iterative_dss_one_degenerate_signal():
    """iterative_dss_one should handle signal killing (norm < 1e-12)."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 3, 100
    X = rng.standard_normal((n_ch, n_times))

    # Stateful denoiser that kills the signal once then works
    class FlakyDenoiser:
        def __init__(self):
            self.killed = False

        def __call__(self, data):
            if not self.killed:
                self.killed = True
                return np.zeros_like(data)
            return data  # Identity map otherwise

    denoiser = FlakyDenoiser()
    w_init = np.array([1.0, 0.0, 0.0])

    w, source, n_iter, converged = iterative_dss_one(
        X, denoiser, w_init=w_init, max_iter=10, random_state=rng
    )

    # It should have reinitialized w (randomly) and then converged
    assert denoiser.killed
    assert not np.allclose(w, w_init)  # Should have changed


def test_iterative_dss_degenerate_orthogonalization():
    """iterative_dss should handle degenerate components during orthogonalization."""
    rng = np.random.default_rng(42)
    n_samples = 100
    # Create rank-deficient data where components are identical
    v = rng.standard_normal(n_samples)
    X = np.vstack([v, v, v])  # Rank 1 data

    # Use a simple identity denoiser
    def identity_denoiser(data):
        return data

    # Mock whitening step to ensure it returns 2 components despite rank 1 data
    X_white_mock = np.zeros((2, n_samples))
    X_white_mock[0] = v

    # Initialize BOTH components to the SAME vector to force collapse after orthogonalization
    w_init_force = np.array([[1.0, 0.0], [1.0, 0.0]])

    with patch("mne_denoise.dss.nonlinear.whiten_from_data_covariance") as mock_whiten:
        mock_whiten.return_value = (
            X_white_mock,
            np.eye(2, 3),  # Fake whitener
            np.eye(3, 2),  # Fake dewhitener
        )

        filters, _, _, _ = iterative_dss(
            X, identity_denoiser, n_components=2, w_init=w_init_force, random_state=rng
        )

    # Should stay at 2 components and re-initialize the degenerate one
    assert filters.shape == (2, 3)
    assert not np.allclose(filters[1], 0)
