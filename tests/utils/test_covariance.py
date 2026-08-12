"""Unit tests for covariance utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise import compute_covariance
from mne_denoise._covariance import _ledoit_wolf_shrinkage


def test_empirical_covariance_shape():
    """Empirical covariance should return correct shape."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 5, 100
    data = rng.standard_normal((n_channels, n_samples))

    cov = compute_covariance(data)

    assert cov.shape == (n_channels, n_channels)
    assert_allclose(cov, cov.T)  # Symmetry


def test_empirical_covariance_value():
    """Empirical covariance should match numpy calculation."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 1000))

    # Center manually for comparison
    data_centered = data - data.mean(axis=1, keepdims=True)
    expected = data_centered @ data_centered.T / 1000

    cov = compute_covariance(data, method="empirical")

    assert_allclose(cov, expected)


def test_empirical_covariance_assume_centered():
    """Already-centered data should not be centered a second time."""
    data = np.array([[2.0, 3.0, 4.0], [-1.0, 0.0, 1.0]])

    cov = compute_covariance(data, assume_centered=True)

    assert_allclose(cov, data @ data.T / data.shape[1])


def test_empirical_covariance_chunked_matches_full():
    """Chunking should only change peak memory use."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((7, 503))
    weights = rng.uniform(0.1, 1.0, data.shape[1])

    full = compute_covariance(data, weights=weights)
    chunked = compute_covariance(data, weights=weights, chunk_size=47)

    assert_allclose(chunked, full, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [0, -1, 2.5, True])
def test_empirical_covariance_rejects_invalid_chunk_size(chunk_size):
    """Chunk size must be a positive integer when supplied."""
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        compute_covariance(np.ones((3, 10)), chunk_size=chunk_size)


def test_non_empirical_covariance_rejects_chunking():
    """Chunking is limited to the empirical covariance path."""
    with pytest.raises(ValueError, match="empirical"):
        compute_covariance(np.ones((3, 10)), method="oas", chunk_size=4)


def test_shrinkage_covariance_identity():
    """Shrinkage should return identity for identity input (mostly)."""
    # Ideally if data is uncorrelated, shrinkage target (diagonal) matches empirical (diagonal)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))  # Uncorrelated

    # With enough samples, empirical is close to identity
    # Shrinkage target is also identity-like
    cov_shrink = compute_covariance(data, method="shrinkage")

    # Check diagonal dominance
    diag = np.diag(cov_shrink)
    off_diag = cov_shrink - np.diag(diag)

    assert np.all(diag > 0.8)
    assert np.all(np.abs(off_diag) < 0.3)


def test_ledoit_wolf_shrinkage_calculation():
    """Test internal LW shrinkage calculation."""
    rng = np.random.default_rng(42)
    mixing = rng.standard_normal((10, 10))
    source = rng.standard_normal((10, 5000))
    data = mixing @ source
    data = data - data.mean(axis=1, keepdims=True)

    shrinkage = _ledoit_wolf_shrinkage(data)

    assert shrinkage < 0.2

    data_small = data[:, :15]  # 15 samples
    data_small = data_small - data_small.mean(axis=1, keepdims=True)
    shrinkage_small = _ledoit_wolf_shrinkage(data_small)
    assert shrinkage_small > shrinkage


def test_covariance_methods():
    """Test that all method strings are accepted."""
    data = np.random.randn(3, 50)

    compute_covariance(data, method="empirical")
    compute_covariance(data, method="shrinkage")

    compute_covariance(data, method="oas")

    # Invalid method
    with pytest.raises(ValueError, match="Unknown covariance method"):
        compute_covariance(data, method="invalid_method")


def test_covariance_mcd_method():
    """Test MCD (Minimum Covariance Determinant) method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))

    cov = compute_covariance(data, method="mcd")

    assert cov.shape == (3, 3)
    assert_allclose(cov, cov.T)  # Symmetric


def test_covariance_3d_data():
    """Test covariance with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))

    cov = compute_covariance(data, method="empirical")

    assert cov.shape == (n_ch, n_ch)


def test_covariance_3d_with_weights():
    """Test covariance with 3D data and per-time-point weights."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))

    # Weights matching n_times (will be tiled)
    weights = rng.uniform(0.5, 1.5, n_times)

    cov = compute_covariance(data, weights=weights)

    assert cov.shape == (n_ch, n_ch)


def test_covariance_3d_with_full_weights():
    """Test covariance with 3D data and full-length weights."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))

    # Weights matching total samples (n_times * n_epochs)
    weights = rng.uniform(0.5, 1.5, n_times * n_epochs)

    cov = compute_covariance(data, weights=weights)

    assert cov.shape == (n_ch, n_ch)


def test_covariance_weights_mismatch_error():
    """Test covariance raises error for weights length mismatch."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    weights = rng.uniform(0, 1, 50)  # Wrong length

    with pytest.raises(ValueError, match="does not match"):
        compute_covariance(data, weights=weights)


def test_covariance_zero_weights_error():
    """Test covariance raises error when sum of weights is zero."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    weights = np.zeros(100)  # All zero weights

    with pytest.raises(ValueError, match="Sum of weights is zero"):
        compute_covariance(data, weights=weights)


def test_covariance_weighted_non_empirical_error():
    """Test that weighted covariance only works with empirical method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    weights = rng.uniform(0.5, 1.5, 100)

    with pytest.raises(ValueError, match="not implemented"):
        compute_covariance(data, weights=weights, method="shrinkage")


def test_covariance_explicit_shrinkage():
    """Test covariance with explicit shrinkage parameter."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))

    cov = compute_covariance(data, method="shrinkage", shrinkage=0.5)

    assert cov.shape == (3, 3)


def test_weighted_covariance_correctness():
    """Test weighted covariance computation produces correct results."""
    n_samples = 100
    n_channels = 3
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_samples)

    # Case 1: Weights = ones -> should equal empirical covariance
    weights_ones = np.ones(n_samples)
    cov_w = compute_covariance(data, weights=weights_ones, method="empirical")

    # Manual empirical
    data_centered = data - data.mean(axis=1, keepdims=True)
    cov_emp = data_centered @ data_centered.T / n_samples
    assert_allclose(cov_w, cov_emp, atol=1e-7, err_msg="Weighted cov (ones) mismatch")

    # Case 2: Zero weights on half the data -> should equal covariance of first half
    weights_half = np.zeros(n_samples)
    weights_half[:50] = 1.0

    # Compute on subset manually
    data_sub = data[:, :50]
    data_sub_c = data_sub - data_sub.mean(axis=1, keepdims=True)
    cov_sub = data_sub_c @ data_sub_c.T / 50.0  # Sum of weights is 50

    cov_w_half = compute_covariance(data, weights=weights_half, method="empirical")
    assert_allclose(
        cov_w_half, cov_sub, atol=1e-7, err_msg="Weighted cov (masking) mismatch"
    )
