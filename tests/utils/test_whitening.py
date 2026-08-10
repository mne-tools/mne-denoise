"""Unit tests for whitening utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._spatial import apply_spatial_transform
from mne_denoise.dss.utils.whitening import (
    apply_covariance_transform,
    compute_data_covariance_whitener,
    compute_mne_sensor_whitener,
    map_spatial_matrices_to_sensor_space,
    whiten_from_data_covariance,
)


@pytest.mark.parametrize("shape", [(3, 20), (3, 10, 2)])
def test_apply_spatial_transform_channel_axis(shape):
    """Spatial matrices should be applied along the first axis."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape)
    matrix = np.diag([2.0, 3.0, 4.0])

    transformed = apply_spatial_transform(matrix, data)

    np.testing.assert_allclose(transformed[0], 2.0 * data[0])
    np.testing.assert_allclose(transformed[1], 3.0 * data[1])
    np.testing.assert_allclose(transformed[2], 4.0 * data[2])


def test_apply_spatial_transform_validates_shapes():
    """Malformed matrices and channel mismatches should be rejected clearly."""
    with pytest.raises(ValueError, match="matrix must be 2D"):
        apply_spatial_transform(np.ones(3), np.ones((3, 10)))
    with pytest.raises(ValueError, match="channel dimensions do not match"):
        apply_spatial_transform(np.eye(2), np.ones((3, 10)))


def test_apply_spatial_transform_chunked_matches_full():
    """Chunking changes memory use without changing the spatial result."""
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((4, 3))
    data = rng.standard_normal((3, 7, 31))

    full = apply_spatial_transform(matrix, data)
    chunked = apply_spatial_transform(matrix, data, chunk_size=19)

    np.testing.assert_allclose(chunked, full)


@pytest.mark.parametrize("chunk_size", [0, -1, 2.5, True])
def test_apply_spatial_transform_rejects_invalid_chunk_size(chunk_size):
    """The shared chunk size has one validated contract."""
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        apply_spatial_transform(np.eye(3), np.ones((3, 10)), chunk_size=chunk_size)


def test_apply_covariance_transform():
    """Covariance transforms should apply a congruence transformation."""
    matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])

    transformed = apply_covariance_transform(matrix, covariance)

    np.testing.assert_allclose(transformed, matrix @ covariance @ matrix.T)
    np.testing.assert_allclose(transformed, transformed.T)


def test_apply_covariance_transform_validates_shapes():
    """Malformed covariance transforms should be rejected clearly."""
    with pytest.raises(ValueError, match="matrix must be 2D"):
        apply_covariance_transform(np.ones(2), np.eye(2))
    with pytest.raises(ValueError, match="covariance must be square"):
        apply_covariance_transform(np.eye(2), np.ones((2, 3)))
    with pytest.raises(ValueError, match="dimensions do not match"):
        apply_covariance_transform(np.eye(2), np.eye(3))


def test_map_spatial_matrices_to_sensor_space():
    """Whitened filters and patterns should map to sensor coordinates."""
    filters = np.array([[1.0, 2.0], [3.0, 4.0]])
    patterns = np.array([[5.0, 6.0], [7.0, 8.0]])
    whitener = np.diag([0.5, 0.25])
    dewhitener = np.diag([2.0, 4.0])

    filters_sensor, patterns_sensor = map_spatial_matrices_to_sensor_space(
        filters,
        patterns,
        whitener=whitener,
        dewhitener=dewhitener,
    )

    np.testing.assert_allclose(filters_sensor, filters @ whitener)
    np.testing.assert_allclose(patterns_sensor, dewhitener @ patterns)


@pytest.mark.parametrize(
    ("filters", "patterns", "whitener", "dewhitener", "error"),
    [
        (np.ones(2), np.eye(2), np.eye(2), np.eye(2), "must be 2D"),
        (np.ones((2, 3)), np.eye(2), np.eye(2), np.eye(2), "filter and whitener"),
        (np.eye(2), np.ones((3, 2)), np.eye(2), np.eye(2), "dewhitener and pattern"),
        (
            np.ones((3, 2)),
            np.ones((2, 2)),
            np.eye(2),
            np.eye(2),
            "same number of components",
        ),
    ],
)
def test_map_spatial_matrices_to_sensor_space_validates_shapes(
    filters, patterns, whitener, dewhitener, error
):
    """Incompatible spatial matrices should fail before multiplication."""
    with pytest.raises(ValueError, match=error):
        map_spatial_matrices_to_sensor_space(
            filters,
            patterns,
            whitener=whitener,
            dewhitener=dewhitener,
        )


def test_compute_mne_sensor_whitener_numpy_scaling():
    """The array fallback should scale every channel independently."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 1000)) * np.array([1.0, 10.0, 100.0])[:, None]

    whitener, colorer = compute_mne_sensor_whitener(data)
    transformed = apply_spatial_transform(whitener, data)

    np.testing.assert_allclose(transformed.std(axis=1), 1.0)
    np.testing.assert_allclose(colorer @ whitener, np.eye(3))


def test_compute_mne_sensor_whitener_rejects_nonfinite_data():
    """Non-finite samples should fail before covariance processing."""
    data = np.ones((2, 10))
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_mne_sensor_whitener(data)


def test_whiten_identity_covariance():
    """Whitened data should have approximately identity covariance."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 16, 5000

    # Create correlated data
    mixing = rng.standard_normal((n_channels, n_channels))
    sources = rng.standard_normal((n_channels, n_samples))
    data = mixing @ sources

    whitened, W, D = whiten_from_data_covariance(data)

    # Check covariance is approximately identity
    cov = whitened @ whitened.T / n_samples
    np.testing.assert_allclose(cov, np.eye(whitened.shape[0]), atol=0.1)


def test_whiten_rank_deficient():
    """Whitening should handle rank-deficient data."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 16, 1000
    true_rank = 8

    # Create rank-deficient data
    sources = rng.standard_normal((true_rank, n_samples))
    mixing = rng.standard_normal((n_channels, true_rank))
    data = mixing @ sources

    whitened, W, D = whiten_from_data_covariance(data)

    # Should auto-detect reduced rank
    assert whitened.shape[0] <= true_rank + 1


def test_whiten_3d_data():
    """Whitening should work on 3D epoched data."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 20

    data = rng.standard_normal((n_channels, n_times, n_epochs))
    whitened, W, D = whiten_from_data_covariance(data)

    assert whitened.ndim == 3
    assert whitened.shape[1:] == (n_times, n_epochs)


def test_compute_data_covariance_whitener_matrices():
    """Whitener and dewhitener should be inverses."""
    rng = np.random.default_rng(42)
    n_channels = 8

    # Create covariance
    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    W, D, eigenvalues = compute_data_covariance_whitener(cov)

    # W @ D should be approximately identity (up to truncation)
    product = W @ D
    np.testing.assert_allclose(product, np.eye(W.shape[0]), atol=1e-10)


def test_compute_data_covariance_whitener_with_rank():
    """Test data-covariance whitening with explicit rank truncation."""
    rng = np.random.default_rng(42)
    n_channels = 10

    # Create full rank covariance
    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    W, D, eigenvalues = compute_data_covariance_whitener(cov, rank=5)

    # Should truncate to specified rank
    assert W.shape[0] == 5
    assert len(eigenvalues) == 5


def test_compute_data_covariance_whitener_no_variance_error():
    """Test data-covariance whitening rejects a zero covariance."""
    # All zeros = no variance
    cov = np.zeros((5, 5))

    with pytest.raises(ValueError, match="no significant variance"):
        compute_data_covariance_whitener(cov)


def test_compute_data_covariance_whitener_is_scale_invariant():
    """Positive covariance magnitude should not determine numerical validity."""
    covariance = np.diag([3.0, 2.0, 1.0]) * 1e-35

    whitener, _, eigenvalues = compute_data_covariance_whitener(covariance)

    np.testing.assert_allclose(
        apply_covariance_transform(whitener, covariance), np.eye(3)
    )
    assert eigenvalues.shape == (3,)


def test_compute_data_covariance_whitener_no_components_error():
    """A cutoff that removes every component should fail explicitly."""
    with pytest.raises(ValueError, match="No components"):
        compute_data_covariance_whitener(np.eye(3), reg=1.0)


@pytest.mark.parametrize(
    ("covariance", "error"),
    [
        (np.ones(3), "must be square"),
        (np.ones((2, 3)), "must be square"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.diag([1.0, -1.0]), "positive semi-definite"),
    ],
)
def test_compute_data_covariance_whitener_validates_covariance(covariance, error):
    """Only finite, square, positive semi-definite covariances are valid."""
    with pytest.raises(ValueError, match=error):
        compute_data_covariance_whitener(covariance)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rank": 0}, "rank must be positive"),
        ({"rank": 1.5}, "rank must be an int"),
        ({"rank": True}, "rank must be an int"),
        ({"reg": -1.0}, "reg must be a finite non-negative"),
        ({"reg": np.inf}, "reg must be a finite non-negative"),
        ({"reg": "small"}, "reg must be a real number"),
    ],
)
def test_compute_data_covariance_whitener_validates_parameters(kwargs, error):
    """Invalid rank and regularization settings should fail explicitly."""
    with pytest.raises((TypeError, ValueError), match=error):
        compute_data_covariance_whitener(np.eye(3), **kwargs)


def test_whiten_from_data_covariance_invalid_ndim():
    """Test data-covariance whitening rejects invalid dimensions."""
    data = np.array([1, 2, 3, 4, 5])  # 1D

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        whiten_from_data_covariance(data)
