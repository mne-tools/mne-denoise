"""Unit tests for covariance utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise import compute_covariance
from mne_denoise._covariance import _ledoit_wolf_shrinkage


def test_empirical_covariance_contract():
    """Empirical covariance has the expected formula, shape, and weighting."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 1000))
    centered = data - data.mean(axis=1, keepdims=True)
    expected = centered @ centered.T / data.shape[1]

    covariance = compute_covariance(data, method="empirical")
    assert covariance.shape == (3, 3)
    assert_allclose(covariance, covariance.T)
    assert_allclose(covariance, expected)

    already_centered = np.array([[2.0, 3.0, 4.0], [-1.0, 0.0, 1.0]])
    assert_allclose(
        compute_covariance(already_centered, assume_centered=True),
        already_centered @ already_centered.T / already_centered.shape[1],
    )

    weights = np.ones(data.shape[1])
    assert_allclose(compute_covariance(data, weights=weights), covariance)
    subset = data[:, :50]
    subset_weights = np.r_[np.ones(50), np.zeros(data.shape[1] - 50)]
    subset_centered = subset - subset.mean(axis=1, keepdims=True)
    assert_allclose(
        compute_covariance(data, weights=subset_weights),
        subset_centered @ subset_centered.T / subset.shape[1],
    )


def test_empirical_covariance_chunking_and_validation():
    """Chunked empirical covariance is exact and rejects unsupported settings."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((7, 503))
    weights = rng.uniform(0.1, 1.0, data.shape[1])
    full = compute_covariance(data, weights=weights)
    chunked = compute_covariance(data, weights=weights, chunk_size=47)
    assert_allclose(chunked, full, rtol=1e-12, atol=1e-12)

    for label, chunk_size in (
        ("zero", 0),
        ("negative", -1),
        ("float", 2.5),
        ("bool", True),
    ):
        with pytest.raises((TypeError, ValueError), match="chunk_size"):
            compute_covariance(np.ones((3, 10)), chunk_size=chunk_size)
    with pytest.raises(ValueError, match="empirical"):
        compute_covariance(np.ones((3, 10)), method="oas", chunk_size=4)


def test_covariance_methods_contract():
    """Supported covariance methods return finite symmetric sensor matrices."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))
    for method in ("shrinkage", "oas", "mcd"):
        covariance = compute_covariance(data, method=method)
        assert covariance.shape == (5, 5), method
        assert np.all(np.isfinite(covariance)), method
        assert_allclose(covariance, covariance.T)

    explicit = compute_covariance(data, method="shrinkage", shrinkage=0.5)
    assert explicit.shape == (5, 5)
    with pytest.raises(ValueError, match="Unknown covariance method"):
        compute_covariance(data, method="invalid_method")

    uncorrelated = compute_covariance(rng.standard_normal((5, 200)), method="shrinkage")
    diagonal = np.diag(uncorrelated)
    off_diagonal = uncorrelated - np.diag(diagonal)
    assert np.all(diagonal > 0.8)
    assert np.all(np.abs(off_diagonal) < 0.3)


def test_ledoit_wolf_shrinkage_calculation():
    """The shrinkage estimate increases as the sample support becomes small."""
    rng = np.random.default_rng(42)
    mixing = rng.standard_normal((10, 10))
    source = rng.standard_normal((10, 5000))
    data = mixing @ source
    data -= data.mean(axis=1, keepdims=True)

    shrinkage = _ledoit_wolf_shrinkage(data)
    small = data[:, :15]
    small -= small.mean(axis=1, keepdims=True)
    assert shrinkage < 0.2
    assert _ledoit_wolf_shrinkage(small) > shrinkage


def test_covariance_3d_weight_contract():
    """Epoched weights broadcast and flatten in the documented C order."""
    data = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]],
            [[3.0, 30.0], [5.0, 50.0], [9.0, 90.0]],
        ]
    )
    time_weights = np.array([1.0, 2.0, 7.0])
    flattened = data.reshape(data.shape[0], -1)
    broadcast = compute_covariance(data, weights=time_weights)
    flat = compute_covariance(flattened, weights=np.repeat(time_weights, data.shape[2]))
    assert broadcast.shape == (2, 2)
    assert_allclose(broadcast, flat)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 5, 4))
    matrix_weights = np.arange(1.0, 21.0).reshape(5, 4)
    assert_allclose(
        compute_covariance(data, weights=matrix_weights),
        compute_covariance(data.reshape(3, -1), weights=matrix_weights.reshape(-1)),
    )
    full_weights = rng.uniform(0.5, 1.5, data.size // data.shape[0])
    assert compute_covariance(data, weights=full_weights).shape == (3, 3)


def test_covariance_weight_validation():
    """Weight shape, sign, finiteness, and estimator compatibility are explicit."""
    data = np.ones((2, 3))
    cases = [
        (np.ones(2), "does not match"),
        (np.zeros(3), "Sum of weights must be positive"),
        (np.array([1.0, -1.0, 1.0]), "non-negative"),
        (np.array([1.0, np.nan, 1.0]), "finite"),
        (np.ones((3, 1)), "one-dimensional"),
    ]
    for weights, message in cases:
        with pytest.raises(ValueError, match=message):
            compute_covariance(data, weights=weights)
    with pytest.raises(ValueError, match="For 3D data"):
        compute_covariance(np.ones((2, 3, 4)), weights=np.ones((4, 3)))
    with pytest.raises(ValueError, match="not implemented"):
        compute_covariance(np.ones((2, 3)), weights=np.ones(3), method="shrinkage")
