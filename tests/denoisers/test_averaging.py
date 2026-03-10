"""Unit tests for averaging denoisers (AverageBias)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.averaging import AverageBias


def test_average_bias_epochs():
    """Test AverageBias(axis='epochs') on simple 3D data."""
    # Create data: (n_channels, n_times, n_epochs)
    n_epochs = 10
    data = np.zeros((1, 5, n_epochs))

    # Trials have a constant component (1) + noise
    # We make trial 0 have value 1, trial 1 have value 3 -> mean 2
    data[0, :, :] = 2

    bias = AverageBias(axis="epochs")
    biased = bias.apply(data)

    # Result should be the mean repeated
    expected = np.ones((1, 5, n_epochs)) * 2
    assert_allclose(biased, expected)


def test_average_bias_epochs_weighted():
    """Test AverageBias(axis='epochs') with weights."""
    # 2 epochs
    data = np.zeros((1, 1, 2))
    data[0, 0, 0] = 10
    data[0, 0, 1] = 20

    # Weighted average: 0.8 * 10 + 0.2 * 20 = 8 + 4 = 12
    weights = [0.8, 0.2]
    bias = AverageBias(axis="epochs", weights=weights)
    biased = bias.apply(data)

    assert_allclose(biased[0, 0, :], 12)


def test_average_bias_epochs_errors():
    """Test error handling for epochs axis."""
    bias = AverageBias(axis="epochs")
    # 2D input should fail (expects epoched)
    data = np.zeros((2, 10))
    # Note: Error message changed to "AverageBias(axis='epochs') requires 3D data" in source
    with pytest.raises(
        ValueError, match="AverageBias.*axis='epochs'.*requires 3D data"
    ):
        bias.apply(data)


def test_average_bias_weight_mismatch():
    """Test error when weights length doesn't match epochs."""
    data = np.zeros((1, 5, 10))  # 10 epochs
    weights = [1, 2, 3]  # Only 3 weights

    bias = AverageBias(axis="epochs", weights=weights)
    with pytest.raises(ValueError, match="weights length.*must match"):
        bias.apply(data)


def test_average_bias_datasets():
    """Test AverageBias(axis='datasets') uniform averaging."""
    rng = np.random.RandomState(0)
    data = rng.randn(4, 3, 50)  # 4 datasets, 3 channels, 50 times
    bias = AverageBias(axis="datasets")
    biased = bias.apply(data)
    expected_mean = data.mean(axis=0)
    for i in range(4):
        assert_allclose(biased[i], expected_mean)


def test_average_bias_datasets_weighted():
    """Test AverageBias(axis='datasets') weighted averaging."""
    rng = np.random.RandomState(1)
    data = rng.randn(3, 2, 20)
    weights = np.array([1.0, 2.0, 3.0])
    bias = AverageBias(axis="datasets", weights=weights)
    biased = bias.apply(data)
    norm_w = weights / weights.sum()
    expected = np.tensordot(norm_w, data, axes=(0, 0))
    for i in range(3):
        assert_allclose(biased[i], expected)


def test_average_bias_datasets_errors():
    """Test error handling for datasets axis."""
    bias = AverageBias(axis="datasets")
    # 2D input should raise ValueError
    with pytest.raises(ValueError, match="requires 3D"):
        bias.apply(np.ones((3, 10)))

    # Weights length mismatch
    data = np.ones((3, 2, 10))
    bias_w = AverageBias(axis="datasets", weights=np.ones(5))
    with pytest.raises(ValueError, match="weights length"):
        bias_w.apply(data)


def test_average_bias_properties():
    """Test general AverageBias properties."""
    # Output shape
    data = np.random.randn(5, 4, 30)
    bias = AverageBias(axis="datasets")
    assert bias.apply(data).shape == data.shape

    # Independence (copy)
    data = np.random.randn(3, 2, 10)
    biased = bias.apply(data)
    biased[0, 0, 0] = 999
    assert data[0, 0, 0] != 999

    # Invalid init
    with pytest.raises(ValueError, match="axis must be"):
        AverageBias(axis="invalid")
