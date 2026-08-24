"""Tests for the shared epoch/continuous reshape helpers in mne_denoise._spatial."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._spatial import continuous_to_epochs, epochs_to_continuous


def test_epochs_to_continuous_concatenates_along_time():
    """Epochs are laid out channel-first and joined end to end."""
    X = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    continuous = epochs_to_continuous(X)
    assert continuous.shape == (3, 8)
    # Channel 0 of epoch 0 followed by channel 0 of epoch 1.
    np.testing.assert_array_equal(continuous[0], np.concatenate([X[0, 0], X[1, 0]]))


def test_round_trip_is_exact():
    """continuous_to_epochs inverts epochs_to_continuous."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4, 30))
    np.testing.assert_array_equal(
        continuous_to_epochs(epochs_to_continuous(X), X.shape), X
    )


def test_continuous_input_passes_through():
    """Two-dimensional data is returned unchanged by both helpers."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 50))
    assert epochs_to_continuous(X) is X
    continuous = epochs_to_continuous(X)
    np.testing.assert_array_equal(continuous_to_epochs(continuous, X.shape), X)


def test_matches_the_manual_idiom():
    """The helpers reproduce the transpose/reshape idiom they replaced."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 6, 20))
    n_epochs, n_channels, n_times = X.shape
    np.testing.assert_array_equal(
        epochs_to_continuous(X), X.transpose(1, 0, 2).reshape(n_channels, -1)
    )
    continuous = epochs_to_continuous(X)
    np.testing.assert_array_equal(
        continuous_to_epochs(continuous, X.shape),
        continuous.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2),
    )


@pytest.mark.parametrize("ndim", [1, 4])
def test_epochs_to_continuous_rejects_other_dimensions(ndim):
    """Only 2-D and 3-D layouts are meaningful."""
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        epochs_to_continuous(np.ones((2,) * ndim))


def test_continuous_to_epochs_rejects_other_shapes():
    """The target shape must describe continuous or epoched data."""
    with pytest.raises(ValueError, match="shape must have 2 or 3 entries"):
        continuous_to_epochs(np.ones((3, 10)), (2, 3, 4, 5))
