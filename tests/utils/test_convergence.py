"""Unit tests for adaptive learning rate helpers."""

import numpy as np

from mne_denoise.dss._convergence import Gamma179, GammaPredictive


def test_gamma179():
    """Test Gamma179 oscillation detection."""
    gamma = Gamma179()
    w_old = np.array([1.0, 0.0])

    # Iteration 1: Init
    g = gamma(np.array([1, 0]), w_old, 1)
    assert g == 1.0

    # Iteration 2: Init delta
    # w_new moves to [1, 0.1] -> delta = [0, -0.1]
    g = gamma(np.array([1.0, 0.1]), w_old, 2)
    assert g == 1.0

    # Iteration 3: Consistent move
    # w moves to [1.0, 0.2] -> delta = [0, -0.1] -> cos=1
    g = gamma(np.array([1.0, 0.2]), np.array([1.0, 0.1]), 3)
    assert g == 1.0

    # Iteration 4: Oscillation (reverse direction)
    # w moves back to [1.0, 0.1] -> delta = [0, 0.1] -> cos=-1
    g = gamma(np.array([1.0, 0.1]), np.array([1.0, 0.2]), 4)
    # Angle > 90 (cos < 0) -> should reduce gamma
    assert g == 0.5


def test_gamma_predictive():
    """Test GammaPredictive controller."""
    gamma = GammaPredictive(min_gamma=0.1)
    # Just verify it runs and returns a float
    w_old = np.array([1.0, 0.0])
    g1 = gamma(np.array([1, 0]), w_old, 1)
    g2 = gamma(np.array([1, 0]), w_old, 2)
    g3 = gamma(np.array([1, 0.1]), np.array([1, 0]), 3)

    assert g1 == 1.0
    assert g2 == 1.0
    assert isinstance(g3, float)
    assert g3 >= 0.1


def test_gamma179_reset():
    """Test Gamma179 reset method."""
    gamma = Gamma179()

    # Run some iterations
    gamma(np.array([1, 0]), np.array([0, 0]), 1)
    gamma(np.array([1, 0]), np.array([0, 0]), 2)
    gamma(np.array([1, 0.5]), np.array([1, 0]), 3)

    # Reset
    gamma.reset()

    assert gamma.gamma == 1.0
    assert gamma.deltaw is None


def test_gamma_predictive_reset():
    """Test GammaPredictive reset method."""
    gamma = GammaPredictive()

    # Run some iterations
    gamma(np.array([1, 0]), np.array([0, 0]), 1)
    gamma(np.array([1, 0]), np.array([0, 0]), 2)
    gamma(np.array([1, 0.5]), np.array([1, 0]), 3)

    # Reset
    gamma.reset()

    assert gamma.gamma == 1.0
    assert gamma.deltaw is None


def test_gamma_predictive_clamp():
    """Test GammaPredictive gamma is clamped to min_gamma."""
    gamma = GammaPredictive(min_gamma=0.5)

    # Force gamma to decrease significantly
    w_old = np.array([1.0, 0.0])
    gamma(np.array([1, 0]), w_old, 1)
    gamma(np.array([0, 1]), w_old, 2)  # Set initial delta

    # Cause a strong negative update to trigger clamping
    # deltaw_old = [1, -1], deltaw = opposite direction
    gamma(np.array([1, 0]), np.array([0, 1]), 3)
    gamma(np.array([0, 1]), np.array([1, 0]), 4)
    gamma(np.array([1, 0]), np.array([0, 1]), 5)

    # Should be clamped at min_gamma
    assert gamma.gamma >= 0.5


def test_gamma179_zero_norm():
    """Test Gamma179 with zero delta norm (no change)."""
    gamma = Gamma179()

    w = np.array([1.0, 0.0])
    gamma(w, w, 1)
    gamma(w, w, 2)  # deltaw = [0, 0]
    g = gamma(w, w, 3)  # Zero norm - should skip angle check

    assert g == 1.0
