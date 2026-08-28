"""Unit tests for ICA-based nonlinearities in dss.denoisers.ica."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.ica import (
    GaussDenoiser,
    KurtosisDenoiser,
    RobustTanhDenoiser,
    SkewDenoiser,
    SmoothTanhDenoiser,
    TanhMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)


def test_tanh_mask_denoiser():
    """Test TanhMaskDenoiser logic."""
    # 1. Basic functional check
    denoiser = TanhMaskDenoiser(alpha=1.0, normalize=False)
    data = np.array([0, 1.0, -1.0])
    expected = np.tanh(data)
    assert_allclose(denoiser.denoise(data), expected)

    # 2. Normalization check
    denoiser_norm = TanhMaskDenoiser(alpha=1.0, normalize=True)
    rng = np.random.default_rng(42)
    data_scaled = rng.normal(0, 5, 100)  # std ~ 5

    # Logic: denoise = tanh(s/std) * std
    std = np.std(data_scaled)
    expected_norm = np.tanh(data_scaled / std) * std
    assert_allclose(denoiser_norm.denoise(data_scaled), expected_norm)


def test_robust_tanh_denoiser():
    """Test RobustTanhDenoiser (s - tanh(s))."""
    denoiser = RobustTanhDenoiser(alpha=1.0)
    data = np.array([-2, 0, 2], dtype=float)
    # expected = s - tanh(s)
    expected = data - np.tanh(data)
    assert_allclose(denoiser.denoise(data), expected)


def test_gauss_denoiser():
    """Test GaussDenoiser (s * exp(-s^2/2))."""
    denoiser = GaussDenoiser(a=1.0)
    data = np.array([1.0])
    # expected = 1 * exp(-0.5) = 0.60653...
    expected = np.exp(-0.5)
    assert_allclose(denoiser.denoise(data), expected)


def test_skew_denoiser():
    """Test SkewDenoiser (s^2)."""
    denoiser = SkewDenoiser()
    data = np.array([-2, 3])
    assert_allclose(denoiser.denoise(data), [4, 9])


def test_kurtosis_denoiser():
    """Test generic KurtosisDenoiser wrappers."""
    # Cube
    k_cube = KurtosisDenoiser("cube")
    assert_allclose(k_cube.denoise(np.array([2])), [8])

    # Tanh
    k_tanh = KurtosisDenoiser("tanh")
    assert_allclose(k_tanh.denoise(np.array([1.0])), [np.tanh(1)])

    # Gauss
    k_gauss = KurtosisDenoiser("gauss")
    assert_allclose(k_gauss.denoise(np.array([1.0])), [np.exp(-0.5)])

    # Error
    with pytest.raises(ValueError):
        KurtosisDenoiser("invalid")


def test_beta_helpers():
    """Test beta calculation logic."""
    # beta_tanh = -mean(1 - tanh(s)^2)
    # If s=0, tanh(0)=0. beta = -mean(1) = -1.
    s_zeros = np.zeros(10)
    assert_allclose(beta_tanh(s_zeros), -1.0)

    # If s is very large, tanh(s)->1. 1-1=0. beta -> 0.
    s_huge = np.ones(10) * 100
    assert_allclose(beta_tanh(s_huge), 0.0, atol=1e-5)

    # beta_pow3 = -3
    assert beta_pow3(s_zeros) == -3.0

    # beta_gauss = -mean((1-a*s^2)*exp(-a*s^2/2))
    # If s=0: -mean(1 * 1) = -1
    assert_allclose(beta_gauss(s_zeros), -1.0)


def test_tanh_mask_denoiser_zero_std():
    """Test TanhMaskDenoiser returns input unchanged when std is near zero."""
    denoiser = TanhMaskDenoiser(alpha=1.0, normalize=True)

    # Create data with essentially zero variance
    data = np.zeros(100) + 1e-20
    result = denoiser.denoise(data)

    # Should return input unchanged (line 70 branch)
    assert_allclose(result, data)


def test_smooth_tanh_denoiser():
    """Test SmoothTanhDenoiser applies smoothing then tanh."""
    rng = np.random.default_rng(42)

    # Create noisy data
    source = rng.normal(0, 1, 100)

    denoiser = SmoothTanhDenoiser(alpha=1.0, window=10)
    result = denoiser.denoise(source)

    # Result should be bounded by tanh range (-1, 1) times some scale
    assert result.shape == source.shape
    # Smoothed signal should be less variable than raw tanh
    raw_tanh = np.tanh(source)
    assert np.std(result) <= np.std(raw_tanh) * 1.2  # Allow some tolerance


def test_kurtosis_denoiser_alpha():
    """Test KurtosisDenoiser with custom alpha."""
    # Tanh with alpha=2
    k_tanh = KurtosisDenoiser("tanh", alpha=2.0)
    data = np.array([0.5])
    expected = np.tanh(2.0 * 0.5)
    assert_allclose(k_tanh.denoise(data), [expected])

    # Gauss with alpha=2
    k_gauss = KurtosisDenoiser("gauss", alpha=2.0)
    data = np.array([1.0])
    expected = 1.0 * np.exp(-0.5 * (2.0 * 1.0) ** 2)
    assert_allclose(k_gauss.denoise(data), [expected])
