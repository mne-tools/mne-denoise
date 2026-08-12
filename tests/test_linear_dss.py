"""Unit tests for DSS module - Linear DSS (compute_dss and DSS class)."""

from __future__ import annotations

import warnings

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss import DSS, compute_dss
from mne_denoise.dss.denoisers.spectral import LineNoiseBias
from mne_denoise.dss.utils.segmentation import CovarianceSegmenter, FixedWindowSegmenter

# =============================================================================
# compute_dss - Core Algorithm Tests
# =============================================================================


def test_compute_dss_shape():
    """compute_dss should return correct shapes."""
    rng = np.random.default_rng(42)
    n_channels = 16

    # Create random covariance matrices
    A = rng.standard_normal((n_channels, n_channels))
    cov0 = A @ A.T  # Baseline covariance
    cov1 = cov0.copy()  # Biased covariance (identity bias)

    filters, patterns, eigenvalues = compute_dss(cov0, cov1, n_components=5)

    assert filters.shape == (5, n_channels)
    assert patterns.shape == (n_channels, 5)
    assert eigenvalues.shape == (5,)


def test_compute_dss_identity_bias():
    """With identity bias (cov0 == cov1), all eigenvalues should be ~1."""
    rng = np.random.default_rng(42)
    n_channels = 8

    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    filters, patterns, eigenvalues = compute_dss(cov, cov)

    # All eigenvalues should be approximately 1
    assert_allclose(eigenvalues, np.ones(n_channels), atol=0.1)


def test_compute_dss_known_signal():
    """compute_dss should maximize biased/baseline variance ratio."""
    np.random.default_rng(42)
    n_channels = 4

    # Create baseline covariance: isotropic (identity-like)
    cov0 = np.eye(n_channels)

    # Create biased covariance: high variance in first direction
    cov1 = np.diag([10.0, 1.0, 1.0, 1.0])  # First component 10x stronger

    filters, patterns, eigenvalues = compute_dss(cov0, cov1, n_components=1)

    # Top filter should align with first basis vector (highest bias)
    top_filter = filters[0]
    expected_direction = np.array([1, 0, 0, 0])
    alignment = np.abs(
        np.dot(top_filter / np.linalg.norm(top_filter), expected_direction)
    )
    assert alignment > 0.95, f"Alignment {alignment} too low"

    # Top eigenvalue should be ~10 (ratio of biased to baseline variance)
    assert eigenvalues[0] > 5, f"Eigenvalue {eigenvalues[0]} too low"


def test_compute_dss_eigenvalue_ordering():
    """Eigenvalues should be in descending order."""
    rng = np.random.default_rng(42)
    n_channels = 6

    A = rng.standard_normal((n_channels, n_channels))
    B = rng.standard_normal((n_channels, n_channels))
    cov0 = A @ A.T
    cov1 = B @ B.T

    _, _, eigenvalues = compute_dss(cov0, cov1)

    # Check descending order
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


def test_compute_dss_orthogonal_filters():
    """DSS filters should be orthogonal in whitened space."""
    rng = np.random.default_rng(42)
    n_channels = 8

    A = rng.standard_normal((n_channels, n_channels))
    cov0 = A @ A.T
    cov1 = cov0 * 1.1  # Slightly different

    filters, _, _ = compute_dss(cov0, cov1)

    # Filters should be approximately orthogonal when projected through cov0
    gram = filters @ cov0 @ filters.T
    off_diag = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diag)) < 0.1


def test_compute_dss_n_components():
    """n_components should limit the output size."""
    rng = np.random.default_rng(42)
    n_channels = 10

    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    for n_comp in [1, 3, 5]:
        filters, patterns, eigenvalues = compute_dss(cov, cov, n_components=n_comp)
        assert filters.shape[0] == n_comp
        assert patterns.shape[1] == n_comp
        assert len(eigenvalues) == n_comp


def test_compute_dss_rank():
    """Rank parameter should limit whitening dimensionality."""
    rng = np.random.default_rng(42)
    n_channels = 10

    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    filters, _, _ = compute_dss(cov, cov, rank=5)

    # Output should be at most rank dimensions
    assert filters.shape[0] <= 5


def test_compute_dss_reconstruction():
    """Patterns @ (filters @ data) should reconstruct centered data."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 6, 500

    data = rng.standard_normal((n_channels, n_samples))
    data_c = data - data.mean(axis=1, keepdims=True)

    cov = data_c @ data_c.T / n_samples

    filters, patterns, _ = compute_dss(cov, cov)

    sources = filters @ data_c
    reconstructed = patterns @ sources

    # Should reconstruct (approximately - some numerical error expected)
    assert_allclose(reconstructed, data_c, atol=0.5)


# =============================================================================
# compute_dss - Error Handling
# =============================================================================


def test_compute_dss_error_shape_mismatch():
    """compute_dss should raise error when covariance shapes mismatch."""
    c1 = np.eye(5)
    c2 = np.eye(6)

    with pytest.raises(ValueError, match="shapes mismatch"):
        compute_dss(c1, c2)


def test_compute_dss_error_not_square():
    """compute_dss should raise error for non-square covariance."""
    c = np.ones((5, 6))

    with pytest.raises(ValueError, match="must be square"):
        compute_dss(c, c)


def test_compute_dss_error_no_variance():
    """compute_dss should raise error when covariance has no variance."""
    c = np.zeros((5, 5))

    with pytest.raises(ValueError, match="no significant variance"):
        compute_dss(c, c)


def test_compute_dss_tiny_positive_covariance_is_scale_invariant():
    """Tiny SI-unit covariances should not be treated as zero variance."""
    cov = np.diag([5.0, 2.0, 1.0, 0.5, 0.25])
    tiny_cov = cov * 1e-26

    filters, patterns, eigenvalues = compute_dss(tiny_cov, tiny_cov)

    assert filters.shape == (5, 5)
    assert patterns.shape == (5, 5)
    assert_allclose(eigenvalues, np.ones(5), atol=1e-12)


# =============================================================================
# DSS Class - Basic Functionality
# =============================================================================


def test_dss_fit_transform():
    """DSS class should support fit_transform workflow."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 1000
    data = rng.standard_normal((n_channels, n_samples))

    dss = DSS(bias=lambda x: x, n_components=3)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, n_samples)
    assert dss.filters_ is not None
    assert dss.filters_.shape == (3, n_channels)
    assert dss.patterns_ is not None
    assert dss.mixing_ is not None
    assert dss.eigenvalues_ is not None


def test_dss_uses_fitted_mean_across_transform_batches():
    """Unrelated transform observations cannot change frozen DSS sources."""
    rng = np.random.default_rng(12)
    train = rng.standard_normal((3, 200)) + np.array([[1.0], [4.0], [-2.0]])
    held_out = rng.standard_normal((3, 40))
    unrelated = rng.standard_normal((3, 60)) * 30.0 + 100.0
    dss = DSS(bias=lambda values: values, n_components=3, normalize_input=False)
    dss.fit(train)

    prefix = dss.transform(held_out)
    combined = dss.transform(np.concatenate([held_out, unrelated], axis=1))

    assert_allclose(combined[:, : held_out.shape[1]], prefix, atol=1e-12)
    assert_allclose(dss.mean_[:, 0], train.mean(axis=1))


def test_dss_center_false_uses_uncentered_second_moments():
    """Explicit uncentered DSS leaves offsets in its component transform."""
    data = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    dss = DSS(
        bias=lambda values: values,
        n_components=2,
        normalize_input=False,
        center=False,
    ).fit(data)

    assert_allclose(dss.mean_, 0.0)
    assert_allclose(dss.transform(data), dss.filters_ @ data)


def test_dss_centering_does_not_change_bias_input():
    """The transform origin must not alter the physical bias operation."""
    data = np.arange(12.0).reshape(3, 4) + 10.0
    received = []

    def bias(values):
        received.append(values.copy())
        return values

    DSS(bias=bias, normalize_input=False, center=True).fit(data)

    assert_allclose(received[0], data)


def test_dss_rejects_non_boolean_center():
    """Centering semantics must be explicit rather than truthy."""
    with pytest.raises(TypeError, match="center must be a bool"):
        DSS(bias=lambda values: values, center=1).fit(np.eye(3))


def test_dss_custom_bias_callable():
    """DSS should accept custom callable bias."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 1000))

    def my_bias(x):
        return x * 2

    dss = DSS(bias=my_bias, n_components=3)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 1000)


def test_dss_denoiser_bias():
    """DSS should accept LinearDenoiser bias."""
    from mne_denoise.dss.denoisers import BandpassBias

    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 2000))

    bias = BandpassBias(freq_band=(8, 12), sfreq=250)
    dss = DSS(bias=bias, n_components=3)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 2000)


def test_dss_3d_data():
    """DSS should handle 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 8, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))

    dss = DSS(bias=lambda x: x, n_components=3)
    dss.fit(data)
    sources = dss.transform(data)

    assert sources.shape == (3, n_times, n_epochs)


def test_dss_without_normalization():
    """DSS should work without input normalization."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, normalize_input=False)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 200)


def test_dss_retain_reconstructs_sensor_data():
    """DSS retention should reconstruct sensor-space data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, component_action="retain")
    dss.fit(data)
    rec = dss.transform(data)

    assert rec.shape == data.shape


def test_dss_cov_method():
    """DSS should accept different covariance methods."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, cov_method="shrinkage")
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 200)


def test_dss_cov_kws():
    """DSS should pass cov_kws to covariance computation."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, cov_kws={"shrinkage": 0.5})
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 200)


# =============================================================================
# DSS Class - inverse_transform
# =============================================================================


def test_dss_inverse_transform_2d():
    """inverse_transform should work with 2D sources."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)
    rec = dss.inverse_transform(sources)

    assert rec.shape == data.shape


def test_dss_inverse_transform_3d():
    """inverse_transform should handle 3D sources."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 100, 4))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)
    rec = dss.inverse_transform(sources)

    assert rec.ndim == 3


def test_dss_inverse_transform_boolean_mask():
    """inverse_transform should work with boolean component mask."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)

    mask = np.array([True, True, True, False, False])
    rec = dss.inverse_transform(sources, component_indices=mask)

    assert rec.shape == data.shape


def test_dss_inverse_transform_integer_indices():
    """inverse_transform should work with integer component indices."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)

    indices = np.array([0, 2])
    rec = dss.inverse_transform(sources, component_indices=indices)

    assert rec.shape == data.shape


# =============================================================================
# DSS Class - Error Handling
# =============================================================================


def test_dss_error_unsupported_type():
    """DSS should raise error for unsupported input types."""
    dss = DSS(bias=lambda x: x, normalize_input=False)

    with pytest.raises(TypeError, match="Unsupported input type"):
        dss.fit("not an array")


def test_dss_error_transform_before_fit():
    """DSS should raise error when transform called before fit."""
    dss = DSS(bias=lambda x: x)
    data = np.random.randn(5, 100)

    with pytest.raises(RuntimeError, match="not fitted"):
        dss.transform(data)


def test_dss_error_inverse_transform_before_fit():
    """DSS should raise error when inverse_transform called before fit."""
    dss = DSS(bias=lambda x: x)
    sources = np.random.randn(3, 100)

    with pytest.raises(RuntimeError, match="not fitted"):
        dss.inverse_transform(sources)


def test_dss_error_mask_length_mismatch():
    """inverse_transform should raise error for wrong mask length."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)

    wrong_mask = np.array([True, True, True])
    with pytest.raises(ValueError, match="Mask length"):
        dss.inverse_transform(sources, component_indices=wrong_mask)


def test_dss_supports_rank_numpy():
    """DSS should support rank parameter with numpy arrays (no warning)."""
    import warnings

    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    # Should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dss = DSS(bias=lambda x: x, n_components=3, rank=5)
        dss.fit(data)

        # Filter out unrelated warnings if any (e.g. from MNE)
        rank_warnings = [
            warning for warning in w if "rank" in str(warning.message).lower()
        ]
        assert len(rank_warnings) == 0


# =============================================================================
# Integration / Functional Tests
# =============================================================================


def test_dss_recovers_narrowband_signal():
    """DSS with bandpass bias should recover narrowband signal."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 2000
    sfreq = 250

    noise = rng.standard_normal((n_channels, n_samples))

    t = np.arange(n_samples) / sfreq
    source = np.sin(2 * np.pi * 10 * t)  # 10 Hz

    mixing = rng.standard_normal(n_channels)
    mixing = mixing / np.linalg.norm(mixing)
    data = noise + 5 * np.outer(mixing, source)

    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)

    dss = DSS(bias=bias, n_components=3)
    sources = dss.fit_transform(data)

    top_source = sources[0]
    correlation = np.abs(np.corrcoef(top_source, source)[0, 1])
    assert correlation > 0.8


def test_dss_evoked_workflow():
    """DSS with trial average bias should recover evoked response."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 50

    noise = rng.standard_normal((n_channels, n_times, n_epochs))

    evoked = np.zeros(n_times)
    evoked[40:60] = np.hanning(20)

    mixing = rng.standard_normal(n_channels)
    mixing = mixing / np.linalg.norm(mixing)
    signal = np.outer(mixing, evoked)[:, :, np.newaxis]

    data = noise + signal

    from mne_denoise.dss.denoisers import AverageBias

    bias = AverageBias(axis="epochs")
    dss = DSS(bias=bias, n_components=3)
    sources = dss.fit_transform(data)

    top_source_avg = sources[0].mean(axis=1)
    correlation = np.abs(np.corrcoef(top_source_avg, evoked)[0, 1])
    assert correlation > 0.7


# =============================================================================
# MNE Integration Tests
# =============================================================================


def test_dss_with_mne_raw():
    """DSS should work with MNE Raw objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 5000
    sfreq = 500.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3)
    sources = dss.fit_transform(raw)

    assert sources.shape == (3, n_samples)
    assert dss.info_ is not None


def test_dss_with_mne_epochs():
    """DSS should work with MNE Epochs objects."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 20
    sfreq = 250.0

    # Create epoch data: MNE expects (n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import AverageBias

    dss = DSS(bias=AverageBias(axis="epochs"), n_components=3)
    sources = dss.fit_transform(epochs)

    # Sources should be (n_epochs, n_components, n_times)
    assert sources.shape == (n_epochs, 3, n_times)


def test_dss_with_mne_evoked():
    """DSS should work with MNE Evoked objects."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 8, 200
    sfreq = 250.0

    data = rng.standard_normal((n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    evoked = mne.EvokedArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3)
    sources = dss.fit_transform(evoked)

    assert sources.shape == (3, n_times)


def test_dss_mne_retain_returns_epochs():
    """DSS retention should preserve an MNE Epochs container."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 4, 50, 10
    sfreq = 100.0

    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3, component_action="retain")
    dss.fit(epochs)
    result = dss.transform(epochs)

    # Should return MNE Epochs object
    assert isinstance(result, mne.epochs.BaseEpochs)


def test_dss_mne_retain_returns_raw():
    """DSS retention should preserve an MNE Raw container."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 2000
    sfreq = 500.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3, component_action="retain")
    dss.fit(raw)
    result = dss.transform(raw)

    # Should return MNE Raw object
    assert isinstance(result, mne.io.BaseRaw)


def test_dss_mne_with_weights():
    """DSS should handle MNE objects with weights (falls back to numpy path)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 1000
    sfreq = 250.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    weights = np.ones(n_samples)
    weights[500:] = 0.5  # Lower weight for second half

    dss = DSS(bias=lambda x: x, n_components=2)
    dss.fit(raw, weights=weights)

    assert dss.filters_.shape == (2, n_channels)


def test_dss_mne_normalization():
    """DSS normalization should work with MNE objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 1000
    sfreq = 250.0

    # Create data with different scales
    data = rng.standard_normal((n_channels, n_samples))
    data[0] *= 1e-6  # Simulate gradiometer scale
    data[1] *= 1e-12  # Simulate magnetometer scale

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3, normalize_input=True)
    sources = dss.fit_transform(raw)

    assert sources.shape == (3, n_samples)
    assert dss.channel_norms_ is not None


# =============================================================================
# More Tests with Known Expected Outputs for DSS class
# =============================================================================


def test_dss_array_extracts_known_signal():
    """DSS should extract a known sinusoidal signal from noise (numpy array)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 2000
    sfreq = 250

    # Create known signal: 10 Hz sinusoid in a known direction
    t = np.arange(n_samples) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)  # Pure 10 Hz

    # Known mixing: signal goes to first 4 channels with weights [1, 0.8, 0.5, 0.2]
    mixing_weights = np.array([1, 0.8, 0.5, 0.2, 0, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Add noise
    noise = rng.standard_normal((n_channels, n_samples)) * 0.5
    data = noise + 3 * np.outer(mixing_weights, signal)

    # Use bandpass bias around 10 Hz
    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(data)

    # Top source should correlate highly with original signal
    correlation = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert correlation > 0.9, f"Correlation {correlation} too low, expected > 0.9"

    # Top filter should align with mixing weights direction
    top_filter = dss.filters_[0]
    alignment = np.abs(np.dot(top_filter / np.linalg.norm(top_filter), mixing_weights))
    assert alignment > 0.8, f"Filter alignment {alignment} too low, expected > 0.8"


def test_dss_array_evoked_extracts_known_erp():
    """DSS with trial average should extract known ERP from epoched data."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 50

    # Create known ERP: Gaussian peak at sample 50
    erp = np.zeros(n_times)
    erp[40:60] = np.hanning(20) * 2  # Peak amplitude 2

    # Known mixing: ERP in first 3 channels
    mixing_weights = np.array([1, 0.7, 0.3, 0, 0, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Add noise
    noise = rng.standard_normal((n_channels, n_times, n_epochs)) * 0.5
    signal = np.outer(mixing_weights, erp)[:, :, np.newaxis]  # (n_ch, n_times, 1)
    data = noise + signal  # Signal replicated across epochs

    from mne_denoise.dss.denoisers import AverageBias

    bias = AverageBias(axis="epochs")

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(data)  # (1, n_times, n_epochs)

    # Average across epochs
    source_avg = sources[0].mean(axis=1)  # (n_times,)

    # Source average should correlate with ERP
    correlation = np.abs(np.corrcoef(source_avg, erp)[0, 1])
    assert correlation > 0.9, f"ERP correlation {correlation} too low"


def test_dss_mne_raw_extracts_line_noise():
    """DSS should extract line noise from MNE Raw (functional test)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 5000
    sfreq = 500.0

    # Create 50 Hz line noise in channels 0-3
    t = np.arange(n_samples) / sfreq
    line_noise = np.sin(2 * np.pi * 50 * t)

    # Known mixing
    mixing_weights = np.zeros(n_channels)
    mixing_weights[:4] = [1, 0.8, 0.5, 0.2]
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Create data
    noise = rng.standard_normal((n_channels, n_samples)) * 0.3
    data = noise + 2 * np.outer(mixing_weights, line_noise)

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    # Use line noise bias (notch method)
    from mne_denoise.dss.denoisers import LineNoiseBias

    bias = LineNoiseBias(freq=50, sfreq=sfreq, method="iir", bandwidth=2)

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(raw)

    # Top source should correlate with line noise
    correlation = np.abs(np.corrcoef(sources[0], line_noise)[0, 1])
    assert correlation > 0.85, f"Line noise correlation {correlation} too low"


def test_dss_mne_epochs_extracts_known_erp():
    """DSS should extract known ERP from MNE Epochs (functional test)."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 6, 100, 40
    sfreq = 100.0

    # Create known ERP
    erp = np.zeros(n_times)
    erp[45:55] = np.hanning(10) * 3  # Strong peak

    # Known mixing: ERP in first 3 channels
    mixing_weights = np.array([1, 0.6, 0.3, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Create data (MNE format: n_epochs, n_channels, n_times)
    noise = rng.standard_normal((n_epochs, n_channels, n_times)) * 0.4
    signal = np.outer(mixing_weights, erp)[np.newaxis, :, :]  # (1, n_ch, n_times)
    data = noise + signal  # Broadcast signal to all epochs

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import AverageBias

    bias = AverageBias(axis="epochs")

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(epochs)  # (n_epochs, 1, n_times)

    # Average across epochs
    source_avg = sources[:, 0, :].mean(axis=0)  # (n_times,)

    # Should correlate with ERP
    correlation = np.abs(np.corrcoef(source_avg, erp)[0, 1])
    assert correlation > 0.9, f"Epochs ERP correlation {correlation} too low"


def test_dss_mne_evoked_extracts_known_signal():
    """DSS should work with MNE Evoked and extract known signal."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 6, 200
    sfreq = 100.0

    # Create known oscillatory signal at 10 Hz
    t = np.arange(n_times) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)

    # Known mixing
    mixing_weights = np.array([1, 0.5, 0.2, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Create data
    noise = rng.standard_normal((n_channels, n_times)) * 0.3
    data = noise + 2 * np.outer(mixing_weights, signal)

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    evoked = mne.EvokedArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(evoked)

    # Top source should correlate with signal
    correlation = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert correlation > 0.85, f"Evoked signal correlation {correlation} too low"


@pytest.mark.parametrize("normalize_input", [False, True])
def test_dss_raw_covariances_use_identical_sample_support(monkeypatch, normalize_input):
    """Baseline and biased Raw covariance must see identical sample support."""
    rng = np.random.default_rng(42)
    info = mne.create_info(["EEG0", "EEG1", "EEG2"], 100.0, "eeg")
    raw = mne.io.RawArray(
        rng.standard_normal((3, 1000)),
        info,
        first_samp=1000,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[2.0],
            duration=[2.0],
            description=["BAD_test"],
        )
    )
    raw.set_eeg_reference(projection=True, verbose=False)

    original_compute = mne.compute_raw_covariance
    calls = []

    def _record_support(inst, *args, **kwargs):
        cov = original_compute(inst, *args, **kwargs)
        calls.append(
            {
                "first_samp": inst.first_samp,
                "annotations": inst.annotations.copy(),
                "nfree": cov.nfree,
                "projectors": [
                    (projector["desc"], projector["active"])
                    for projector in inst.info["projs"]
                ],
            }
        )
        return cov

    monkeypatch.setattr(mne, "compute_raw_covariance", _record_support)
    DSS(
        bias=lambda data: data,
        normalize_input=normalize_input,
    ).fit(raw)

    assert len(calls) == 2
    assert calls[0]["first_samp"] == calls[1]["first_samp"] == raw.first_samp
    assert calls[0]["annotations"] == calls[1]["annotations"]
    assert calls[0]["nfree"] == calls[1]["nfree"]
    assert calls[0]["nfree"] < raw.n_times - 1
    assert calls[0]["projectors"] == calls[1]["projectors"]


@pytest.mark.parametrize("normalize_input", [False, True])
def test_dss_raw_bad_channels_are_excluded_and_preserved(normalize_input):
    """DSS must fit good channels and pass bad Raw channels through unchanged."""
    rng = np.random.default_rng(43)
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    raw = mne.io.RawArray(
        rng.standard_normal((4, 1000)),
        info,
        first_samp=250,
        verbose=False,
    )
    raw.info["bads"] = ["EEG3"]
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0],
            duration=[0.5],
            description=["marker"],
        )
    )

    dss = DSS(
        bias=lambda data: data,
        n_components=3,
        normalize_input=normalize_input,
        component_action="retain",
    ).fit(raw)
    transformed = dss.transform(raw)

    assert dss._mne_ch_names_ == ["EEG0", "EEG1", "EEG2"]
    assert dss.filters_.shape == (3, 3)
    assert transformed.first_samp == raw.first_samp
    assert transformed.annotations == raw.annotations
    assert transformed.info["bads"] == raw.info["bads"]
    assert_allclose(transformed.get_data(picks=["EEG3"]), raw.get_data(picks=["EEG3"]))


def test_dss_fit_transform_preserves_bad_raw_channels():
    """Artifact subtraction must operate only on the fitted good channels."""
    rng = np.random.default_rng(44)
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 1000)), info, verbose=False)
    raw.info["bads"] = ["EEG3"]

    cleaned = DSS(
        bias=lambda data: data,
        n_components=3,
        n_select=1,
        normalize_input=False,
        component_action="subtract",
    ).fit_transform(raw)

    assert cleaned.info["bads"] == ["EEG3"]
    assert_allclose(cleaned.get_data(picks=["EEG3"]), raw.get_data(picks=["EEG3"]))


@pytest.mark.parametrize("fit_kws", [{"smooth": 5}, {"whiten": True}])
def test_dss_alternate_fit_paths_preserve_bad_raw_channels(fit_kws):
    """Smoothing and whitening must follow the same good-channel contract."""
    rng = np.random.default_rng(47)
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 1000)), info, verbose=False)
    raw.info["bads"] = ["EEG3"]

    dss = DSS(
        bias=lambda data: data,
        n_components=3,
        normalize_input=False,
        component_action="retain",
        **fit_kws,
    ).fit(raw)
    transformed = dss.transform(raw)

    assert dss._mne_ch_names_ == ["EEG0", "EEG1", "EEG2"]
    assert dss.info_["ch_names"] == dss._mne_ch_names_
    assert dss.filters_.shape == (3, 3)
    assert_allclose(transformed.get_data(picks=["EEG3"]), raw.get_data(picks=["EEG3"]))


def test_dss_transform_aligns_reordered_mne_channels_by_name():
    """Transform must align fitted channels by name, independent of input order."""
    rng = np.random.default_rng(48)
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 1000)), info, verbose=False)
    raw.info["bads"] = ["EEG3"]
    reordered = raw.copy().reorder_channels(["EEG2", "EEG0", "EEG3", "EEG1"])

    dss = DSS(
        bias=lambda data: data,
        n_components=3,
        normalize_input=False,
        component_action="retain",
    ).fit(raw)
    transformed = dss.transform(reordered)

    assert transformed.ch_names == reordered.ch_names
    assert_allclose(transformed.get_data(), reordered.get_data(), atol=1e-12)


def test_dss_transform_rejects_missing_fitted_channel():
    """Missing fitted channels must raise an actionable error."""
    rng = np.random.default_rng(49)
    info = mne.create_info(["EEG0", "EEG1", "EEG2"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((3, 1000)), info, verbose=False)
    dss = DSS(bias=lambda data: data, normalize_input=False).fit(raw)

    with pytest.raises(ValueError, match="missing required channels.*EEG2"):
        dss.transform(raw.copy().drop_channels(["EEG2"]))


def test_dss_fit_rejects_all_bad_data_channels():
    """DSS must fail clearly when no usable fitted channels remain."""
    rng = np.random.default_rng(50)
    info = mne.create_info(["EEG0", "EEG1"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((2, 1000)), info, verbose=False)
    raw.info["bads"] = raw.ch_names

    with pytest.raises(ValueError, match="No good data channels remain"):
        DSS(bias=lambda data: data, normalize_input=False).fit(raw)


def test_dss_epochs_bad_channels_are_excluded_and_preserved():
    """DSS must preserve bad Epochs channels and epoch bookkeeping."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(45)
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    events = np.column_stack(
        [np.arange(5) * 100, np.zeros(5, dtype=int), np.ones(5, dtype=int)]
    )
    epochs = mne.EpochsArray(
        rng.standard_normal((5, 4, 100)),
        info,
        events=events,
        event_id={"stim": 1},
        verbose=False,
    )
    epochs.info["bads"] = ["EEG3"]
    epochs.metadata = pd.DataFrame({"condition": list("abcde")})

    dss = DSS(
        bias=lambda data: data,
        n_components=3,
        normalize_input=False,
        component_action="retain",
    ).fit(epochs)
    transformed = dss.transform(epochs)

    assert dss._mne_ch_names_ == ["EEG0", "EEG1", "EEG2"]
    assert dss.filters_.shape == (3, 3)
    assert transformed.info["bads"] == epochs.info["bads"]
    assert_allclose(
        transformed.get_data(picks=["EEG3"]), epochs.get_data(picks=["EEG3"])
    )
    assert_allclose(transformed.events, epochs.events)
    assert transformed.event_id == epochs.event_id
    assert transformed.metadata.equals(epochs.metadata)


def test_dss_evoked_bad_channels_are_excluded_and_preserved():
    """The NumPy Evoked path must honor the same fitted-channel contract."""
    rng = np.random.default_rng(46)
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    evoked = mne.EvokedArray(
        rng.standard_normal((4, 500)),
        info,
        tmin=-0.2,
        comment="condition",
        nave=17,
        verbose=False,
    )
    evoked.info["bads"] = ["EEG3"]

    dss = DSS(
        bias=lambda data: data,
        n_components=3,
        normalize_input=False,
        component_action="retain",
    ).fit(evoked)
    transformed = dss.transform(evoked)

    assert dss._mne_ch_names_ == ["EEG0", "EEG1", "EEG2"]
    assert dss.filters_.shape == (3, 3)
    assert transformed.info["bads"] == evoked.info["bads"]
    assert transformed.comment == evoked.comment
    assert transformed.nave == evoked.nave
    assert_allclose(
        transformed.get_data(picks=["EEG3"]), evoked.get_data(picks=["EEG3"])
    )


def test_dss_reconstruction_preserves_signal():
    """DSS transform + inverse_transform should preserve signal content."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 500

    # Create data with known structure
    t = np.linspace(0, 1, n_samples)
    signal1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz
    signal2 = np.sin(2 * np.pi * 10 * t)  # 10 Hz

    data = np.zeros((n_channels, n_samples))
    data[0] = signal1 * 2
    data[1] = signal1 + signal2
    data[2:] = rng.standard_normal((n_channels - 2, n_samples)) * 0.1

    dss = DSS(bias=lambda x: x, n_components=n_channels, normalize_input=False)
    sources = dss.fit_transform(data)
    reconstructed = dss.inverse_transform(sources)

    # Reconstruction should match centered original
    data_centered = data - data.mean(axis=1, keepdims=True)
    assert_allclose(reconstructed, data_centered, atol=0.1)


def test_dss_mne_epochs_inverse_transform_with_normalization():
    """inverse_transform should work with MNE Epochs format sources."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 6, 50, 10
    sfreq = 100.0

    # Create data (MNE format: n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import AverageBias

    dss = DSS(bias=AverageBias(axis="epochs"), n_components=3, normalize_input=True)

    # Fit and transform - sources will be (n_epochs, n_components, n_times)
    sources = dss.fit_transform(epochs)

    assert sources.shape == (n_epochs, 3, n_times)

    # Now inverse_transform with MNE epochs format sources
    reconstructed = dss.inverse_transform(sources)

    # Should produce (n_epochs, n_channels, n_times)
    assert reconstructed.shape == (n_epochs, n_channels, n_times)


def test_dss_inverse_transform_mne_format_3d():
    """inverse_transform should detect MNE epochs format (n_epochs, n_comps, n_times)."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 5
    n_components = 4

    # Create numpy 3D data (channels, times, epochs)
    data = rng.standard_normal((n_channels, n_times, n_epochs))

    dss = DSS(bias=lambda x: x, n_components=n_components, normalize_input=True)
    sources = dss.fit_transform(data)  # (n_components, n_times, n_epochs)

    assert sources.shape == (n_components, n_times, n_epochs)

    # Now manually transpose to MNE epochs format
    sources_mne_format = np.transpose(
        sources, (2, 0, 1)
    )  # (n_epochs, n_comps, n_times)

    # inverse_transform should detect this and handle it
    reconstructed = dss.inverse_transform(sources_mne_format)

    # Should produce (n_epochs, n_channels, n_times) since it detected MNE format
    assert reconstructed.shape == (n_epochs, n_channels, n_times)


def test_dss_normalization_with_different_scales():
    """Test DSS normalization with channels at vastly different scales."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_channels = 3

    # Create data with vastly different scales
    data = rng.randn(n_channels, n_samples)
    data[0] *= 1e-6  # "Gradiometer" scale
    data[1] *= 1  # "Magnetometer" scale
    data[2] *= 1000  # "EEG" scale

    class SimpleBias:
        def apply(self, x):
            return x

    # Fit without normalization
    dss_raw = DSS(n_components=3, bias=SimpleBias(), normalize_input=False)
    dss_raw.fit(data)

    # Fit with normalization
    dss_norm = DSS(n_components=3, bias=SimpleBias(), normalize_input=True)
    dss_norm.fit(data)

    # Check that channel norms were computed correctly
    assert dss_norm.channel_norms_ is not None
    assert dss_norm.channel_norms_.shape == (n_channels,)
    # Norms should reflect the scales
    assert (
        dss_norm.channel_norms_[0]
        < dss_norm.channel_norms_[1]
        < dss_norm.channel_norms_[2]
    )

    # Transform and reconstruct
    sources_norm = dss_norm.transform(data)
    assert sources_norm.shape == (3, n_samples)

    rec_norm = dss_norm.inverse_transform(sources_norm)
    data_centered = data - data.mean(axis=1, keepdims=True)
    assert_allclose(data_centered, rec_norm, atol=1e-5 * 1000, rtol=1e-5)


def test_dss_weighted_fit_ignores_outliers():
    """Test DSS fit with weights to mask out outliers."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_channels = 4
    data = rng.randn(n_channels, n_samples)

    # Create data with outliers in second half
    data_clean = data.copy()
    data_bad = data.copy()
    data_bad[0, 100:] = 1e6  # Huge outlier

    weights = np.ones(n_samples)
    weights[100:] = 0  # Ignore outlier region

    # Fit DSS on bad data with weights masking outliers
    dss = DSS(n_components=2, bias=lambda x: x, normalize_input=False)
    dss.fit(data_bad, weights=weights)

    # Fit DSS on clean data (first half only)
    dss_clean = DSS(n_components=2, bias=lambda x: x, normalize_input=False)
    dss_clean.fit(data_clean[:, :100])

    # Filters should be nearly identical (up to sign flip)
    f1 = dss.filters_
    f2 = dss_clean.filters_

    for i in range(2):
        corr = np.corrcoef(f1[i], f2[i])[0, 1]
        assert abs(corr) > 0.99, f"Filter {i} correlation {corr} too low"


def test_dss_cov_method_options():
    """Test DSS with different covariance method options."""
    rng = np.random.RandomState(42)
    n_samples = 2000
    n_channels = 3
    data = rng.randn(n_channels, n_samples)

    # Test numpy path with shrinkage
    dss = DSS(
        n_components=2,
        bias=lambda x: x,
        cov_method="shrinkage",
        cov_kws={"shrinkage": 0.1},
    )
    dss.fit(data)
    assert dss.filters_.shape == (2, 3)

    # Test MNE path with auto method
    info = mne.create_info(n_channels, 1000.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    dss_mne = DSS(
        n_components=2,
        bias=lambda x: x,
        cov_method="auto",
        cov_kws=None,
    )
    dss_mne.cov_method = "empirical"
    dss_mne.fit(raw)
    assert dss_mne.filters_.shape == (2, 3)


def test_dss_preserves_scale():
    """DSS reconstruction should preserve physical signal scale (Microvolts)."""
    sfreq = 1000
    n_channels = 10
    n_times = 5000
    t = np.arange(n_times) / sfreq

    signal_scale = 5e-6
    data = np.random.randn(n_channels, n_times) * 1e-7  # noise
    data[0:3, :] += signal_scale * np.sin(2 * np.pi * 10 * t)

    from mne_denoise.dss.denoisers import LinearDenoiser

    class IdentityBias(LinearDenoiser):
        def apply(self, data):
            return data

    bias = IdentityBias()
    dss = DSS(
        bias=bias,
        n_components=n_channels,
        normalize_input=False,
        component_action="retain",
    )
    reconstructed = dss.fit_transform(data)

    rms_orig = np.sqrt(np.mean(data**2))
    rms_rec = np.sqrt(np.mean(reconstructed**2))

    assert_allclose(rms_orig, rms_rec, rtol=0.05)


def test_dss_get_normalized_patterns():
    """Test the newly added get_normalized_patterns method in DSS."""
    from mne_denoise.dss.denoisers import LinearDenoiser

    class IdentityBias(LinearDenoiser):
        def apply(self, data):
            return data

    data = np.random.randn(10, 1000)
    dss = DSS(bias=IdentityBias(), n_components=2)
    dss.fit(data)
    norm_patterns = dss.get_normalized_patterns()
    assert norm_patterns.shape == (10, 2)
    assert_allclose(np.linalg.norm(norm_patterns, axis=0), 1.0)


def test_compute_dss_warns_on_heavy_rank_reduction(caplog):
    """``compute_dss`` warns when ``reg`` discards >75% of components.

    Mimics MEG-style fT-scale covariances whose eigenvalues span many
    decades: only a handful survive the default relative ``reg`` threshold,
    and we want the user to see actionable workaround suggestions.
    """
    import logging

    n_channels = 40
    # Six eigenvalues at 1.0, the rest at 1e-15 -- well below the default
    # reg=1e-9 cutoff, so n_keep ends up at 6 (< n_channels // 4 = 10).
    eigvals = np.concatenate([np.ones(6), np.full(n_channels - 6, 1e-15)])
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
    cov = Q @ np.diag(eigvals) @ Q.T
    cov = (cov + cov.T) / 2

    with caplog.at_level(logging.WARNING, logger="mne_denoise.dss.linear"):
        compute_dss(cov, cov)

    assert any(
        "components kept after rank reduction" in rec.getMessage()
        for rec in caplog.records
    )
    assert any("lowering reg" in rec.getMessage() for rec in caplog.records)
    assert not any("raising reg" in rec.getMessage() for rec in caplog.records)


# =============================================================================
# DSS - Multi-modal joint decomposition via whitening (Issue #37)
# =============================================================================


def _mixed_sensor_raw(seed=0, n_t=2000, sfreq=200.0):
    """Raw with mag + grad + eeg channels at very different physical scales."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)
    ch_types = ["mag"] * 6 + ["grad"] * 6 + ["eeg"] * 6
    scales = np.array([1e-12] * 6 + [1e-11] * 6 + [1e-5] * 6)
    n_ch = len(ch_types)
    mixing = rng.standard_normal(n_ch)
    data = (np.outer(mixing, signal) + 0.3 * rng.standard_normal((n_ch, n_t))) * scales[
        :, None
    ]
    info = mne.create_info([f"C{i}" for i in range(n_ch)], sfreq, ch_types)
    return mne.io.RawArray(data, info, verbose=False), signal, data


def test_dss_whiten_decomposes_mixed_types_without_warning():
    """whiten=True fits all data channel types jointly and stays quiet."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, signal, _ = _mixed_sensor_raw()
    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dss = DSS(bias=bias, whiten=True)
        sources = dss.fit_transform(raw)

    assert dss.filters_.shape == (18, 18)  # all data channels, jointly
    assert sources.shape == (18, raw.n_times)
    top_corr = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert top_corr > 0.9


def test_dss_whiten_reconstruction_is_faithful_across_units():
    """Whitening is undone on reconstruction even for very different units."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, _, data = _mixed_sensor_raw()
    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])

    dss = DSS(bias=bias, whiten=True, component_action="retain").fit(raw)
    rec = dss.transform(raw).get_data()

    rel_error = np.linalg.norm(rec - data) / np.linalg.norm(data)
    assert rel_error < 1e-6


def test_dss_whiten_with_noise_cov():
    """A provided noise covariance is used to build the whitener."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, signal, data = _mixed_sensor_raw()
    picks = mne.pick_types(raw.info, meg=True, eeg=True)
    cov_data = np.cov(data[picks]) + np.eye(len(picks)) * 1e-30
    noise_cov = mne.Covariance(
        cov_data,
        [raw.ch_names[p] for p in picks],
        raw.info["bads"],
        raw.info["projs"],
        nfree=raw.n_times,
    )

    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])
    dss = DSS(bias=bias, whiten=True, noise_cov=noise_cov)
    sources = dss.fit_transform(raw)

    assert sources.shape == (18, raw.n_times)
    assert np.abs(np.corrcoef(sources[0], signal)[0, 1]) > 0.9

    ranked_sources = DSS(
        bias=bias, whiten=True, noise_cov=noise_cov, rank=10
    ).fit_transform(raw)
    assert ranked_sources.shape == (10, raw.n_times)


def test_dss_whiten_epochs():
    """whiten=True works on epoched (3D) data."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, _, data = _mixed_sensor_raw(n_t=1000)
    epoch_data = np.stack([data, data * 1.1], axis=0)
    epochs = mne.EpochsArray(epoch_data, raw.info, verbose=False)

    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])
    dss = DSS(bias=bias, whiten=True, component_action="retain").fit(epochs)
    rec = dss.transform(epochs).get_data()

    assert rec.shape == epoch_data.shape
    assert np.linalg.norm(rec - epoch_data) / np.linalg.norm(epoch_data) < 1e-6


def test_dss_whiten_numpy_array():
    """whiten=True works on a plain NumPy array (diagonal whitener)."""
    from mne_denoise.dss.denoisers import BandpassBias

    _, signal, data = _mixed_sensor_raw()
    # Rescale to comparable magnitudes for the array path.
    arr = data / data.std(axis=1, keepdims=True)
    bias = BandpassBias(freq_band=(8, 12), sfreq=200.0)

    dss = DSS(bias=bias, whiten=True)
    sources = dss.fit_transform(arr)

    assert sources.shape == (18, arr.shape[1])
    assert np.abs(np.corrcoef(sources[0], signal)[0, 1]) > 0.9


def test_dss_whiten_false_still_isolates_single_type():
    """whiten=False keeps the homogeneous-type behaviour and warning."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, _, _ = _mixed_sensor_raw()
    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DSS(bias=bias).fit(raw)
    assert any("multiple data channel types" in str(w.message) for w in caught)


def test_dss_whiten_noise_cov_missing_channels_raises():
    """A noise_cov missing data channels is rejected."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, _, _ = _mixed_sensor_raw()
    small = mne.Covariance(
        np.eye(2),
        raw.ch_names[:2],
        raw.info["bads"],
        raw.info["projs"],
        nfree=raw.n_times,
    )
    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])
    with pytest.raises(ValueError, match="missing required channels"):
        DSS(bias=bias, whiten=True, noise_cov=small).fit(raw)


def test_dss_whiten_noise_cov_requires_mne_input():
    """noise_cov cannot be aligned to a bare NumPy array."""
    from mne_denoise.dss.denoisers import BandpassBias

    _, _, data = _mixed_sensor_raw()
    noise_cov = mne.Covariance(
        np.eye(18),
        [f"C{i}" for i in range(18)],
        [],
        [],
        nfree=data.shape[1],
    )
    bias = BandpassBias(freq_band=(8, 12), sfreq=200.0)
    with pytest.raises(ValueError, match="named channels"):
        DSS(bias=bias, whiten=True, noise_cov=noise_cov).fit(data)


def test_dss_whiten_rejects_non_mne_noise_covariance():
    """noise_cov should use MNE's covariance type and semantics."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, _, _ = _mixed_sensor_raw()
    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])
    with pytest.raises(TypeError, match="mne.Covariance"):
        DSS(bias=bias, whiten=True, noise_cov=np.eye(18)).fit(raw)


def test_dss_whiten_no_data_channels_raises():
    """whiten=True needs at least one data channel."""
    from mne_denoise.dss.denoisers import BandpassBias

    rng = np.random.default_rng(0)
    info = mne.create_info(["S0", "S1"], 200.0, ["stim", "stim"])
    raw = mne.io.RawArray(rng.standard_normal((2, 500)), info, verbose=False)
    bias = BandpassBias(freq_band=(8, 12), sfreq=200.0)
    with pytest.raises(ValueError, match="No data channels"):
        DSS(bias=bias, whiten=True).fit(raw)


def test_dss_whiten_rejects_unsupported_input():
    """A non-array, non-MNE input is rejected in the whitening path."""
    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=200.0)
    with pytest.raises(TypeError, match="Unsupported input type"):
        DSS(bias=bias, whiten=True).fit("not data")


def test_dss_whiten_inverse_transform_recovers_sensor_data():
    """inverse_transform un-whitens sources back to sensor space."""
    from mne_denoise.dss.denoisers import BandpassBias

    raw, _, data = _mixed_sensor_raw()
    bias = BandpassBias(freq_band=(8, 12), sfreq=raw.info["sfreq"])
    dss = DSS(bias=bias, whiten=True, component_action="extract").fit(raw)

    sources = dss.transform(raw)
    rec = dss.inverse_transform(sources)

    data_centered = data - data.mean(axis=1, keepdims=True)
    assert rec.shape == data.shape
    assert np.linalg.norm(rec - data_centered) / np.linalg.norm(data_centered) < 1e-6


# =============================================================================
# Adaptive DSS – Segmented Mode
# =============================================================================


def _make_nonstationary_line_noise(
    n_channels=16,
    sfreq=250.0,
    duration=120.0,
    freq=50.0,
    snr_first_half=0.8,
    snr_second_half=0.3,
    seed=42,
):
    """Create synthetic non-stationary line noise (different amplitude halves).

    Returns data (n_channels, n_times) and sfreq.
    """
    rng = np.random.default_rng(seed)
    n_times = int(sfreq * duration)
    half = n_times // 2
    t = np.arange(n_times) / sfreq

    # Spatial mixing for line noise (rank-1)
    topo = rng.standard_normal(n_channels)
    topo /= np.linalg.norm(topo)

    # Line noise source
    source = np.sin(2 * np.pi * freq * t)

    # Background EEG (pink-ish noise)
    eeg = rng.standard_normal((n_channels, n_times)) * 0.5

    # Inject different amplitudes per half
    noise = np.outer(topo, source)
    noise[:, :half] *= snr_first_half
    noise[:, half:] *= snr_second_half

    return eeg + noise, sfreq


class TestSegmentedDSS:
    """Tests for DSS with adaptive=True."""

    def test_fit_produces_global_fit(self):
        """fit() in adaptive mode yields a usable global fit, not an error.

        Segmented adaptation happens in fit_transform; fit alone must still
        honour the sklearn contract so DSS stays usable inside a Pipeline.
        """
        data = np.random.default_rng(0).standard_normal((8, 5000))
        bias = LineNoiseBias(freq=50.0, sfreq=250.0)
        dss = DSS(bias, component_action="subtract", adaptive=True, n_components=2)
        dss.fit(data)
        assert dss.filters_ is not None
        assert dss.filters_.shape == (2, 8)
        assert dss.segment_results_ is None

    def test_fit_transform_raises_without_sfreq(self):
        """fit_transform needs an sfreq it can discover for segmentation."""

        def bias_without_sfreq(x):
            return x

        data = np.random.default_rng(0).standard_normal((8, 5000))
        dss = DSS(
            bias_without_sfreq,
            component_action="subtract",
            adaptive=True,
            n_components=2,
        )
        with pytest.raises(ValueError, match="sfreq"):
            dss.fit_transform(data)

    def test_adaptive_clone_forwards_params(self):
        """The per-segment estimator inherits every constructor parameter."""
        bias = LineNoiseBias(freq=50.0, sfreq=250.0)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            n_components=2,
            reg=1e-7,
            normalize_input=False,
            selection_threshold=2.5,
            knee_min_ratio=4.0,
        )
        est = dss._make_segment_estimator()
        assert est.adaptive is False
        assert est.reg == 1e-7
        assert est.normalize_input is False
        assert est.selection_threshold == 2.5
        assert est.knee_min_ratio == 4.0
        # adaptive mode with n_select unset must not silently select nothing
        assert est.n_select == "auto"

    def test_fit_transform_runs(self):
        """fit_transform should run to completion in adaptive mode."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=2,
        )
        result = dss.fit_transform(raw)
        assert result is not None

    def test_segment_results_populated(self):
        """After segmented fit, segment_results_ should be populated."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=2,
        )
        dss.fit_transform(raw)
        assert hasattr(dss, "segment_results_")
        assert len(dss.segment_results_) >= 2

    def test_n_selected_is_max(self):
        """n_selected_ should be the max across segments, not the sum."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=4,
            n_select="auto",
        )
        dss.fit_transform(raw)
        per_seg_n = [r["n_selected"] for r in dss.segment_results_]
        assert dss.n_selected_ == max(per_seg_n)

    def test_reduces_artifact(self):
        """Segmented DSS should reduce line noise power."""
        from scipy.signal import welch

        data, sfreq = _make_nonstationary_line_noise(freq=50.0)
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=4,
            n_select=1,
        )
        cleaned = dss.fit_transform(raw)
        cleaned_data = cleaned.get_data()

        # Compare average 50 Hz power before and after
        def avg_power_at_freq(d, freq, sfreq):
            f, psd = welch(d, fs=sfreq, nperseg=min(1024, d.shape[1]))
            idx = np.argmin(np.abs(f - freq))
            return psd[:, idx].mean()

        pwr_before = avg_power_at_freq(data, 50.0, sfreq)
        pwr_after = avg_power_at_freq(cleaned_data, 50.0, sfreq)
        assert pwr_after < pwr_before

    def test_covariance_segmenter(self):
        """CovarianceSegmenter as segmenter should work."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=CovarianceSegmenter(sfreq=sfreq, min_chunk_len=20.0),
            n_components=2,
        )
        result = dss.fit_transform(raw)
        assert result is not None

    def test_epochs_3d(self):
        """Segmented DSS should reject 3-D data (Epochs) gracefully or run."""
        rng = np.random.default_rng(42)
        n_ch, n_times, sfreq = 8, 250, 250.0
        n_epochs = 10
        data_3d = rng.standard_normal((n_ch, n_times * n_epochs))
        info = mne.create_info(n_ch, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data_3d, info, verbose=False)
        events = np.column_stack(
            [
                np.arange(0, n_times * n_epochs, n_times),
                np.zeros(n_epochs, int),
                np.ones(n_epochs, int),
            ]
        )
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=(n_times - 1) / sfreq,
            baseline=None,
            preload=True,
            verbose=False,
        )
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, component_action="subtract", adaptive=True, n_components=2)
        # Epochs should either work (segmentation on concatenated) or raise
        try:
            dss.fit_transform(epochs)
        except (ValueError, RuntimeError):
            pass  # Acceptable: adaptive mode may not support Epochs


class TestAutoSelect:
    """Tests for automatic component selection (n_select='outlier', etc.)."""

    def test_auto(self):
        """n_select='auto' should set n_selected_ >= 0."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=4, n_select="auto")
        dss.fit(raw)
        assert dss.n_selected_ is not None
        assert dss.n_selected_ >= 0

    def test_auto_matches_robust_selector(self):
        """auto_select delegates to auto_select_components_robust."""
        from mne_denoise.dss.utils.selection import auto_select_components_robust

        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=4, n_select="auto")
        dss.fit(raw)
        assert dss.n_selected_ == auto_select_components_robust(
            dss.eigenvalues_,
            sigma=dss.selection_threshold,
            knee_rel_floor=dss.knee_rel_floor,
            knee_min_ratio=dss.knee_min_ratio,
        )

    def test_auto_selects_nothing_on_clean_data(self):
        """A smoothly-decaying spectrum must not trigger false removals."""
        rng = np.random.default_rng(0)
        sfreq = 250.0
        data = rng.standard_normal((12, int(60 * sfreq))) * 1e-6
        info = mne.create_info(12, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_select="auto")
        dss.fit(raw)
        assert dss.n_selected_ == 0

    def test_int_passthrough(self):
        """n_select=int should directly set n_selected_."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=4, n_select=2)
        dss.fit(raw)
        assert dss.n_selected_ == 2

    def test_invalid_n_select(self):
        """A non-int, non-'auto' n_select should raise ValueError."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=4, n_select="nonexistent_method")
        with pytest.raises(ValueError, match="n_select"):
            dss.fit(raw)

    def test_manual_override(self):
        """n_select=None should leave n_selected_=None."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=4, n_select=None)
        dss.fit(raw)
        assert dss.n_selected_ is None


class TestSmoothingDecomposition:
    """Tests for smooth parameter (smoothing decomposition)."""

    def test_smooth_int_fit_transform(self):
        """smooth=int should run fit_transform without error."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=2, smooth=5)
        result = dss.fit_transform(raw)
        assert result is not None

    def test_fit_then_transform_preserves_smooth(self):
        """fit() then transform() should NOT lose the smooth component.

        Regression test: previously transform() discarded data_smooth.
        """
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(bias, n_components=2, smooth=5, component_action="retain")
        dss.fit(raw)
        result = dss.transform(raw)
        result_data = result.get_data()
        # The result should have similar scale to the input (because smooth is
        # added back). If smooth were lost, the result would be much smaller.
        ratio = np.std(result_data) / np.std(data)
        assert ratio > 0.3, f"Smooth likely lost: ratio={ratio:.3f}"

    def test_smooth_adaptive_cleans_artifact(self):
        """Smooth + adaptive should still reduce artifact power."""
        from scipy.signal import welch

        data, sfreq = _make_nonstationary_line_noise(freq=50.0)
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=4,
            n_select=1,
            smooth=5,
        )
        cleaned = dss.fit_transform(raw)
        cleaned_data = cleaned.get_data()

        # Compare 50 Hz power
        def avg_power_at_freq(d, freq, sfreq):
            f, psd = welch(d, fs=sfreq, nperseg=min(1024, d.shape[1]))
            idx = np.argmin(np.abs(f - freq))
            return psd[:, idx].mean()

        pwr_before = avg_power_at_freq(data, 50.0, sfreq)
        pwr_after = avg_power_at_freq(cleaned_data, 50.0, sfreq)
        assert pwr_after < pwr_before


class TestCapAndFloor:
    """Tests for max_prop_remove and min_select."""

    def test_max_prop_remove_caps(self):
        """max_prop_remove=0.1 on 32ch should cap at 3 components."""
        rng = np.random.default_rng(42)
        n_ch = 32
        data = rng.standard_normal((n_ch, int(250 * 120)))
        sfreq = 250.0
        # Inject strong line noise to get many components selected
        t = np.arange(data.shape[1]) / sfreq
        topo = rng.standard_normal(n_ch)
        for h in range(1, 6):  # 5 harmonics
            data += np.outer(topo * h, np.sin(2 * np.pi * 50 * h * t))
        info = mne.create_info(n_ch, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=10,
            n_select="auto",
            max_prop_remove=0.1,
        )
        dss.fit_transform(raw)
        max_cap = int(n_ch * 0.1)
        for r in dss.segment_results_:
            assert r["n_selected"] <= max_cap

    def test_min_select_floor(self):
        """min_select=2 should ensure at least 2 components removed."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=4,
            n_select="auto",
            min_select=2,
        )
        dss.fit_transform(raw)
        for r in dss.segment_results_:
            assert r["n_selected"] >= 2


class TestCrossfade:
    """Tests for cross-fade overlap-add blending at segment boundaries."""

    def test_crossfade_output_shape(self):
        """Output shape must match input shape regardless of crossfade."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            crossfade=1.0,
            n_components=4,
            n_select="auto",
        )
        cleaned = dss.fit_transform(raw)
        assert cleaned.get_data().shape == data.shape

    def test_crossfade_no_boundary_jump(self):
        """Derivative at segment boundaries should not spike."""
        data, sfreq = _make_nonstationary_line_noise(duration=120.0)
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)

        # Without cross-fade
        dss_hard = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            crossfade=0.0,
            n_components=4,
            n_select="auto",
        )
        cleaned_hard = dss_hard.fit_transform(raw).get_data()

        # With cross-fade
        dss_xfade = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            crossfade=1.0,
            n_components=4,
            n_select="auto",
        )
        cleaned_xfade = dss_xfade.fit_transform(raw).get_data()

        # Check derivative at boundary (30s = 7500 samples)
        boundary = int(30 * sfreq)
        win = 10  # samples around boundary
        far = 500  # samples away for reference

        diff_hard = np.abs(np.diff(cleaned_hard, axis=1))
        diff_xfade = np.abs(np.diff(cleaned_xfade, axis=1))

        # Max derivative at boundary vs. reference region
        bnd_hard = diff_hard[:, boundary - win : boundary + win].max()
        ref_hard = np.median(diff_hard[:, boundary - far : boundary - 2 * win])
        bnd_xfade = diff_xfade[:, boundary - win : boundary + win].max()
        ref_xfade = np.median(diff_xfade[:, boundary - far : boundary - 2 * win])

        ratio_hard = bnd_hard / (ref_hard + 1e-12)
        ratio_xfade = bnd_xfade / (ref_xfade + 1e-12)

        # Cross-fade boundary ratio should be no worse than hard boundary
        # (and often much better)
        assert ratio_xfade <= ratio_hard + 1.0 or ratio_xfade < 10.0

    def test_crossfade_single_segment_matches_hard(self):
        """With only one segment, crossfade has no effect."""
        # Short data → single segment (< min_chunk_len * 2)
        rng = np.random.default_rng(99)
        sfreq = 250.0
        data = rng.standard_normal((8, int(40 * sfreq)))
        t = np.arange(data.shape[1]) / sfreq
        data += 0.5 * np.sin(2 * np.pi * 50 * t)[np.newaxis, :]

        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        seg = FixedWindowSegmenter(sfreq=sfreq, window_len=60.0)

        dss_hard = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=seg,
            crossfade=0.0,
            n_components=4,
            n_select="auto",
        )
        dss_xfade = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=seg,
            crossfade=1.0,
            n_components=4,
            n_select="auto",
        )

        out_hard = dss_hard.fit_transform(data)
        out_xfade = dss_xfade.fit_transform(data)
        np.testing.assert_array_equal(out_hard, out_xfade)

    def test_crossfade_zero_backward_compat(self):
        """crossfade=0 must produce same result as before (hard concat)."""
        data, sfreq = _make_nonstationary_line_noise()
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        seg = FixedWindowSegmenter(sfreq=sfreq, window_len=30.0)

        # Default crossfade (0.0)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=seg,
            n_components=4,
            n_select="auto",
        )
        out_default = dss.fit_transform(data)

        # Explicit crossfade=0.0
        dss0 = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=seg,
            crossfade=0.0,
            n_components=4,
            n_select="auto",
        )
        out_zero = dss0.fit_transform(data)
        np.testing.assert_array_equal(out_default, out_zero)

    def test_crossfade_preserves_energy(self):
        """Total signal power should not change dramatically with crossfade."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)

        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            crossfade=1.0,
            n_components=4,
            n_select="auto",
        )
        cleaned = dss.fit_transform(raw).get_data()

        # Power ratio should be between 0.3 and 1.5
        # (cleaning removes artifact, but broadband should be preserved)
        power_in = np.mean(data**2)
        power_out = np.mean(cleaned**2)
        ratio = power_out / power_in
        assert 0.3 < ratio < 1.5, f"Power ratio {ratio:.2f} out of range"

    def test_crossfade_overlap_clamped(self):
        """When crossfade is longer than half a segment, it gets clamped."""
        rng = np.random.default_rng(42)
        sfreq = 250.0
        # 20s data with 10s segments → crossfade=8s should get clamped
        data = rng.standard_normal((8, int(20 * sfreq)))
        t = np.arange(data.shape[1]) / sfreq
        data += 0.3 * np.sin(2 * np.pi * 50 * t)[np.newaxis, :]

        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=10.0),
            crossfade=8.0,  # way too long
            n_components=4,
            n_select="auto",
        )
        # Should not crash — overlap gets clamped internally
        cleaned = dss.fit_transform(data)
        assert cleaned.shape == data.shape


def test_narrowband_scan_rejects_adaptive():
    """narrowband_scan should raise ValueError with adaptive=True."""
    from mne_denoise.dss.variants.narrowband import narrowband_scan

    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 5000))
    with pytest.raises(ValueError, match="adaptive"):
        narrowband_scan(data, sfreq=250.0, adaptive=True)


class TestSmootherResolution:
    """_as_smoother coerces the ``smooth`` parameter to a denoiser."""

    def test_none_returns_none(self):
        from mne_denoise.dss.linear import _as_smoother

        assert _as_smoother(None) is None

    def test_int_becomes_smoothing_bias(self):
        from mne_denoise.dss.denoisers.temporal import SmoothingBias
        from mne_denoise.dss.linear import _as_smoother

        smoother = _as_smoother(7)
        assert isinstance(smoother, SmoothingBias)
        assert smoother.window == 7

    def test_numpy_integer_accepted(self):
        """np.int64 windows arise naturally from int(sfreq / f_line)."""
        from mne_denoise.dss.denoisers.temporal import SmoothingBias
        from mne_denoise.dss.linear import _as_smoother

        assert isinstance(_as_smoother(np.int64(7)), SmoothingBias)

    def test_duck_typed_denoiser_passes_through(self):
        from mne_denoise.dss.linear import _as_smoother

        class Custom:
            def apply(self, data):
                return data

        custom = Custom()
        assert _as_smoother(custom) is custom

    def test_unusable_value_raises(self):
        from mne_denoise.dss.linear import _as_smoother

        with pytest.raises(TypeError, match="smooth must be"):
            _as_smoother("boxcar")


def test_auto_select_before_fit_raises():
    dss = DSS(bias=lambda x: x, n_select="auto")
    with pytest.raises(RuntimeError, match="not fitted"):
        dss.auto_select()


def test_get_normalized_patterns_before_fit_raises():
    dss = DSS(bias=lambda x: x)
    with pytest.raises(RuntimeError, match="not fitted"):
        dss.get_normalized_patterns()


def test_get_normalized_patterns_returns_unit_columns():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 3000)) * 1e-6
    dss = DSS(bias=lambda x: x, n_components=3).fit(data)

    patterns = dss.get_normalized_patterns()
    assert patterns.shape == dss.patterns_.shape
    assert_allclose(np.linalg.norm(patterns, axis=0), 1.0, rtol=1e-10)
    # The stored patterns keep their physical scale
    assert not np.allclose(np.linalg.norm(dss.patterns_, axis=0), 1.0)


def test_fit_transform_subtracts_selected_components():
    """Explicit subtraction removes selected components."""
    sfreq = 250.0
    n_times = int(20 * sfreq)
    t = np.arange(n_times) / sfreq
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, n_times)) * 0.2
    data += np.sin(2 * np.pi * 50.0 * t) * 4.0

    bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
    dss = DSS(bias, n_select=2, component_action="subtract")
    cleaned = dss.fit_transform(data)

    assert cleaned.shape == data.shape
    assert dss.n_selected_ == 2
    # Some artifact was actually removed
    assert not np.allclose(cleaned, data)


def test_fit_transform_without_selection_is_passthrough():
    """n_select=None means nothing is subtracted, so data returns unchanged."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 2000))

    dss = DSS(bias=lambda x: x, component_action="subtract")
    cleaned = dss.fit_transform(data)

    assert dss.n_selected_ is None
    assert_allclose(cleaned, data)


def test_empty_segmentation_raises():
    class EmptySegmenter:
        def segment(self, data):
            return []

    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 5000))
    bias = LineNoiseBias(freq=50.0, sfreq=250.0)
    dss = DSS(
        bias, component_action="subtract", adaptive=True, segmenter=EmptySegmenter()
    )

    with pytest.raises(ValueError, match="no segments"):
        dss.fit_transform(data)


def test_default_segmenter_band_limits_around_bias_frequency():
    bias = LineNoiseBias(freq=50.0, sfreq=250.0)
    segmenter = DSS(bias, adaptive=True)._resolve_segmenter(250.0)

    assert isinstance(segmenter, CovarianceSegmenter)
    assert segmenter.bandpass == (47.0, 53.0)


def test_default_segmenter_has_no_band_without_bias_frequency():
    """A bias with no .freq gives a broadband segmenter rather than crashing."""
    segmenter = DSS(bias=lambda x: x, adaptive=True)._resolve_segmenter(250.0)

    assert isinstance(segmenter, CovarianceSegmenter)
    assert segmenter.bandpass is None


def test_normalize_preserves_raw_annotations():
    rng = np.random.default_rng(0)
    info = mne.create_info(4, 250.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 2500)) * 1e-6, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[1.0], duration=[0.5], description=["bad"])
    )

    normalized = DSS(bias=lambda x: x)._normalize(raw, fit=True)

    assert len(normalized.annotations) == 1
    assert normalized.annotations.description[0] == "bad"


def test_normalize_preserves_epochs_metadata():
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(0)
    info = mne.create_info(4, 250.0, "eeg")
    epochs = mne.EpochsArray(
        rng.standard_normal((5, 4, 200)) * 1e-6, info, verbose=False
    )
    epochs.metadata = pd.DataFrame({"cond": list("abcde")})

    normalized = DSS(bias=lambda x: x)._normalize(epochs, fit=True)

    assert normalized.metadata is not None
    assert list(normalized.metadata["cond"]) == list("abcde")
