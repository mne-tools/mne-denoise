"""Unit tests for DSS module - Linear DSS (compute_dss and DSS class)."""

from __future__ import annotations

import warnings

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss import DSS, compute_dss
from mne_denoise.dss.denoisers.spectral import LineNoiseBias
from mne_denoise.dss.segmentation import FixedWindowSegmenter

# =============================================================================
# compute_dss - Core Algorithm Tests
# =============================================================================


def test_compute_dss_eigensystem_contract():
    """DSS orders a known biased eigensystem and keeps its metric orthogonal."""
    cov0 = np.eye(4)
    cov1 = np.diag([10.0, 4.0, 1.0, 0.5])
    filters, patterns, eigenvalues = compute_dss(cov0, cov1)
    alignment = np.abs(filters[0] / np.linalg.norm(filters[0]))
    assert alignment[0] > 0.95
    assert eigenvalues[0] == pytest.approx(10.0)
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:])
    gram = filters @ cov0 @ filters.T
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diagonal)) < 1e-10
    filters_again, _, eigenvalues_again = compute_dss(cov0, cov1)
    assert_allclose(np.abs(filters_again), np.abs(filters))
    assert_allclose(eigenvalues_again, eigenvalues)
    identity_values = compute_dss(np.eye(4), np.eye(4))[2]
    assert_allclose(identity_values, np.ones(4), atol=1e-12)
    assert patterns.shape == (4, 4)


def test_compute_dss_rank_and_reconstruction_contract():
    """Component truncation, rank limits, and full reconstruction agree."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 10))
    cov = A @ A.T
    filters, patterns, eigenvalues = compute_dss(cov, cov, n_components=3)
    assert filters.shape == (3, 10)
    assert patterns.shape == (10, 3)
    assert eigenvalues.shape == (3,)
    ranked_filters, _, _ = compute_dss(cov, cov, rank=5)
    assert ranked_filters.shape[0] <= 5

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


def test_compute_dss_covariance_validation():
    """Shape, square, and variance failures are one covariance contract."""
    cases = [
        (np.eye(5), np.eye(6), "shapes mismatch"),
        (np.ones((5, 6)), np.ones((5, 6)), "must be square"),
        (np.zeros((5, 5)), np.zeros((5, 5)), "no significant variance"),
    ]
    for cov0, cov1, message in cases:
        with pytest.raises(ValueError, match=message):
            compute_dss(cov0, cov1)


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
    covariance = data @ data.T / data.shape[1]
    assert_allclose(
        dss.explained_variance_,
        np.diag(dss.filters_ @ covariance @ dss.filters_.T),
    )
    assert_allclose(dss.eigenvalues_, np.ones(2), atol=1e-10)


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


def test_dss_raw_covariances_use_identical_sample_support(monkeypatch):
    """Baseline and biased Raw covariance must see identical sample support."""
    original_compute = mne.compute_raw_covariance
    for normalize_input in (False, True):
        rng = np.random.default_rng(42)
        info = mne.create_info(["EEG0", "EEG1", "EEG2"], 100.0, "eeg")
        raw = mne.io.RawArray(
            rng.standard_normal((3, 1000)),
            info,
            first_samp=1000,
            verbose=False,
        )
        raw.set_annotations(
            mne.Annotations(onset=[2.0], duration=[2.0], description=["BAD_test"])
        )
        raw.set_eeg_reference(projection=True, verbose=False)

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
        DSS(bias=lambda data: data, normalize_input=normalize_input).fit(raw)

        assert len(calls) == 2
        assert calls[0]["first_samp"] == calls[1]["first_samp"] == raw.first_samp
        assert calls[0]["annotations"] == calls[1]["annotations"]
        assert calls[0]["nfree"] == calls[1]["nfree"]
        assert calls[0]["nfree"] < raw.n_times - 1
        assert calls[0]["projectors"] == calls[1]["projectors"]


def test_dss_alternate_fit_paths_preserve_bad_raw_channels():
    """Smoothing and whitening must follow the same good-channel contract."""
    for fit_kws in ({"smooth": 5}, {"whiten": True}):
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
        assert_allclose(
            transformed.get_data(picks=["EEG3"]), raw.get_data(picks=["EEG3"])
        )


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


def test_dss_fit_rejects_all_bad_data_channels():
    """DSS must fail clearly when no usable fitted channels remain."""
    rng = np.random.default_rng(50)
    info = mne.create_info(["EEG0", "EEG1"], 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((2, 1000)), info, verbose=False)
    raw.info["bads"] = raw.ch_names

    with pytest.raises(ValueError, match="No good data channels remain"):
        DSS(bias=lambda data: data, normalize_input=False).fit(raw)


def test_dss_mne_bad_channels_are_excluded_and_preserved():
    """DSS excludes bad channels while preserving Raw, Epochs, and Evoked data."""
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


# =============================================================================
# DSS - Multi-modal joint decomposition via whitening
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


def test_dss_whiten_noise_cov_validation():
    """Noise covariance validation covers coverage, type, names, and data channels."""
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
    with pytest.raises(ValueError, match="noise_cov is missing required channels"):
        DSS(bias=bias, whiten=True, noise_cov=small).fit(raw)

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
    with pytest.raises(TypeError, match="mne.Covariance"):
        DSS(bias=bias, whiten=True, noise_cov=np.eye(18)).fit(raw)

    rng = np.random.default_rng(0)
    info = mne.create_info(["S0", "S1"], 200.0, ["stim", "stim"])
    raw = mne.io.RawArray(rng.standard_normal((2, 500)), info, verbose=False)
    with pytest.raises(ValueError, match="No data channels"):
        DSS(bias=bias, whiten=True).fit(raw)


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


def _make_small_adaptive_dss_case():
    """Create a fast deterministic case for adaptive callback tests."""
    rng = np.random.default_rng(123)
    sfreq = 100.0
    n_times = 600
    times = np.arange(n_times) / sfreq
    data = rng.standard_normal((4, n_times)) * 0.1
    topo = rng.standard_normal(4)
    topo /= np.linalg.norm(topo)
    data += 2.0 * np.outer(topo, np.sin(2 * np.pi * 20.0 * times))
    bias = LineNoiseBias(
        freq=20.0,
        sfreq=sfreq,
        n_harmonics=1,
        nfft=128,
    )
    segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=2.0)
    return data, sfreq, bias, segmenter


class TestSegmentedDSS:
    """Tests for DSS with adaptive=True."""

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

    def test_segment_results_populated(self):
        """Adaptive fitting keeps global state and local segment topology."""
        data, sfreq = _make_nonstationary_line_noise()
        info = mne.create_info(data.shape[0], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        bias = LineNoiseBias(freq=50.0, sfreq=sfreq)
        global_est = DSS(
            bias, component_action="subtract", adaptive=True, n_components=2
        ).fit(data)
        assert global_est.filters_.shape == (2, data.shape[0])
        assert global_est.segment_results_ is None
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=FixedWindowSegmenter(sfreq=sfreq, window_len=30.0),
            n_components=2,
        )
        dss.fit_transform(raw)
        assert dss.filters_.shape == (2, data.shape[0])
        assert dss.segment_results_ is not None
        assert len(dss.segment_results_) == 4
        assert [
            (result["start"], result["end"]) for result in dss.segment_results_
        ] == [
            (0, 7500),
            (7500, 15000),
            (15000, 22500),
            (22500, 30000),
        ]
        assert all(
            result["filters"].shape == dss.filters_.shape
            for result in dss.segment_results_
        )
        assert all(
            {"eigenvalues", "patterns", "filters"} <= result.keys()
            for result in dss.segment_results_
        )
        assert any(
            not np.allclose(np.abs(result["filters"]), np.abs(global_est.filters_))
            for result in dss.segment_results_
        )

    def test_adaptive_progress_reports_completed_segments(self):
        """Adaptive DSS emits one event per completed segment."""
        data, sfreq, bias, segmenter = _make_small_adaptive_dss_case()
        dss = DSS(
            bias,
            adaptive=True,
            component_action="subtract",
            segmenter=segmenter,
            n_components=2,
            n_select=1,
            normalize_input=False,
        )
        events = []

        dss.fit_transform(data, callback=events.append)

        n_segments = len(dss.segment_results_)
        assert len(events) == n_segments
        assert [event.current for event in events] == list(range(1, n_segments + 1))
        assert [event.total for event in events] == [n_segments] * n_segments
        assert all(event.method == "dss" for event in events)
        assert all(event.stage == "segment" for event in events)
        assert all(event.component is None for event in events)
        assert [event.metric for event in events] == [
            float(result["n_selected"]) for result in dss.segment_results_
        ]

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


class TestAutoSelect:
    """Tests for automatic component selection (n_select='outlier', etc.)."""

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

    def test_fit_then_transform_preserves_smooth(self):
        """fit() followed by transform() preserves the smooth component."""
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

        # Cross-fade stays within a deterministic tolerance of the hard boundary.
        assert ratio_xfade <= ratio_hard + 1.0

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


def test_get_normalized_patterns_returns_unit_columns():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 3000)) * 1e-6
    dss = DSS(bias=lambda x: x, n_components=3).fit(data)

    patterns = dss.get_normalized_patterns()
    assert patterns.shape == dss.patterns_.shape
    assert_allclose(np.linalg.norm(patterns, axis=0), 1.0, rtol=1e-10)
    # The stored patterns keep their physical scale
    assert not np.allclose(np.linalg.norm(dss.patterns_, axis=0), 1.0)


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
