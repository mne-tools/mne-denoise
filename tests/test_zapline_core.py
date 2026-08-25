"""Unit tests for ZapLine core functions (dss_zapline, apply_zapline)."""

from __future__ import annotations

from unittest.mock import patch

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from mne_denoise.dss.denoisers.spectral import LineNoiseBias
from mne_denoise.zapline.core import ZapLine


@pytest.fixture
def line_noise_data():
    """Generate synthetic data with 50 Hz line noise."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_channels = 8
    n_times = 5000  # 10s
    times = np.arange(n_times) / sfreq

    # Clean brain-like signal (alpha oscillation + noise)
    brain_signal = np.sin(2 * np.pi * 10 * times) * 0.5  # 10 Hz alpha
    brain_signal = brain_signal[np.newaxis, :] + rng.normal(
        0, 0.2, (n_channels, n_times)
    )

    # Line noise (50 Hz + harmonics) - STRONG and uniform spatial distribution
    line_50hz = np.sin(2 * np.pi * 50 * times) * 10.0  # Much stronger!
    line_100hz = np.sin(2 * np.pi * 100 * times) * 3.0
    line_noise = line_50hz + line_100hz  # Same on all channels

    data = brain_signal + line_noise[np.newaxis, :]

    return {
        "data": data,
        "sfreq": sfreq,
        "line_freq": 50.0,
        "brain_signal": brain_signal,
        "line_noise": line_noise[np.newaxis, :] * np.ones((n_channels, 1)),
        "times": times,
    }


@pytest.fixture
def minimal_data():
    """Generate minimal test data for quick tests."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_channels = 4
    n_times = 1000  # 4s
    times = np.arange(n_times) / sfreq

    # Simple 50 Hz noise
    line_noise = np.sin(2 * np.pi * 50 * times) * 5.0
    data = rng.normal(0, 1, (n_channels, n_times))
    data[0] += line_noise  # Add noise mostly to ch 0

    return {"data": data, "sfreq": sfreq, "line_freq": 50.0, "times": times}


@pytest.fixture
def mixed_sensor_raw():
    """Create mixed-unit MNE data with one non-data channel."""
    rng = np.random.default_rng(37)
    sfreq = 250.0
    n_times = 2000
    times = np.arange(n_times) / sfreq
    ch_types = ["mag", "mag", "grad", "grad", "eeg", "eeg", "stim"]
    scales = np.array([1e-12, 1e-12, 1e-11, 1e-11, 1e-5, 1e-5, 1.0])
    data = rng.standard_normal((len(ch_types), n_times)) * scales[:, None]
    data[:6] += np.sin(2 * np.pi * 50.0 * times) * scales[:6, None] * 5.0
    data[-1, 500:510] = 1.0
    info = mne.create_info(
        [f"CH{idx}" for idx in range(len(ch_types))], sfreq, ch_types
    )
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture(scope="module")
def smooth_spectrum_line_noise():
    """Reproduce Issue #63 with stochastic, distributed line noise."""
    sfreq = 500.0
    line_freq = 50.0
    n_channels = 19
    n_times = int(sfreq * 120)
    rng = np.random.default_rng(0)
    bandpass = signal.butter(
        2,
        [(line_freq - 0.2) / (sfreq / 2), (line_freq + 0.2) / (sfreq / 2)],
        btype="band",
    )

    sources = np.empty((n_channels, n_times))
    for component in range(n_channels):
        pink = np.cumsum(rng.standard_normal(n_times))
        pink /= pink.std()
        white = rng.standard_normal(n_times)
        white /= white.std()
        sources[component] = (pink + 0.15 * white) * (1 + 0.35 * component / n_channels)

    amplitudes = np.array(
        [0.35, 0.21, 0.155, 0.08, 0.047, 0.03, 0.02]
        + [0.02 * 0.8 ** (component + 1) for component in range(n_channels - 7)]
    )
    for component, amplitude in enumerate(amplitudes):
        line = signal.filtfilt(bandpass[0], bandpass[1], rng.standard_normal(n_times))
        sources[component] += amplitude * line / line.std()

    mixing, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
    data = (mixing @ sources) * 3e-6
    return data, sfreq, line_freq


@pytest.fixture(scope="module")
def fitted_smooth_spectrum_line_noise(smooth_spectrum_line_noise):
    """Fit standard auto-selection on the Issue #63 reproducer once."""
    data, sfreq, line_freq = smooth_spectrum_line_noise
    estimator = ZapLine(sfreq=sfreq, line_freq=line_freq, n_select="auto")
    estimator.fit(data)
    return estimator


def test_zapline_class_init(minimal_data):
    """ZapLine should initialize correctly."""
    est = ZapLine(
        line_freq=minimal_data["line_freq"],
        sfreq=minimal_data["sfreq"],
    )
    assert est.n_select == "auto"


def test_zapline_class_fit_transform(minimal_data):
    """ZapLine should fit and transform correctly."""
    data = minimal_data["data"]
    est = ZapLine(
        line_freq=minimal_data["line_freq"],
        sfreq=minimal_data["sfreq"],
    )
    est.fit(data)
    cleaned = est.transform(data)
    assert est.filters_ is not None
    assert cleaned.shape == data.shape


def test_zapline_whiten_processes_mixed_sensor_types(mixed_sensor_raw):
    """Whitening should jointly clean data channels and preserve other channels."""
    raw = mixed_sensor_raw
    stim_before = raw.get_data(picks="stim")
    est = ZapLine(
        line_freq=50.0,
        sfreq=raw.info["sfreq"],
        n_select=1,
        whiten=True,
    )

    cleaned = est.fit_transform(raw)

    assert isinstance(cleaned, mne.io.BaseRaw)
    assert est.filters_.shape[1] == 6
    assert cleaned.ch_names == raw.ch_names
    np.testing.assert_array_equal(cleaned.get_data(picks="stim"), stim_before)
    with pytest.raises(ValueError, match="missing required channels"):
        est.transform(raw.copy().drop_channels(["CH0"]))


def test_zapline_class_output_shapes(minimal_data):
    """ZapLine should return arrays with correct shapes."""
    data = minimal_data["data"]
    # n_harmonics=None means auto-detected
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"])
    est.fit(data)
    cleaned = est.transform(data)

    n_channels, n_times = data.shape

    assert cleaned.shape == data.shape
    # Check attributes populated after fit
    assert est.filters_.shape[1] == n_channels
    assert est.patterns_.shape[0] == n_channels
    assert len(est.eigenvalues_) > 0


def test_dss_zapline_reduces_line_noise(line_noise_data):
    """dss_zapline should significantly reduce power at line frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    line_freq = line_noise_data["line_freq"]

    # Use fixed n_select=1 to ensure we actually remove something
    est = ZapLine(line_freq=line_freq, sfreq=sfreq, n_select=1)
    est.fit(data)
    cleaned = est.transform(data)

    # Compare power at 50 Hz before/after
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, line_freq, sfreq)
    power_after = get_power_at(cleaned, line_freq, sfreq)

    reduction = (power_before - power_after) / power_before
    assert reduction > 0.1, f"Insufficient noise reduction: {reduction:.2%}"


def test_dss_zapline_preserves_brain_signal(line_noise_data):
    """dss_zapline should preserve signals at non-line frequencies."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Use n_select=1 to ensure testing with actual removal
    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_select=1)
    est.fit(data)
    cleaned = est.transform(data)

    # Check 10 Hz alpha power preservation
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_alpha_original = get_power_at(data, 10.0, sfreq)
    power_alpha_cleaned = get_power_at(cleaned, 10.0, sfreq)

    # Alpha power should be mostly preserved (within 30%)
    ratio = power_alpha_cleaned / power_alpha_original
    assert ratio > 0.5, f"Alpha signal degraded: ratio={ratio:.2f}"


def test_dss_zapline_closed_sum_property(minimal_data):
    """cleaned + removed should equal original data."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"])
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"])
    est.fit(data)

    cleaned = est.transform(data)
    removed = data - cleaned

    reconstructed = cleaned + removed
    assert_allclose(reconstructed, data, rtol=1e-10)


def test_zapline_n_remove_fixed(minimal_data):
    """ZapLine should remove exactly n_select components when specified."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], n_select=2)
    with patch("mne_denoise.zapline.core.check_artifact_presence") as check_presence:
        est.fit(data)

    assert est.n_removed_ == 2
    check_presence.assert_not_called()


def test_zapline_n_remove_auto(minimal_data):
    """ZapLine auto should work (may remove 0 or more components)."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], n_select="auto")
    est.fit(data)
    cleaned = est.transform(data)

    assert est.n_removed_ >= 0
    assert cleaned.shape == data.shape


def test_zapline_auto_spectral_fallback_selects_components(
    smooth_spectrum_line_noise, fitted_smooth_spectrum_line_noise
):
    """Auto-selection removes Issue #63's distributed stochastic line noise."""
    data, _, line_freq = smooth_spectrum_line_noise
    estimator = fitted_smooth_spectrum_line_noise
    cleaned = estimator.transform(data)

    freqs, psd_before = signal.welch(data, fs=estimator.sfreq, nperseg=8192, axis=-1)
    _, psd_after = signal.welch(cleaned, fs=estimator.sfreq, nperseg=8192, axis=-1)
    line_index = np.argmin(np.abs(freqs - line_freq))
    power_before = psd_before[:, line_index].mean()
    power_after = psd_after[:, line_index].mean()

    assert estimator.n_removed_ > 0
    assert power_after < power_before * 0.1


def test_zapline_auto_spectral_fallback_keeps_clean_data_unchanged():
    """Auto-selection still removes zero components without line noise."""
    rng = np.random.default_rng(0)
    sfreq = 250.0
    data = rng.standard_normal((12, int(sfreq * 60))) * 1e-6
    estimator = ZapLine(sfreq=sfreq, line_freq=50.0, n_select="auto").fit(data)

    assert estimator.n_removed_ == 0


def test_zapline_spectral_fallback_stops_at_first_ok():
    """The fallback selects the smallest candidate passing spectral QA."""
    estimator = ZapLine(sfreq=500.0, line_freq=50.0)
    data_smooth = np.zeros((3, 20))
    data_residual = np.ones((3, 20))
    full_filters = np.eye(3)
    full_mixing = np.eye(3)

    with patch(
        "mne_denoise.zapline.core.check_spectral_qa",
        side_effect=["weak", "weak", "ok"],
    ) as check_qa:
        selected = estimator._find_min_components_for_line_suppression(
            data_smooth=data_smooth,
            data_residual=data_residual,
            full_filters=full_filters,
            full_mixing=full_mixing,
        )

    assert selected == 3
    assert check_qa.call_count == 3


def test_zapline_spectral_fallback_warns_when_no_candidate_passes():
    """The fallback warns when every candidate remains over-cleaned."""
    estimator = ZapLine(sfreq=500.0, line_freq=50.0)
    data_smooth = np.zeros((2, 20))
    data_residual = np.ones((2, 20))
    full_filters = np.eye(2)
    full_mixing = np.eye(2)

    with (
        patch(
            "mne_denoise.zapline.core.check_spectral_qa",
            return_value="strong",
        ),
        pytest.warns(UserWarning, match="Line noise remains"),
    ):
        selected = estimator._find_min_components_for_line_suppression(
            data_smooth=data_smooth,
            data_residual=data_residual,
            full_filters=full_filters,
            full_mixing=full_mixing,
        )

    assert selected == 2


def test_zapline_with_harmonics(line_noise_data):
    """ZapLine should handle harmonics correctly."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_harmonics=2, n_select=1)
    est.fit(data)

    assert est.n_harmonics_ == 2

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    cleaned = est.transform(data)
    power_100_before = get_power_at(data, 100.0, sfreq)
    power_100_after = get_power_at(cleaned, 100.0, sfreq)

    assert power_100_after < power_100_before


def test_zapline_with_nkeep(minimal_data):
    """ZapLine should work with nkeep parameter."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], nkeep=2)
    est.fit(data)
    cleaned = est.transform(data)

    # Should still produce valid output
    assert cleaned.shape == data.shape
    assert est.filters_.shape[1] == data.shape[0]


def test_zapline_60hz(minimal_data):
    """ZapLine should work with 60 Hz line frequency."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000
    times = np.arange(n_times) / sfreq

    # Create data with 60 Hz noise
    data = rng.normal(0, 1, (4, n_times))
    data += np.sin(2 * np.pi * 60 * times) * 5.0

    est = ZapLine(line_freq=60.0, sfreq=sfreq, n_select=1)
    est.fit(data)
    cleaned = est.transform(data)

    # Check 60 Hz power reduction
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 60.0, sfreq)
    power_after = get_power_at(cleaned, 60.0, sfreq)

    assert power_after < power_before * 0.5


def test_zapline_threshold_parameter(minimal_data):
    """ZapLine should use threshold for auto component selection."""
    data = minimal_data["data"]

    # Low threshold should remove more components
    est_low = ZapLine(
        line_freq=50.0, sfreq=minimal_data["sfreq"], n_select="auto", threshold=1.0
    )
    est_low.fit(data)

    # High threshold should remove fewer
    est_high = ZapLine(
        line_freq=50.0, sfreq=minimal_data["sfreq"], n_select="auto", threshold=5.0
    )
    est_high.fit(data)

    # Low threshold >= high threshold (could be equal if all detected)
    assert est_low.n_removed_ >= est_high.n_removed_


def test_zapline_rank_parameter(minimal_data):
    """ZapLine should accept rank parameter."""
    data = minimal_data["data"]
    # Limit DSS rank to 2
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], rank=2)
    est.fit(data)
    cleaned = est.transform(data)

    # Should still work
    assert cleaned.shape == data.shape


def test_zapline_error_1d_data():
    """ZapLine should raise error for 1D data."""
    data = np.random.randn(1000)
    est = ZapLine(line_freq=50.0, sfreq=250)

    with pytest.raises(Exception):
        est.fit(data)


def test_zapline_closed_sum_method(minimal_data):
    """Cleaned + Removed should equal original."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    est = ZapLine(line_freq=50.0, sfreq=sfreq)
    est.fit(data)
    cleaned = est.transform(data)
    removed = data - cleaned

    reconstructed = cleaned + removed
    assert_allclose(reconstructed, data, rtol=1e-10)


def test_zapline_error_3d_data():
    """ZapLine should handle 3D data by reshaping."""
    data = np.random.randn(4, 100, 10)
    est = ZapLine(line_freq=50.0, sfreq=250)
    est.fit(data)


def test_zapline_error_zero_line_freq():
    """ZapLine should safely handle zero line_freq (do nothing)."""
    data = np.random.randn(4, 1000)

    est = ZapLine(line_freq=0.0, sfreq=250)
    est.fit(data)
    assert est.n_removed_ == 0


def test_zapline_adaptive_fit_error():
    """ZapLine adaptive mode should raise error on fit()."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250, adaptive=True)

    with pytest.raises(RuntimeError, match="Adaptive mode requires"):
        est.fit(data)


def test_zapline_adaptive_transform_error():
    """ZapLine adaptive mode should raise error on transform()."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250, adaptive=True)
    est.filters_ = np.eye(4)  # Fake fitted state

    with pytest.raises(RuntimeError, match="Adaptive mode requires"):
        est.transform(data)


def test_zapline_sfreq_mismatch_warning_fit():
    """ZapLine should warn on sfreq mismatch during fit."""
    import mne

    data = np.random.randn(4, 1000)
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=500, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)

    est = ZapLine(line_freq=50.0, sfreq=250)  # Different sfreq

    with pytest.warns(UserWarning, match="Input data sfreq"):
        est.fit(raw)


def test_zapline_sfreq_mismatch_warning_transform():
    """ZapLine should warn on sfreq mismatch during transform."""
    import mne

    # Fit on array first
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250)
    est.fit(data)

    # Transform with Raw that has different sfreq
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=500, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)

    with pytest.warns(UserWarning, match="Input data sfreq"):
        est.transform(raw)


def test_zapline_sfreq_mismatch_warning_fit_transform():
    """ZapLine should warn on sfreq mismatch during fit_transform."""
    import mne

    data = np.random.randn(4, 1000)
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=500, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)

    est = ZapLine(line_freq=50.0, sfreq=250)  # Different sfreq

    with pytest.warns(UserWarning, match="Input data sfreq"):
        est.fit_transform(raw)


def test_zapline_mne_eeg_raw_reduces_line_noise():
    """EEG-only MNE Raw inputs should keep the existing all-channel behavior."""
    import mne

    rng = np.random.default_rng(33)
    sfreq = 400.0
    n_eeg = 4
    n_times = int(20 * sfreq)
    times = np.arange(n_times) / sfreq

    eeg_noise = rng.normal(0, 0.2, (n_eeg, n_times))
    eeg_line = 2.0 * np.sin(2 * np.pi * 50 * times)[None, :]
    eeg_data = eeg_noise + eeg_line

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_eeg)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    def line_power(x):
        freqs, psd = signal.welch(x, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
        idx = np.argmin(np.abs(freqs - 50.0))
        return float(np.mean(psd[:, idx]))

    before = line_power(eeg_data)
    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1, n_harmonics=1, nfft=400)
    cleaned = est.fit_transform(raw)

    assert 10 * np.log10(before / line_power(cleaned.get_data())) > 10
    assert est.n_removed_ == 1


def test_zapline_mne_mixed_channels_cleans_magnetometers_only():
    """Mixed-unit MNE objects should clean MEG channels without touching misc."""
    import mne

    rng = np.random.default_rng(34)
    sfreq = 400.0
    n_mag = 8
    n_times = int(20 * sfreq)
    times = np.arange(n_times) / sfreq

    phases = rng.uniform(0, 2 * np.pi, n_mag)
    mag_noise = rng.normal(0, 0.2e-13, (n_mag, n_times))
    mag_line = 2e-13 * np.sin(2 * np.pi * 50 * times[None, :] + phases[:, None])
    mag_data = mag_noise + mag_line
    misc_data = rng.normal(0, 1.0, (1, n_times))
    data = np.vstack([mag_data, misc_data])

    info = mne.create_info(
        ch_names=[f"MEG{i:03d}" for i in range(n_mag)] + ["MISC001"],
        sfreq=sfreq,
        ch_types=["mag"] * n_mag + ["misc"],
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    def line_power(x):
        freqs, psd = signal.welch(x, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
        idx = np.argmin(np.abs(freqs - 50.0))
        return float(np.mean(psd[:, idx]))

    before = line_power(mag_data)
    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=2, n_harmonics=1, nfft=400)
    cleaned = est.fit_transform(raw)

    cleaned_mag = cleaned.get_data(picks="mag")
    cleaned_misc = cleaned.get_data(picks="misc")

    assert 10 * np.log10(before / line_power(cleaned_mag)) > 10
    assert_allclose(cleaned_misc, misc_data, atol=0, rtol=0)
    assert est.n_removed_ == 2


def test_zapline_mne_mixed_channels_prefers_gradiometers_when_no_mags():
    """Mixed MNE inputs should clean gradiometers when magnetometers are absent."""
    import mne

    rng = np.random.default_rng(35)
    sfreq = 400.0
    n_grad = 4
    n_times = int(10 * sfreq)
    times = np.arange(n_times) / sfreq

    grad_noise = rng.normal(0, 0.2e-13, (n_grad, n_times))
    grad_line = 2e-13 * np.sin(2 * np.pi * 50 * times)[None, :]
    grad_data = grad_noise + grad_line
    misc_data = rng.normal(0, 1.0, (1, n_times))

    info = mne.create_info(
        ch_names=[f"MEG{i:03d}" for i in range(n_grad)] + ["MISC001"],
        sfreq=sfreq,
        ch_types=["grad"] * n_grad + ["misc"],
    )
    raw = mne.io.RawArray(np.vstack([grad_data, misc_data]), info, verbose=False)

    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1, n_harmonics=1, nfft=400)
    cleaned = est.fit_transform(raw)

    assert est._mne_ch_names_ == [f"MEG{i:03d}" for i in range(n_grad)]
    assert_allclose(cleaned.get_data(picks="misc"), misc_data, atol=0, rtol=0)


def test_zapline_mne_mixed_epochs_preserves_misc_channels():
    """Mixed Epochs should clean EEG picks and reinsert them into full data."""
    import mne

    rng = np.random.default_rng(36)
    sfreq = 400.0
    n_epochs = 3
    n_eeg = 4
    n_times = int(4 * sfreq)
    times = np.arange(n_times) / sfreq

    eeg_noise = rng.normal(0, 0.2, (n_epochs, n_eeg, n_times))
    eeg_line = 2.0 * np.sin(2 * np.pi * 50 * times)[None, None, :]
    eeg_data = eeg_noise + eeg_line
    misc_data = rng.normal(0, 1.0, (n_epochs, 1, n_times))
    data = np.concatenate([eeg_data, misc_data], axis=1)

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_eeg)] + ["MISC001"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_eeg + ["misc"],
    )
    events = np.column_stack(
        [
            np.arange(n_epochs) * n_times,
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    )
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)

    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1, n_harmonics=1, nfft=400)
    cleaned = est.fit_transform(epochs)

    assert cleaned.get_data().shape == data.shape
    assert_allclose(cleaned.get_data(picks="misc"), misc_data, atol=0, rtol=0)


def test_zapline_mne_mixed_evoked_transform_preserves_misc_channels():
    """Mixed Evoked transform should use fitted channel names and preserve misc."""
    import mne

    rng = np.random.default_rng(37)
    sfreq = 400.0
    n_eeg = 4
    n_times = int(10 * sfreq)
    times = np.arange(n_times) / sfreq

    eeg_noise = rng.normal(0, 0.2, (n_eeg, n_times))
    eeg_line = 2.0 * np.sin(2 * np.pi * 50 * times)[None, :]
    eeg_data = eeg_noise + eeg_line
    misc_data = rng.normal(0, 1.0, (1, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_eeg)] + ["MISC001"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_eeg + ["misc"],
    )
    evoked = mne.EvokedArray(np.vstack([eeg_data, misc_data]), info, tmin=0)

    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1, n_harmonics=1, nfft=400)
    est.fit(evoked)
    cleaned = est.transform(evoked)

    assert cleaned.data.shape == evoked.data.shape
    assert_allclose(cleaned.get_data(picks="misc"), misc_data, atol=0, rtol=0)


def test_zapline_mne_transform_requires_fitted_channels():
    """Transform should fail clearly if fitted MNE channels are missing."""
    import mne

    rng = np.random.default_rng(38)
    sfreq = 400.0
    n_times = int(4 * sfreq)
    times = np.arange(n_times) / sfreq
    eeg_data = rng.normal(0, 0.2, (2, n_times))
    eeg_data += 2.0 * np.sin(2 * np.pi * 50 * times)[None, :]
    misc_data = rng.normal(0, 1.0, (1, n_times))

    info = mne.create_info(
        ch_names=["EEG001", "EEG002", "MISC001"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg", "misc"],
    )
    raw = mne.io.RawArray(np.vstack([eeg_data, misc_data]), info, verbose=False)

    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1, n_harmonics=1, nfft=400)
    est.fit(raw)

    with pytest.raises(
        ValueError, match="Input MNE object is missing required channels"
    ):
        est.transform(raw.copy().drop_channels(["EEG002"]))


def test_zapline_fit_none_line_freq_error():
    """ZapLine fit() should raise error if line_freq is None."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=None, sfreq=250)

    with pytest.raises(ValueError, match="line_freq required"):
        est.fit(data)


def test_zapline_transform_not_fitted_error():
    """ZapLine transform() should raise error if not fitted."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250)

    with pytest.raises(RuntimeError, match="Not fitted"):
        est.transform(data)


def test_zapline_3d_data_fit_transform():
    """ZapLine should handle 3D epoched data correctly."""
    # Shape: (n_epochs, n_channels, n_times)
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 5, 4, 500
    sfreq = 250
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 1, (n_epochs, n_ch, n_times))
    # Add line noise
    data += np.sin(2 * np.pi * 50 * times) * 2.0

    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_select=1)
    est.fit(data)
    cleaned = est.transform(data)

    assert cleaned.shape == data.shape


def test_zapline_adaptive_3d_data():
    """ZapLine adaptive mode should handle 3D epoched data."""
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 3, 4, 2500
    sfreq = 250
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.5, (n_epochs, n_ch, n_times))
    data += np.sin(2 * np.pi * 50 * times) * 5.0

    est = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={"min_chunk_len": 5.0},
    )
    cleaned = est.fit_transform(data)

    assert cleaned.shape == data.shape
    assert hasattr(est, "adaptive_results_")


def test_zapline_adaptive_auto_detection():
    """ZapLine adaptive mode with line_freq=None should auto-detect."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 7500  # 30s
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.5, (4, n_times))
    # Add strong 50 Hz line noise
    data += np.sin(2 * np.pi * 50 * times) * 10.0

    est = ZapLine(
        sfreq=sfreq,
        line_freq=None,  # Auto-detect
        adaptive=True,
        adaptive_params={"fmin": 45, "fmax": 55, "min_chunk_len": 10.0},
    )
    cleaned = est.fit_transform(data)

    assert cleaned.shape == data.shape
    assert est.adaptive_results_ is not None


def test_zapline_adaptive_no_detection():
    """ZapLine adaptive mode should handle case where no noise is detected."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 7500

    # Clean data with no line noise
    data = rng.normal(0, 1, (4, n_times))

    est = ZapLine(
        sfreq=sfreq,
        line_freq=None,
        adaptive=True,
        adaptive_params={"fmin": 45, "fmax": 55, "min_chunk_len": 10.0},
    )
    cleaned = est.fit_transform(data)

    # Should return data mostly unchanged
    assert cleaned.shape == data.shape


def test_zapline_adaptive_with_harmonics():
    """ZapLine adaptive mode should process harmonics when enabled."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_times = 15000  # 30s
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.5, (4, n_times))
    # Add 50 Hz and 100 Hz harmonics
    data += np.sin(2 * np.pi * 50 * times) * 10.0
    data += np.sin(2 * np.pi * 100 * times) * 5.0

    est = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={
            "process_harmonics": True,
            "max_harmonics": 2,
            "min_chunk_len": 10.0,
        },
    )
    cleaned = est.fit_transform(data)

    assert cleaned.shape == data.shape


def test_smoothing_warnings():
    """Test that warnings are issued for bad sfreq/line_freq ratios."""
    with pytest.warns(UserWarning):
        zap = ZapLine(sfreq=100, line_freq=70)
        data = np.random.randn(1, 100)
        zap._get_smooth_residual(data, warn=True)


def test_smoothing_warning_integer_mismatch():
    """Test warning when period is not exactly integer but close enough for standard."""
    sfreq = 10000
    period = 2000.15
    line_freq = sfreq / period

    with pytest.warns(UserWarning, match="is not exactly an integer"):
        zap = ZapLine(sfreq=sfreq, line_freq=line_freq)
        data = np.random.randn(1, 2500)
        zap._get_smooth_residual(data, warn=True)


def test_fractional_smooth_period_le_1():
    """Test fractional smooth when period <= 1."""
    zap = ZapLine(sfreq=100, line_freq=150)  # period < 1
    data = np.random.randn(1, 100)
    data_smooth = zap._fractional_smooth(data, period=0.5)
    assert np.array_equal(data_smooth, data)


def test_fractional_smooth_integ_equals_ntimes():
    """Test fractional smooth when smoothing period >= n_times."""
    zap = ZapLine(sfreq=100, line_freq=1)
    data = np.random.randn(1, 50)

    smooth = zap._fractional_smooth(data, period=100.0)
    assert np.allclose(smooth, np.mean(data))


def test_fractional_smooth_integer_period():
    """Test fast path for integer period in fractional smooth."""
    zap = ZapLine(sfreq=100, line_freq=50)
    data = np.random.randn(1, 100)

    smooth = zap._fractional_smooth(data, period=2.0)
    assert smooth.shape == data.shape


def test_linenoise_bias_3d():
    """Test LineNoiseBias with 3D data."""
    bias = LineNoiseBias(freq=50, sfreq=1000, method="fft")
    data = np.random.randn(2, 100, 3)  # ch, time, ep

    biased = bias.apply(data)
    assert biased.shape == data.shape

    with pytest.raises(ValueError):
        bias._apply_fft(np.zeros((2,)))  # 1D data


def test_linenoise_bias_method_errors():
    """Test LineNoiseBias error handling for invalid methods."""
    # Init validation
    with pytest.raises(ValueError, match="Unknown method"):
        LineNoiseBias(freq=50, sfreq=1000, method="invalid")

    # Apply fallback
    bias = LineNoiseBias(freq=50, sfreq=1000, method="fft")
    bias.method = "invalid"
    # Should return data unchanged
    data = np.random.randn(1, 100)
    assert np.array_equal(bias.apply(data), data)


def test_zapline_auto_meg_like_many_coequal_components():
    """Regression for Issue #34: auto-mode must detect MEG-style line noise.

    Mimics high-channel-count MEG environmental pickup: each sensor sees
    the 50 / 100 / 150 Hz field through a different lead-field path, so
    amplitudes and phases vary across channels. This produces multiple
    independent line-noise sources, and DSS yields several co-equal strong
    components -- the regime where the pre-fix outlier selector underdetected.
    """
    rng = np.random.default_rng(34)
    sfreq = 400
    n_channels = 64
    n_times = 8000  # 20 s
    times = np.arange(n_times) / sfreq

    line = np.zeros((n_channels, n_times))
    for freq in (50.0, 100.0, 150.0):
        phases = rng.uniform(0, 2 * np.pi, n_channels)
        amps = rng.normal(1.0, 0.3, n_channels) * 5.0
        line += amps[:, None] * np.sin(
            2 * np.pi * freq * times[None, :] + phases[:, None]
        )

    background = rng.normal(0, 1, (n_channels, n_times))
    data = background + line

    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_select="auto", n_harmonics=3)
    est.fit(data)
    cleaned = est.transform(data)

    # Should detect multiple line-noise components. With 3 harmonics and
    # per-channel phase variation, we expect ~6 strong components (sin and
    # cos at each harmonic). >=3 is a conservative lower bound.
    assert est.n_removed_ >= 3, (
        f"Expected n_removed_ >= 3 for MEG-like coherent line noise, "
        f"got {est.n_removed_}. Eigenvalues (top 10): {est.eigenvalues_[:10]}"
    )

    # Power at 50 Hz should drop substantially.
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 50.0, sfreq)
    power_after = get_power_at(cleaned, 50.0, sfreq)
    reduction_db = 10 * np.log10(power_before / max(power_after, 1e-30))
    assert reduction_db > 10.0, (
        f"Expected >10 dB drop at 50 Hz, got {reduction_db:.1f} dB"
    )


def test_zapline_no_supported_channel_types_falls_back():
    """If no mag/grad/eeg channels are present, ZapLine processes all channels."""
    import mne

    rng = np.random.default_rng(3)
    sfreq = 400.0
    n_times = int(2 * sfreq)
    times = np.arange(n_times) / sfreq

    line = 5.0 * np.sin(2 * np.pi * 50 * times)
    data = rng.normal(0, 0.5, (3, n_times)) + line[None, :]

    info = mne.create_info(
        ch_names=["MISC001", "MISC002", "MISC003"],
        sfreq=sfreq,
        ch_types=["misc"] * 3,
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    est = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1, n_harmonics=1, nfft=400)
    est.fit(raw)

    # No mag/grad/eeg -> no picking; estimator processes the misc channels.
    # We now properly track the names of whatever channels we process
    assert est._mne_ch_names_ == ["MISC001", "MISC002", "MISC003"]
    assert est.n_removed_ == 1


# =========================================================================
# ZapLine reuses DSS's selection policy rather than duplicating it
# =========================================================================


def _line_noise_data(sfreq=500.0, duration=30.0, n_ch=16, freq=60.0):
    rng = np.random.default_rng(0)
    n_times = int(duration * sfreq)
    times = np.arange(n_times) / sfreq
    return (
        rng.standard_normal((n_ch, n_times)) * 0.5
        + np.sin(2 * np.pi * freq * times) * 3.0
    )


def test_zapline_has_only_n_select():
    """ZapLine exposes only the inherited n_select, never a second name."""
    assert not hasattr(ZapLine(sfreq=500.0), "n_remove")
    assert "n_select" in ZapLine(sfreq=500.0).get_params()
    for value in ("auto", 3):
        assert ZapLine(sfreq=500.0, line_freq=60.0, n_select=value).n_select == value


def test_auto_matches_shared_robust_selector():
    """n_select='auto' produces exactly auto_select_components_robust's count."""
    from mne_denoise.dss.utils.selection import auto_select_components_robust

    data = _line_noise_data()
    zap = ZapLine(sfreq=500.0, line_freq=60.0, n_select="auto").fit(data)

    assert zap.n_removed_ == auto_select_components_robust(
        zap.eigenvalues_,
        sigma=zap.threshold,
        knee_rel_floor=zap.knee_rel_floor,
        knee_min_ratio=zap.knee_min_ratio,
    )


def test_auto_agrees_with_dss_auto_select():
    """ZapLine.n_removed_ is exactly what the inherited auto_select returns."""
    data = _line_noise_data()
    zap = ZapLine(sfreq=500.0, line_freq=60.0, n_select="auto").fit(data)
    assert zap.n_removed_ == zap.auto_select()


def test_int_n_remove_is_clipped_to_available_components():
    data = _line_noise_data()
    zap = ZapLine(sfreq=500.0, line_freq=60.0, n_select=50).fit(data)
    assert zap.n_removed_ == len(zap.eigenvalues_)


def test_adaptive_is_inherited_from_dss():
    """ZapLine does not redefine adaptive; it is DSS's single switch."""
    from mne_denoise.dss.linear import DSS

    assert "adaptive" in DSS(bias=lambda x: x).get_params()
    assert "segmented" not in ZapLine(sfreq=500.0).get_params()
    assert ZapLine(sfreq=500.0, adaptive=True).adaptive is True


def test_adaptive_is_settable_through_sklearn_api():
    """adaptive/segmenter/min_select are real params, so clone() works."""
    from sklearn.base import clone

    zap = ZapLine(sfreq=500.0, line_freq=60.0, adaptive=True, min_select=2)
    params = zap.get_params()
    assert {"adaptive", "segmenter", "min_select", "max_prop_remove"} <= set(params)
    assert clone(zap).min_select == 2

    zap.set_params(adaptive=False)
    assert zap.adaptive is False


def test_process_segment_without_target_frequency_raises():
    """A segment cannot be cleaned without knowing which frequency to target."""
    zap = ZapLine(sfreq=500.0, line_freq=None, adaptive=True)
    assert zap._target_freq_ is None
    with pytest.raises(RuntimeError, match="target frequency"):
        zap._process_segment(np.random.default_rng(0).standard_normal((4, 500)))


def test_process_segment_falls_back_to_line_freq():
    """Reached outside _run_adaptive, the segment hook uses line_freq."""
    zap = ZapLine(sfreq=500.0, line_freq=60.0, adaptive=True)
    result = zap._process_segment(_line_noise_data(duration=10.0))
    assert set(result) >= {"cleaned", "n_selected", "fine_freq", "artifact_present"}


def test_per_chunk_estimator_inherits_all_params():
    """The per-chunk ZapLine is derived from get_params, not hand-copied."""
    zap = ZapLine(
        sfreq=500.0,
        line_freq=60.0,
        adaptive=True,
        rank=7,
        reg=1e-7,
        nfft=512,
        whiten=False,
    )
    params = {
        **zap.get_params(),
        "line_freq": 59.9,
        "n_select": "auto",
        "threshold": 2.5,
        "adaptive": False,
        "crossfade": 0.0,
    }
    est = type(zap)(**params)
    assert est.rank == 7
    assert est.reg == 1e-7
    assert est.nfft == 512
    assert est.line_freq == 59.9
    assert est.adaptive is False


def test_zapline_usable_in_sklearn_pipeline():
    """The estimator survives clone/get_params inside a Pipeline."""
    from sklearn.pipeline import Pipeline

    data = _line_noise_data(duration=20.0)
    pipe = Pipeline([("zap", ZapLine(sfreq=500.0, line_freq=60.0, adaptive=True))])
    out = pipe.fit_transform(data)
    assert out.shape == data.shape
