"""Tests for ZapLine's line-noise-specific behavior."""

from __future__ import annotations

from unittest.mock import patch

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from mne_denoise.zapline.core import ZapLine


@pytest.fixture
def line_noise_data():
    """Generate synthetic data with 50 Hz line noise and a 10 Hz signal."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_channels = 8
    n_times = 5000
    times = np.arange(n_times) / sfreq

    brain_signal = np.sin(2 * np.pi * 10 * times) * 0.5
    brain_signal = brain_signal[np.newaxis, :] + rng.normal(
        0, 0.2, (n_channels, n_times)
    )

    line_50hz = np.sin(2 * np.pi * 50 * times) * 10.0
    line_100hz = np.sin(2 * np.pi * 100 * times) * 3.0
    line_noise = line_50hz + line_100hz

    return {
        "data": brain_signal + line_noise[np.newaxis, :],
        "sfreq": sfreq,
        "line_freq": 50.0,
        "brain_signal": brain_signal,
        "times": times,
    }


@pytest.fixture
def minimal_data():
    """Generate small data with a single line-noise component."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_channels = 4
    n_times = 1000
    times = np.arange(n_times) / sfreq

    line_noise = np.sin(2 * np.pi * 50 * times) * 5.0
    data = rng.normal(0, 1, (n_channels, n_times))
    data[0] += line_noise

    return {"data": data, "sfreq": sfreq, "line_freq": 50.0, "times": times}


@pytest.fixture
def mixed_sensor_raw():
    """Create mixed-unit MNE data for the whitening path."""
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
    """Generate stochastic line noise distributed across a smooth spectrum."""
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
    return (mixing @ sources) * 3e-6, sfreq, line_freq


def _power_at(data, freq, sfreq):
    """Return mean Welch power at the closest frequency bin."""
    freqs, psd = signal.welch(
        data, fs=sfreq, nperseg=min(int(sfreq), data.shape[-1]), axis=-1
    )
    index = np.argmin(np.abs(freqs - freq))
    return float(np.mean(psd[..., index]))


def _make_mixed_meg_raw(sensor_type, n_channels, seed):
    """Create a mixed MEG/misc Raw for channel-picking policy tests."""
    rng = np.random.default_rng(seed)
    sfreq = 400.0
    n_times = int(20 * sfreq)
    times = np.arange(n_times) / sfreq
    scale = 1e-12 if sensor_type == "mag" else 1e-11
    phases = rng.uniform(0, 2 * np.pi, n_channels)
    sensor_data = rng.normal(0, 0.2 * scale, (n_channels, n_times))
    sensor_data += 2 * scale * np.sin(2 * np.pi * 50 * times[None, :] + phases[:, None])
    misc_data = rng.normal(0, 1.0, (1, n_times))
    info = mne.create_info(
        [f"MEG{i:03d}" for i in range(n_channels)] + ["MISC001"],
        sfreq,
        [sensor_type] * n_channels + ["misc"],
    )
    return mne.io.RawArray(np.vstack([sensor_data, misc_data]), info, verbose=False)


def test_zapline_suppresses_line_noise_and_preserves_non_target(line_noise_data):
    """Cleaning reduces target power while retaining a non-target rhythm."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = ZapLine(
        line_freq=50.0, sfreq=sfreq, n_select=1, n_harmonics=1
    ).fit_transform(data)

    power_before = _power_at(data, 50.0, sfreq)
    power_after = _power_at(cleaned, 50.0, sfreq)
    alpha_before = _power_at(data, 10.0, sfreq)
    alpha_after = _power_at(cleaned, 10.0, sfreq)

    assert power_after < power_before * 0.9
    assert alpha_after / alpha_before > 0.5


def test_zapline_reconstruction_is_exact(minimal_data):
    """The cleaned and removed parts reconstruct the input exactly."""
    data = minimal_data["data"]
    cleaned = ZapLine(
        line_freq=minimal_data["line_freq"],
        sfreq=minimal_data["sfreq"],
        n_select=1,
    ).fit_transform(data)
    removed = data - cleaned

    assert_allclose(cleaned + removed, data, rtol=1e-10)


def test_zapline_explicit_component_count_is_clipped_to_rank(minimal_data):
    """An explicit component count is honored and bounded by the fit rank."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    selected = ZapLine(line_freq=50.0, sfreq=sfreq, n_select=2).fit(data)
    clipped = ZapLine(line_freq=50.0, sfreq=sfreq, n_select=50).fit(data)

    assert selected.n_removed_ == 2
    assert clipped.n_removed_ == len(clipped.eigenvalues_)


def test_zapline_auto_fallback_removes_distributed_line_noise(
    smooth_spectrum_line_noise,
):
    """Auto-selection suppresses distributed stochastic line noise."""
    data, sfreq, line_freq = smooth_spectrum_line_noise
    estimator = ZapLine(sfreq=sfreq, line_freq=line_freq, n_select="auto").fit(data)
    cleaned = estimator.transform(data)

    power_before = _power_at(data, line_freq, sfreq)
    power_after = _power_at(cleaned, line_freq, sfreq)

    assert estimator.n_removed_ > 0
    assert power_after < power_before * 0.1


def test_zapline_auto_selection_leaves_clean_data_unchanged():
    """Automatic selection does not remove components from clean data."""
    rng = np.random.default_rng(0)
    sfreq = 250.0
    data = rng.standard_normal((12, int(sfreq * 60))) * 1e-6
    estimator = ZapLine(sfreq=sfreq, line_freq=50.0, n_select="auto").fit(data)

    assert estimator.n_removed_ == 0


def test_zapline_spectral_fallback_selects_first_acceptable_candidate():
    """The line-noise fallback stops at the first spectrally acceptable count."""
    estimator = ZapLine(sfreq=500.0, line_freq=50.0)
    data_smooth = np.zeros((3, 20))
    data_residual = np.ones((3, 20))

    with patch(
        "mne_denoise.zapline.core.check_spectral_qa",
        side_effect=["weak", "weak", "ok"],
    ) as check_qa:
        selected = estimator._find_min_components_for_line_suppression(
            data_smooth=data_smooth,
            data_residual=data_residual,
            full_filters=np.eye(3),
            full_mixing=np.eye(3),
        )

    assert selected == 3
    assert check_qa.call_count == 3


def test_zapline_spectral_fallback_uses_last_candidate_when_none_passes():
    """The line-noise fallback remains bounded when no candidate passes QA."""
    estimator = ZapLine(sfreq=500.0, line_freq=50.0)

    with (
        patch(
            "mne_denoise.zapline.core.check_spectral_qa",
            return_value="strong",
        ),
        pytest.warns(UserWarning, match="Line noise remains"),
    ):
        selected = estimator._find_min_components_for_line_suppression(
            data_smooth=np.zeros((2, 20)),
            data_residual=np.ones((2, 20)),
            full_filters=np.eye(2),
            full_mixing=np.eye(2),
        )

    assert selected == 2


def test_zapline_removes_requested_harmonics(line_noise_data):
    """The line-noise bias can target the fundamental and its harmonics."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    estimator = ZapLine(line_freq=50.0, sfreq=sfreq, n_harmonics=2, n_select=1).fit(
        data
    )
    cleaned = estimator.transform(data)

    assert estimator.n_harmonics_ == 2
    assert _power_at(cleaned, 100.0, sfreq) < _power_at(data, 100.0, sfreq)


def test_zapline_supports_requested_line_frequency():
    """The same line-noise operation works for a 60 Hz target."""
    rng = np.random.default_rng(42)
    sfreq = 250.0
    times = np.arange(2000) / sfreq
    data = rng.normal(0, 1, (4, len(times)))
    data += np.sin(2 * np.pi * 60 * times) * 5.0

    cleaned = ZapLine(
        line_freq=60.0, sfreq=sfreq, n_select=1, n_harmonics=1
    ).fit_transform(data)

    assert _power_at(cleaned, 60.0, sfreq) < _power_at(data, 60.0, sfreq) * 0.5


def test_zapline_requires_line_frequency_in_standard_mode():
    """Standard fitting requires a target frequency."""
    data = np.random.default_rng(0).standard_normal((4, 1000))

    with pytest.raises(ValueError, match="line_freq required"):
        ZapLine(line_freq=None, sfreq=250.0).fit(data)


def test_zapline_whiten_handles_mixed_sensor_types(mixed_sensor_raw):
    """Whitening permits one joint ZapLine fit across mixed sensor units."""
    raw = mixed_sensor_raw
    stim_before = raw.get_data(picks="stim")
    estimator = ZapLine(
        line_freq=50.0,
        sfreq=raw.info["sfreq"],
        n_select=1,
        whiten=True,
    )

    cleaned = estimator.fit_transform(raw)

    assert isinstance(cleaned, mne.io.BaseRaw)
    assert estimator.filters_.shape[1] == 6
    assert cleaned.ch_names == raw.ch_names
    np.testing.assert_array_equal(cleaned.get_data(picks="stim"), stim_before)


def test_zapline_mne_uses_magnetometers_then_gradiometers_when_needed():
    """MNE channel policy prefers mags and falls back to grads."""
    for sensor_type, n_channels, n_select, seed in (
        ("mag", 8, 2, 34),
        ("grad", 4, 1, 35),
    ):
        raw = _make_mixed_meg_raw(sensor_type, n_channels, seed)
        sensor_before = raw.get_data(picks=sensor_type)
        misc_before = raw.get_data(picks="misc")
        estimator = ZapLine(
            sfreq=raw.info["sfreq"],
            line_freq=50.0,
            n_select=n_select,
            n_harmonics=1,
            nfft=400,
        )
        cleaned = estimator.fit_transform(raw)

        expected_names = [f"MEG{i:03d}" for i in range(n_channels)]
        assert estimator._mne_ch_names_ == expected_names
        assert _power_at(cleaned.get_data(picks=sensor_type), 50.0, 400.0) < (
            _power_at(sensor_before, 50.0, 400.0) * 0.1
        )
        np.testing.assert_array_equal(cleaned.get_data(picks="misc"), misc_before)


def test_zapline_auto_meg_like_many_coequal_components():
    """Auto-selection suppresses line noise across many MEG-like components."""
    rng = np.random.default_rng(34)
    sfreq = 400.0
    n_channels = 64
    n_times = 8000
    times = np.arange(n_times) / sfreq

    line = np.zeros((n_channels, n_times))
    for freq in (50.0, 100.0, 150.0):
        phases = rng.uniform(0, 2 * np.pi, n_channels)
        amps = rng.normal(1.0, 0.3, n_channels) * 5.0
        line += amps[:, None] * np.sin(
            2 * np.pi * freq * times[None, :] + phases[:, None]
        )

    data = rng.normal(0, 1, (n_channels, n_times)) + line
    estimator = ZapLine(
        line_freq=50.0, sfreq=sfreq, n_select="auto", n_harmonics=3
    ).fit(data)
    cleaned = estimator.transform(data)

    assert estimator.n_removed_ >= 3
    reduction_db = 10 * np.log10(
        _power_at(data, 50.0, sfreq) / max(_power_at(cleaned, 50.0, sfreq), 1e-30)
    )
    assert reduction_db > 10.0


def test_zapline_exposes_only_n_select():
    """The public component-selection parameter is the inherited n_select."""
    estimator = ZapLine(sfreq=500.0)

    assert not hasattr(estimator, "n_remove")
    assert "n_select" in estimator.get_params()
    assert ZapLine(sfreq=500.0, line_freq=60.0, n_select="auto").n_select == "auto"
    assert ZapLine(sfreq=500.0, line_freq=60.0, n_select=3).n_select == 3
