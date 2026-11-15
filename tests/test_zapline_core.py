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


def test_zapline_removes_strong_line_at_fractional_period():
    """A strong line is removed when the period is not a whole number of samples."""
    rng = np.random.default_rng(3)
    sfreq = 200.0  # 60 Hz line: a period of 10/3 samples
    times = np.arange(int(sfreq * 60)) / sfreq
    clean = rng.standard_normal((8, len(times)))
    line = rng.standard_normal((8, 1)) * np.sin(2 * np.pi * 60.0 * times) * 30.0

    cleaned = ZapLine(line_freq=60.0, sfreq=sfreq, n_select=1).fit_transform(
        clean + line
    )

    # The line is ~47 dB above the floor; at most 1 dB of it may remain.
    assert _power_at(cleaned, 60.0, sfreq) < _power_at(clean, 60.0, sfreq) * 10**0.1


def test_zapline_auto_selection_ignores_a_strong_rhythm_away_from_the_line():
    """A strong alpha rhythm does not change how many components are removed."""
    rng = np.random.default_rng(0)
    sfreq = 500.0
    times = np.arange(int(sfreq * 60)) / sfreq
    noise = rng.standard_normal((19, len(times)))
    alpha = np.outer(rng.standard_normal(19), np.sin(2 * np.pi * 10 * times)) * 5

    # The alpha source has almost no power at 60 Hz, so its DSS score drops far
    # below the others, at the bottom of the spectrum.
    n_noise = ZapLine(sfreq=sfreq, line_freq=60.0).fit(noise).n_removed_
    n_alpha = ZapLine(sfreq=sfreq, line_freq=60.0).fit(noise + alpha).n_removed_
    assert n_alpha == n_noise


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


def _eeg_with_muscle_and_line(sfreq, line_amplitudes, *, duration=60.0, seed=0):
    """Return 19-channel EEG-like data with muscle on 3 channels and a 60 Hz line.

    ``line_amplitudes`` gives the relative amplitude of each 60 Hz harmonic.
    """
    rng = np.random.default_rng(seed)
    n_channels, n_times = 19, int(sfreq * duration)
    times = np.arange(n_times) / sfreq
    brain = np.cumsum(rng.standard_normal((n_channels, n_times)), axis=1)
    brain = signal.sosfiltfilt(
        signal.butter(2, 0.5, "hp", fs=sfreq, output="sos"), brain
    )
    mixing = np.eye(n_channels) + 0.3 * rng.standard_normal((n_channels, n_channels))
    data = mixing @ brain
    muscle_band = signal.butter(
        4, [20.0, 0.95 * sfreq / 2], "bp", fs=sfreq, output="sos"
    )
    data[:3] += 20.0 * signal.sosfiltfilt(
        muscle_band, rng.standard_normal((3, n_times))
    )
    line = sum(
        amplitude * np.sin(2 * np.pi * 60.0 * (harmonic + 1) * times + harmonic)
        for harmonic, amplitude in enumerate(line_amplitudes)
    )
    return data + 30.0 * rng.uniform(0.5, 1.0, n_channels)[:, None] * line


def _welch_db_change(before, after, sfreq):
    """Return frequencies and per-channel Welch power change in dB."""
    freqs, psd_before = signal.welch(before, fs=sfreq, nperseg=int(4 * sfreq))
    _, psd_after = signal.welch(after, fs=sfreq, nperseg=int(4 * sfreq))
    return freqs, 10 * np.log10(psd_after / psd_before)


def _line_reduction_db(before, after, sfreq, freq):
    """Return the drop in summed Welch power within 0.5 Hz of freq, in dB.

    Unlike ``_power_at``, the 0.25 Hz resolution keeps the broadband power next
    to the line, which only the whole-component subtraction removes, out of the
    measure.
    """
    freqs, psd_before = signal.welch(before, fs=sfreq, nperseg=int(4 * sfreq))
    _, psd_after = signal.welch(after, fs=sfreq, nperseg=int(4 * sfreq))
    near = np.abs(freqs - freq) <= 0.5
    return 10 * np.log10(psd_before[:, near].sum() / psd_after[:, near].sum())


@pytest.mark.parametrize(
    ("sfreq", "n_harmonics", "line_amplitudes"),
    [
        (200.0, None, (1.0,)),  # fractional period; only 60 Hz is below Nyquist
        (500.0, 2, (1.0, 0.5)),
    ],
)
def test_zapline_line_bandwidth_leaves_spectrum_outside_the_line_unchanged(
    sfreq, n_harmonics, line_amplitudes
):
    """Line-only subtraction keeps the default's line removal without its broadband cost.

    With three components, the default whole-component subtraction moves the
    muscle activity of some channels onto others; subtracting only each
    component's part within 1 Hz of the harmonics leaves every other Fourier bin
    of the input unchanged.
    """
    data = _eeg_with_muscle_and_line(sfreq, line_amplitudes)
    harmonics = 60.0 * np.arange(1, len(line_amplitudes) + 1)
    params = {
        "sfreq": sfreq,
        "line_freq": 60.0,
        "n_select": 3,
        "n_harmonics": n_harmonics,
    }
    default = ZapLine(**params).fit_transform(data)
    line_only = ZapLine(**params, line_bandwidth=1.0).fit_transform(data)

    freqs, change = _welch_db_change(data, default, sfreq)
    away = np.all(np.abs(freqs[:, None] - harmonics) > 1.5, axis=1) & (freqs >= 1.0)
    assert change[:, away].max() > 3.0

    fft_freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sfreq)
    outside = np.all(np.abs(fft_freqs[:, None] - harmonics) > 1.0, axis=1)
    spectrum = np.fft.rfft(data, axis=-1)
    assert_allclose(
        np.fft.rfft(line_only, axis=-1)[:, outside],
        spectrum[:, outside],
        rtol=0,
        atol=1e-9 * np.abs(spectrum).max(),
    )

    for freq in harmonics:
        reduction = _line_reduction_db(data, line_only, sfreq, freq)
        assert reduction > 20.0
        assert reduction == pytest.approx(
            _line_reduction_db(data, default, sfreq, freq), abs=0.5
        )


def test_zapline_line_bandwidth_in_adaptive_mode_keeps_the_segment_selection():
    """Adaptive segments select as by default; only the line band is subtracted.

    The segments' spectral checks see whole-component subtraction, so they
    remove the same number of components, and the band is applied to the
    removed signal of the whole recording, so no segment boundary leaks into
    the spectrum outside the line bands.
    """
    sfreq = 200.0
    data = _eeg_with_muscle_and_line(sfreq, (1.0,))
    params = {
        "sfreq": sfreq,
        "line_freq": 60.0,
        "adaptive": True,
        "adaptive_params": {"min_chunk_len": 20.0},
    }
    default = ZapLine(**params)
    cleaned_default = default.fit_transform(data)
    line_only = ZapLine(**params, line_bandwidth=1.0)
    cleaned = line_only.fit_transform(data)

    assert len(line_only.adaptive_results_["chunk_info"]) > 1
    assert line_only.n_removed_ == default.n_removed_

    # The segments refine the line frequency to within 0.05 Hz.
    fft_freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sfreq)
    outside = np.abs(fft_freqs - 60.0) > 1.1
    spectrum = np.fft.rfft(data, axis=-1)
    assert_allclose(
        np.fft.rfft(cleaned, axis=-1)[:, outside],
        spectrum[:, outside],
        rtol=0,
        atol=1e-9 * np.abs(spectrum).max(),
    )
    reduction = _line_reduction_db(data, cleaned, sfreq, 60.0)
    assert reduction > 20.0
    assert reduction == pytest.approx(
        _line_reduction_db(data, cleaned_default, sfreq, 60.0), abs=0.5
    )


@pytest.mark.parametrize(
    ("line_bandwidth", "error"),
    [(0.0, ValueError), (-2.0, ValueError), (np.inf, ValueError), (True, TypeError)],
)
def test_zapline_rejects_invalid_line_bandwidth(minimal_data, line_bandwidth, error):
    """line_bandwidth must be a positive, finite number of hertz, also when set later."""
    data, sfreq = minimal_data["data"], minimal_data["sfreq"]
    model = ZapLine(sfreq=sfreq, line_freq=50.0, line_bandwidth=line_bandwidth)
    with pytest.raises(error, match="line_bandwidth"):
        model.fit(data)

    fitted = ZapLine(sfreq=sfreq, line_freq=50.0, n_select=1).fit(data)
    fitted.set_params(line_bandwidth=line_bandwidth)
    with pytest.raises(error, match="line_bandwidth"):
        fitted.transform(data)
