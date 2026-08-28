from unittest.mock import patch

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from mne_denoise.zapline import ZapLine
from mne_denoise.zapline.adaptive import (
    apply_cleanline_notch,
    apply_hybrid_cleanup,
    check_artifact_presence,
    check_spectral_qa,
    detect_harmonics,
    find_fine_peak,
    find_noise_freqs,
)


@pytest.fixture
def line_noise_data():
    """Generate synthetic data with 50 Hz line noise."""
    rng = np.random.default_rng(42)
    sfreq = 250.0
    n_channels = 4
    n_times = 2500
    times = np.arange(n_times) / sfreq
    brain = rng.normal(0, 0.5, (n_channels, n_times))
    line_noise = np.sin(2 * np.pi * 50 * times) * 2.0
    return {
        "data": brain + line_noise[np.newaxis, :],
        "sfreq": sfreq,
        "line_freq": 50.0,
        "times": times,
    }


def _power_at(data, freq, sfreq):
    """Return mean Welch power at the closest frequency bin."""
    freqs, psd = signal.welch(
        data, fs=sfreq, nperseg=min(int(sfreq), data.shape[-1]), axis=-1
    )
    index = np.argmin(np.abs(freqs - freq))
    return float(np.mean(psd[..., index]))


def _nonstationary_line_data(sfreq=250.0, duration=60.0, n_ch=4):
    """Create line noise with a spatially changing amplitude."""
    rng = np.random.default_rng(0)
    n_times = int(duration * sfreq)
    times = np.arange(n_times) / sfreq
    data = rng.standard_normal((n_ch, n_times)) * 0.1
    amplitude = np.concatenate(
        [np.full(n_times // 2, 5.0), np.full(n_times - n_times // 2, 1.0)]
    )
    data += np.sin(2 * np.pi * 50.0 * times) * amplitude
    return data, sfreq


def _frequency_drift_data():
    """Create two strong, spatially distinct line frequencies and a quiet tail."""
    rng = np.random.default_rng(42)
    sfreq = 250.0
    n_times = int(120 * sfreq)
    times = np.arange(n_times) / sfreq
    first_end = n_times // 3
    second_end = 2 * n_times // 3
    data = rng.standard_normal((4, n_times)) * 0.05
    data[:2, :first_end] += 10 * np.sin(2 * np.pi * 50.0 * times[:first_end])
    data[2:, first_end:second_end] += 10 * np.sin(
        2 * np.pi * 50.04 * times[first_end:second_end]
    )
    return data, sfreq, first_end, second_end


def test_adaptive_detects_line_frequency_and_artifact():
    """Adaptive helpers identify a line peak but not an unrelated frequency."""
    rng = np.random.default_rng(42)
    sfreq = 1000.0
    times = np.arange(10000) / sfreq
    data = rng.standard_normal((4, len(times))) * 0.1
    data += 2.0 * np.sin(2 * np.pi * 50 * times)

    frequencies = find_noise_freqs(data, sfreq, fmin=45.0, fmax=55.0)

    assert frequencies
    assert np.isclose(frequencies[0], 50.0, atol=1.0)
    assert check_artifact_presence(data, sfreq, target_freq=50.0)
    assert not check_artifact_presence(data, sfreq, target_freq=60.0)


def test_adaptive_tracks_frequency_drift_and_preserves_quiet_tail():
    """Adaptive cleaning follows segment-specific targets and leaves quiet data usable."""
    data, sfreq, first_end, second_end = _frequency_drift_data()
    zap = ZapLine(
        sfreq=sfreq,
        line_freq=None,
        adaptive=True,
        adaptive_params={
            "fmin": 49.0,
            "fmax": 51.0,
            "min_chunk_len": 10.0,
            "n_remove_params": {"sigma": 3.0},
            "qa_params": {"max_sigma": 4.0},
        },
    )

    cleaned = zap.fit_transform(data)
    chunks = zap.adaptive_results_["chunk_info"]
    active = [chunk for chunk in chunks if chunk["artifact_present"]]
    early = [chunk for chunk in active if chunk["start"] < first_end]
    shifted = [chunk for chunk in active if first_end <= chunk["start"] < second_end]

    assert early and shifted
    assert np.isclose(
        np.median([chunk["fine_freq"] for chunk in early]), 50.0, atol=0.03
    )
    assert np.isclose(
        np.median([chunk["fine_freq"] for chunk in shifted]), 50.04, atol=0.04
    )
    assert (
        _power_at(cleaned[:, :first_end], 50.0, sfreq)
        < _power_at(data[:, :first_end], 50.0, sfreq) * 0.1
    )
    assert (
        _power_at(cleaned[:, first_end:second_end], 50.04, sfreq)
        < _power_at(data[:, first_end:second_end], 50.04, sfreq) * 0.1
    )
    assert np.isclose(
        np.var(cleaned[:, second_end:]), np.var(data[:, second_end:]), rtol=0.5
    )


def test_cleanline_notch_suppresses_target_and_preserves_other_signal(line_noise_data):
    """The notch fallback removes target power without damaging alpha power."""
    data = line_noise_data["data"].copy()
    sfreq = line_noise_data["sfreq"]
    data[0] += np.sin(2 * np.pi * 10.0 * line_noise_data["times"])

    cleaned = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0)

    assert _power_at(cleaned, 50.0, sfreq) < _power_at(data, 50.0, sfreq) * 0.1
    assert _power_at(cleaned[:1], 10.0, sfreq) / _power_at(data[:1], 10.0, sfreq) > 0.8


def test_cleanline_notch_handles_nyquist_and_invalid_bandwidth():
    """Notch construction remains bounded at Nyquist and invalid edges."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 1000))
    sfreq = 100.0

    near_nyquist = apply_cleanline_notch(data, sfreq=sfreq, freq=49.0, bandwidth=0.5)
    invalid_bandwidth = apply_cleanline_notch(
        data, sfreq=sfreq, freq=49.9, bandwidth=10.0
    )
    collapsed_band = apply_cleanline_notch(
        data, sfreq=sfreq, freq=49.99, bandwidth=0.001
    )

    assert near_nyquist.shape == data.shape
    assert np.isfinite(near_nyquist).all()
    assert invalid_bandwidth.shape == data.shape
    assert np.isfinite(invalid_bandwidth).all()
    assert_allclose(collapsed_band, data)


def test_hybrid_cleanup_suppresses_target(line_noise_data):
    """Hybrid cleanup reduces line power when a notch is safe to apply."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0)

    assert cleaned.shape == data.shape
    assert _power_at(cleaned, 50.0, sfreq) < _power_at(data, 50.0, sfreq)


def test_hybrid_cleanup_falls_back_for_short_or_overcleaned_data():
    """Hybrid cleanup retains data when QA cannot justify the fallback."""
    rng = np.random.default_rng(42)
    data = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000.0)[None, :]

    with patch("mne_denoise.zapline.adaptive.welch") as mock_welch:
        freqs = np.linspace(0, 100, 100)
        mock_welch.side_effect = [
            (freqs, np.ones((1, 100))),
            (freqs, np.full((1, 100), 1e-10)),
        ]
        cleaned = apply_hybrid_cleanup(data, sfreq=1000.0, freq=50.0)

    short = rng.standard_normal((4, 100))
    short_cleaned = apply_hybrid_cleanup(short, sfreq=100.0, freq=25.0)

    assert_allclose(cleaned, data)
    assert short_cleaned.shape == short.shape


def test_frequency_detection_handles_empty_and_small_windows():
    """Frequency detection handles empty search bands and insufficient windows."""
    rng = np.random.default_rng(42)
    empty_band = rng.standard_normal((4, 1024))
    small_window = rng.standard_normal((4, 1024))

    assert find_noise_freqs(empty_band, 250.0, fmin=200.0, fmax=250.0) == []
    assert isinstance(
        find_noise_freqs(
            small_window,
            256.0,
            fmin=50.0,
            fmax=51.0,
            window_length=0.1,
        ),
        list,
    )


def test_fine_peak_returns_coarse_frequency_outside_spectrum():
    """Fine target estimation falls back when the search band is unusable."""
    data = np.random.default_rng(42).standard_normal((4, 1000))

    assert find_fine_peak(data, 250.0, coarse_freq=200.0, search_width=0.05) == 200.0


def test_harmonic_detection_respects_presence_and_nyquist():
    """Harmonic detection finds real harmonics and stops before Nyquist."""
    rng = np.random.default_rng(42)
    sfreq = 500.0
    times = np.arange(5000) / sfreq
    data = rng.normal(0, 0.1, (4, len(times)))
    data += 10.0 * np.sin(2 * np.pi * 50 * times)
    data += 8.0 * np.sin(2 * np.pi * 100 * times)

    harmonics = detect_harmonics(data, sfreq, fundamental=50.0, max_harmonics=3)
    none = detect_harmonics(
        rng.normal(0, 1, (4, 5000)), sfreq, fundamental=50.0, max_harmonics=3
    )
    near_nyquist = detect_harmonics(
        rng.normal(0, 1, (4, 1024)),
        250.0,
        fundamental=60.0,
        max_harmonics=5,
    )

    assert any(np.isclose(harmonics, 100.0))
    assert none == []
    assert all(harmonic < 125.0 for harmonic in near_nyquist)


def test_spectral_qa_returns_ok_for_insufficient_data():
    """Spectral QA does not reject a segment with too little data."""
    data = np.random.default_rng(42).standard_normal((4, 50))

    assert check_spectral_qa(data, 100.0, target_freq=50.0) == "ok"


def test_adaptive_segments_cover_recording_and_report_line_targets():
    """Adaptive chunks tile the recording and expose ZapLine diagnostics."""
    data, sfreq = _nonstationary_line_data()
    zap = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={"min_chunk_len": 20.0},
    )

    cleaned = zap.fit_transform(data)
    results = zap.segment_results_
    assert len(results) > 1
    assert results[0]["start"] == 0
    assert results[-1]["end"] == data.shape[1]
    assert all(
        left["end"] == right["start"] for left, right in zip(results, results[1:])
    )
    assert all(
        "fine_freq" in result and "artifact_present" in result for result in results
    )
    assert all(48.0 <= result["fine_freq"] <= 52.0 for result in results)
    assert zap.adaptive_results_["line_freq"] == 50.0
    assert cleaned.shape == data.shape


def test_adaptive_progress_reports_one_frequency_event_for_many_segments():
    """A single target frequency emits one outer event over all segments."""
    data, sfreq = _nonstationary_line_data()
    zap = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={"min_chunk_len": 20.0},
    )
    events = []

    zap.fit_transform(data, callback=events.append)

    assert len(zap.segment_results_) > 1
    assert [(event.current, event.total, event.metric) for event in events] == [
        (1, 1, 50.0)
    ]


def test_adaptive_progress_reports_harmonic_frequency_events():
    """Frequency progress includes each ZapLine harmonic target."""
    sfreq = 500.0
    times = np.arange(5000) / sfreq
    data = np.random.default_rng(0).standard_normal((4, len(times))) * 0.1
    data += 3.0 * np.sin(2 * np.pi * 50.0 * times)
    data += 1.5 * np.sin(2 * np.pi * 100.0 * times)
    zap = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={
            "process_harmonics": True,
            "max_harmonics": 2,
            "min_chunk_len": 10.0,
        },
    )
    events = []

    with patch("mne_denoise.zapline.core.detect_harmonics", return_value=[100.0]):
        cleaned = zap.fit_transform(data, callback=events.append)

    assert [(event.current, event.total, event.metric) for event in events] == [
        (1, 2, 50.0),
        (2, 2, 100.0),
    ]
    assert _power_at(cleaned, 50.0, sfreq) < _power_at(data, 50.0, sfreq) * 0.5
    assert _power_at(cleaned, 100.0, sfreq) < _power_at(data, 100.0, sfreq) * 0.5


def test_adaptive_crossfade_smooths_segment_boundaries():
    """Crossfade reduces discontinuities between adaptive segment outputs."""
    data, sfreq = _nonstationary_line_data()
    params = {"min_chunk_len": 20.0}

    hard = ZapLine(
        sfreq=sfreq, line_freq=50.0, adaptive=True, adaptive_params=params
    ).fit_transform(data)
    faded = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        crossfade=1.0,
        adaptive_params=params,
    ).fit_transform(data)

    assert faded.shape == data.shape
    assert np.max(np.abs(np.diff(faded, axis=1))) <= np.max(
        np.abs(np.diff(hard, axis=1))
    )


def test_adaptive_target_frequency_resets_after_success_and_interruption():
    """The per-frequency marker is cleared after normal and interrupted runs."""
    data, sfreq = _nonstationary_line_data(duration=40.0)
    kwargs = {
        "sfreq": sfreq,
        "line_freq": 50.0,
        "adaptive": True,
        "adaptive_params": {"min_chunk_len": 10.0},
    }

    completed = ZapLine(**kwargs)
    completed.fit_transform(data)
    assert completed._target_freq_ is None

    interrupted = ZapLine(**kwargs)

    def callback(_event):
        raise RuntimeError("ZapLine callback failed")

    with pytest.raises(RuntimeError):
        interrupted.fit_transform(data, callback=callback)
    assert interrupted._target_freq_ is None


def test_adaptive_warns_on_sfreq_mismatch():
    """Adaptive MNE input warns when its sampling rate differs from init."""
    sfreq_init, sfreq_data = 500.0, 250.0
    n_times = int(20 * sfreq_data)
    times = np.arange(n_times) / sfreq_data
    data = np.random.default_rng(0).standard_normal((4, n_times)) * 0.3
    data += np.sin(2 * np.pi * 50.0 * times) * 3.0
    raw = mne.io.RawArray(data, mne.create_info(4, sfreq_data, "eeg"), verbose=False)
    zap = ZapLine(
        sfreq=sfreq_init,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={"min_chunk_len": 5.0},
    )

    with pytest.warns(UserWarning, match="differs from init sfreq"):
        zap.fit_transform(raw)


def test_adaptive_no_detected_frequencies_is_passthrough():
    """When no line frequency is detected, adaptive mode is a no-op."""
    data = np.random.default_rng(0).standard_normal((4, 5000)) * 0.1
    zap = ZapLine(sfreq=250.0, line_freq=None, adaptive=True)

    with patch("mne_denoise.zapline.core.find_noise_freqs", return_value=[]):
        cleaned = zap.fit_transform(data)

    assert_allclose(cleaned, data)
    assert zap.n_removed_ == 0


def _mixed_sensor_line_raw(sfreq=250.0, duration=30.0, seed=0):
    """Raw with mixed sensor units and shared line noise."""
    rng = np.random.default_rng(seed)
    n_times = int(duration * sfreq)
    times = np.arange(n_times) / sfreq
    ch_types = ["mag"] * 4 + ["grad"] * 4 + ["eeg"] * 4
    scales = np.array([1e-12] * 4 + [1e-11] * 4 + [1e-5] * 4)
    mixing = rng.standard_normal(len(ch_types))
    line = np.sin(2 * np.pi * 50.0 * times)
    data = (
        np.outer(mixing, line) * 3.0 + rng.standard_normal((len(ch_types), n_times))
    ) * scales[:, None]
    info = mne.create_info([f"C{i}" for i in range(len(ch_types))], sfreq, ch_types)
    return mne.io.RawArray(data, info, verbose=False)


def test_adaptive_whitening_returns_sensor_space():
    """Adaptive whitening cleans mixed units and reconstructs sensor units."""
    raw = _mixed_sensor_line_raw()
    zap = ZapLine(
        sfreq=raw.info["sfreq"],
        line_freq=50.0,
        adaptive=True,
        whiten=True,
        adaptive_params={"min_chunk_len": 10.0},
    )

    cleaned = zap.fit_transform(raw)
    data = raw.get_data()

    assert cleaned.get_data().shape == data.shape
    for picks in (slice(0, 4), slice(4, 8), slice(8, 12)):
        ratio = np.std(cleaned.get_data()[picks]) / np.std(data[picks])
        assert 0.1 < ratio < 10.0
    assert zap.adaptive_results_["removed"].shape == data.shape
