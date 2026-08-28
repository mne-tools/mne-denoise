"""Scientific and method-specific contracts for Basic SSA."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise.ssa import (
    SingularSpectrumAnalysis,
    compute_basic_ssa,
    ssa_clean_channel,
    ssa_decompose,
    ssa_w_correlation,
)

from ._utils import band_power


def test_ssa_reconstruction_uses_hankel_trajectory_and_is_deterministic():
    """Hankel SSA has a deterministic, additive elementary reconstruction."""
    sfreq = 100.0
    times = np.arange(31) / sfreq
    signal = np.sin(2 * np.pi * 3.0 * times) + 0.25 * np.cos(2 * np.pi * 7.0 * times)

    first, first_info = ssa_decompose(signal, window_length=16)
    second, second_info = ssa_decompose(signal, window_length=16)

    assert first.shape == (16, signal.size)
    assert first_info["trajectory_shape"] == (16, 16)
    assert first_info["window_length"] == 16
    assert np.all(np.diff(first_info["singular_values"]) <= 0)
    np.testing.assert_allclose(first.sum(axis=0), signal, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        first_info["singular_values"], second_info["singular_values"]
    )


def test_constant_decomposition_and_w_correlation():
    """A constant series is rank one and has finite weighted diagnostics."""
    signal = np.full(20, 3.0)
    components, info = ssa_decompose(signal, 5)

    assert info["rank"] == 1
    np.testing.assert_allclose(components[0], signal, atol=1e-13)
    np.testing.assert_allclose(components[1:], 0.0, atol=1e-13)

    correlation = ssa_w_correlation(
        np.vstack((signal, np.zeros((4, signal.size)))),
        5,
    )
    assert correlation.shape == (5, 5)
    assert correlation[0, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(correlation[1:], 0.0)
    assert np.isfinite(correlation).all()


def test_sample_and_second_windows_are_equivalent():
    """Equivalent sample- and duration-based embeddings give the same result."""
    sfreq = 100.0
    times = np.arange(80) / sfreq
    signal = np.sin(2 * np.pi * 2.0 * times) + 0.2 * np.sin(2 * np.pi * 11.0 * times)

    by_sample, sample_info = ssa_decompose(signal, 20, sfreq=sfreq)
    by_time, time_info = ssa_decompose(signal, window_seconds=0.2, sfreq=sfreq)

    np.testing.assert_allclose(by_sample, by_time)
    assert sample_info["window_length"] == time_info["window_length"] == 20
    assert sample_info["trajectory_shape"] == time_info["trajectory_shape"]


def test_ssa_clean_channel_removes_drift_and_reports_additive_artifact(drift_data):
    """Frequency grouping removes drift, preserves alpha, and is additive."""
    data, sfreq = drift_data
    signal = data[0]
    cleaned, info = ssa_clean_channel(
        signal,
        sfreq,
        window_length=40,
        drop_freq_max=3.0,
        return_info=True,
    )

    assert cleaned.shape == signal.shape
    np.testing.assert_allclose(cleaned + info["artifact"], signal)
    np.testing.assert_allclose(info["components"].sum(axis=0), signal)
    assert np.all(info["dropped_frequencies"] <= 3.0)

    low_before = band_power(signal, sfreq, 0.0, 3.0)
    low_after = band_power(cleaned, sfreq, 0.0, 3.0)
    alpha_before = band_power(signal, sfreq, 8.0, 12.0)
    alpha_after = band_power(cleaned, sfreq, 8.0, 12.0)
    assert low_after < 0.5 * low_before
    assert alpha_after > 0.7 * alpha_before


def test_compute_basic_ssa_known_frequency_drives_selection_and_diagnostics():
    """Dominant-frequency grouping exposes the expected counts and frequencies."""
    sfreq = 100.0
    times = np.arange(500) / sfreq
    data = np.vstack(
        (
            np.sin(2 * np.pi * 1.0 * times),
            np.sin(2 * np.pi * 10.0 * times),
        )
    )

    cleaned, info = compute_basic_ssa(
        data,
        sfreq,
        window_length=50,
        drop_freq_max=2.0,
        n_check=2,
    )

    assert cleaned.shape == data.shape
    np.testing.assert_array_equal(info["dropped_counts"], [2, 0])
    np.testing.assert_allclose(info["dropped_frequencies"][0], [1.0, 1.0])
    assert info["dropped_frequencies"][1].size == 0
    np.testing.assert_allclose(info["dominant_frequencies"][0][:2], [1.0, 1.0])
    np.testing.assert_allclose(info["dominant_frequencies"][1][:2], [10.0, 10.0])
    np.testing.assert_allclose(cleaned[0], 0.0, atol=1e-12)
    np.testing.assert_array_equal(cleaned[1], data[1])


def test_compute_basic_ssa_drop_band_targets_a_non_default_band():
    """An explicit band removes its rhythm while preserving alpha activity."""
    sfreq = 250.0
    times = np.arange(2000) / sfreq
    cardiac = np.sin(2 * np.pi * 1.2 * times)
    alpha = np.sin(2 * np.pi * 10.0 * times)
    data = (3.0 * cardiac + alpha)[None, :]

    cleaned, info = compute_basic_ssa(data, sfreq, drop_band=(0.8, 1.6))

    assert info["dropped_counts"].shape == (1,)
    assert info["dropped_frequencies"][0].size >= 1
    assert np.all(info["dropped_frequencies"][0] >= 0.8)
    assert np.all(info["dropped_frequencies"][0] <= 1.6)
    cardiac_before = band_power(data, sfreq, 0.8, 1.6)
    cardiac_after = band_power(cleaned, sfreq, 0.8, 1.6)
    alpha_before = band_power(data, sfreq, 8.0, 12.0)
    alpha_after = band_power(cleaned, sfreq, 8.0, 12.0)
    assert cardiac_after < 0.5 * cardiac_before
    assert alpha_after > 0.7 * alpha_before


def test_compute_basic_ssa_progress_reports_channel_drop_counts():
    """The functional callback reports one channel event with its drop metric."""
    sfreq = 100.0
    times = np.arange(500) / sfreq
    data = np.vstack(
        (
            np.sin(2 * np.pi * 1.0 * times),
            np.sin(2 * np.pi * 10.0 * times),
        )
    )
    events = []

    cleaned, info = compute_basic_ssa(
        data,
        sfreq,
        window_length=50,
        drop_freq_max=2.0,
        n_check=2,
        callback=events.append,
    )

    assert cleaned.shape == data.shape
    assert len(events) == data.shape[0]
    assert [event.method for event in events] == ["basic_ssa"] * data.shape[0]
    assert [event.stage for event in events] == ["channel"] * data.shape[0]
    assert [event.current for event in events] == [1, 2]
    assert [event.total for event in events] == [2, 2]
    assert all(event.component is None for event in events)
    np.testing.assert_array_equal(
        [event.metric for event in events], info["dropped_counts"].astype(float)
    )


def test_ssa_estimator_uses_fitted_operating_point_and_exposes_diagnostics(
    drift_data,
):
    """The estimator records its fitted frequency and scientific diagnostics."""
    data, sfreq = drift_data
    with pytest.raises(ValueError, match="sfreq is required"):
        SingularSpectrumAnalysis(window_length=40).fit(data[:2, :500])

    estimator = SingularSpectrumAnalysis(
        sfreq,
        window_length=40,
        drop_freq_max=3.0,
        n_check=5,
    )
    cleaned = estimator.fit_transform(data[:2, :500])

    assert cleaned.shape == (2, 500)
    assert estimator.sfreq_ == sfreq
    assert estimator.n_channels_in_ == 2
    assert estimator.diagnostics_["window_length"] == 40
    assert len(estimator.diagnostics_["singular_values"][0]) == 40
    assert estimator.diagnostics_["frequency_resolution"] == pytest.approx(sfreq / 500)
    assert estimator.dropped_counts_.shape == (2,)
    assert len(estimator.dropped_frequencies_) == 2
    assert [
        len(frequencies) for frequencies in estimator.dropped_frequencies_
    ] == estimator.dropped_counts_.tolist()


def test_ssa_estimator_distinguishes_continuous_and_epoch_progress_topology(
    drift_data,
):
    """Continuous records use channel events; epochs use epoch events."""
    data, sfreq = drift_data
    data = data[:2, :240]
    continuous_events = []
    continuous_estimator = SingularSpectrumAnalysis(
        sfreq=sfreq,
        window_length=20,
        drop_freq_max=3.0,
        n_check=5,
    )
    continuous = continuous_estimator.fit_transform(
        data, callback=continuous_events.append
    )

    assert len(continuous_events) == data.shape[0]
    assert all(event.method == "basic_ssa" for event in continuous_events)
    assert [event.stage for event in continuous_events] == ["channel", "channel"]
    assert [event.current for event in continuous_events] == [1, 2]
    assert all(event.total == 2 for event in continuous_events)
    assert all(event.component is None for event in continuous_events)
    np.testing.assert_array_equal(
        [event.metric for event in continuous_events],
        continuous_estimator.dropped_counts_.astype(float),
    )

    epochs = data.reshape(2, 2, 120).transpose(1, 0, 2)
    epoch_events = []
    epoch_estimator = SingularSpectrumAnalysis(
        sfreq=sfreq,
        window_length=20,
        drop_freq_max=3.0,
        n_check=5,
    )
    epoched = epoch_estimator.fit_transform(epochs, callback=epoch_events.append)

    assert epoched.shape == epochs.shape
    assert len(epoch_events) == epochs.shape[0]
    assert all(event.method == "basic_ssa" for event in epoch_events)
    assert [event.stage for event in epoch_events] == ["epoch", "epoch"]
    assert [event.current for event in epoch_events] == [1, 2]
    assert all(event.total == 2 for event in epoch_events)
    assert all(event.component is None for event in epoch_events)
    assert all(event.metric is None for event in epoch_events)
    assert epoch_estimator.dropped_counts_.shape == (2, 2)
    assert len(epoch_estimator.dropped_frequencies_) == 2
    assert all(
        len(frequencies) == 2 for frequencies in epoch_estimator.dropped_frequencies_
    )

    split = epoched.transpose(1, 0, 2).reshape(data.shape)
    assert not np.allclose(continuous, split)


def test_ssa_mne_sfreq_inference_controls_frequency_grouping(drift_data):
    """MNE metadata supplies the sampling rate and conflicting values fail."""
    mne = pytest.importorskip("mne")
    data, sfreq = drift_data
    info = mne.create_info(["EEG0", "EEG1"], sfreq, "eeg")
    raw = mne.io.RawArray(data[:2, :500], info, verbose=False)

    estimator = SingularSpectrumAnalysis(window_length=40, drop_freq_max=3.0)
    cleaned = estimator.fit_transform(raw)

    assert estimator.sfreq_ == sfreq
    assert np.all(
        [
            frequency <= 3.0
            for frequencies in estimator.dropped_frequencies_
            for frequency in frequencies
        ]
    )
    low_before = band_power(data[:2, :500], sfreq, 0.0, 3.0)
    low_after = band_power(cleaned.get_data(), sfreq, 0.0, 3.0)
    assert low_after < low_before

    with pytest.raises(ValueError, match="disagrees"):
        SingularSpectrumAnalysis(sfreq=sfreq / 2, window_length=40).fit(raw)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sfreq": 250.0, "window_length": 1}, "window_length"),
        ({"sfreq": 250.0, "window_length": 101}, "window_length"),
        ({"sfreq": 250.0, "max_window": 1}, "max_window"),
        ({"sfreq": 250.0, "n_check": 0}, "n_check"),
        ({"sfreq": 250.0, "drop_freq_max": 126}, "Nyquist"),
        ({"sfreq": 250.0, "drop_band": (2.0, 1.0)}, "drop_band"),
    ],
    ids=[
        "window-lower",
        "window-upper",
        "max-window",
        "n-check",
        "frequency",
        "band",
    ],
)
def test_ssa_rejects_invalid_operating_points(kwargs, match):
    """Distinct embedding, frequency, and selection constraints fail clearly."""
    with pytest.raises(ValueError, match=match):
        SingularSpectrumAnalysis(**kwargs).fit(np.ones((2, 200)))


def test_ssa_handles_dc_and_zero_energy_components():
    """DC can be selected explicitly while zero-energy components stay absent."""
    constant = np.full(200, 3.0)
    cleaned, constant_info = ssa_clean_channel(
        constant,
        100.0,
        window_length=50,
        drop_freq_max=0.0,
        n_check=1,
        return_info=True,
    )
    assert np.linalg.norm(cleaned) < 1e-10
    np.testing.assert_array_equal(constant_info["dropped_frequencies"], [0.0])

    zero, zero_info = ssa_clean_channel(
        np.zeros(100),
        100.0,
        window_length=20,
        drop_freq_max=3.0,
        n_check=5,
        return_info=True,
    )
    np.testing.assert_array_equal(zero, 0.0)
    assert zero_info["rank"] == 0
    assert zero_info["dropped_frequencies"].size == 0


def test_ssa_verbose_reports_one_aggregate_summary(drift_data, caplog):
    """Opt-in logging exposes one useful Basic SSA run summary."""
    data, sfreq = drift_data
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        SingularSpectrumAnalysis(
            sfreq=sfreq,
            window_length=40,
            verbose=True,
        ).fit_transform(data[:2, :500])

    summaries = [
        record for record in caplog.records if record.message.startswith("Basic SSA:")
    ]
    assert len(summaries) == 1
    for token in ("window=", "channels=", "dropped="):
        assert token in summaries[0].message
