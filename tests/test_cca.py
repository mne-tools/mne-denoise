"""Tests for explicit-lag reference-free CCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mne_denoise.cca import LaggedCCA, compute_lagged_cca
from mne_denoise.cca.core import _lagged_pairs


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


@pytest.fixture()
def muscle_data(rng):
    """Synthetic narrowband neural sources plus broadband muscle sources."""
    from scipy.signal import butter, filtfilt

    sfreq = 250.0
    n_times = 2000
    n_channels = 12
    times = np.arange(n_times) / sfreq
    neural = np.vstack(
        [
            np.sin(2 * np.pi * freq * times + rng.uniform(0, 2 * np.pi))
            for freq in (9.0, 10.0, 11.0)
        ]
    )
    b, a = butter(4, 30.0 / (0.5 * sfreq), btype="high")
    muscle = filtfilt(b, a, rng.standard_normal((3, n_times)), axis=-1)
    clean = rng.standard_normal((n_channels, 3)) @ neural
    artifact = 1.5 * rng.standard_normal((n_channels, 3)) @ muscle
    contaminated = clean + artifact + 0.05 * rng.standard_normal(clean.shape)
    return contaminated, clean, sfreq


def _band_power(X, sfreq, fmin, fmax):
    spectrum = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(X.shape[-1], 1.0 / sfreq)
    mask = (frequencies >= fmin) & (frequencies < fmax)
    return float(spectrum[..., mask].sum())


def test_function_requires_one_explicit_lag(rng):
    X = rng.standard_normal((4, 100))
    with pytest.raises(ValueError, match="exactly one"):
        compute_lagged_cca(X)
    with pytest.raises(ValueError, match="exactly one"):
        compute_lagged_cca(X, lag_samples=1, lag_seconds=0.01, sfreq=100)


def test_sample_and_physical_lag_are_equivalent(rng):
    X = rng.standard_normal((6, 500))
    by_sample, sample_info = compute_lagged_cca(X, lag_samples=2, n_keep=3)
    by_time, time_info = compute_lagged_cca(X, lag_seconds=0.008, sfreq=250.0, n_keep=3)
    np.testing.assert_allclose(by_sample, by_time)
    assert sample_info["lag_samples"] == time_info["lag_samples"] == 2
    assert time_info["lag_seconds"] == pytest.approx(0.008)


def test_lagged_pairs_never_wrap(rng):
    X = rng.standard_normal((3, 20))
    current, past = _lagged_pairs(X, 4)
    np.testing.assert_array_equal(current, X[:, 4:].T)
    np.testing.assert_array_equal(past, X[:, :-4].T)


def test_epoched_pairs_do_not_cross_boundaries():
    epochs = np.stack(
        [
            np.tile(np.arange(6, dtype=float), (2, 1)),
            np.tile(np.arange(100, 106, dtype=float), (2, 1)),
        ]
    )
    current, past = _lagged_pairs(epochs, 1)
    assert current.shape == past.shape == (10, 2)
    assert not np.any((past[:, 0] == 5) & (current[:, 0] == 100))
    np.testing.assert_array_equal(current[:5, 0], np.arange(1, 6))
    np.testing.assert_array_equal(current[5:, 0], np.arange(101, 106))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lag_samples": 0}, "positive"),
        ({"lag_samples": 99}, "fewer than two"),
        ({"lag_seconds": 0.0001, "sfreq": 100.0}, "less than one"),
        ({"lag_seconds": 0.01}, "sfreq is required"),
    ],
)
def test_invalid_lags_fail_loudly(rng, kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_lagged_cca(rng.standard_normal((3, 100)), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rho_threshold": -0.1}, "between 0 and 1"),
        ({"rho_threshold": 1.1}, "between 0 and 1"),
        ({"n_keep": 0}, "positive integer"),
        ({"n_keep": 20}, "exceeds"),
    ],
)
def test_invalid_selection_fails_loudly(rng, kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_lagged_cca(rng.standard_normal((5, 300)), lag_samples=1, **kwargs)


def test_nonfinite_and_zero_rank_inputs_are_inadmissible(rng):
    nonfinite = rng.standard_normal((4, 100))
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_lagged_cca(nonfinite, lag_samples=1)
    with pytest.raises(ValueError, match="zero-rank"):
        compute_lagged_cca(np.zeros((4, 100)), lag_samples=1)


def test_function_shapes_diagnostics_and_exact_n_keep(rng):
    X = rng.standard_normal((8, 1000))
    cleaned, info = compute_lagged_cca(X, lag_samples=3, n_keep=4)
    assert cleaned.shape == X.shape
    assert info["cleaning_matrix"].shape == (8, 8)
    assert info["n_kept"] == 4
    assert info["n_kept"] + info["n_removed"] == len(info["correlations"])
    assert info["lag_samples"] == 3


def test_estimator_is_cloneable_and_fit_transform_is_compositional(rng):
    X = rng.standard_normal((7, 600))
    estimator = LaggedCCA(lag_samples=2, rho_threshold=0.8, n_keep=4)
    assert clone(estimator).get_params() == estimator.get_params()
    combined = estimator.fit_transform(X)
    separate = LaggedCCA(lag_samples=2, rho_threshold=0.8, n_keep=4).fit(X).transform(X)
    np.testing.assert_allclose(combined, separate)


def test_estimator_transform_uses_the_training_operator(rng):
    train = rng.standard_normal((8, 1000))
    evaluation = rng.standard_normal((8, 500))
    estimator = LaggedCCA(lag_samples=1, n_keep=4).fit(train)
    cleaned = estimator.transform(evaluation)
    mean = evaluation.mean(axis=1, keepdims=True)
    expected = estimator.cleaning_matrix_ @ (evaluation - mean) + mean
    np.testing.assert_allclose(cleaned, expected)


def test_transform_before_fit_and_channel_mismatch_raise(rng):
    with pytest.raises(NotFittedError):
        LaggedCCA(lag_samples=1).transform(rng.standard_normal((4, 100)))
    estimator = LaggedCCA(lag_samples=1).fit(rng.standard_normal((4, 100)))
    with pytest.raises(ValueError, match="channels"):
        estimator.transform(rng.standard_normal((3, 100)))


def test_epoched_array_round_trip(rng):
    epochs = rng.standard_normal((5, 6, 200))
    cleaned = LaggedCCA(lag_samples=2, n_keep=4).fit_transform(epochs)
    assert cleaned.shape == epochs.shape


def test_broadband_attenuation_and_alpha_preservation(muscle_data):
    X, _clean, sfreq = muscle_data
    cleaned = LaggedCCA(lag_samples=1, rho_threshold=0.9).fit_transform(X)
    assert _band_power(cleaned, sfreq, 30.0, 120.0) < 0.8 * _band_power(
        X, sfreq, 30.0, 120.0
    )
    assert _band_power(cleaned, sfreq, 8.0, 12.0) > 0.7 * _band_power(
        X, sfreq, 8.0, 12.0
    )


def test_mne_raw_preserves_container_metadata_and_nonpicked_channels(muscle_data):
    mne = pytest.importorskip("mne")
    X, _clean, sfreq = muscle_data
    stim = np.zeros((1, X.shape[1]))
    data = np.vstack((X, stim))
    info = mne.create_info(
        [*(f"EEG{idx:02d}" for idx in range(X.shape[0])), "STI 014"],
        sfreq,
        [*("eeg" for _ in range(X.shape[0])), "stim"],
    )
    raw = mne.io.RawArray(data, info, first_samp=123, verbose=False)
    raw.set_annotations(mne.Annotations([0.1], [0.2], ["test"]))
    cleaned = LaggedCCA(lag_seconds=1.0 / sfreq).fit_transform(raw)
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.first_samp == raw.first_samp
    assert cleaned.annotations == raw.annotations
    assert cleaned.ch_names == raw.ch_names
    np.testing.assert_array_equal(cleaned.get_data(picks=["STI 014"]), stim)
    assert not np.allclose(cleaned.get_data(picks="eeg"), X)


def test_mne_epochs_use_within_epoch_pairs_and_preserve_events(rng):
    mne = pytest.importorskip("mne")
    data = rng.standard_normal((4, 5, 120))
    info = mne.create_info([f"EEG{idx}" for idx in range(5)], 200.0, "eeg")
    events = np.column_stack((np.arange(4) * 200, np.zeros(4, int), np.ones(4, int)))
    epochs = mne.EpochsArray(
        data, info, events=events, event_id={"event": 1}, verbose=False
    )
    cleaned = LaggedCCA(lag_seconds=0.01, n_keep=3).fit_transform(epochs)
    assert isinstance(cleaned, mne.BaseEpochs)
    np.testing.assert_array_equal(cleaned.events, epochs.events)
    assert cleaned.event_id == epochs.event_id
    assert cleaned.get_data(copy=False).shape == data.shape


def test_mne_sfreq_conflict_is_rejected(muscle_data):
    mne = pytest.importorskip("mne")
    X, _clean, sfreq = muscle_data
    raw = mne.io.RawArray(
        X,
        mne.create_info([f"EEG{idx}" for idx in range(X.shape[0])], sfreq, "eeg"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="disagrees"):
        LaggedCCA(lag_seconds=0.004, sfreq=100.0).fit(raw)


def test_transform_sfreq_must_match_fitted_mne_data(muscle_data):
    mne = pytest.importorskip("mne")
    X, _clean, sfreq = muscle_data
    names = [f"EEG{idx}" for idx in range(X.shape[0])]
    train = mne.io.RawArray(X, mne.create_info(names, sfreq, "eeg"), verbose=False)
    different_rate = mne.io.RawArray(
        X, mne.create_info(names, sfreq * 2, "eeg"), verbose=False
    )
    estimator = LaggedCCA(lag_seconds=0.004).fit(train)
    with pytest.raises(ValueError, match="transform sfreq"):
        estimator.transform(different_rate)
