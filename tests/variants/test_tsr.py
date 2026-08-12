import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from mne_denoise.dss import DSS, AverageBias, TimeShiftDSS
from mne_denoise.dss.denoisers.temporal import LagAverageBias, SmoothingBias
from mne_denoise.dss.variants.tsr import (
    _lag_augment,
    _observation_weights,
    smooth_dss,
)


@pytest.fixture
def slow_data_generator():
    rng = np.random.default_rng(42)
    n_times = 500

    # Slow wave (high autocorrelation)
    t = np.linspace(0, 4 * np.pi, n_times)
    slow_signal = np.sin(t)

    def get_data(shape):
        noise = rng.normal(0, 0.5, shape)  # White noise (low autocorr)
        data = noise.copy()

        # Add slow signal
        if len(shape) == 2:  # (n_ch, n_times)
            data[0] += slow_signal
        elif len(shape) == 3:  # (n_epochs, n_ch, n_times)
            data[:, 0, :] += slow_signal

        return data, slow_signal

    return get_data


def test_tsr_array(slow_data_generator):
    data, slow = slow_data_generator((3, 500))
    dss = DSS(bias=LagAverageBias(lags=10), n_components=3)
    dss.fit(data)

    sources = dss.transform(data)
    # Slow component should be first (highest eigenvalue/score)
    corr = np.abs(np.corrcoef(sources[0], slow)[0, 1])
    assert corr > 0.8


def test_tsr_raw(slow_data_generator):
    data, slow = slow_data_generator((3, 500))
    info = mne.create_info(3, 100, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=LagAverageBias(lags=10), n_components=3)
    dss.fit(raw)

    sources = dss.transform(raw)
    corr = np.abs(np.corrcoef(sources[0], slow)[0, 1])
    assert corr > 0.8


def test_tsr_epochs(slow_data_generator):
    data, slow = slow_data_generator((5, 3, 500))
    info = mne.create_info(3, 100, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)

    dss = DSS(bias=LagAverageBias(lags=10), n_components=2)
    dss.fit(epochs)

    sources = dss.transform(epochs)
    assert sources.shape == (5, 2, 500)

    corr = np.abs(np.corrcoef(sources[0, 0], slow)[0, 1])
    assert corr > 0.8


def test_smooth_dss_evoked(slow_data_generator):
    # smooth_dss specifically targets low frequency
    data, slow = slow_data_generator((5, 3, 500))
    info = mne.create_info(3, 100, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    evoked = epochs.average()

    dss = smooth_dss(window=20, n_components=1)
    dss.fit(evoked)

    src = dss.transform(evoked)
    assert isinstance(src, np.ndarray)
    assert src.shape == (1, 500)

    corr = np.abs(np.corrcoef(src[0], slow)[0, 1])
    assert corr > 0.8


def test_tsr_3d_bias_unit():
    """Test explicit 3D data handling in bias classes."""
    rng = np.random.default_rng(42)
    data_3d = rng.standard_normal((3, 20, 5))  # (ch, times, epochs)

    # 1. LagAverageBias
    bias = LagAverageBias(lags=2)
    out_3d = bias.apply(data_3d)
    assert out_3d.shape == data_3d.shape
    assert out_3d.ndim == 3

    # 2. SmoothingBias
    bias_smooth = SmoothingBias(window=3)
    out_smooth = bias_smooth.apply(data_3d)
    assert out_smooth.shape == data_3d.shape
    assert out_smooth.ndim == 3


def test_tsr_inverse_lag_weighting(slow_data_generator):
    """Test inverse-lag weighting."""
    data, slow = slow_data_generator((3, 500))

    dss = DSS(
        bias=LagAverageBias(lags=10, weighting="inverse_lag"),
        n_components=3,
    )
    dss.fit(data)

    sources = dss.transform(data)
    # Should still extract the slow component
    corr = np.abs(np.corrcoef(sources[0], slow)[0, 1])
    assert corr > 0.8


def _data(seed=0, shape=(3, 80, 8), offset=True):
    """Create deterministic channel-by-time-by-epoch data."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape)
    if offset:
        data += np.linspace(-2.0, 3.0, shape[0])[:, np.newaxis, np.newaxis]
    return data


def _estimator(**kwargs):
    """Build a small explicit-rank estimator."""
    params = {"lag_samples": [0, 1], "n_components": 4, "rank": 4}
    params.update(kwargs)
    return TimeShiftDSS(**params)


def test_lag_augmentation_sign_and_epoch_isolation():
    """Positive lags use past samples and never borrow adjacent epochs."""
    data = np.array([[[0.0, 100.0], [1.0, 101.0], [2.0, 102.0], [3.0, 103.0]]])

    augmented, start, stop = _lag_augment(data, (-1, 0, 1))

    assert (start, stop) == (1, 3)
    assert_array_equal(augmented[0], [[2.0, 102.0], [3.0, 103.0]])
    assert_array_equal(augmented[1], [[1.0, 101.0], [2.0, 102.0]])
    assert_array_equal(augmented[2], [[0.0, 100.0], [1.0, 101.0]])


@pytest.mark.parametrize(
    "lag_kws, error, match",
    [
        ({}, ValueError, "exactly one"),
        (
            {"lag_samples": [0, 1], "lag_times": [0.0, 0.01], "sfreq": 100.0},
            ValueError,
            "exactly one",
        ),
        ({"lag_samples": "0,1"}, TypeError, "one-dimensional sequence"),
        ({"lag_samples": []}, ValueError, "non-empty"),
        ({"lag_samples": [0, True]}, TypeError, "only integers"),
        ({"lag_samples": [1, 2]}, ValueError, "contain zero"),
        (
            {"lag_times": "0,0.01", "sfreq": 100.0},
            TypeError,
            "one-dimensional sequence",
        ),
        (
            {"lag_times": [0.0, np.nan], "sfreq": 100.0},
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_lag_declarations(lag_kws, error, match):
    """Lag declarations reject ambiguous, malformed, and non-finite grids."""
    with pytest.raises(error, match=match):
        TimeShiftDSS(n_components=1, rank=1, **lag_kws).fit(_data())


@pytest.mark.parametrize(
    "data, match",
    [
        (np.ones((3, 20)), "requires epoched data"),
        (np.empty((0, 20, 2)), "dimensions must be non-empty"),
        (np.ones((3, 1, 2)), "dimensions must be non-empty"),
        (np.ones((3, 20, 1)), "at least two repeated epochs"),
        (np.full((3, 20, 2), np.nan), "finite values"),
    ],
)
def test_invalid_epoched_array_geometry(data, match):
    """The repeated-trial array contract is checked before decomposition."""
    with pytest.raises(ValueError, match=match):
        _estimator().fit(data)


def test_lag_span_must_leave_two_common_samples():
    """Lag grids cannot consume the complete within-epoch support."""
    with pytest.raises(ValueError, match="fewer than two common"):
        TimeShiftDSS(
            lag_samples=[0, 19],
            n_components=1,
            rank=1,
        ).fit(_data(shape=(3, 20, 2)))


def test_lag_weights_use_minimum_across_touched_samples():
    """A zero-weight source sample invalidates every lag window touching it."""
    weights = np.ones((6, 2))
    weights[2, 0] = 0.0

    valid = _observation_weights(
        weights,
        n_times=6,
        n_epochs=2,
        lags=(0, 1, 2),
        start=2,
        stop=6,
    )

    # Reference times are 2..5. The bad source sample at t=2 is touched by
    # reference windows t=2, 3, and 4, but only in epoch 0.
    assert_array_equal(valid[:, 0], [0.0, 0.0, 0.0, 1.0])
    assert_array_equal(valid[:, 1], np.ones(4))


def test_extract_shapes_and_explicit_fitted_geometry():
    """Extraction exposes source components only on common valid support."""
    data = _data()
    est = _estimator().fit(data)

    sources = est.transform(data)

    assert sources.shape == (4, 79, 8)
    assert est.n_augmented_features_ == 6
    assert est.positive_weight_observations_ == 79 * 8
    assert est.effective_observations_ == pytest.approx(79 * 8)
    assert est.valid_slice_ == slice(1, 80)
    assert est.feature_mean_.shape == (6, 1)
    assert_array_equal(est.feature_mean_, 0.0)


def test_wrapper_composes_trial_average_dss_on_lag_features():
    """TimeShiftDSS adds lags and delegates decomposition to the DSS estimator."""
    data = _data(seed=30)
    augmented, _, _ = _lag_augment(data, (0, 1))
    weights = np.ones(augmented.shape[1:])
    reference = DSS(
        bias=AverageBias(axis="epochs", weights=weights),
        n_components=4,
        rank=4,
        reg=1e-9,
        normalize_input=False,
        center=False,
    ).fit(augmented, weights=weights)

    fitted = _estimator().fit(data)

    assert isinstance(fitted.dss_, DSS)
    assert isinstance(fitted.dss_.bias, AverageBias)
    assert fitted.dss_.bias.axis == "epochs"
    assert_allclose(fitted.eigenvalues_, reference.eigenvalues_, rtol=1e-12, atol=1e-12)
    assert_allclose(
        np.abs(fitted.dss_.filters_), np.abs(reference.filters_), atol=1e-12
    )


def test_sensor_patterns_equal_direct_training_least_squares():
    """Sensor projection is explicit least squares, not a lag-pattern shortcut."""
    data = _data(seed=31, offset=False)
    fitted = _estimator().fit(data)
    sources = fitted.transform(data).reshape(4, -1)
    sensors = data[:, 1:, :].reshape(3, -1)
    expected = sensors @ np.linalg.pinv(sources)

    assert_allclose(fitted.patterns_, expected, rtol=1e-10, atol=1e-10)


def test_centering_is_training_fitted_and_transform_batch_invariant():
    """Unrelated transform epochs cannot change an unchanged source prefix."""
    fit_data = _data(seed=1)
    held_out = _data(seed=2, shape=(3, 80, 2))
    unrelated = _data(seed=3, shape=(3, 80, 5)) * 20.0 + 100.0
    est = _estimator(center=True).fit(fit_data)

    prefix = est.transform(held_out)
    combined = est.transform(np.concatenate([held_out, unrelated], axis=2))

    assert_allclose(combined[:, :, :2], prefix, rtol=0.0, atol=1e-12)
    assert not np.allclose(est.feature_mean_, 0.0)


@pytest.mark.parametrize("center", [False, True])
def test_full_rank_retain_reconstructs_valid_interval(center):
    """Weighted least-squares patterns reconstruct the zero-lag sensor block."""
    data = _data(shape=(3, 60, 8))
    original = data.copy()
    est = TimeShiftDSS(
        lag_samples=[0, 1],
        n_components=6,
        rank=6,
        n_select=6,
        component_action="retain",
        center=center,
    )

    retained = est.fit_transform(data)

    assert_allclose(retained[:, 1:, :], data[:, 1:, :], rtol=1e-9, atol=1e-9)
    assert_array_equal(retained[:, :1, :], data[:, :1, :])
    assert_array_equal(data, original)


def test_subtract_matches_selected_sensor_projection():
    """Subtraction removes exactly the selected fitted component activity."""
    data = _data()
    est = _estimator(n_select=2, component_action="extract").fit(data)
    sources = est.transform(data)
    selected = est.patterns_[:, :2] @ sources[:2].reshape(2, -1)
    selected = selected.reshape(3, 79, 8)

    subtracted = est.set_params(component_action="subtract").transform(data)

    assert_allclose(subtracted[:, 1:, :], data[:, 1:, :] - selected)
    assert_array_equal(subtracted[:, 0, :], data[:, 0, :])


def test_weight_broadcast_matches_explicit_time_epoch_matrix():
    """One-dimensional time weights broadcast identically over epochs."""
    data = _data(offset=False)
    weights = np.linspace(0.2, 1.0, data.shape[1])
    matrix = np.broadcast_to(weights[:, np.newaxis], data.shape[1:])

    time_fit = _estimator().fit(data, sample_weight=weights)
    matrix_fit = _estimator().fit(data, sample_weight=matrix)

    assert_allclose(time_fit.eigenvalues_, matrix_fit.eigenvalues_)
    assert_allclose(np.abs(time_fit.filters_), np.abs(matrix_fit.filters_))


@pytest.mark.parametrize(
    "weights, match",
    [
        (np.ones((8, 80)), "sample_weight"),
        (np.full(80, np.nan), "finite"),
        (np.r_[np.ones(79), -1.0], "non-negative"),
        (np.zeros(80), "no positive-weight"),
    ],
)
def test_invalid_sample_weights_are_rejected(weights, match):
    """Fit weights have an explicit time-by-epoch contract."""
    with pytest.raises(ValueError, match=match):
        _estimator().fit(_data(), sample_weight=weights)


def test_seconds_lags_resolve_against_sampling_grid():
    """Physical lags become exact integer sample offsets."""
    est = TimeShiftDSS(
        lag_times=[0.0, 0.01],
        sfreq=100.0,
        n_components=4,
        rank=4,
    ).fit(_data())

    assert est.lag_samples_ == (0, 1)
    assert est.lag_times_ == (0.0, 0.01)

    with pytest.raises(ValueError, match="sampling grid"):
        TimeShiftDSS(
            lag_times=[0.0, 0.015],
            sfreq=100.0,
            n_components=4,
            rank=4,
        ).fit(_data())
    with pytest.raises(TypeError, match="booleans"):
        TimeShiftDSS(
            lag_times=[False, 0.01],
            sfreq=100.0,
            n_components=4,
            rank=4,
        ).fit(_data())


def test_explicit_rank_and_selection_contracts():
    """The estimator has no in-sample automatic component-selection path."""
    data = _data()
    with pytest.raises(ValueError, match="n_select is required"):
        _estimator(component_action="retain").fit(data)
    with pytest.raises(ValueError, match="cannot exceed rank"):
        TimeShiftDSS(lag_samples=[0, 1], n_components=5, rank=4).fit(data)
    with pytest.raises(ValueError, match="exceeds.*augmented"):
        TimeShiftDSS(lag_samples=[0, 1], n_components=4, rank=7).fit(data)
    with pytest.raises(ValueError, match="n_select cannot exceed"):
        _estimator(n_select=5).fit(data)
    assert not hasattr(_estimator(), "auto_select")


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"component_action": "invalid"}, ValueError, "component_action"),
        ({"center": 1}, TypeError, "center"),
        ({"distortion_control": "invalid"}, ValueError, "distortion_control"),
        (
            {"distortion_control": "cca", "n_select": 2},
            ValueError,
            "CCA distortion control",
        ),
        ({"reg": True}, TypeError, "reg"),
        ({"reg": 0.0}, ValueError, "reg"),
        ({"reg": np.nan}, ValueError, "reg"),
    ],
)
def test_invalid_estimator_parameters(kwargs, error, match):
    """Constructor choices fail explicitly before fitting numerical state."""
    with pytest.raises(error, match=match):
        _estimator(**kwargs).fit(_data())


def test_high_parameter_ratio_warns_instead_of_auto_selecting():
    """A high-dimensional fit is visibly risky and never promotes components."""
    data = _data(shape=(4, 14, 3), offset=False)
    with pytest.warns(UserWarning, match="high risk of overfitting"):
        est = TimeShiftDSS(
            lag_samples=[0, 1, 2, 3, 4],
            n_components=8,
            rank=8,
        ).fit(data)
    rank_deficient = np.broadcast_to(data[:1], (3, *data.shape[1:])).copy()
    with pytest.raises(ValueError, match="numerical whitening rank"):
        _estimator(n_components=3, rank=4).fit(rank_deficient)

    assert est.n_augmented_features_ == 20
    assert not hasattr(est, "n_selected_")


def test_cca_is_training_only_and_returns_one_component():
    """CCA freezes its fitted rotation and produces one canonical variate."""
    fit_data = _data(seed=10)
    held_out = _data(seed=11, shape=(3, 80, 2))
    unrelated = _data(seed=12, shape=(3, 80, 3)) * 30.0
    est = _estimator(distortion_control="cca").fit(fit_data)

    prefix = est.transform(held_out)
    combined = est.transform(np.concatenate([held_out, unrelated], axis=2))

    assert prefix.shape == (1, 79, 2)
    assert est.cca_rotation_.shape == (1, 4)
    assert est.filters_.shape == (1, 6)
    assert est.dss_.filters_.shape == (4, 6)
    assert est.patterns_.shape == (3, 1)
    assert 0 <= est.cca_correlations_[0] <= 1
    assert_allclose(combined[:, :, :2], prefix, rtol=0.0, atol=1e-12)


def test_score_uses_a_fixed_leading_subspace_on_held_out_trials():
    """Evaluation returns one finite score for the declared leading subspace."""
    fit_data = _data(seed=20)
    held_out = _data(seed=21, shape=(3, 80, 4))
    est = _estimator(n_select=2).fit(fit_data)

    score = est.score(held_out)

    assert np.isfinite(score)
    assert score >= 0


def test_score_accepts_a_different_epoch_length():
    """Held-out scoring derives lag support from the evaluated data."""
    fit_data = _data(seed=22, shape=(3, 80, 8))
    held_out = _data(seed=23, shape=(3, 50, 4))
    est = _estimator(n_select=2).fit(fit_data)

    score = est.score(held_out)

    assert np.isfinite(score)


def test_score_requires_a_predeclared_subspace():
    """Validation never chooses a component count from held-out observations."""
    est = _estimator().fit(_data(seed=24))

    with pytest.raises(ValueError, match="explicit n_select"):
        est.score(_data(seed=25, shape=(3, 80, 4)))


def test_score_returns_zero_for_zero_power_held_out_data():
    """A valid held-out set with no component power has a finite zero score."""
    est = _estimator(n_select=2).fit(_data(seed=26))

    assert est.score(np.zeros((3, 80, 4))) == 0.0


def test_transform_rejects_container_and_channel_contract_changes():
    """A fitted estimator freezes both input family and channel geometry."""
    est = _estimator().fit(_data())
    info = mne.create_info(3, 100.0, "eeg")
    epochs = mne.EpochsArray(
        np.transpose(_data(shape=(3, 80, 2)), (2, 0, 1)),
        info,
        verbose=False,
    )

    with pytest.raises(TypeError, match="container family"):
        est.transform(epochs)
    with pytest.raises(TypeError, match="supports MNE Epochs or NumPy arrays"):
        est.transform([[[1.0]]])
    with pytest.raises(ValueError, match="channels; fitted data had"):
        est.transform(_data(shape=(4, 80, 2)))


def test_mne_epochs_extract_returns_epoch_major_sources():
    """MNE extraction returns an array in epochs-by-components orientation."""
    data = np.transpose(_data(shape=(3, 40, 5)), (2, 0, 1)) * 1e-6
    epochs = mne.EpochsArray(
        data,
        mne.create_info(3, 100.0, "eeg"),
        verbose=False,
    )

    sources = _estimator().fit_transform(epochs)

    assert isinstance(sources, np.ndarray)
    assert sources.shape == (5, 4, 39)


def test_empty_cca_solution_is_rejected(monkeypatch):
    """CCA distortion control rejects a numerically empty canonical space."""

    def _empty_cca(*args, **kwargs):
        return None, None, np.array([]), None, None

    monkeypatch.setattr(
        "mne_denoise.dss.variants.tsr.canonical_correlation",
        _empty_cca,
    )
    with pytest.raises(ValueError, match="CCA input has no variance"):
        _estimator(distortion_control="cca").fit(_data())


def test_mne_epochs_preserve_metadata_and_bad_channels():
    """Sensor output uses shared reconstruction and leaves excluded bads unchanged."""
    data = np.transpose(_data(shape=(4, 80, 8)), (2, 0, 1)) * 1e-6
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 100.0, "eeg")
    epochs = mne.EpochsArray(data, info, tmin=-0.2, verbose=False)
    epochs.info["bads"] = ["EEG3"]
    events = epochs.events.copy()
    metadata = epochs.metadata
    est = TimeShiftDSS(
        lag_times=[0.0, 0.01],
        n_components=4,
        rank=4,
        n_select=1,
        component_action="subtract",
    )

    cleaned = est.fit_transform(epochs)

    assert isinstance(cleaned, mne.BaseEpochs)
    assert_array_equal(cleaned.events, events)
    assert cleaned.metadata is metadata
    assert cleaned.tmin == epochs.tmin
    assert cleaned.info["bads"] == epochs.info["bads"]
    assert_array_equal(
        cleaned.get_data(picks=["EEG3"]), epochs.get_data(picks=["EEG3"])
    )


def test_array_integer_input_produces_float_without_mutation():
    """Integer observations cannot truncate fitted or reconstructed values."""
    data = np.rint(_data() * 4).astype(np.int64)
    original = data.copy()
    out = _estimator(n_select=1, component_action="subtract").fit_transform(data)

    assert out.dtype == np.float64
    assert_array_equal(data, original)


def test_estimator_is_cloneable():
    """All scientific choices remain explicit sklearn parameters."""
    estimator = _estimator(center=True, distortion_control="cca")
    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
