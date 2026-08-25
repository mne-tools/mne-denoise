"""Tests for the mne_denoise.bss_cca module (reference-free BSS-CCA)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mne_denoise.bss_cca import BSSCCA, compute_bss_cca
from mne_denoise.bss_cca import core as bss_cca_core
from mne_denoise.bss_cca.core import _lagged_pairs, _segment_bounds

SFREQ = 250.0


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def muscle_data(rng):
    """Well-separated brain sources plus broadband muscle sources.

    The neural sources are narrow-band so their lag-1 correlation is high and
    the three EMG sources sit clearly at the bottom of the ordering.
    """
    n_times = int(SFREQ * 10)
    t = np.arange(n_times) / SFREQ
    brain = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 3 * t + 0.4),
            np.sin(2 * np.pi * 21 * t + 1.1),
        ]
    )
    emg = rng.standard_normal((3, n_times))
    clean = rng.standard_normal((21, 3)) @ brain
    observed = clean + 0.8 * (rng.standard_normal((21, 3)) @ emg)
    return observed, clean, SFREQ


@pytest.fixture()
def realistic_eeg(rng):
    """Broadband neural background plus high-frequency EMG.

    Unlike ``muscle_data`` the canonical correlations here are compressed into
    a narrow band, which is what real recordings look like. Any selection rule
    that silently assumes a wide separation fails on this fixture.
    """
    n_times = int(SFREQ * 10)
    freqs = np.fft.rfftfreq(n_times, 1.0 / SFREQ)
    brain = np.zeros((6, n_times))
    for index in range(6):
        spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
            freqs.size
        )
        spectrum[(freqs < 1.0) | (freqs > 45.0)] = 0.0
        brain[index] = np.fft.irfft(spectrum, n=n_times)
    emg = np.zeros((3, n_times))
    for index in range(3):
        spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
            freqs.size
        )
        spectrum[freqs < 30.0] = 0.0
        emg[index] = np.fft.irfft(spectrum, n=n_times)
    brain /= np.linalg.norm(brain)
    emg /= np.linalg.norm(emg)
    clean = rng.standard_normal((21, 6)) @ brain
    observed = clean + 1.2 * (rng.standard_normal((21, 3)) @ emg)
    return observed, clean, SFREQ


def _band_power(X, sfreq, fmin, fmax):
    spectrum = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], 1.0 / sfreq)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(spectrum[..., band].sum())


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["full_rank", "low_rank", "average_reference", "duplicate", "constant"],
)
def test_keeping_every_component_is_the_identity(rng, name):
    """n_remove=0 must reproduce the input, including rank-deficient input.

    This is the regression guarding the back-projection: inverting the
    canonical filters instead of regressing against the data destroys whole
    channels here, silently, on the paper's own average-reference montage.
    """
    if name == "full_rank":
        data = rng.standard_normal((8, 4000))
    elif name == "low_rank":
        data = rng.standard_normal((8, 5)) @ rng.standard_normal((5, 4000))
    elif name == "average_reference":
        data = rng.standard_normal((21, 4000))
        data = data - data.mean(axis=0, keepdims=True)
    elif name == "duplicate":
        data = rng.standard_normal((6, 4000))
        data[5] = data[0]
    else:
        data = rng.standard_normal((6, 3000))
        data[3] = 4.2

    cleaned, info = compute_bss_cca(data, n_remove=0, preserve_mean=True)
    np.testing.assert_allclose(cleaned, data, atol=1e-9)
    assert info["n_removed"] == 0
    assert info["input_rank"] == info["kept_mask"].size


def test_analytical_subspace_removal(rng):
    """A known artifact subspace is removed and the rest is preserved exactly."""
    n_times = 4000
    t = np.arange(n_times) / SFREQ
    smooth = np.vstack([np.sin(2 * np.pi * 4 * t), np.cos(2 * np.pi * 7 * t)])
    white = rng.standard_normal((2, n_times))
    mixing = rng.standard_normal((6, 4))
    observed = mixing @ np.vstack([smooth, white])

    cleaned, info = compute_bss_cca(observed, n_remove=2, preserve_mean=False)

    smooth_only = mixing[:, :2] @ smooth
    smooth_only = smooth_only - smooth_only.mean(axis=1, keepdims=True)
    error = np.linalg.norm(cleaned - smooth_only) / np.linalg.norm(smooth_only)
    assert error < 0.05, f"residual {error:.3f}"
    assert info["kept_mask"].tolist() == [True, True, False, False]


def test_reconstruction_matrix_orientation(rng):
    """cleaning_matrix_ is a channel-space operator applied on the left."""
    data = rng.standard_normal((7, 2000))
    cleaned, info = compute_bss_cca(data, n_remove=2, preserve_mean=False)
    operator = info["cleaning_matrix"]
    assert operator.shape == (7, 7)
    centered = data - data.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(cleaned, operator @ centered, atol=1e-10)


def test_correlations_are_descending_and_selection_drops_the_tail(muscle_data):
    """Components are ordered by decreasing correlation; n_remove drops the tail."""
    observed, _clean, _sfreq = muscle_data
    _cleaned, info = compute_bss_cca(observed, n_remove=3)
    assert np.all(np.diff(info["correlations"]) <= 1e-12)
    assert not info["kept_mask"][-3:].any()
    assert info["kept_mask"][:-3].all()


def test_threshold_and_count_can_select_the_same_components(muscle_data):
    """A threshold placed between two correlations equals the matching count."""
    observed, _clean, _sfreq = muscle_data
    _cleaned, info = compute_bss_cca(observed, n_remove=3)
    rho = info["correlations"]
    threshold = 0.5 * (rho[-4] + rho[-3])
    _cleaned_t, info_t = compute_bss_cca(observed, rho_threshold=threshold)
    np.testing.assert_array_equal(info_t["kept_mask"], info["kept_mask"])


def test_signed_autocorrelation_exposes_near_nyquist_aliasing(rng):
    """A component at f_s/2 has rho ~ 1 but a negative signed autocorrelation.

    Canonical correlations come from singular values and cannot be negative,
    so anti-correlated components rank as the most 'brain-like'. The signed
    diagnostic is what lets a user notice.
    """
    n_times = 2000
    sources = np.zeros((6, n_times))
    sources[0] = (-1.0) ** np.arange(n_times)
    sources[1:] = rng.standard_normal((5, n_times))
    observed = rng.standard_normal((6, 6)) @ sources

    _cleaned, info = compute_bss_cca(observed, n_remove=1)
    assert np.all(info["correlations"] >= -1e-12)
    aliased = int(np.argmin(info["autocorrelations"]))
    assert info["autocorrelations"][aliased] < -0.9
    assert info["correlations"][aliased] > 0.9


def test_filter_asymmetry_is_reported_per_component(muscle_data):
    """The two canonical filters are compared as a validity diagnostic."""
    observed, _clean, _sfreq = muscle_data
    _cleaned, info = compute_bss_cca(observed, n_remove=3)
    assert info["filter_asymmetry"].shape == info["correlations"].shape
    assert np.all(info["filter_asymmetry"] >= 0.0)
    assert np.all(info["filter_asymmetry"] <= 2.0 + 1e-9)


# ---------------------------------------------------------------------------
# Lag semantics
# ---------------------------------------------------------------------------


def test_lag_defaults_to_one_sample(rng):
    """The paper fixes the lag at one sample; that is the default."""
    data = rng.standard_normal((5, 1000))
    _cleaned, info = compute_bss_cca(data, n_remove=1)
    assert info["lag_samples"] == 1


def test_sample_and_physical_lag_agree(rng):
    """lag_seconds resolves to the equivalent sample lag."""
    data = rng.standard_normal((5, 1000))
    by_sample, info_a = compute_bss_cca(data, lag_samples=3, n_remove=1)
    by_time, info_b = compute_bss_cca(
        data, lag_seconds=3.0 / SFREQ, sfreq=SFREQ, n_remove=1
    )
    assert info_a["lag_samples"] == info_b["lag_samples"] == 3
    np.testing.assert_allclose(by_sample, by_time, atol=1e-12)
    assert info_b["lag_seconds"] == pytest.approx(3.0 / SFREQ)


def test_lagged_pairs_do_not_wrap(rng):
    """Pairs are truncated at the endpoints rather than wrapped."""
    data = rng.standard_normal((3, 50))
    current, past = _lagged_pairs(data, 4)
    assert current.shape == past.shape == (46, 3)
    np.testing.assert_array_equal(current, data[:, 4:].T)
    np.testing.assert_array_equal(past, data[:, :-4].T)


def test_lagged_pairs_never_cross_epoch_boundaries():
    """Every emitted pair stays inside one epoch and honours the lag."""
    lag = 2
    X = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    current, past = _lagged_pairs(X, lag)

    lookup = {
        value: (epoch, time)
        for epoch, row in enumerate(X[:, 0, :])
        for time, value in enumerate(row)
    }
    pairs = [(lookup[c[0]], lookup[p[0]]) for c, p in zip(current, past, strict=True)]
    assert len(pairs) == X.shape[0] * (X.shape[2] - lag)
    assert all(a[0] == b[0] for a, b in pairs), "pair crossed an epoch boundary"
    assert all(a[1] - b[1] == lag for a, b in pairs), "pair used the wrong lag"


# ---------------------------------------------------------------------------
# Selection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{}, {"n_remove": 1, "rho_threshold": 0.9}])
def test_selection_rule_must_be_explicit(rng, kwargs):
    """Neither a silent default nor two competing rules are accepted."""
    with pytest.raises(ValueError, match="exactly one of"):
        compute_bss_cca(rng.standard_normal((5, 1000)), **kwargs)


def test_unreachable_threshold_warns_and_removes_everything(rng, caplog):
    """An unreachable threshold is reported, never silently reinterpreted."""
    data = rng.standard_normal((6, 3000))
    with caplog.at_level(logging.WARNING, logger="mne_denoise.bss_cca.core"):
        cleaned, info = compute_bss_cca(data, rho_threshold=0.9, preserve_mean=False)
    assert "no component reaches rho_threshold" in caplog.text
    assert info["n_kept"] == 0
    np.testing.assert_allclose(cleaned, 0.0, atol=1e-9)


def test_realistic_correlation_spectrum_is_not_silently_destroyed(realistic_eeg):
    """On realistic data a paper-style count beats leaving the data alone.

    Real recordings produce a compressed correlation spectrum, so a rule that
    keeps only what exceeds a high fixed threshold can retain almost nothing.
    """
    observed, clean, _sfreq = realistic_eeg
    _cleaned, info = compute_bss_cca(observed, n_remove=3)
    assert info["correlations"].max() < 0.99

    cleaned, _info = compute_bss_cca(observed, n_remove=3, preserve_mean=False)
    centered = observed - observed.mean(axis=1, keepdims=True)
    clean_c = clean - clean.mean(axis=1, keepdims=True)
    assert _corr(cleaned, clean_c) > _corr(centered, clean_c)


def test_n_remove_cannot_exceed_the_fitted_rank(rng):
    """Asking to remove more components than exist fails loudly."""
    data = rng.standard_normal((4, 5)) @ rng.standard_normal((5, 2000))
    with pytest.raises(ValueError, match="exceeds the fitted CCA rank"):
        compute_bss_cca(data, n_remove=99)


# ---------------------------------------------------------------------------
# Degenerate and invalid input
# ---------------------------------------------------------------------------


def test_undersampled_data_is_rejected(rng):
    """Fewer pairs than channels saturates every correlation at one."""
    with pytest.raises(ValueError, match="more lagged pairs than channels"):
        compute_bss_cca(rng.standard_normal((16, 12)), n_remove=1)


def test_scarce_samples_warn(rng, caplog):
    """A thin but usable sample count is flagged rather than rejected."""
    with caplog.at_level(logging.WARNING, logger="mne_denoise.bss_cca.core"):
        compute_bss_cca(rng.standard_normal((8, 40)), n_remove=1)
    assert "lagged pairs for" in caplog.text


def test_all_constant_input_is_rejected():
    """Zero-rank data cannot be decomposed."""
    with pytest.raises(ValueError, match="zero-rank"):
        compute_bss_cca(np.ones((4, 500)), n_remove=1)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.ones(10), "2-D"),
        (np.ones((2, 2, 2, 2)), "2-D"),
        (np.ones((1, 500)), "at least two channels"),
        (np.ones((4, 1)), "at least two time samples"),
        (np.full((4, 500), np.nan), "finite"),
        (np.full((4, 500), np.inf), "finite"),
    ],
)
def test_invalid_data_is_rejected(data, message):
    """Shape, size, and finiteness are checked at the public boundary."""
    with pytest.raises(ValueError, match=message):
        compute_bss_cca(data, n_remove=1)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"lag_samples": 0}, ValueError, "lag_samples"),
        ({"lag_samples": 1.5}, TypeError, "lag_samples"),
        ({"lag_samples": True}, TypeError, "lag_samples"),
        ({"lag_samples": 5000}, ValueError, "leaves no paired samples"),
        ({"lag_samples": 1, "lag_seconds": 0.1}, ValueError, "at most one"),
        ({"lag_seconds": 0.1}, ValueError, "sfreq is required"),
        ({"lag_seconds": np.nan, "sfreq": SFREQ}, TypeError, "lag_seconds"),
        ({"lag_seconds": -0.1, "sfreq": SFREQ}, ValueError, "positive"),
        ({"lag_seconds": 1e-9, "sfreq": SFREQ}, ValueError, "less than one sample"),
        ({"lag_seconds": 0.1, "sfreq": 0.0}, ValueError, "sfreq must be a positive"),
        ({"lag_seconds": 0.1, "sfreq": True}, TypeError, "sfreq"),
    ],
)
def test_invalid_lag_declarations(rng, kwargs, error, message):
    """Lag scalars reject booleans, non-finite, and out-of-range values."""
    with pytest.raises(error, match=message):
        compute_bss_cca(rng.standard_normal((4, 1000)), n_remove=1, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_remove": -1}, ValueError, "non-negative"),
        ({"n_remove": 1.5}, TypeError, "n_remove"),
        ({"n_remove": True}, TypeError, "n_remove"),
        ({"rho_threshold": 1.5}, ValueError, "between 0 and 1"),
        ({"rho_threshold": np.nan}, TypeError, "rho_threshold"),
        ({"rho_threshold": True}, TypeError, "rho_threshold"),
    ],
)
def test_invalid_selection_parameters(rng, kwargs, error, message):
    """Selection scalars are validated the same way as the rest of the package."""
    with pytest.raises(error, match=message):
        compute_bss_cca(rng.standard_normal((4, 1000)), **kwargs)


def test_numpy_scalars_are_accepted(rng):
    """NumPy scalars are ordinary numbers, not invalid input."""
    data = rng.standard_normal((5, 1000))
    _cleaned, info = compute_bss_cca(
        data,
        lag_samples=np.int64(2),
        rho_threshold=np.float64(0.1),
    )
    assert info["lag_samples"] == 2
    _cleaned_b, info_b = compute_bss_cca(
        data,
        lag_seconds=np.float64(0.008),
        sfreq=np.float64(SFREQ),
        n_remove=np.int64(1),
    )
    assert info_b["lag_samples"] == 2


def test_input_is_not_modified(rng):
    """Neither the function nor the estimator mutates its input."""
    data = rng.standard_normal((5, 1000))
    original = data.copy()
    compute_bss_cca(data, n_remove=1)
    BSSCCA(n_remove=1).fit_transform(data)
    np.testing.assert_array_equal(data, original)


# ---------------------------------------------------------------------------
# Shapes and epochs
# ---------------------------------------------------------------------------


def test_continuous_and_epoched_shapes_are_preserved(rng):
    """Output shape always matches input shape."""
    continuous = rng.standard_normal((6, 1000))
    epoched = rng.standard_normal((4, 6, 250))
    assert compute_bss_cca(continuous, n_remove=1)[0].shape == continuous.shape
    assert compute_bss_cca(epoched, n_remove=1)[0].shape == epoched.shape


def test_epoched_result_matches_manual_concatenation(rng):
    """Epoched input is decomposed on within-epoch pairs, applied per epoch."""
    epoched = rng.standard_normal((3, 5, 400))
    cleaned, info = compute_bss_cca(epoched, n_remove=2, preserve_mean=False)
    operator = info["cleaning_matrix"]
    mean = info["training_mean"]
    for index in range(epoched.shape[0]):
        expected = operator @ (epoched[index] - mean)
        np.testing.assert_allclose(cleaned[index], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Fitted behaviour and leakage
# ---------------------------------------------------------------------------


def test_training_mean_is_fixed_during_transform(rng):
    """The centering statistic comes from fit, never from the evaluation set."""
    train = rng.standard_normal((6, 4000))
    estimator = BSSCCA(n_remove=1).fit(train)
    np.testing.assert_allclose(
        estimator.training_mean_, train.mean(axis=1, keepdims=True)
    )

    shifted = rng.standard_normal((6, 500)) + 100.0
    expected = estimator.cleaning_matrix_ @ (shifted - estimator.training_mean_)
    if estimator.preserve_mean:
        expected = expected + estimator.training_mean_
    np.testing.assert_allclose(estimator.transform(shifted), expected, atol=1e-10)


@pytest.mark.parametrize("preserve_mean", [False, True])
def test_continuous_transform_is_batch_invariant(rng, preserve_mean):
    """A sample's result does not depend on what else is in the call."""
    train = rng.standard_normal((6, 3000))
    evaluation = rng.standard_normal((6, 1000)) + np.arange(6)[:, None]
    estimator = BSSCCA(n_remove=1, preserve_mean=preserve_mean).fit(train)
    whole = estimator.transform(evaluation)
    chunked = np.concatenate(
        [
            estimator.transform(evaluation[:, :400]),
            estimator.transform(evaluation[:, 400:]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(whole, chunked, atol=1e-12)


@pytest.mark.parametrize("preserve_mean", [False, True])
def test_epoch_transform_is_batch_invariant(rng, preserve_mean):
    """Epochs transform identically alone or alongside others."""
    train = rng.standard_normal((6, 3000))
    epochs = rng.standard_normal((4, 6, 300))
    epochs[2] += 5.0
    estimator = BSSCCA(n_remove=1, preserve_mean=preserve_mean).fit(train)
    together = estimator.transform(epochs)
    alone = np.stack([estimator.transform(epochs[i]) for i in range(4)])
    np.testing.assert_allclose(together, alone, atol=1e-12)


def test_train_and_evaluation_sets_stay_separate(rng):
    """The operator learned on train is what acts on evaluation data."""
    train = rng.standard_normal((6, 3000))
    evaluation = rng.standard_normal((6, 3000))
    estimator = BSSCCA(n_remove=2).fit(train)
    refit = BSSCCA(n_remove=2).fit(evaluation)
    assert not np.allclose(estimator.cleaning_matrix_, refit.cleaning_matrix_)
    np.testing.assert_allclose(
        estimator.transform(evaluation),
        estimator.cleaning_matrix_ @ (evaluation - estimator.training_mean_)
        + estimator.training_mean_,
        atol=1e-10,
    )


def test_preserve_mean_controls_the_offset(rng):
    """preserve_mean adds exactly the fitted mean back."""
    data = rng.standard_normal((5, 2000)) + 7.0
    with_mean, _ = compute_bss_cca(data, n_remove=1, preserve_mean=True)
    without_mean, info = compute_bss_cca(data, n_remove=1, preserve_mean=False)
    np.testing.assert_allclose(
        with_mean - without_mean,
        np.broadcast_to(info["training_mean"], data.shape),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Estimator contract
# ---------------------------------------------------------------------------


def test_estimator_delegates_to_the_public_function(monkeypatch, rng):
    """fit() routes through compute_bss_cca rather than a private twin."""
    calls = {}
    original = bss_cca_core.compute_bss_cca

    def spy(X, **kwargs):
        calls["kwargs"] = kwargs
        return original(X, **kwargs)

    monkeypatch.setattr(bss_cca_core, "compute_bss_cca", spy)
    BSSCCA(n_remove=2, lag_samples=3, preserve_mean=False).fit(
        rng.standard_normal((5, 1000))
    )
    assert calls, "fit() did not call compute_bss_cca"
    assert calls["kwargs"]["n_remove"] == 2
    assert calls["kwargs"]["lag_samples"] == 3
    assert calls["kwargs"]["preserve_mean"] is False


def test_estimator_is_cloneable_and_fit_transform_composes(rng):
    """Standard scikit-learn cloning and fit_transform semantics hold."""
    data = rng.standard_normal((5, 1000))
    estimator = BSSCCA(n_remove=2, lag_samples=2)
    copy = clone(estimator)
    assert copy.get_params() == estimator.get_params()
    np.testing.assert_allclose(
        estimator.fit_transform(data), copy.fit(data).transform(data), atol=1e-12
    )


def test_fitted_attributes(muscle_data):
    """The documented fitted attributes are present and consistent."""
    observed, _clean, sfreq = muscle_data
    estimator = BSSCCA(n_remove=3, sfreq=sfreq).fit(observed)
    n_channels, n_components = observed.shape[0], estimator.correlations_.size
    assert estimator.cleaning_matrix_.shape == (n_channels, n_channels)
    assert estimator.filters_.shape == (n_components, n_channels)
    assert estimator.patterns_.shape == (n_channels, n_components)
    assert estimator.autocorrelations_.shape == (n_components,)
    assert estimator.filter_asymmetry_.shape == (n_components,)
    assert estimator.kept_mask_.shape == (n_components,)
    assert estimator.training_mean_.shape == (n_channels, 1)
    assert estimator.n_channels_in_ == n_channels
    assert estimator.input_rank_ == n_components
    assert estimator.n_kept_ + estimator.n_removed_ == n_components
    assert estimator.n_removed_ == 3
    assert estimator.feature_names_in_ is None


def test_transform_before_fit_raises(rng):
    """An unfitted estimator refuses to transform."""
    with pytest.raises(NotFittedError):
        BSSCCA(n_remove=1).transform(rng.standard_normal((5, 100)))


def test_transform_rejects_a_channel_count_mismatch(rng):
    """Channel counts must match the fitted layout."""
    estimator = BSSCCA(n_remove=1).fit(rng.standard_normal((5, 1000)))
    with pytest.raises(ValueError, match="channels; fitted data had"):
        estimator.transform(rng.standard_normal((4, 1000)))


def test_transform_rejects_non_finite_evaluation_data(rng):
    """Evaluation data is validated as strictly as training data."""
    estimator = BSSCCA(n_remove=1).fit(rng.standard_normal((5, 1000)))
    bad = rng.standard_normal((5, 100))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.transform(bad)


def test_fit_transform_rejects_unexpected_fit_params(rng):
    """Unsupported fit parameters fail with a clear message."""
    with pytest.raises(TypeError, match="Unexpected fit parameters"):
        BSSCCA(n_remove=1).fit_transform(rng.standard_normal((5, 500)), bogus=1)


def test_fit_reports_the_resolved_operating_point(rng, caplog):
    """The fitted lag, block count, and component counts are logged."""
    data = rng.standard_normal((5, 1000))
    with caplog.at_level(logging.INFO, logger="mne_denoise.bss_cca.core"):
        BSSCCA(n_remove=2, lag_samples=2, verbose=True).fit(data)
    assert "BSS-CCA: lag=2 sample(s), 1 block(s), removed 2 of 5" in caplog.text


@pytest.mark.parametrize(
    ("verbose", "expected"),
    [("ERROR", logging.ERROR), (True, logging.INFO), (False, logging.WARNING)],
)
def test_verbose_sets_the_package_logger_level(rng, verbose, expected):
    """MNE-style verbosity routes through the shared level helper."""
    package_logger = logging.getLogger("mne_denoise")
    previous = package_logger.level
    try:
        BSSCCA(n_remove=1, verbose=verbose).fit(rng.standard_normal((5, 500)))
        assert package_logger.level == expected
    finally:
        package_logger.setLevel(previous)


def test_verbose_none_leaves_external_configuration_alone(rng):
    """verbose=None respects whatever level the caller configured."""
    package_logger = logging.getLogger("mne_denoise")
    previous = package_logger.level
    try:
        package_logger.setLevel(logging.CRITICAL)
        BSSCCA(n_remove=1, verbose=None).fit(rng.standard_normal((5, 500)))
        assert package_logger.level == logging.CRITICAL
    finally:
        package_logger.setLevel(previous)


# ---------------------------------------------------------------------------
# Block-wise operation
# ---------------------------------------------------------------------------


def test_segment_bounds_tile_exactly_without_overlap():
    """Contiguous blocks cover every sample exactly once."""
    bounds = _segment_bounds(1000, n_block=250, hop=250)
    own = [(own_start, own_end) for _e0, _e1, own_start, own_end in bounds]
    assert own == [(0, 250), (250, 500), (500, 750), (750, 1000)]


def test_segment_bounds_extend_the_final_ragged_block():
    """A ragged tail is fitted on a full-length block ending at the last sample."""
    bounds = _segment_bounds(900, n_block=250, hop=250)
    assert bounds[-1][0] == 650 and bounds[-1][1] == 900
    assert bounds[-1][3] == 900
    assert bounds[0][2] == 0


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_block_wise_keeping_everything_is_the_identity(rng, overlap):
    """Blocked operation is still a no-op when nothing is removed."""
    data = rng.standard_normal((8, int(SFREQ * 35)))
    cleaned, info = compute_bss_cca(
        data,
        sfreq=SFREQ,
        segment_len=10.0,
        overlap=overlap,
        n_remove=0,
        preserve_mean=True,
    )
    np.testing.assert_allclose(cleaned, data, atol=1e-9)
    assert info["n_blocks"] > 1


def test_one_block_covering_everything_equals_the_global_fit(rng):
    """A block longer than the recording degenerates to one operator."""
    data = rng.standard_normal((6, 2000))
    blocked, info = compute_bss_cca(data, sfreq=SFREQ, segment_len=1000.0, n_remove=2)
    globally, _ = compute_bss_cca(data, n_remove=2)
    np.testing.assert_allclose(blocked, globally, atol=1e-10)
    assert info["n_blocks"] == 1


def test_block_wise_diagnostics_are_per_block(rng):
    """Blocked mode reports one operator and span per block."""
    data = rng.standard_normal((6, int(SFREQ * 30)))
    _cleaned, info = compute_bss_cca(data, sfreq=SFREQ, segment_len=10.0, n_remove=1)
    assert info["n_blocks"] == 3
    assert len(info["cleaning_matrix"]) == 3
    assert len(info["correlations"]) == 3
    assert info["spans"] == ((0, 2500), (2500, 5000), (5000, 7500))


def test_block_wise_estimator_is_tied_to_its_timeline(rng):
    """A piecewise operator refuses data it cannot be aligned to."""
    data = rng.standard_normal((6, int(SFREQ * 30)))
    estimator = BSSCCA(sfreq=SFREQ, segment_len=10.0, n_remove=1).fit(data)
    np.testing.assert_allclose(
        estimator.transform(data),
        compute_bss_cca(data, sfreq=SFREQ, segment_len=10.0, n_remove=1)[0],
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="tied to the timeline"):
        estimator.transform(data[:, :1000])


def test_segment_len_is_rejected_for_epoched_input(rng):
    """Epoched data is already segmented."""
    with pytest.raises(ValueError, match="only supported for 2-D"):
        compute_bss_cca(
            rng.standard_normal((3, 5, 500)), sfreq=SFREQ, segment_len=1.0, n_remove=1
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"segment_len": 10.0}, ValueError, "sfreq is required"),
        ({"segment_len": 0.0, "sfreq": SFREQ}, ValueError, "positive"),
        ({"segment_len": 0.001, "sfreq": SFREQ}, ValueError, "use a longer block"),
        ({"segment_len": True, "sfreq": SFREQ}, TypeError, "segment_len"),
        ({"overlap": 1.0}, ValueError, r"\[0, 1\)"),
        ({"overlap": -0.1}, ValueError, r"\[0, 1\)"),
        ({"overlap": True}, TypeError, "overlap"),
    ],
)
def test_invalid_blocking_parameters(rng, kwargs, error, message):
    """Blocking parameters are validated before any decomposition runs."""
    with pytest.raises(error, match=message):
        compute_bss_cca(rng.standard_normal((5, 2000)), n_remove=1, **kwargs)


# ---------------------------------------------------------------------------
# Signal behaviour
# ---------------------------------------------------------------------------


def test_broadband_attenuation_preserves_neural_bands(muscle_data):
    """Broadband power drops while representative neural bands survive."""
    observed, clean, sfreq = muscle_data
    cleaned, _info = compute_bss_cca(observed, n_remove=3, preserve_mean=False)

    broadband_before = _band_power(observed, sfreq, 60.0, 120.0)
    broadband_after = _band_power(cleaned, sfreq, 60.0, 120.0)
    assert broadband_after < 0.2 * broadband_before

    for fmin, fmax in ((2.0, 4.0), (9.0, 11.0), (20.0, 22.0)):
        retained = _band_power(cleaned, sfreq, fmin, fmax)
        reference = _band_power(clean, sfreq, fmin, fmax)
        assert retained > 0.7 * reference, f"lost the {fmin}-{fmax} Hz band"


def test_waveform_is_recovered(muscle_data):
    """The cleaned signal correlates with the known clean waveform."""
    observed, clean, _sfreq = muscle_data
    cleaned, _info = compute_bss_cca(observed, n_remove=3, preserve_mean=False)
    clean_c = clean - clean.mean(axis=1, keepdims=True)
    observed_c = observed - observed.mean(axis=1, keepdims=True)
    assert _corr(cleaned, clean_c) > 0.95
    assert _corr(cleaned, clean_c) > _corr(observed_c, clean_c)


# ---------------------------------------------------------------------------
# MNE integration
# ---------------------------------------------------------------------------


def test_mne_raw_round_trip_preserves_metadata_and_other_channels(muscle_data):
    """Raw output keeps timing, annotations, bads, and unselected channels."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    stim = np.zeros((1, observed.shape[1]))
    info = mne.create_info(
        [*(f"EEG{i:02d}" for i in range(observed.shape[0])), "STI 014"],
        sfreq,
        [*("eeg" for _ in range(observed.shape[0])), "stim"],
    )
    raw = mne.io.RawArray(
        np.vstack([observed, stim]), info, first_samp=123, verbose=False
    )
    raw.set_annotations(mne.Annotations([0.1], [0.2], ["test"]))
    raw.info["bads"] = ["EEG02"]
    before = raw.get_data()

    cleaned = BSSCCA(n_remove=3).fit_transform(raw)

    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.first_samp == raw.first_samp
    assert cleaned.annotations == raw.annotations
    assert cleaned.ch_names == raw.ch_names
    assert cleaned.info["bads"] == ["EEG02"]
    np.testing.assert_array_equal(cleaned.get_data(picks=["STI 014"]), stim)
    np.testing.assert_array_equal(raw.get_data(), before)
    assert not np.allclose(cleaned.get_data(picks="eeg"), observed)


def test_mne_epochs_round_trip_preserves_events_and_metadata(rng):
    """Epochs output keeps events, event_id, metadata, and timing."""
    mne = pytest.importorskip("mne")
    pd = pytest.importorskip("pandas")
    data = rng.standard_normal((4, 5, 300))
    info = mne.create_info([f"EEG{i}" for i in range(5)], 200.0, "eeg")
    events = np.column_stack((np.arange(4) * 400, np.zeros(4, int), np.ones(4, int)))
    metadata = pd.DataFrame({"trial": np.arange(4)})
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"event": 1},
        tmin=-0.1,
        metadata=metadata,
        verbose=False,
    )
    cleaned = BSSCCA(n_remove=2).fit_transform(epochs)

    assert isinstance(cleaned, mne.BaseEpochs)
    np.testing.assert_array_equal(cleaned.events, epochs.events)
    assert cleaned.event_id == epochs.event_id
    assert cleaned.tmin == epochs.tmin
    pd.testing.assert_frame_equal(cleaned.metadata, epochs.metadata)
    assert cleaned.get_data(copy=False).shape == data.shape


def test_mne_evoked_round_trip_preserves_identity_fields(rng):
    """Evoked output keeps comment, nave, and timing."""
    mne = pytest.importorskip("mne")
    info = mne.create_info(["EEG0", "EEG1", "EEG2", "EEG3"], 200.0, "eeg")
    evoked = mne.EvokedArray(
        rng.standard_normal((4, 400)),
        info,
        tmin=-0.1,
        comment="condition",
        nave=12,
        verbose=False,
    )
    cleaned = BSSCCA(n_remove=1).fit_transform(evoked)
    assert isinstance(cleaned, mne.Evoked)
    assert cleaned.comment == evoked.comment
    assert cleaned.nave == evoked.nave
    assert cleaned.first == evoked.first
    assert cleaned.data.shape == evoked.data.shape


def test_mne_channel_names_must_match_fit(muscle_data):
    """Transforming a different channel layout fails loudly."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    names = [f"EEG{i:02d}" for i in range(observed.shape[0])]
    train = mne.io.RawArray(
        observed, mne.create_info(names, sfreq, "eeg"), verbose=False
    )
    renamed = mne.io.RawArray(
        observed,
        mne.create_info([f"X{i:02d}" for i in range(observed.shape[0])], sfreq, "eeg"),
        verbose=False,
    )
    estimator = BSSCCA(n_remove=3).fit(train)
    with pytest.raises(ValueError, match="missing required channels"):
        estimator.transform(renamed)


def test_mne_feature_names_are_recorded(muscle_data):
    """Fitting on an MNE object records the selected channel names."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    names = [f"EEG{i:02d}" for i in range(observed.shape[0])]
    raw = mne.io.RawArray(observed, mne.create_info(names, sfreq, "eeg"), verbose=False)
    estimator = BSSCCA(n_remove=3).fit(raw)
    assert estimator.feature_names_in_ == tuple(names)


def test_mne_sfreq_conflict_is_rejected(muscle_data):
    """A declared sfreq must agree with the container's."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    raw = mne.io.RawArray(
        observed,
        mne.create_info([f"EEG{i}" for i in range(observed.shape[0])], sfreq, "eeg"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="disagrees"):
        BSSCCA(lag_seconds=0.004, sfreq=100.0, n_remove=1).fit(raw)


def test_transform_sfreq_must_match_the_fitted_rate(muscle_data):
    """Evaluation data recorded at another rate is rejected."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    names = [f"EEG{i}" for i in range(observed.shape[0])]
    train = mne.io.RawArray(
        observed, mne.create_info(names, sfreq, "eeg"), verbose=False
    )
    other = mne.io.RawArray(
        observed, mne.create_info(names, sfreq * 2, "eeg"), verbose=False
    )
    estimator = BSSCCA(n_remove=3).fit(train)
    with pytest.raises(ValueError, match="transform sfreq"):
        estimator.transform(other)


def test_mne_lag_seconds_uses_container_sfreq(muscle_data):
    """MNE inputs supply their own sampling rate for a physical lag."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    raw = mne.io.RawArray(
        observed,
        mne.create_info([f"EEG{i}" for i in range(observed.shape[0])], sfreq, "eeg"),
        verbose=False,
    )
    estimator = BSSCCA(lag_seconds=2.0 / sfreq, n_remove=3).fit(raw)
    assert estimator.lag_samples_ == 2
    assert estimator.sfreq_ == pytest.approx(sfreq)


def _drift_and_muscle(rng, n_times=4000, sfreq=200.0):
    """Mixture with one strongly autocorrelated source and one white source."""
    t = np.arange(n_times) / sfreq
    drift = np.sin(2 * np.pi * 0.3 * t)
    muscle = rng.standard_normal(n_times)
    brain = np.sin(2 * np.pi * 10.0 * t)
    mixing = rng.standard_normal((8, 3))
    return mixing @ np.vstack([drift, muscle, brain]), drift, muscle


def test_reject_high_drops_the_autocorrelated_end():
    """``reject='high'`` removes drift; ``reject='low'`` removes muscle."""
    rng = np.random.default_rng(0)
    observed, drift, muscle = _drift_and_muscle(rng)

    low, _ = compute_bss_cca(observed, lag_samples=1, n_remove=2, reject="low")
    high, _ = compute_bss_cca(observed, lag_samples=1, n_remove=2, reject="high")

    # Dropping the low-autocorrelation end leaves the drift behind.
    assert abs(np.corrcoef(low[0], drift)[0, 1]) > 0.9
    assert abs(np.corrcoef(low[0], muscle)[0, 1]) < 0.1
    # Dropping the high-autocorrelation end leaves the muscle behind.
    assert abs(np.corrcoef(high[0], muscle)[0, 1]) > 0.9
    assert abs(np.corrcoef(high[0], drift)[0, 1]) < 0.1


def test_reject_defaults_to_low_and_is_backward_compatible():
    """Omitting ``reject`` reproduces the previous behaviour exactly."""
    rng = np.random.default_rng(1)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    without, _ = compute_bss_cca(observed, lag_samples=1, n_remove=2)
    explicit, _ = compute_bss_cca(observed, lag_samples=1, n_remove=2, reject="low")
    np.testing.assert_allclose(without, explicit)


def test_reject_threshold_selects_opposite_ends():
    """The two modes remove disjoint component sets.

    Not a strict partition: a component whose correlation sits exactly on the
    threshold is kept by both, since ``low`` keeps ``rho >= t`` and ``high``
    keeps ``rho <= t``. Disjointness of the *removed* sets is the real invariant.
    """
    rng = np.random.default_rng(2)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    low = BSSCCA(rho_threshold=0.5, reject="low").fit(observed)
    high = BSSCCA(rho_threshold=0.5, reject="high").fit(observed)
    assert low.n_removed_ >= 1 and high.n_removed_ >= 1
    n_components = low.n_kept_ + low.n_removed_
    assert low.n_removed_ + high.n_removed_ <= n_components


def test_reject_rejects_unknown_value():
    rng = np.random.default_rng(3)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    with pytest.raises(ValueError, match="reject must be 'low' or 'high'"):
        compute_bss_cca(observed, lag_samples=1, n_remove=1, reject="sideways")


def test_threshold_on_rsq_matches_squared_rho():
    """``rsq`` thresholds the squared correlation, not the correlation itself."""
    rng = np.random.default_rng(4)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    # rho >= sqrt(0.36) == 0.6 selects the same set as rho**2 >= 0.36.
    on_rho = BSSCCA(rho_threshold=0.6, threshold_on="rho").fit(observed)
    on_rsq = BSSCCA(rho_threshold=0.36, threshold_on="rsq").fit(observed)
    assert on_rho.n_removed_ == on_rsq.n_removed_
    assert on_rho.n_kept_ == on_rsq.n_kept_


def test_threshold_on_defaults_to_rho():
    """Omitting ``threshold_on`` leaves the correlation scale unchanged."""
    rng = np.random.default_rng(5)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    without, _ = compute_bss_cca(observed, lag_samples=1, rho_threshold=0.5)
    explicit, _ = compute_bss_cca(
        observed, lag_samples=1, rho_threshold=0.5, threshold_on="rho"
    )
    np.testing.assert_allclose(without, explicit)


def test_threshold_on_rejects_unknown_value():
    rng = np.random.default_rng(6)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    with pytest.raises(ValueError, match="threshold_on must be 'rho' or 'rsq'"):
        compute_bss_cca(observed, lag_samples=1, rho_threshold=0.5, threshold_on="r2")


def test_estimator_forwards_reject_and_threshold_on():
    """The estimator must honour both knobs, not just the functional API.

    Regression: ``BSSCCA.__init__`` stored ``reject``/``threshold_on`` but
    ``fit`` did not pass them to ``compute_bss_cca``, so the estimator silently
    used the defaults -- a threshold meant as r-squared was applied to rho
    instead, removing far fewer components than intended.
    """
    rng = np.random.default_rng(11)
    observed, _drift, _muscle = _drift_and_muscle(rng)

    # rho ordering is fixed, so thresholding rho**2 at t must remove at least as
    # many components as thresholding rho at the same t.
    on_rho = BSSCCA(rho_threshold=0.59, threshold_on="rho").fit(observed)
    on_rsq = BSSCCA(rho_threshold=0.59, threshold_on="rsq").fit(observed)
    assert on_rsq.n_removed_ >= on_rho.n_removed_
    assert (on_rsq.n_removed_, on_rho.n_removed_) != (0, 0)

    # And the estimator must agree with the function it delegates to.
    for kwargs in (
        {"reject": "high"},
        {"threshold_on": "rsq"},
        {"reject": "high", "threshold_on": "rsq"},
    ):
        est = BSSCCA(rho_threshold=0.59, **kwargs).fit(observed)
        _cleaned, info = compute_bss_cca(observed, rho_threshold=0.59, **kwargs)
        np.testing.assert_array_equal(est.kept_mask_, info["kept_mask"])
