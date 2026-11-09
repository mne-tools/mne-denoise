"""Tests for the mne_denoise.bss_cca module (reference-free BSS-CCA)."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._blending import overlap_add_combine
from mne_denoise.bss_cca import BSSCCA, _lagged_pairs, _segment_bounds, compute_bss_cca

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
    """Filters and patterns compose into a left-applied sensor operator."""
    data = rng.standard_normal((7, 2000))
    cleaned, info = compute_bss_cca(data, n_remove=2, preserve_mean=False)
    operator = info["cleaning_matrix"]
    filters = info["filters"]
    patterns = info["patterns"]
    assert operator.shape == (7, 7)
    assert filters.shape[1] == data.shape[0]
    assert patterns.shape[0] == data.shape[0]
    centered = data - data.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(cleaned, operator @ centered, atol=1e-10)
    np.testing.assert_allclose(
        operator,
        patterns @ (info["kept_mask"][:, np.newaxis] * filters),
        atol=1e-10,
    )


def test_correlations_are_descending_and_selection_drops_the_tail(muscle_data):
    """Components are ordered by decreasing correlation; n_remove drops the tail."""
    observed, _clean, _sfreq = muscle_data
    _cleaned, info = compute_bss_cca(observed, n_remove=3)
    _repeat_cleaned, repeat_info = compute_bss_cca(observed, n_remove=3)
    assert np.all(np.diff(info["correlations"]) <= 1e-12)
    assert not info["kept_mask"][-3:].any()
    assert info["kept_mask"][:-3].all()
    np.testing.assert_allclose(info["correlations"], repeat_info["correlations"])
    np.testing.assert_array_equal(info["kept_mask"], repeat_info["kept_mask"])
    rho = info["correlations"]
    threshold = 0.5 * (rho[-4] + rho[-3])
    _, threshold_info = compute_bss_cca(observed, rho_threshold=threshold)
    np.testing.assert_array_equal(threshold_info["kept_mask"], info["kept_mask"])


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
    """The per-component filter-alignment diagnostic is finite and nontrivial."""
    observed, _clean, _sfreq = muscle_data
    _cleaned, info = compute_bss_cca(observed, n_remove=3)
    assert info["filter_asymmetry"].shape == info["correlations"].shape
    assert np.all(np.isfinite(info["filter_asymmetry"]))
    assert np.any(info["filter_asymmetry"] > 1e-8)


# ---------------------------------------------------------------------------
# Lag semantics
# ---------------------------------------------------------------------------


def test_lag_semantics(rng):
    """The default, sample-based, and physical lag declarations agree."""
    data = rng.standard_normal((5, 1000))
    _cleaned, info = compute_bss_cca(data, n_remove=1)
    assert info["lag_samples"] == 1
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


def test_selection_rule_must_be_explicit(rng):
    """Neither a silent default nor two competing rules are accepted."""
    data = rng.standard_normal((5, 1000))
    for kwargs in (
        {},
        {"n_remove": 1, "rho_threshold": 0.9},
        {"n_remove": 1, "slope_threshold": 0.0, "sfreq": SFREQ},
    ):
        with pytest.raises(ValueError, match="exactly one of"):
            compute_bss_cca(data, **kwargs)


def test_unreachable_threshold_removes_everything(rng, caplog):
    """An unreachable threshold has an explicit all-components outcome."""
    data = rng.standard_normal((6, 3000))
    for kwargs, message in (
        ({"rho_threshold": 0.9}, "rho_threshold"),
        ({"slope_threshold": -5.0, "sfreq": SFREQ}, "spectral slope above"),
    ):
        caplog.clear()
        with caplog.at_level("WARNING", logger="mne_denoise"):
            cleaned, info = compute_bss_cca(data, preserve_mean=False, **kwargs)
        assert info["n_kept"] == 0
        np.testing.assert_allclose(cleaned, 0.0, atol=1e-9)
        assert message in caplog.text


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


def test_invalid_lag_declarations(rng):
    """Lag scalars reject invalid, non-finite, and out-of-range values."""
    data = rng.standard_normal((4, 1000))
    cases = [
        ({"lag_samples": 0}, ValueError, "lag_samples"),
        ({"lag_samples": 1.5}, TypeError, "lag_samples"),
        ({"lag_samples": 5000}, ValueError, "leaves no paired samples"),
        ({"lag_samples": 1, "lag_seconds": 0.1}, ValueError, "at most one"),
        ({"lag_seconds": 0.1}, ValueError, "sfreq is required"),
        ({"lag_seconds": 1e-9, "sfreq": SFREQ}, ValueError, "less than one sample"),
        ({"lag_seconds": 0.1, "sfreq": 0.0}, ValueError, "sfreq must be a positive"),
    ]
    for kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            compute_bss_cca(data, n_remove=1, **kwargs)


def test_invalid_selection_parameters(rng):
    """Selection scalars are validated the same way as the rest of the package."""
    data = rng.standard_normal((4, 1000))
    cases = [
        ({"n_remove": -1}, ValueError, "non-negative"),
        ({"n_remove": 1.5}, TypeError, "n_remove"),
        ({"rho_threshold": 1.5}, ValueError, "between 0 and 1"),
        ({"rho_threshold": np.nan}, TypeError, "rho_threshold"),
        ({"slope_threshold": 0.0}, ValueError, "sfreq is required"),
        ({"slope_threshold": 0.0, "sfreq": 100.0}, ValueError, "Nyquist"),
        (
            {"slope_threshold": 0.0, "sfreq": SFREQ, "slope_band": 5.0},
            TypeError,
            "pair",
        ),
        (
            {"slope_threshold": 0.0, "sfreq": SFREQ, "slope_band": (50.0, 5.0)},
            ValueError,
            "fmin < fmax",
        ),
        (
            {"slope_threshold": 0.0, "sfreq": SFREQ, "slope_band": (10.0, 10.2)},
            ValueError,
            "bin",
        ),
        (
            {"slope_threshold": 0.0, "sfreq": SFREQ, "reject": "high"},
            ValueError,
            "reject='high' cannot",
        ),
    ]
    for kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            compute_bss_cca(data, **kwargs)


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


# ---------------------------------------------------------------------------
# Block-wise operation
# ---------------------------------------------------------------------------


def test_segment_bounds_tile_and_cover_a_ragged_tail():
    """Contiguous blocks tile the recording and extend the final tail."""
    bounds = _segment_bounds(1000, n_block=250, hop=250)
    own = [(own_start, own_end) for _e0, _e1, own_start, own_end in bounds]
    assert own == [(0, 250), (250, 500), (500, 750), (750, 1000)]
    ragged = _segment_bounds(900, n_block=250, hop=250)
    # The tail block overlaps its neighbour on 650-750 and takes over at 700.
    assert ragged[-2:] == [(500, 750, 500, 700), (650, 900, 700, 900)]


@pytest.mark.parametrize(
    ("n_times", "n_block", "hop"),
    [(9000, 3000, 1500), (9700, 3000, 1500), (5000, 1000, 250), (5000, 1000, 1000)],
)
def test_overlapping_blocks_blend_without_jumps(n_times, n_block, hop):
    """Owned spans tile the recording and the blend weights change smoothly.

    Each block outputs a constant equal to its index, so the blended signal is
    the local mixing weight. A block that took full weight at its first sample
    while its predecessor still had full weight would jump by 0.5 there.
    """
    bounds = _segment_bounds(n_times, n_block=n_block, hop=hop)
    owned = [(own_start, own_end) for _e0, _e1, own_start, own_end in bounds]
    assert owned[0][0] == 0
    assert owned[-1][1] == n_times
    assert all(a[1] == b[0] for a, b in zip(owned[:-1], owned[1:]))
    for ext_start, ext_end, own_start, own_end in bounds:
        assert ext_start <= own_start <= own_end <= ext_end

    chunks = [
        {
            "data": np.full((1, ext_end - ext_start), float(index)),
            "ext_start": ext_start,
            "ext_end": ext_end,
            "start": own_start,
            "end": own_end,
        }
        for index, (ext_start, ext_end, own_start, own_end) in enumerate(bounds)
    ]
    blended = overlap_add_combine((1, n_times), chunks)[0]
    overlap = n_block - hop
    # Each raised-cosine ramp spans half the overlap, so one weight changes by
    # at most pi / overlap per sample; a factor 2 covers more than two blocks
    # sharing a sample at high overlap. Without overlap the blocks are simply
    # concatenated.
    max_step = 2 * np.pi / overlap if overlap else 1.0
    assert np.abs(np.diff(blended)).max() <= max_step + 1e-12


def test_one_block_covering_everything_equals_the_global_fit(rng):
    """A block longer than the recording degenerates to one operator."""
    data = rng.standard_normal((6, 2000))
    blocked, info = compute_bss_cca(data, sfreq=SFREQ, segment_len=1000.0, n_remove=2)
    globally, _ = compute_bss_cca(data, n_remove=2)
    np.testing.assert_allclose(blocked, globally, atol=1e-10)
    assert info["n_blocks"] == 1

    events = []
    blocked_with_callback, _ = compute_bss_cca(
        data,
        sfreq=SFREQ,
        segment_len=1000.0,
        n_remove=2,
        callback=events.append,
    )
    assert len(events) == 1
    assert events[0].method == "bss_cca"
    assert events[0].stage == "block"
    assert events[0].current == 1
    assert events[0].total == 1
    assert events[0].component is None
    assert events[0].metric == pytest.approx(float(np.mean(info["correlations"][0])))
    np.testing.assert_allclose(blocked_with_callback, blocked, atol=1e-10)


def test_block_wise_diagnostics_are_per_block(rng):
    """Blocked mode reports one operator and span per block."""
    data = rng.standard_normal((6, int(SFREQ * 30)))
    _cleaned, info = compute_bss_cca(data, sfreq=SFREQ, segment_len=10.0, n_remove=1)
    assert info["n_blocks"] == 3
    assert len(info["cleaning_matrix"]) == 3
    assert len(info["correlations"]) == 3
    assert info["spans"] == ((0, 2500), (2500, 5000), (5000, 7500))


def test_segmented_callback_reports_completed_operator_blocks(rng):
    """Segmented progress follows fitted operators, including overlap."""
    data = rng.standard_normal((5, 1500))
    overlap = 0.5
    events = []
    _cleaned, info = compute_bss_cca(
        data,
        sfreq=SFREQ,
        segment_len=2.0,
        overlap=0.5,
        n_remove=1,
        callback=events.append,
    )

    n_block = int(SFREQ * 2.0)
    hop = max(1, n_block - int(np.floor(overlap * n_block + 0.5)))
    bounds = _segment_bounds(data.shape[-1], n_block=n_block, hop=hop)

    assert len(events) == len(bounds) == info["n_blocks"]
    assert [(event.current, event.total) for event in events] == [
        (index, len(bounds)) for index in range(1, len(bounds) + 1)
    ]
    assert all(event.method == "bss_cca" for event in events)
    assert all(event.stage == "block" for event in events)
    assert all(event.component is None for event in events)
    np.testing.assert_allclose(
        [event.metric for event in events],
        [float(np.mean(correlations)) for correlations in info["correlations"]],
    )
    assert info["spans"] == tuple(bound[:2] for bound in bounds)


def test_segmented_callback_exception_stops_after_first_block(rng):
    """A callback interruption leaves the segmented topology at block one."""
    data = rng.standard_normal((5, 1500))

    class CallbackSentinel(RuntimeError):
        pass

    error = CallbackSentinel("BSS-CCA callback failed")
    events = []

    def callback(event):
        events.append(event)
        raise error

    with pytest.raises(CallbackSentinel):
        compute_bss_cca(
            data,
            sfreq=SFREQ,
            segment_len=2.0,
            n_remove=1,
            callback=callback,
        )

    assert [(event.method, event.stage, event.current) for event in events] == [
        ("bss_cca", "block", 1)
    ]


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


def test_invalid_blocking_parameters(rng):
    """Blocking parameters are validated before any decomposition runs."""
    data = rng.standard_normal((5, 2000))
    cases = [
        ({"segment_len": 10.0}, ValueError, "sfreq is required"),
        ({"segment_len": 0.0, "sfreq": SFREQ}, ValueError, "positive"),
        ({"segment_len": 0.001, "sfreq": SFREQ}, ValueError, "use a longer block"),
        ({"overlap": 1.0}, ValueError, r"\[0, 1\)"),
    ]
    for kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            compute_bss_cca(data, n_remove=1, **kwargs)


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
    clean_c = clean - clean.mean(axis=1, keepdims=True)
    observed_c = observed - observed.mean(axis=1, keepdims=True)
    assert _corr(cleaned, clean_c) > 0.95
    assert _corr(cleaned, clean_c) > _corr(observed_c, clean_c)


# ---------------------------------------------------------------------------
# Spectral-slope selection (Janani et al., 2018)
# ---------------------------------------------------------------------------


def test_slope_rule_removes_the_flat_spectrum_sources(realistic_eeg):
    """Components with flat (muscle-like) spectra are removed, brain kept."""
    observed, clean, sfreq = realistic_eeg
    cleaned, info = compute_bss_cca(
        observed, sfreq=sfreq, slope_threshold=0.0, preserve_mean=False
    )
    assert info["slope_band"] == (7.0, 75.0)
    assert info["n_removed"] == 3
    np.testing.assert_array_equal(info["kept_mask"], info["slopes"] <= 0.0)
    assert _band_power(cleaned, sfreq, 50.0, 120.0) < 0.05 * _band_power(
        observed, sfreq, 50.0, 120.0
    )
    assert _corr(cleaned, clean - clean.mean(axis=1, keepdims=True)) > 0.95

    model = BSSCCA(sfreq=sfreq, slope_threshold=0.0).fit(observed)
    np.testing.assert_array_equal(model.kept_mask_, info["kept_mask"])
    np.testing.assert_allclose(model.slopes_, info["slopes"])

    # Epoched input: spectra are computed within epochs and averaged.
    epochs = observed.reshape(21, 5, -1).transpose(1, 0, 2)
    _cleaned, epoched = compute_bss_cca(epochs, sfreq=sfreq, slope_threshold=0.0)
    assert epoched["n_removed"] == 3


def test_slope_rule_is_independent_of_the_correlation_ranking(rng):
    """A strongly autocorrelated source with a rising spectrum is removed.

    A 70 Hz oscillation ranks above white noise by correlation, so a rank cut
    could not remove it without brain-like sources ranked below it.
    """
    n_times = int(SFREQ * 20)
    t = np.arange(n_times) / SFREQ
    pink = np.cumsum(rng.standard_normal((3, n_times)), axis=1)
    pink -= pink.mean(axis=1, keepdims=True)
    hf = np.sin(2 * np.pi * 70 * t) * (1 + 0.5 * rng.standard_normal(n_times))
    sources = np.vstack([pink / pink.std(axis=1, keepdims=True), hf])
    data = rng.standard_normal((10, 4)) @ sources
    _cleaned, info = compute_bss_cca(data, sfreq=SFREQ, slope_threshold=0.0)
    assert info["n_removed"] == 1
    assert np.all(info["slopes"][info["kept_mask"]] < -1.0)


def test_slope_matches_the_reference_welch_fit(rng):
    """Slopes use 1 s Hann Welch spectra and a log-log line, as in the reference."""
    from scipy.signal import welch

    data = np.cumsum(rng.standard_normal((4, int(SFREQ * 20))), axis=1)
    data += 0.5 * rng.standard_normal(data.shape)
    _cleaned, info = compute_bss_cca(data, sfreq=SFREQ, slope_threshold=0.0)
    sources = info["filters"] @ (data - info["training_mean"])
    freqs, psd = welch(sources, fs=SFREQ, nperseg=int(SFREQ))
    band = (freqs >= 7.0) & (freqs <= 75.0)
    expected = np.polyfit(np.log(freqs[band]), np.log(psd[:, band]).T, 1)[0]
    np.testing.assert_allclose(info["slopes"], expected, rtol=1e-10)


def test_combined_rule_removes_the_union_of_both_stages(realistic_eeg):
    """rho_threshold with slope_threshold is Janani et al.'s two-stage rule."""
    observed, _clean, sfreq = realistic_eeg
    _c, by_rho = compute_bss_cca(observed, sfreq=sfreq, rho_threshold=0.9)
    _c, by_slope = compute_bss_cca(observed, sfreq=sfreq, slope_threshold=0.0)
    _c, both = compute_bss_cca(
        observed, sfreq=sfreq, rho_threshold=0.9, slope_threshold=0.0
    )
    np.testing.assert_array_equal(
        both["kept_mask"], by_rho["kept_mask"] & by_slope["kept_mask"]
    )


# ---------------------------------------------------------------------------
# MNE integration
# ---------------------------------------------------------------------------


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
    assert np.all(low.correlations_[low.kept_mask_] >= 0.5)
    assert np.all(low.correlations_[~low.kept_mask_] < 0.5)
    assert np.all(high.correlations_[high.kept_mask_] <= 0.5)
    assert np.all(high.correlations_[~high.kept_mask_] > 0.5)


def test_selection_scale_options_reject_unknown_value():
    """Selection direction and scale accept only their documented values."""
    observed, _drift, _muscle = _drift_and_muscle(np.random.default_rng(3))
    cases = [
        ({"reject": "sideways", "n_remove": 1}, "reject must be 'low' or 'high'"),
        (
            {"threshold_on": "r2", "rho_threshold": 0.5},
            "threshold_on must be 'rho' or 'rsq'",
        ),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            compute_bss_cca(observed, lag_samples=1, **kwargs)


def test_threshold_on_rsq_matches_squared_rho():
    """``rsq`` thresholds the squared correlation, not the correlation itself."""
    rng = np.random.default_rng(4)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    # rho >= sqrt(0.36) == 0.6 selects the same set as rho**2 >= 0.36.
    on_rho = BSSCCA(rho_threshold=0.6, threshold_on="rho").fit(observed)
    on_rsq = BSSCCA(rho_threshold=0.36, threshold_on="rsq").fit(observed)
    np.testing.assert_array_equal(on_rho.kept_mask_, on_rsq.kept_mask_)


def test_threshold_on_defaults_to_rho():
    """Omitting ``threshold_on`` leaves the correlation scale unchanged."""
    rng = np.random.default_rng(5)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    without, _ = compute_bss_cca(observed, lag_samples=1, rho_threshold=0.5)
    explicit, _ = compute_bss_cca(
        observed, lag_samples=1, rho_threshold=0.5, threshold_on="rho"
    )
    np.testing.assert_allclose(without, explicit)


def test_estimator_forwards_reject_and_threshold_on():
    """The estimator and functional API select the same BSS-CCA components."""
    rng = np.random.default_rng(11)
    observed, _drift, _muscle = _drift_and_muscle(rng)
    kwargs = {"rho_threshold": 0.59, "reject": "high", "threshold_on": "rsq"}
    estimator = BSSCCA(**kwargs).fit(observed)
    _cleaned, info = compute_bss_cca(observed, **kwargs)
    np.testing.assert_array_equal(estimator.kept_mask_, info["kept_mask"])


# ---------------------------------------------------------------------------
# Bad channels
# ---------------------------------------------------------------------------


def test_bad_channels_are_left_out_of_the_decomposition(muscle_data):
    """Bad channels neither shape the operator nor get modified."""
    mne = pytest.importorskip("mne")
    observed, _clean, sfreq = muscle_data
    info = mne.create_info([f"EEG{i:02d}" for i in range(21)], sfreq, "eeg")
    raw = mne.io.RawArray(observed * 1e-6, info, verbose=False)
    raw._data[3] *= 1e3  # a broken electrode
    raw.info["bads"] = ["EEG03"]

    model = BSSCCA(n_remove=3).fit(raw)
    cleaned = model.transform(raw)
    assert "EEG03" not in model.feature_names_in_
    np.testing.assert_array_equal(
        cleaned.get_data(picks="EEG03"), raw.get_data(picks="EEG03")
    )
    assert cleaned.info["bads"] == ["EEG03"]

    good = raw.copy().drop_channels(["EEG03"])
    reference = BSSCCA(n_remove=3).fit_transform(good)
    np.testing.assert_allclose(
        cleaned.get_data(picks=good.ch_names), reference.get_data(), rtol=1e-10, atol=0
    )
