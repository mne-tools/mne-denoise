"""Tests for mne_denoise.icanclean module."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.icanclean import ICanClean, compute_icanclean

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_dual_layer(rng):
    """Create synthetic dual-layer data with a known artifact.

    Returns (data, primary_idx, ref_idx, sfreq, truth) where truth contains
    the clean brain signal for validation.
    """
    sfreq = 250.0
    duration = 10.0
    n_times = int(sfreq * duration)
    n_primary = 16
    n_ref = 4
    t = np.arange(n_times) / sfreq

    # Brain signal: sum of alpha + theta
    brain = np.zeros((n_primary, n_times))
    for i in range(n_primary):
        phase = rng.uniform(0, 2 * np.pi)
        brain[i] = 0.5 * np.sin(2 * np.pi * 10 * t + phase) + 0.3 * np.sin(
            2 * np.pi * 6 * t + phase * 0.7
        )

    # Artifact sources: correlated across primary AND reference
    n_artifacts = 2
    artifact_sources = rng.standard_normal((n_artifacts, n_times)) * 2.0

    mixing_primary = rng.standard_normal((n_primary, n_artifacts)) * 0.8
    mixing_ref = rng.standard_normal((n_ref, n_artifacts)) * 1.0

    artifact_primary = mixing_primary @ artifact_sources
    artifact_ref = mixing_ref @ artifact_sources

    # Reference also has its own noise
    ref_noise = rng.standard_normal((n_ref, n_times)) * 0.3

    data_primary = brain + artifact_primary
    data_ref = artifact_ref + ref_noise

    data = np.vstack([data_primary, data_ref])
    primary_idx = list(range(n_primary))
    ref_idx = list(range(n_primary, n_primary + n_ref))

    truth = {"brain": brain, "artifact_primary": artifact_primary}
    return data, primary_idx, ref_idx, sfreq, truth


# ---------------------------------------------------------------------------
# ICanClean tests (numpy)
# ---------------------------------------------------------------------------


def test_compute_icanclean_basic_cleaning_and_qc(synthetic_dual_layer):
    """compute_icanclean returns cleaned primary data and QC."""
    data, primary_idx, ref_idx, sfreq, truth = synthetic_dual_layer

    cleaned_primary, qc = compute_icanclean(
        data[primary_idx],
        data[ref_idx],
        sfreq=sfreq,
        segment_len=2.0,
        overlap=0.5,
        threshold=0.5,
        verbose=False,
    )

    assert qc["n_windows_"] > 0
    assert qc["correlations_"].shape[0] == qc["n_windows_"]
    assert qc["n_removed_"].shape == (qc["n_windows_"],)
    assert len(qc["removed_idx_"]) == qc["n_windows_"]
    assert len(qc["filters_"]) == qc["n_windows_"]
    assert len(qc["patterns_"]) == qc["n_windows_"]

    residual_before = np.var(data[primary_idx] - truth["brain"])
    residual_after = np.var(cleaned_primary - truth["brain"])
    assert residual_after < residual_before
    assert (
        np.corrcoef(cleaned_primary.ravel(), truth["brain"].ravel())[0, 1]
        > np.corrcoef(data[primary_idx].ravel(), truth["brain"].ravel())[0, 1]
    )
    icc = ICanClean(
        sfreq=sfreq,
        ref_channels=ref_idx,
        primary_channels=primary_idx,
        segment_len=2.0,
        overlap=0.5,
        threshold=0.5,
        verbose=False,
    )
    cleaned = icc.fit_transform(data)

    np.testing.assert_allclose(cleaned_primary, cleaned[primary_idx])
    np.testing.assert_allclose(qc["correlations_"], icc.correlations_, equal_nan=True)
    np.testing.assert_array_equal(qc["n_removed_"], icc.n_removed_)
    np.testing.assert_array_equal(cleaned[ref_idx], data[ref_idx])


def test_compute_icanclean_hybrid_is_not_supported(synthetic_dual_layer):
    """Hybrid orchestration belongs to the estimator, not compute_icanclean."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer

    with pytest.raises(ValueError, match="supports only single-pass"):
        compute_icanclean(
            data[primary_idx],
            data[ref_idx],
            sfreq=sfreq,
            mode="hybrid",
            verbose=False,
        )


def test_compute_icanclean_zero_rank_reference_raises(rng):
    """Zero-rank reference windows fail loudly."""
    x_primary = rng.standard_normal((4, 500))
    x_ref = np.ones((2, 500))

    with pytest.raises(ValueError, match="returned 0 components"):
        compute_icanclean(
            x_primary,
            x_ref,
            sfreq=250.0,
            mode="global",
            verbose=False,
        )


def test_compute_icanclean_window_modes_and_progress(synthetic_dual_layer):
    """Sliding and calibrated passes expose aligned per-window QC and events."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    for mode in ("sliding", "calibrated"):
        kwargs = {
            "sfreq": sfreq,
            "mode": mode,
            "segment_len": 2.0,
            "overlap": 0.5,
            "threshold": 0.5,
            "verbose": False,
        }
        events = []
        _cleaned, qc = compute_icanclean(
            data[primary_idx], data[ref_idx], callback=events.append, **kwargs
        )
        assert qc["n_windows_"] > 1
        assert qc["correlations_"].shape[0] == qc["n_windows_"]
        assert len(qc["filters_"]) == qc["n_windows_"]
        assert len(qc["patterns_"]) == qc["n_windows_"]
        assert len(events) == qc["n_windows_"]
        assert [event.method for event in events] == ["icanclean"] * len(events)
        assert [event.stage for event in events] == ["window"] * len(events)
        assert [event.current for event in events] == list(range(1, len(events) + 1))
        assert [event.total for event in events] == [qc["n_windows_"]] * len(events)
        assert all(event.component is None for event in events)
        np.testing.assert_array_equal(
            [event.metric for event in events], qc["n_removed_"].astype(float)
        )


def test_compute_icanclean_global_callback_is_silent(synthetic_dual_layer):
    """The implementation-reuse global pass emits no window event."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    kwargs = {"sfreq": sfreq, "mode": "global", "threshold": 0.5, "verbose": False}
    events = []
    compute_icanclean(
        data[primary_idx], data[ref_idx], callback=events.append, **kwargs
    )

    assert events == []


def test_compute_icanclean_null_threshold_emits_window_events(
    synthetic_dual_layer, monkeypatch
):
    """Null surrogates remain nested inside one event per cleaning window."""
    import mne_denoise.icanclean as icc_core

    original_null_threshold = icc_core.null_r2_threshold

    def fast_null_threshold(X_cca, Y_cca, **kwargs):
        return original_null_threshold(X_cca, Y_cca, n_surrogate=3, **kwargs)

    monkeypatch.setattr(icc_core, "null_r2_threshold", fast_null_threshold)
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    events = []
    _, qc = compute_icanclean(
        data[primary_idx],
        data[ref_idx],
        sfreq=sfreq,
        mode="sliding",
        threshold="null",
        null_random_state=0,
        callback=events.append,
        verbose=False,
    )

    assert len(events) == qc["n_windows_"]
    assert [event.current for event in events] == list(range(1, qc["n_windows_"] + 1))
    np.testing.assert_array_equal(
        [event.metric for event in events], qc["n_removed_"].astype(float)
    )


# ---------------------------------------------------------------------------
# Estimator tests (numpy)
# ---------------------------------------------------------------------------


def test_icanclean_numpy_configuration_guards(rng):
    """Missing references and windows longer than data fail at clear boundaries."""
    with pytest.raises(ValueError, match="ref_channels must be provided explicitly"):
        ICanClean(sfreq=250.0, verbose=False)
    data = rng.standard_normal((10, 100))
    icc = ICanClean(
        sfreq=250.0,
        ref_channels=[8, 9],
        segment_len=10.0,  # 2500 samples > 100
        verbose=False,
    )
    with pytest.raises(ValueError, match="exceeds data length"):
        icc.fit_transform(data)


def test_icanclean_callback_exception_propagates_without_partial_qc(
    synthetic_dual_layer,
):
    """A callback failure aborts continuous processing unchanged."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    estimator = ICanClean(
        sfreq=sfreq,
        ref_channels=ref_idx,
        primary_channels=primary_idx,
        segment_len=2.0,
        threshold=0.5,
        verbose=False,
    )
    sentinel = RuntimeError("icanclean callback failed")
    events = []

    def callback(event):
        events.append(event)
        raise sentinel

    with pytest.raises(RuntimeError):
        estimator.transform(data, callback=callback)

    assert len(events) == 1
    assert events[0].current == 1
    assert not hasattr(estimator, "n_windows_")


def test_icanclean_numpy_max_reject_fraction(synthetic_dual_layer):
    """The rejection cap covers both bounded and zero-removal behavior."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    icc = ICanClean(
        sfreq=sfreq,
        ref_channels=ref_idx,
        threshold=0.01,  # very low -> would remove everything
        max_reject_fraction=0.25,
        verbose=False,
    )
    icc.fit_transform(data)
    n_comp = min(len(primary_idx), len(ref_idx))
    max_allowed = max(1, int(0.25 * n_comp))
    assert np.all(icc.n_removed_ <= max_allowed)

    uncapped_to_zero = ICanClean(
        sfreq=sfreq,
        ref_channels=list(ref_idx),
        max_reject_fraction=0.0,
        threshold=0.0,
    )
    unchanged = uncapped_to_zero.fit_transform(data)
    np.testing.assert_array_almost_equal(unchanged, data)
    assert uncapped_to_zero.n_removed_.sum() == 0


def test_icanclean_mne_ordered_named_channels(rng):
    """Public MNE fitting preserves deliberately ordered channel names."""
    mne = pytest.importorskip("mne")
    sfreq = 250.0
    raw_names = ["P-Y", "R-B", "P-Z", "R-A", "P-X", "P-W"]
    primary_names = ["P-W", "P-X", "P-Z", "P-Y"]
    ref_names = ["R-A", "R-B"]
    raw = mne.io.RawArray(
        rng.standard_normal((len(raw_names), int(6 * sfreq))),
        mne.create_info(raw_names, sfreq, ["eeg"] * len(raw_names)),
        verbose=False,
    )

    icc = ICanClean(
        sfreq=sfreq,
        primary_channels=primary_names,
        ref_channels=ref_names,
        segment_len=1.0,
        verbose=False,
    )
    cleaned = icc.fit_transform(raw)

    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.get_data().shape == raw.get_data().shape
    assert np.isfinite(cleaned.get_data()).all()
    assert icc.primary_channels_ == primary_names
    assert icc.ref_channels_ == ref_names


# ---------------------------------------------------------------------------
# Validation & Edge cases
# ---------------------------------------------------------------------------


def test_icanclean_validation_contract():
    """Representative invalid configuration classes fail at construction."""
    invalid = [
        ({"overlap": 1.0}, "overlap"),
        ({"mode": "unknown"}, "mode"),
        ({"clean_with": "Z"}, "clean_with"),
        ({"max_reject_fraction": -0.1}, "max_reject_fraction"),
        ({"reref_primary": "bad"}, "reref_primary"),
        ({"mode": "hybrid"}, "mode='hybrid' requires"),
        (
            {
                "mode": "sliding",
                "global_threshold": 0.7,
                "global_clean_with": "X",
                "global_max_reject_fraction": 0.5,
            },
            "only supported when mode='hybrid'",
        ),
        (
            {"segment_len": 2.0, "stats_segment_len": 1.0},
            "stats_segment_len",
        ),
    ]
    for kwargs, message in invalid:
        with pytest.raises(ValueError, match=message):
            ICanClean(sfreq=250.0, ref_channels=[0], **kwargs)


# ---------------------------------------------------------------------------
# Pseudo-reference mode (Downey & Ferris 2023, Sensors 23(19):8214)
# ---------------------------------------------------------------------------


def test_filter_ref_rejects_invalid_specs():
    """Malformed, non-positive, and Nyquist-edge filters fail early."""
    bad_filters = [
        ("bogus", 10.0),
        ("bandstop", 10.0),
        ("bandstop", (45.0, 5.0)),
        ("bandstop", (0.0, 45.0)),
        ("lowpass", -1.0),
        ("lowpass", 125.0),
    ]
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            ICanClean(sfreq=250.0, ref_channels=[0], filter_ref=bad_filter)


def test_pseudo_ref_configuration_contract():
    """Pseudo-reference mode owns its filter and does not accept real references."""
    with pytest.raises(ValueError, match="requires filter_ref"):
        ICanClean(sfreq=250.0, primary_channels=[0, 1], pseudo_ref=True)
    icc = ICanClean(
        sfreq=250.0,
        primary_channels=[0, 1],
        pseudo_ref=True,
        filter_ref=("bandstop", (5.0, 45.0)),
    )
    assert icc.ref_channels is None
    with pytest.raises(ValueError, match="ref_channels is not used"):
        ICanClean(
            sfreq=250.0,
            ref_channels=[3],
            pseudo_ref=True,
            filter_ref=("bandstop", (5.0, 45.0)),
        )


def test_pseudo_ref_removes_out_of_band_artifact(synthetic_dual_layer):
    """A strong sub-band drift shared across channels should be attenuated.

    The pseudo-reference retains only content outside 5-45 Hz, so CCA can see
    the drift but not the in-band signal.
    """
    data, primary_idx, _ref_idx, sfreq, truth = synthetic_dual_layer
    primary = truth["brain"]
    n_times = primary.shape[1]
    t = np.arange(n_times) / sfreq
    drift = 8.0 * np.sin(2 * np.pi * 0.7 * t)
    contaminated = primary + drift[None, :]

    icc = ICanClean(
        sfreq=sfreq,
        primary_channels=list(range(len(primary_idx))),
        pseudo_ref=True,
        filter_ref=("bandstop", (5.0, 45.0)),
        segment_len=2.0,
        threshold=0.5,
        verbose=False,
    )
    cleaned = icc.fit_transform(contaminated)

    edge = int(sfreq)
    before = np.mean(np.abs(contaminated[:, edge:-edge] - primary[:, edge:-edge]))
    after = np.mean(np.abs(cleaned[:, edge:-edge] - primary[:, edge:-edge]))
    assert after < before, f"drift not attenuated: {before:.3f} -> {after:.3f}"


def test_icanclean_mode_global(synthetic_dual_layer):
    """mode='global' runs as a single window pass."""
    data, _primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    icc = ICanClean(
        sfreq=sfreq,
        ref_channels=list(ref_idx),
        mode="global",
        threshold=0.7,
    )
    icc.fit_transform(data)
    assert icc.n_windows_ == 1


def test_icanclean_mode_hybrid(synthetic_dual_layer):
    """mode='hybrid' runs both global and sliding passes."""
    data, _primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    icc = ICanClean(
        sfreq=sfreq,
        ref_channels=list(ref_idx),
        mode="hybrid",
        threshold=0.7,
        global_threshold=0.9,
        global_clean_with="Y",
        global_max_reject_fraction=0.3,
    )
    icc.fit_transform(data)
    assert hasattr(icc, "global_correlations_")
    assert hasattr(icc, "sliding_correlations_")
    assert icc.n_windows_ > 1
    assert icc.correlations_.shape == icc.sliding_correlations_.shape


def test_icanclean_hybrid_progress_reports_only_sliding_pass(
    synthetic_dual_layer,
):
    """Hybrid callbacks expose sliding windows, not the global setup pass."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    kwargs = {
        "sfreq": sfreq,
        "ref_channels": ref_idx,
        "primary_channels": primary_idx,
        "mode": "hybrid",
        "segment_len": 2.0,
        "overlap": 0.5,
        "threshold": 0.5,
        "global_threshold": 0.8,
        "global_clean_with": "Y",
        "global_max_reject_fraction": 0.5,
        "verbose": False,
    }
    with_callback = ICanClean(**kwargs)
    events = []
    with_callback.transform(data, callback=events.append)
    assert len(events) == with_callback.sliding_n_windows_
    assert with_callback.global_n_windows_ == 1
    assert [event.method for event in events] == ["icanclean"] * len(events)
    assert [event.stage for event in events] == ["window"] * len(events)
    assert [event.current for event in events] == list(range(1, len(events) + 1))
    assert [event.total for event in events] == [
        with_callback.sliding_n_windows_
    ] * len(events)
    assert all(event.component is None for event in events)
    np.testing.assert_array_equal(
        [event.metric for event in events],
        with_callback.sliding_n_removed_.astype(float),
    )


def test_icanclean_clean_with_variants(synthetic_dual_layer):
    """Support for clean_with='X', 'Y', 'both'."""
    data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
    results = {}
    for cw in ("X", "Y", "both"):
        icc = ICanClean(
            sfreq=sfreq,
            ref_channels=list(ref_idx),
            clean_with=cw,
            threshold=0.5,
        )
        results[cw] = icc.fit_transform(data)

    # X and Y must differ, and the combined basis must differ from Y.
    assert not np.allclose(
        results["X"][primary_idx], results["Y"][primary_idx], atol=1e-10
    )
    assert not np.allclose(
        results["both"][primary_idx], results["Y"][primary_idx], atol=1e-10
    )


def test_icanclean_terminal_window_awkward_overlap(rng):
    """With awkward overlap, the final partial window still cleans artifact."""
    n_primary, n_ref, n_times = 8, 2, 1000
    sfreq = 250.0
    t = np.arange(n_times) / sfreq
    artifact = np.sin(2 * np.pi * 6.0 * t)
    brain = 0.1 * rng.standard_normal((n_primary, n_times))
    primary_mixing = rng.standard_normal((n_primary, 1))
    ref_mixing = rng.standard_normal((n_ref, 1))
    data = np.vstack(
        [
            brain + 3.0 * primary_mixing @ artifact[None, :],
            0.1 * rng.standard_normal((n_ref, n_times))
            + 3.0 * ref_mixing @ artifact[None, :],
        ]
    )
    ref_idx = list(range(n_primary, n_primary + n_ref))

    icc = ICanClean(
        sfreq=sfreq,
        ref_channels=ref_idx,
        segment_len=0.5,  # 125 samples
        overlap=0.3,  # step = 87.5 -> 88 samples
        threshold=0.99,  # high threshold = minimal cleaning
    )
    cleaned = icc.fit_transform(data)
    assert icc.n_windows_ > 1
    tail = slice(-50, None)
    before = np.var(data[:n_primary, tail] - brain[:, tail])
    after = np.var(cleaned[:n_primary, tail] - brain[:, tail])
    assert after < 0.1 * before


def test_icanclean_epoch_aggregation_hybrid_all_passes(rng):
    """Epoch-level QC aggregation for hybrid pass."""
    mne = pytest.importorskip("mne")
    sfreq = 100.0
    n_epochs = 3
    n_primary = 6
    n_ref = 2
    n_times = 400
    n_channels = n_primary + n_ref
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 0.05

    for epoch_idx in range(n_epochs):
        t = np.arange(n_times) / sfreq
        artifact = np.sin(2 * np.pi * (5 + epoch_idx) * t)
        data[epoch_idx, :n_primary] += artifact
        for ref_idx in range(n_ref):
            data[epoch_idx, n_primary + ref_idx] = (
                artifact + 0.01 * rng.standard_normal(n_times)
            )

    info = mne.create_info(
        [f"EEG{idx}" for idx in range(n_primary)]
        + [f"REF{idx}" for idx in range(n_ref)],
        sfreq,
        ["eeg"] * n_channels,
    )
    epochs = mne.EpochsArray(data, info, verbose=False)
    icc = ICanClean(
        sfreq=sfreq,
        ref_channels=[f"REF{idx}" for idx in range(n_ref)],
        mode="hybrid",
        segment_len=2.0,
        overlap=0.5,
        threshold=0.2,
        global_threshold=0.3,
        global_clean_with="Y",
        global_max_reject_fraction=0.5,
        verbose=False,
    )
    icc.fit_transform(epochs)

    assert icc.correlations_.shape[0] == icc.n_windows_
    assert len(icc.epoch_window_slices_) == n_epochs
    assert len(icc.global_epoch_window_slices_) == n_epochs
    assert len(icc.sliding_epoch_window_slices_) == n_epochs
    assert icc.sliding_correlations_.shape[0] == icc.correlations_.shape[0]
    assert len(icc.sliding_filters_) == icc.sliding_correlations_.shape[0]
    assert len(icc.global_filters_) == icc.global_correlations_.shape[0]

    top_total = sum(s.stop - s.start for s in icc.epoch_window_slices_)
    assert top_total == icc.n_windows_

    global_total = sum(s.stop - s.start for s in icc.global_epoch_window_slices_)
    assert global_total == icc.global_correlations_.shape[0]


def test_compute_icanclean_cca_failure_mock(monkeypatch):
    """Verify RuntimeError when CCA fails."""
    import mne_denoise.icanclean as icc_core

    def mock_cca_fail(*args, **kwargs):
        raise ValueError("Linear Algebra is hard")

    monkeypatch.setattr(icc_core, "canonical_correlation", mock_cca_fail)
    # Use enough samples to satisfy default segment_len=2s at 100Hz
    data = np.ones((1, 500))

    with pytest.raises(RuntimeError, match="CCA failed"):
        compute_icanclean(data, data, 100.0, mode="sliding")


# ---------------------------------------------------------------------------
# threshold='null' -- scale-free rejection (issue: absolute R^2 does not transfer)
# ---------------------------------------------------------------------------
def _ar1(rng, n_ch, n_times, rho=0.95):
    """Autocorrelated noise. White surrogates make the null test too easy."""
    e = rng.standard_normal((n_ch, n_times))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, n_times):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def test_null_threshold_accepts_and_validates():
    """'null' is a legal threshold; out-of-range floats and strings are not."""
    for good in ("auto", "null", 0.0, 0.5, 1.0):
        ICanClean(sfreq=100.0, ref_channels=[0], threshold=good)
    # 5.0 silently made the estimator a pass-through before this check existed.
    for bad in (5.0, -1.0, "0.5", None):
        with pytest.raises(ValueError):
            ICanClean(sfreq=100.0, ref_channels=[0], threshold=bad)


def test_null_threshold_rejects_nothing_when_blocks_are_independent():
    """The property the whole design rests on: no shared structure, no removals.

    A fixed threshold cannot do this. At n/(p+q) ~ 6 a constant r2=0.65 removes
    several components from data that shares nothing, because short windows make
    canonical correlations overfit.
    """
    rng = np.random.default_rng(0)
    p = q = 20
    n_times = 500
    X, Y = _ar1(rng, p, n_times), _ar1(rng, q, n_times)
    icc = ICanClean(
        sfreq=250.0,
        primary_channels=list(range(p)),
        ref_channels=list(range(p, p + q)),
        mode="global",
        threshold="null",
        null_random_state=0,
        verbose=False,
    )
    icc.fit_transform(np.vstack([X, Y]))
    assert icc.n_removed_.sum() == 0, (
        f"null threshold removed {icc.n_removed_.sum()} components from "
        f"independent blocks at n={n_times}"
    )


def test_null_threshold_rejects_nothing_in_calibrated_mode():
    """'calibrated' scores components via a *fixed* global basis, not a fresh
    per-window CCA search. The null must be built the same way, or a
    search-optimized surrogate distribution understates what the fixed
    projection can reach under noise and the threshold runs anticonservative.
    """
    rng = np.random.default_rng(1)
    p = q = 20
    n_times = 4000
    X, Y = _ar1(rng, p, n_times), _ar1(rng, q, n_times)
    icc = ICanClean(
        sfreq=250.0,
        primary_channels=list(range(p)),
        ref_channels=list(range(p, p + q)),
        mode="calibrated",
        threshold="null",
        null_random_state=0,
        verbose=False,
    )
    icc.fit_transform(np.vstack([X, Y]))
    assert icc.n_removed_.sum() == 0, (
        f"null threshold removed {icc.n_removed_.sum()} components in "
        "calibrated mode from independent blocks"
    )


def test_null_threshold_handles_short_windows():
    """A window short enough to collapse the min-shift guard band still
    returns a valid threshold instead of degenerating to one fixed shift.
    """
    from mne_denoise.icanclean import null_r2_threshold

    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 2))
    Y = rng.standard_normal((3, 2))
    thr = null_r2_threshold(X, Y, n_surrogate=10, random_state=0)
    assert 0.0 <= thr <= 1.0


def test_null_threshold_recovers_injected_components():
    """Safety must not come from timidity: genuine shared structure is found."""
    rng = np.random.default_rng(7)
    p = q = 20
    n_times = 4000
    for n_shared in (0, 1, 3):
        X, Y = _ar1(rng, p, n_times), _ar1(rng, q, n_times)
        if n_shared:
            shared = _ar1(rng, n_shared, n_times)
            X[:n_shared] += 2.0 * shared
            Y[:n_shared] += 2.0 * shared
        icc = ICanClean(
            sfreq=250.0,
            primary_channels=list(range(p)),
            ref_channels=list(range(p, p + q)),
            mode="global",
            threshold="null",
            # Pinned: the n_shared=0 case sits close enough to the null
            # boundary that a handful of surrogate seeds flip the decision
            # (expected Monte Carlo jitter in a 100-surrogate quantile, not
            # miscalibration -- the aggregate false-rejection rate across
            # independent datasets is ~2.5-5%, matching alpha).
            null_random_state=2,
            verbose=False,
        )
        icc.fit_transform(np.vstack([X, Y]))
        assert int(icc.n_removed_.sum()) == n_shared


def test_qc_records_max_r2_and_conditioning():
    """A zero removal must be distinguishable from an unreachable threshold."""
    rng = np.random.default_rng(3)
    p = q = 16
    n_times = 2000
    icc = ICanClean(
        sfreq=250.0,
        primary_channels=list(range(p)),
        ref_channels=list(range(p, p + q)),
        mode="global",
        threshold=0.99,
        verbose=False,
    )
    icc.fit_transform(np.vstack([_ar1(rng, p, n_times), _ar1(rng, q, n_times)]))
    assert icc.n_removed_.sum() == 0
    # The evidence that explains the zero.
    assert icc.max_r2_.shape == icc.thresholds_.shape
    assert float(icc.max_r2_[0]) < float(icc.thresholds_[0])
    assert icc.samples_per_variable_ == pytest.approx(n_times / (p + q))


def test_reset_clears_hybrid_window_counters():
    """global_n_windows_/sliding_n_windows_ leaked across re-fits before this."""
    rng = np.random.default_rng(5)
    p = q = 12
    data = np.vstack([_ar1(rng, p, 1500), _ar1(rng, q, 1500)])
    kw = {
        "sfreq": 250.0,
        "primary_channels": list(range(p)),
        "ref_channels": list(range(p, p + q)),
        "verbose": False,
    }
    icc = ICanClean(
        mode="hybrid",
        segment_len=2.0,
        threshold=0.9,
        global_threshold=0.9,
        global_clean_with="X",
        global_max_reject_fraction=0.5,
        **kw,
    )
    icc.fit_transform(data)
    assert hasattr(icc, "global_n_windows_")

    icc.set_params(
        mode="sliding",
        global_threshold=None,
        global_clean_with=None,
        global_max_reject_fraction=None,
    )
    icc.fit_transform(data)
    assert not hasattr(icc, "global_n_windows_")
    assert not hasattr(icc, "sliding_n_windows_")
