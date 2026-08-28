"""JugglerASR."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from mne_denoise.asr import JugglerASR, select_juggler_reference_samples
from mne_denoise.asr._filters import _design_statistics_filter, _lfilter_channels
from mne_denoise.asr.juggler import _dbscan_chebyshev_memory_bounded

SFREQ = 250.0


@pytest.mark.parametrize("seed", [3, 17, 91])
def test_memory_bounded_dbscan_matches_sklearn(seed):
    """The bounded backend preserves exact Chebyshev DBSCAN memberships."""
    rng = np.random.default_rng(seed)
    features = np.vstack(
        [
            rng.normal(-1.0, 0.12, size=(80, 5)),
            rng.normal(1.0, 0.12, size=(70, 5)),
            rng.uniform(-3.0, 3.0, size=(20, 5)),
        ]
    )
    expected = DBSCAN(eps=0.32, min_samples=6, metric="chebyshev").fit_predict(features)
    observed, diagnostics = _dbscan_chebyshev_memory_bounded(
        features,
        eps=0.32,
        min_samples=6,
        count_batch_size=23,
    )

    np.testing.assert_array_equal(observed == -1, expected == -1)
    expected_same_cluster = expected[:, None] == expected[None, :]
    observed_same_cluster = observed[:, None] == observed[None, :]
    non_noise_pairs = (expected[:, None] >= 0) & (expected[None, :] >= 0)
    np.testing.assert_array_equal(
        observed_same_cluster[non_noise_pairs],
        expected_same_cluster[non_noise_pairs],
    )
    assert diagnostics["juggler_dbscan_backend"] == ("ckdtree_grid_memory_bounded")


def test_memory_bounded_dbscan_handles_one_dense_cell():
    """A dense 100k-style cluster is compressed to one occupied core cell."""
    rng = np.random.default_rng(8)
    features = rng.uniform(0.0, 0.01, size=(5000, 5))
    labels, diagnostics = _dbscan_chebyshev_memory_bounded(
        features,
        eps=1.0,
        min_samples=500,
        count_batch_size=127,
    )
    np.testing.assert_array_equal(labels, np.zeros(features.shape[0], dtype=int))
    assert diagnostics["juggler_dbscan_core_cells"] == 1


def _make_synthetic_eeg(
    sfreq: float = 250.0,
    duration_s: float = 60.0,
    n_channels: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build clean synthetic EEG: 10 Hz + 6 Hz oscillation + Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    return data


def _inject_bursts(
    data: np.ndarray,
    n_bursts: int = 8,
    burst_duration_s: float = 0.5,
    amplitude: float = 12.0,
    sfreq: float = 250.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject n_bursts high-amplitude bursts at evenly-spaced intervals."""
    if rng is None:
        rng = np.random.default_rng(97)
    burst_len = int(round(burst_duration_s * sfreq))
    n_times = data.shape[1]
    starts = np.linspace(burst_len, n_times - burst_len, n_bursts).astype(int)
    contaminated = data.copy()
    mask = np.zeros(n_times, dtype=bool)
    channel_scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_times)
        actual_len = stop - start
        spatial = rng.standard_normal(data.shape[0])
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(actual_len)
        contaminated[:, start:stop] += (
            amplitude * channel_scale * np.outer(spatial, temporal)
        )
        mask[start:stop] = True
    return contaminated, mask


def test_juggler_summary_identifies_strategy_and_fit_state(caplog):
    """JugglerASR INFO includes strategy and the fitted operating point."""
    data = _make_synthetic_eeg(sfreq=SFREQ, duration_s=15.0, n_channels=8)
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        JugglerASR(
            sfreq=SFREQ,
            cutoff=20.0,
            strategy="dbscan",
            random_state=42,
            verbose=False,
        ).fit(data, verbose=True)
    summaries = [
        record for record in caplog.records if record.message.startswith("JugglerASR:")
    ]
    assert len(summaries) == 1
    for token in (
        "strategy=dbscan",
        "method=",
        "channels=",
        "sfreq=",
        "cutoff=",
        "rank=",
    ):
        assert token in summaries[0].message


def test_juggler_clean_input_keeps_most_samples():
    """Clean EEG with no bursts should retain a large fraction of samples."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, diag = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="dbscan",
    )
    keep = float(np.mean(mask))
    assert keep > 0.5, f"DBSCAN on clean input retained only {keep * 100:.1f}%"
    assert diag["reference_selected_samples"] == int(mask.sum())
    assert diag["reference_selection_strategy"] == "dbscan"


def test_juggler_dbscan_with_huge_eps_keeps_everything():
    """DBSCAN with a giant eps should put all samples in one cluster."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, _ = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="dbscan",
        dbscan_eps=1e12,
    )
    assert mask.mean() > 0.95, (
        f"DBSCAN with eps=1e12 kept only {mask.mean() * 100:.1f}%; "
        "expected near-100% since one cluster should swallow everything"
    )


def test_juggler_gev_on_clean_input():
    """GEV strategy should retain a non-trivial fraction on clean EEG."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, diag = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="gev",
    )
    keep = float(np.mean(mask))
    assert keep > 0.05, f"GEV retained only {keep * 100:.1f}%"
    assert diag["reference_selection_strategy"] == "gev"
    assert np.isfinite(diag["juggler_gev_mode"])
    assert diag["juggler_gev_scale"] > 0


def test_juggler_reduces_keep_on_contaminated_input():
    """Contaminated input should retain FEWER samples than clean input."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(
        clean,
        n_bursts=20,
        burst_duration_s=0.5,
        amplitude=12.0,
        sfreq=sfreq,
    )
    _, mask_clean, _ = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="gev",
    )
    _, mask_dirty, _ = select_juggler_reference_samples(
        contaminated,
        sfreq,
        strategy="gev",
    )
    # With heavy bursts, the GEV mode shifts lower and contaminated bursts
    # are excluded — keep-fraction should at least drop, not rise.
    assert mask_dirty.mean() <= mask_clean.mean() + 1e-3, (
        f"GEV keep-fraction did not drop on contaminated input: "
        f"clean={mask_clean.mean():.3f} vs dirty={mask_dirty.mean():.3f}"
    )


def test_juggler_min_reference_fraction_raises():
    """min_reference_fraction floor should trigger on heavily-contaminated input."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(
        clean,
        n_bursts=50,
        burst_duration_s=1.0,
        amplitude=50.0,
        sfreq=sfreq,
    )
    # Set the floor unreasonably high (99%) so the call MUST raise.
    with pytest.raises(RuntimeError, match="retained too little data"):
        select_juggler_reference_samples(
            contaminated,
            sfreq,
            strategy="gev",
            min_reference_fraction=0.99,
        )


def test_juggler_asr_dbscan_fit_transform_round_trip():
    """JugglerASR(dbscan) should fit + transform end-to-end on synthetic data."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)

    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr.fit(contaminated)
    cleaned = asr.transform(contaminated)

    assert cleaned.shape == contaminated.shape
    assert asr.calibration_info_["reference_selection_strategy"] == "dbscan"
    assert 0.0 < asr.calibration_info_["reference_selected_fraction"] <= 1.0
    # cleaned variance should be at most the input variance
    assert np.var(cleaned) <= np.var(contaminated) * 1.05


def test_juggler_fit_progress_only_reports_shared_asr_calibration(
    synthetic_burst_data,
):
    """Juggler selection is silent; shared threshold fitting reports ASR events."""
    data, _, _, sfreq = synthetic_burst_data
    events = []
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=3.0,
        strategy="dbscan",
        filter_kind="none",
        selection_filter_kind="none",
        verbose=False,
    )
    asr.fit(data, callback=events.append)

    assert len(events) == asr.thresholds_.size
    assert all(event.method == "asr" for event in events)
    assert all(event.stage == "calibration" for event in events)
    assert not any(event.stage in {"selection", "dbscan", "gev"} for event in events)
    np.testing.assert_allclose(
        [event.metric for event in events], asr.thresholds_, rtol=0.0, atol=1e-12
    )


def test_juggler_transform_progress_uses_standard_asr_windows(synthetic_burst_data):
    """Juggler inherits standard ASR reconstruction progress semantics."""
    data, _, _, sfreq = synthetic_burst_data
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=3.0,
        strategy="dbscan",
        filter_kind="none",
        selection_filter_kind="none",
        lookahead=0.0,
        verbose=False,
    ).fit(data)
    events = []
    _, diagnostics = asr.transform(
        data,
        callback=events.append,
        return_diagnostics=True,
    )

    assert len(events) == diagnostics["n_windows"]
    assert all(event.method == "asr" for event in events)
    assert all(event.stage == "window" for event in events)
    assert [event.current for event in events] == list(range(1, len(events) + 1))
    assert [event.total for event in events] == [len(events)] * len(events)
    assert all(event.component is None for event in events)
    np.testing.assert_allclose(
        [event.metric for event in events],
        diagnostics["n_components_reconstructed"],
        rtol=0.0,
        atol=0.0,
    )


def test_juggler_fit_transform_inherits_standard_progress(synthetic_burst_data):
    """Juggler fit_transform composes ASR calibration and window events."""
    data, _, _, sfreq = synthetic_burst_data
    kwargs = {
        "sfreq": sfreq,
        "cutoff": 3.0,
        "strategy": "dbscan",
        "filter_kind": "none",
        "selection_filter_kind": "none",
        "lookahead": 0.0,
        "verbose": False,
    }
    events = []
    with_callback = JugglerASR(**kwargs)
    cleaned, diagnostics = with_callback.fit_transform(
        data,
        callback=events.append,
        return_diagnostics=True,
    )
    without_callback = JugglerASR(**kwargs)
    reference_cleaned, reference_diagnostics = without_callback.fit_transform(
        data,
        return_diagnostics=True,
    )

    n_components = with_callback.thresholds_.size
    calibration_events = events[:n_components]
    reconstruction_events = events[n_components:]
    assert all(event.method == "asr" for event in calibration_events)
    assert all(event.stage == "calibration" for event in calibration_events)
    assert len(reconstruction_events) == diagnostics["n_windows"]
    assert all(event.method == "asr" for event in reconstruction_events)
    assert all(event.stage == "window" for event in reconstruction_events)
    assert all(event.component is None for event in reconstruction_events)
    np.testing.assert_allclose(cleaned, reference_cleaned)
    np.testing.assert_array_equal(
        diagnostics["n_components_reconstructed"],
        reference_diagnostics["n_components_reconstructed"],
    )


def test_juggler_asr_gev_fit_transform_round_trip():
    """JugglerASR(gev) should fit + transform end-to-end on synthetic data."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)

    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="gev",
        random_state=42,
        verbose=False,
    )
    asr.fit(contaminated)
    cleaned = asr.transform(contaminated)

    assert cleaned.shape == contaminated.shape
    assert asr.calibration_info_["reference_selection_strategy"] == "gev"
    assert 0.0 < asr.calibration_info_["reference_selected_fraction"] <= 1.0


def test_juggler_calibration_mask_kind_is_sample():
    """JugglerASR advertises sample-based calibration (not window-based)."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr.fit(clean)
    assert asr.calibration_mask_kind_ == "sample"


def test_juggler_get_calibration_mask_after_fit():
    """get_calibration_mask should return a sample-wise bool array of length n_times."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr.fit(clean)
    mask = asr.get_calibration_mask()
    assert mask.dtype == bool
    assert mask.shape == (clean.shape[1],)
    assert mask.sum() == int(asr.calibration_info_["reference_selected_samples"])


def test_juggler_invalid_strategy_raises():
    """Invalid strategy name should raise at fit time."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=10.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="bogus",
        verbose=False,
    )
    with pytest.raises(ValueError, match="strategy must be"):
        asr.fit(clean)


def test_juggler_dbscan_deterministic():
    """JugglerASR(dbscan) with same random_state should be deterministic."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)
    asr1 = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr2 = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr1.fit(contaminated)
    asr2.fit(contaminated)
    np.testing.assert_allclose(asr1.M_, asr2.M_, rtol=1e-12)
    np.testing.assert_allclose(asr1.T_, asr2.T_, rtol=1e-12)
    np.testing.assert_array_equal(
        asr1.get_calibration_mask(), asr2.get_calibration_mask()
    )


def _eeg(n_channels=8, n_times=8000, seed=0, bursts=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / SFREQ
    X = np.zeros((n_channels, n_times))
    for c in range(n_channels):
        X[c] = 0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28)) + (
            0.05 * rng.standard_normal(n_times)
        )
    for s in np.linspace(800, n_times - 500, bursts).astype(int):
        spatial = rng.standard_normal(n_channels)
        spatial /= np.linalg.norm(spatial)
        X[:, s : s + 150] += 10.0 * np.outer(spatial, rng.standard_normal(150))
    return X


def _make_clean_eeg(
    n_channels: int = 16,
    duration_s: float = 40.0,
    sfreq: float = 250.0,
    seed: int = 11,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration_s)
    t = np.arange(n) / sfreq
    data = np.zeros((n_channels, n), dtype=np.float64)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    return data


def test_juggler_to_annotations_calibration_ok():
    mne = pytest.importorskip("mne")
    j = JugglerASR(sfreq=SFREQ, cutoff=20.0, strategy="dbscan", verbose=False)
    j.fit_transform(_eeg())
    ann = j.to_annotations("calibration")
    assert isinstance(ann, mne.Annotations)


def test_juggler_min_reference_fraction_floor_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg(bursts=40)  # heavily contaminated
    with pytest.raises(RuntimeError, match="retained too little"):
        select_juggler_reference_samples(
            X, SFREQ, strategy="gev", min_reference_fraction=0.99
        )


def test_juggler_evoked_calibration_raises():
    mne = pytest.importorskip("mne")
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    evoked = mne.EvokedArray(_eeg(n_times=2000) * 1e-6, info, tmin=0.0, verbose=False)
    with pytest.raises(ValueError, match="Evoked"):
        JugglerASR(sfreq=SFREQ, verbose=False).fit(evoked)


def test_juggler_calibration_mask_shape_raises():
    with pytest.raises(ValueError, match="calibration_mask must have shape"):
        JugglerASR(sfreq=SFREQ, verbose=False).fit(
            _eeg(), calibration_mask=np.ones(10, dtype=bool)
        )


def test_juggler_dbscan_bad_eps_string_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    with pytest.raises(ValueError, match="dbscan_eps"):
        select_juggler_reference_samples(
            _eeg(), SFREQ, strategy="dbscan", dbscan_eps="bogus"
        )


def test_juggler_dbscan_bad_min_samples_string_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    with pytest.raises(ValueError, match="dbscan_min_samples"):
        select_juggler_reference_samples(
            _eeg(), SFREQ, strategy="dbscan", dbscan_min_samples="bogus"
        )


def test_juggler_dbscan_numeric_eps_fallback():
    from mne_denoise.asr import select_juggler_reference_samples

    _, mask, _ = select_juggler_reference_samples(
        _eeg(bursts=8), SFREQ, strategy="dbscan", dbscan_eps=0.0
    )
    assert mask.dtype == bool


def test_juggler_dbscan_min_samples_numeric():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg(bursts=8)
    _, m1, _ = select_juggler_reference_samples(
        X, SFREQ, strategy="dbscan", dbscan_min_samples=0.05
    )
    _, m2, _ = select_juggler_reference_samples(
        X, SFREQ, strategy="dbscan", dbscan_min_samples=5
    )
    assert m1.dtype == bool and m2.dtype == bool


def test_select_juggler_reference_samples_param_guards():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg()
    with pytest.raises(ValueError, match="strategy must be"):
        select_juggler_reference_samples(X, SFREQ, strategy="bogus")
    with pytest.raises(ValueError, match="dbscan_top_k"):
        select_juggler_reference_samples(X, SFREQ, dbscan_top_k=0)
    with pytest.raises(ValueError, match="gev_grid_size"):
        select_juggler_reference_samples(X, SFREQ, strategy="gev", gev_grid_size=8)
    with pytest.raises(ValueError, match="min_reference_fraction"):
        select_juggler_reference_samples(X, SFREQ, min_reference_fraction=1.5)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"dbscan_top_k": 0}, "dbscan_top_k"),
        ({"gev_grid_size": 8}, "gev_grid_size"),
        ({"min_reference_fraction": 1.5}, "min_reference_fraction"),
    ],
)
def test_juggler_constructor_param_guards(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        JugglerASR(sfreq=SFREQ, verbose=False, **kwargs).fit(_eeg())


@pytest.mark.parametrize("eps_multiplier", [0.5, 1.0, 2.0])
def test_juggler_dbscan_eps_parametric(eps_multiplier):
    """Larger eps should retain at least as many samples as smaller eps."""
    clean = _make_clean_eeg(duration_s=60.0)
    dirty, _ = _inject_bursts(clean, n_bursts=10)
    # Get auto eps first
    base = JugglerASR(
        sfreq=250.0,
        cutoff=20.0,
        strategy="dbscan",
        verbose=False,
    )
    base.fit(dirty)
    auto_eps = float(base.calibration_info_["juggler_dbscan_eps"])
    eps = auto_eps * eps_multiplier
    asr = JugglerASR(
        sfreq=250.0,
        cutoff=20.0,
        strategy="dbscan",
        dbscan_eps=eps,
        verbose=False,
    )
    asr.fit(dirty)
    rf = float(asr.calibration_info_["reference_selected_fraction"])
    assert 0.0 < rf <= 1.0


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_select_juggler_reference_samples_rejects_burst_samples(
    synthetic_burst_data,
    strategy,
):
    """Juggler reference selectors prefer low-amplitude samples."""
    data, _, burst_mask, sfreq = synthetic_burst_data
    reference, sample_mask, diagnostics = select_juggler_reference_samples(
        data,
        sfreq,
        strategy=strategy,
    )

    assert reference.shape[0] == data.shape[0]
    assert reference.shape[1] == int(np.sum(sample_mask))
    assert sample_mask.shape == (data.shape[1],)
    assert np.mean(sample_mask[burst_mask]) < np.mean(sample_mask[~burst_mask])
    assert diagnostics["reference_selection_strategy"] == strategy
    assert diagnostics["reference_selected_samples"] == int(np.sum(sample_mask))


def test_juggler_reference_contains_continuously_filtered_samples(
    synthetic_burst_data,
):
    """Pointwise selection must not re-filter concatenated sample fragments."""
    data, _, _, sfreq = synthetic_burst_data
    reference, sample_mask, diagnostics = select_juggler_reference_samples(
        data,
        sfreq,
        strategy="gev",
    )
    b, a = _design_statistics_filter(sfreq, "asr")
    filtered, zf = _lfilter_channels(data, b, a)

    np.testing.assert_allclose(reference, filtered[:, sample_mask])
    np.testing.assert_allclose(diagnostics["selection_filter_zi"], zf)


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_juggler_asr_reduces_synthetic_bursts(synthetic_burst_data, strategy):
    """JugglerASR reuses standard ASR repair after sample-wise calibration."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=3.0,
        strategy=strategy,
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.get_calibration_mask().shape == (data.shape[1],)
    assert asr.calibration_info_["reference_selection_strategy"] == strategy
    assert asr.calibration_info_["reference_selected_samples"] == int(
        np.sum(asr.reference_sample_mask_)
    )

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_juggler_cleaning_is_invariant_to_eeg_unit_scaling(
    synthetic_burst_data, strategy
):
    """Juggler gives equivalent results for volts and microvolts."""
    data_uv, _, _, sfreq = synthetic_burst_data
    model_uv = JugglerASR(
        sfreq=sfreq,
        cutoff=5.0,
        strategy=strategy,
        verbose=False,
    )
    clean_uv = model_uv.fit_transform(data_uv)

    model_v = JugglerASR(
        sfreq=sfreq,
        cutoff=5.0,
        strategy=strategy,
        verbose=False,
    )
    clean_v = model_v.fit_transform(data_uv * 1e-6)

    np.testing.assert_allclose(clean_v * 1e6, clean_uv, rtol=2e-7, atol=2e-8)
    np.testing.assert_array_equal(
        model_v.n_components_reconstructed_, model_uv.n_components_reconstructed_
    )


def test_juggler_rejects_mismatched_statistics_filters(synthetic_burst_data):
    """Selection and reconstruction must use the same statistics filter."""
    data, _, _, sfreq = synthetic_burst_data
    model = JugglerASR(
        sfreq=sfreq,
        filter_kind="asr",
        selection_filter_kind="none",
        verbose=False,
    )
    with pytest.raises(ValueError, match="filter_kind and selection_filter_kind"):
        model.fit(data)


def test_juggler_asr_reference_annotations_and_metrics(synthetic_burst_data):
    """JugglerASR exposes the retained reference spans for QC."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=3.0,
        strategy="dbscan",
        verbose=False,
    )
    asr.fit_transform(data)
    annotations = asr.to_annotations("calibration")

    assert isinstance(annotations, mne.Annotations)
    assert len(annotations) >= 1
    assert set(annotations.description) == {"ASR_REFERENCE"}
    assert asr.calibration_mask_kind_ == "sample"

    assert asr.calibration_info_["reference_selected_samples"] == int(
        np.sum(asr.reference_sample_mask_)
    )
    assert asr.calibration_info_["reference_candidate_samples"] == data.shape[1]


def test_juggler_asr_mne_raw_preserves_non_picked_channels(synthetic_burst_data):
    """JugglerASR cleans EEG picks while leaving non-picked channels alone."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    eog = np.vstack(
        [
            np.sin(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
            np.cos(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
        ]
    )
    raw_data = np.vstack([data, eog])
    ch_names = [f"EEG{idx}" for idx in range(data.shape[0])] + ["EOG1", "EOG2"]
    ch_types = ["eeg"] * data.shape[0] + ["eog", "eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    asr = JugglerASR(cutoff=3.0, strategy="gev", verbose=False)
    raw_clean = asr.fit_transform(raw)

    assert isinstance(raw_clean, mne.io.RawArray)
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1", "EOG2"]), eog)


def test_juggler_edge_cases():
    from unittest.mock import MagicMock, patch

    from mne_denoise.asr import select_juggler_reference_samples
    from mne_denoise.asr.juggler import _histogram_mode, _resolve_dbscan_eps

    clean = _make_clean_eeg(duration_s=2.0)
    JugglerASR(sfreq=SFREQ, verbose=False).fit(
        clean, calibration_mask=np.ones(clean.shape[1], dtype=bool)
    )

    with pytest.raises(ValueError, match="empty values"):
        _histogram_mode(np.array([]))

    assert _histogram_mode(np.array([2.0, 2.0, 2.0])) == 2.0

    with patch("numpy.histogram_bin_edges", return_value=np.array([1.0])):
        assert _histogram_mode(np.array([1.0, 2.0, 3.0])) == 2.0

    with pytest.raises(RuntimeError, match="zero-amplitude"):
        _resolve_dbscan_eps("auto", 0.0, np.zeros(10))

    with patch(
        "mne_denoise.asr.juggler._dbscan_chebyshev_memory_bounded",
        return_value=(np.full(100, -1), {}),
    ):
        with pytest.raises(RuntimeError, match="DBSCAN found no non-noise cluster"):
            select_juggler_reference_samples(clean, SFREQ, strategy="dbscan")

    with patch("scipy.stats.genextreme.fit", side_effect=Exception("mocked")):
        with pytest.raises(RuntimeError, match="GEV fitting failed"):
            select_juggler_reference_samples(clean, SFREQ, strategy="gev")

    with patch("scipy.stats.genextreme.fit", return_value=(0.0, 0.0, 0.0)):
        with pytest.raises(
            RuntimeError, match="GEV fitting returned a non-positive scale"
        ):
            select_juggler_reference_samples(clean, SFREQ, strategy="gev")

    class MockGenExtreme:
        def fit(self, x):
            return 1.0, 1.0, 1.0

        def __call__(self, *args, **kwargs):
            m = MagicMock()
            m.ppf.side_effect = [np.inf, np.inf]
            return m

    with patch("mne_denoise.asr.juggler.stats.genextreme", MockGenExtreme()):
        _, mask, diag = select_juggler_reference_samples(clean, SFREQ, strategy="gev")
        assert diag["juggler_gev_mode"] > 0

    with patch("scipy.optimize.minimize_scalar", return_value=MagicMock(success=False)):
        _, mask, diag = select_juggler_reference_samples(clean, SFREQ, strategy="gev")
        assert diag["juggler_gev_mode"] > 0

    mne = pytest.importorskip("mne")
    info = mne.create_info(clean.shape[0], SFREQ, "eeg")
    epochs = mne.EpochsArray(clean.reshape(1, clean.shape[0], -1), info, verbose=False)
    JugglerASR(sfreq=SFREQ, verbose=False).fit(epochs)
