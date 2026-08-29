import numpy as np
import pytest

from mne_denoise.asr._windowing import (
    _compute_window_diagnostics,
    _compute_window_rms,
    _concatenate_windows,
    _create_good_sample_mask_from_mne,
    _create_sample_mask_from_windows,
    _extract_clean_calibration_samples,
    _get_fractional_window_starts,
    _get_window_starts,
    _get_window_weights,
    _mask_to_sample_spans,
    _merge_sample_spans,
    _resolve_bad_channel_count,
    _select_clean_windows,
    compute_clean_window_mask,
)


def test_windowing_helper_contract() -> None:
    """Window primitives agree on starts, weights, masks, and span layout."""
    starts = _get_window_starts(n_times=100, win_len=10, overlap=0.5)
    assert starts.tolist() == list(range(0, 91, 5))
    assert _get_fractional_window_starts(100, 10, 0.5)[0] == 0
    assert np.all((_get_window_weights(10) >= 0) & (_get_window_weights(10) <= 1))
    assert np.all(_get_window_weights(2) == 1.0)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    rms = _compute_window_rms(data, np.array([0, 10, 20]), win_len=10)
    assert rms.shape == (3, 3)
    assert np.all(rms >= 0)
    assert _resolve_bad_channel_count(0.1, 100) == 10
    assert _resolve_bad_channel_count(0.5, 10) == 5
    assert _resolve_bad_channel_count(5, 10) == 5
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_bad_channel_count(-1, 10)

    data = rng.standard_normal((2, 50))
    concatenated = _concatenate_windows(data, np.array([0, 20]), 10)
    assert concatenated.shape == (2, 20)
    np.testing.assert_allclose(concatenated, np.hstack([data[:, :10], data[:, 20:30]]))
    mask = _create_sample_mask_from_windows(
        50, np.array([0, 10, 20]), 10, np.array([False, True, False])
    )
    assert np.all(mask[:10]) and not np.any(mask[10:20]) and np.all(mask[20:])
    assert _merge_sample_spans([(0, 10), (5, 15), (20, 30)]) == [(0, 15), (20, 30)]
    assert _merge_sample_spans([]) == []
    assert _mask_to_sample_spans(np.array([True, True, False, True, True])) == [
        (0, 2),
        (3, 5),
    ]


def test_clean_window_diagnostics_contract() -> None:
    """Window selection identifies bursts and reports aligned diagnostics."""
    data = np.ones((2, 1000))
    data[:, 100:110] = 100.0
    starts = np.arange(0, 1000, 10)
    clean, _ = _select_clean_windows(
        X=data,
        starts=starts,
        win_len=10,
        ref_max_bad_channels=0.5,
        ref_tolerances=(-5.0, 5.0),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert clean.shape == (100,)
    assert not clean[10]
    diagnostics = _compute_window_diagnostics(
        X=data,
        starts=starts,
        win_len=10,
        max_bad_channels=0,
        zthresholds=(-5.0, 5.0),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert diagnostics["window_remove_mask"].shape == (100,)
    assert diagnostics["window_remove_mask"][10]
    assert diagnostics["window_rms"].shape == (100, 2)


def test_create_good_sample_mask_from_mne() -> None:
    """BAD annotations remove exactly their sample interval."""
    import mne

    info = mne.create_info(3, 100, "eeg")
    raw = mne.io.RawArray(np.zeros((3, 100)), info)
    raw.set_annotations(
        mne.Annotations(onset=[0.2], duration=[0.2], description=["BAD_test"])
    )
    mask = _create_good_sample_mask_from_mne(raw, ("bad",))
    assert mask.shape == (100,)
    assert np.all(mask[:20]) and not np.any(mask[20:40]) and np.all(mask[40:])


def test_clean_window_mask_and_calibration_contract(synthetic_burst_data):
    """Burst rejection produces clean calibration samples and useful diagnostics."""
    data, _, burst_mask, sfreq = synthetic_burst_data
    sample_mask, diagnostics = compute_clean_window_mask(
        data,
        sfreq,
        max_bad_channels=0.25,
        zthresholds=(-np.inf, 3.5),
        window_length=0.5,
        window_overlap=0.66,
    )
    assert sample_mask.shape == (data.shape[1],)
    assert diagnostics["n_rejected_windows"] > 0
    assert diagnostics["window_keep_mask"].shape[0] == diagnostics["n_windows"]
    assert np.mean(sample_mask[burst_mask]) < np.mean(sample_mask[~burst_mask])

    clean, mask, calibration = _extract_clean_calibration_samples(
        X=data,
        sfreq=sfreq,
        window_length=0.5,
        window_overlap=0.66,
        max_bad_channels=0.25,
        zthresholds=(-np.inf, 3.5),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert clean.shape[1] == mask.sum() > 0
    np.testing.assert_allclose(clean, data[:, mask])
    assert isinstance(calibration, dict)
    assert np.mean(mask[burst_mask]) < np.mean(mask[~burst_mask])


def test_windowing_edge_cases():
    """Boundary validation and forced minimum retention remain explicit."""
    with pytest.raises(ValueError, match="exceeds data length"):
        _get_window_starts(10, 20, 0.5)
    with pytest.raises(ValueError, match="exceeds data length"):
        _get_fractional_window_starts(10, 20, 0.5)
    with pytest.raises(ValueError, match="at least 2"):
        _get_fractional_window_starts(100, 1, 0.5)
    assert _get_window_starts(10, 4, 0.0).tolist() == [0, 4, 6]

    X = np.random.default_rng(0).standard_normal((3, 1000))
    with pytest.raises(ValueError, match="expects a 2D array"):
        compute_clean_window_mask(X[:, :, np.newaxis], sfreq=100)
    with pytest.raises(ValueError, match="at least one channel"):
        compute_clean_window_mask(np.empty((0, 1000)), sfreq=100)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    _, diagnostics = compute_clean_window_mask(X_nan, sfreq=100, window_length=0.5)
    assert not np.isnan(diagnostics["window_rms"]).any()
    assert _mask_to_sample_spans(np.array([], dtype=bool)) == []

    class DummyRaw:
        def __init__(self):
            self.n_times = 100
            self.info = {"sfreq": 100}

    raw = DummyRaw()
    assert np.all(_create_good_sample_mask_from_mne(raw, ("bad",)))

    class DummyAnnotations:
        def __init__(self, onset, duration, description):
            self.onset = onset
            self.duration = duration
            self.description = description

        def __len__(self):
            return len(self.onset)

    raw.annotations = DummyAnnotations([0.1], [0.1], ["good"])
    assert np.all(_create_good_sample_mask_from_mne(raw, ("bad",)))
    starts = _get_fractional_window_starts(X.shape[1], 10, 0.5)
    diagnostics = _compute_window_diagnostics(
        X,
        starts,
        10,
        max_bad_channels=10,
        zthresholds=(-3.5, 3.5),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert np.all(diagnostics["window_keep_mask"])
    clean, _ = _select_clean_windows(
        X,
        starts,
        10,
        ref_max_bad_channels=0,
        ref_tolerances=(-0.001, 0.001),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert clean.sum() == 1
    clean_data, mask, diagnostics = _extract_clean_calibration_samples(
        X, 100, 0.1, 0.5, 0, (-0.001, 0.001), 0.1, 0.25
    )
    assert clean_data.shape[1] == mask.sum()
    assert diagnostics["window_keep_mask"].sum() == 1
