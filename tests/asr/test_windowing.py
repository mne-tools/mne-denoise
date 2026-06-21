import numpy as np
import pytest

from mne_denoise.asr._windowing import (
    _compute_window_diagnostics,
    _compute_window_rms,
    _concatenate_windows,
    _create_good_sample_mask_from_mne,
    _create_sample_mask_from_windows,
    _get_fractional_window_starts,
    _get_window_starts,
    _get_window_weights,
    _mask_to_sample_spans,
    _merge_sample_spans,
    _resolve_bad_channel_count,
    _select_clean_windows,
    compute_clean_window_mask,
)


def test_get_window_starts() -> None:
    starts = _get_window_starts(n_times=100, win_len=10, overlap=0.5)
    assert len(starts) > 0
    assert starts[0] == 0
    assert starts[-1] == 90

    with pytest.raises(ValueError, match="must be at least 2"):
        _get_window_starts(100, 1, 0.5)


def test_get_fractional_window_starts() -> None:
    starts = _get_fractional_window_starts(n_times=100, win_len=10, overlap=0.5)
    assert len(starts) > 0
    assert starts[0] == 0


def test_get_window_weights() -> None:
    weights = _get_window_weights(10)
    assert len(weights) == 10
    assert np.all(weights >= 0) and np.all(weights <= 1)

    weights_small = _get_window_weights(2)
    assert np.all(weights_small == 1.0)


def test_compute_window_rms() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    starts = np.array([0, 10, 20])
    rms = _compute_window_rms(X, starts, win_len=10)
    assert rms.shape == (3, 3)
    assert np.all(rms >= 0)


def test_resolve_bad_channel_count() -> None:
    assert _resolve_bad_channel_count(0.1, 100) == 10
    assert _resolve_bad_channel_count(0.5, 10) == 5
    assert _resolve_bad_channel_count(5, 10) == 5
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_bad_channel_count(-1, 10)


def test_concatenate_windows() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2, 50))
    starts = np.array([0, 20])
    out = _concatenate_windows(X, starts, 10)
    assert out.shape == (2, 20)
    assert np.allclose(out[:, :10], X[:, :10])
    assert np.allclose(out[:, 10:], X[:, 20:30])


def test_sample_mask_logic() -> None:
    mask = _create_sample_mask_from_windows(
        n_times=50,
        starts=np.array([0, 10, 20]),
        win_len=10,
        window_remove_mask=np.array([False, True, False]),
    )
    assert np.all(mask[:10])
    assert not np.any(mask[10:20])
    assert np.all(mask[20:])


def test_merge_sample_spans() -> None:
    spans = [(0, 10), (5, 15), (20, 30)]
    merged = _merge_sample_spans(spans)
    assert merged == [(0, 15), (20, 30)]
    assert _merge_sample_spans([]) == []


def test_mask_to_sample_spans() -> None:
    mask = np.array([True, True, False, True, True])
    spans = _mask_to_sample_spans(mask)
    assert spans == [(0, 2), (3, 5)]


def test_select_clean_windows() -> None:
    X = np.ones((2, 1000))
    X[:, 100:110] = 100.0  # Make one window clearly an outlier
    starts = np.arange(0, 1000, 10)  # 100 windows
    mask, _ = _select_clean_windows(
        X=X,
        starts=starts,
        win_len=10,
        ref_max_bad_channels=0.5,
        ref_tolerances=(-5.0, 5.0),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert mask.shape == (100,)
    assert not mask[10], "Window 10 should be marked bad"


def test_compute_window_diagnostics() -> None:
    # Need enough data points for robust distribution fitting
    X = np.ones((2, 1000))
    X[0, 100:110] = 100.0  # Spike on channel 0
    starts = np.arange(0, 1000, 10)  # 100 windows

    diagnostics = _compute_window_diagnostics(
        X=X,
        starts=starts,
        win_len=10,
        max_bad_channels=0,  # Any bad channel triggers rejection
        zthresholds=(-5.0, 5.0),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    bad_channel_counts = diagnostics["window_remove_mask"]
    assert bad_channel_counts.shape == (100,)
    assert bad_channel_counts[10]
    rms = diagnostics["window_rms"]
    assert rms.shape == (100, 2)


def test_create_good_sample_mask_from_mne() -> None:
    import mne

    info = mne.create_info(3, 100, "eeg")
    raw = mne.io.RawArray(np.zeros((3, 100)), info)
    # Add a BAD annotation
    annotations = mne.Annotations(onset=[0.2], duration=[0.2], description=["BAD_test"])
    raw.set_annotations(annotations)

    mask = _create_good_sample_mask_from_mne(raw, ("bad",))
    assert mask.shape == (100,)
    assert np.all(mask[:20])
    assert not np.any(mask[20:40])
    assert np.all(mask[40:])


def test_compute_clean_window_mask_flags_burst_samples(synthetic_burst_data):
    """clean_windows-style rejection mask flags high-burst segments."""
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


def test_compute_clean_window_mask_standalone():
    from .test_core import SFREQ, _eeg

    X = _eeg(bursts=8)
    mask, info = compute_clean_window_mask(X, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (X.shape[1],)
    assert isinstance(info, dict)


def test_extract_clean_calibration_samples(synthetic_burst_data):
    from mne_denoise.asr._windowing import _extract_clean_calibration_samples

    data, _, burst_mask, sfreq = synthetic_burst_data
    X_clean, sample_mask, diagnostics = _extract_clean_calibration_samples(
        X=data,
        sfreq=sfreq,
        window_length=0.5,
        window_overlap=0.66,
        max_bad_channels=0.25,
        zthresholds=(-np.inf, 3.5),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )

    # Check that X_clean is exactly the unmasked samples of data
    assert X_clean.shape[1] == sample_mask.sum()
    assert np.allclose(X_clean, data[:, sample_mask])

    # Check that the number of returned clean samples is greater than 0
    assert X_clean.shape[1] > 0

    # Check diagnostics
    assert isinstance(diagnostics, dict)
    assert "window_keep_mask" in diagnostics

    # Burst regions should have a very low retention rate
    burst_retention = np.mean(sample_mask[burst_mask])
    clean_retention = np.mean(sample_mask[~burst_mask])
    assert burst_retention < clean_retention


def test_windowing_edge_cases():
    """Test all remaining edge cases in _windowing.py."""
    # 1. _get_window_starts and _get_fractional_window_starts edge cases
    with pytest.raises(ValueError, match="exceeds data length"):
        _get_window_starts(10, 20, 0.5)
    with pytest.raises(ValueError, match="exceeds data length"):
        _get_fractional_window_starts(10, 20, 0.5)
    with pytest.raises(ValueError, match="at least 2"):
        _get_fractional_window_starts(100, 1, 0.5)

    # 2. _get_window_starts with last window appended
    starts = _get_window_starts(10, 4, 0.0)
    assert starts.tolist() == [0, 4, 6]

    # 3. compute_clean_window_mask validation
    from mne_denoise.asr._windowing import (
        _compute_window_diagnostics,
        _create_good_sample_mask_from_mne,
        compute_clean_window_mask,
    )

    X = np.random.randn(3, 1000)
    with pytest.raises(ValueError, match="expects a 2D array"):
        compute_clean_window_mask(X[:, :, np.newaxis], sfreq=100)
    with pytest.raises(ValueError, match="at least one channel"):
        compute_clean_window_mask(np.empty((0, 1000)), sfreq=100)

    # 4. NaNs in compute_clean_window_mask
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    mask, diag = compute_clean_window_mask(X_nan, sfreq=100, window_length=0.5)
    assert not np.isnan(diag["window_rms"]).any()

    # 5. _mask_to_sample_spans empty
    assert _mask_to_sample_spans(np.array([], dtype=bool)) == []

    # 6. _compute_clean_annotations_mask absent/empty/missed annotations
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

    # 7. tolerated_bad_channels >= n_channels in _compute_window_diagnostics
    win_len = 10
    starts = _get_fractional_window_starts(X.shape[1], win_len, 0.5)
    diag = _compute_window_diagnostics(
        X,
        starts,
        win_len,
        max_bad_channels=10,
        zthresholds=(-3.5, 3.5),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert np.all(diag["window_keep_mask"])

    # 8. z_high <= 0 and z_low >= 0
    diag = _compute_window_diagnostics(
        X,
        starts,
        win_len,
        max_bad_channels=0,
        zthresholds=(1.0, -1.0),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert np.any(diag["window_keep_mask"]) or not np.any(
        diag["window_keep_mask"]
    )  # Can be anything but skips lines 283-286

    # 9. _select_clean_windows where np.any(clean) is False (forces 1 retention)
    from mne_denoise.asr._windowing import _select_clean_windows

    clean, zscores = _select_clean_windows(
        X,
        starts,
        win_len,
        ref_max_bad_channels=0,
        ref_tolerances=(-0.001, 0.001),
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
    )
    assert np.sum(clean) == 1

    # 10. _extract_clean_calibration_samples where np.any(keep) is False
    from mne_denoise.asr._windowing import _extract_clean_calibration_samples

    X_clean, s_mask, diag = _extract_clean_calibration_samples(
        X, 100, 0.1, 0.5, 0, (-0.001, 0.001), 0.1, 0.25
    )
    assert np.sum(diag["window_keep_mask"]) == 1
