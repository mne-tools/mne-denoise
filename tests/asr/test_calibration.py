from unittest.mock import patch

import numpy as np
import pytest
from scipy import signal

from mne_denoise.asr import calibrate_asr
from mne_denoise.asr._filters import _design_statistics_filter
from mne_denoise.asr._types import ASRState

SFREQ = 250.0


def _eeg():
    return np.random.default_rng(42).standard_normal((3, 1000))


@pytest.mark.parametrize("calibration", ["auto", "manual"])
def test_calibrate_asr_returns_state_and_diagnostics(synthetic_burst_data, calibration):
    """Automatic and manual calibration expose the fitted ASR state."""
    data, _, _, sfreq = synthetic_burst_data
    state, diagnostics = calibrate_asr(
        data,
        sfreq,
        cutoff=4.0,
        calibration=calibration,
        filter_kind="none",
    )

    assert isinstance(state, ASRState)
    assert state.M.shape == (data.shape[0], data.shape[0])
    assert state.T.shape == (data.shape[0], data.shape[0])
    assert state.thresholds.shape == (data.shape[0],)
    assert diagnostics["clean_window_mask"].ndim == 1
    if calibration == "auto":
        assert diagnostics["n_clean_windows"] > 0
    else:
        assert diagnostics["n_clean_windows"] == 0
        np.testing.assert_array_equal(diagnostics["clean_sample_mask"], True)
    assert diagnostics["threshold_mu"].shape == (data.shape[0],)
    assert diagnostics["threshold_sigma"].shape == (data.shape[0],)
    assert diagnostics["threshold_beta"].shape == (data.shape[0],)
    assert diagnostics["threshold_fit_interval"].shape == (data.shape[0], 2)
    assert np.all(state.thresholds > 0)
    assert state.rank == diagnostics["rank"]
    np.testing.assert_allclose(
        state.T,
        np.diag(state.thresholds) @ state.calibration_patterns.T,
    )


def test_calibrate_asr_progress_reports_fitted_component_thresholds():
    """Calibration emits one completed event for each fitted component."""
    events = []
    state, _ = calibrate_asr(
        _eeg(),
        SFREQ,
        cutoff=5.0,
        calibration="manual",
        filter_kind="none",
        callback=events.append,
    )

    assert len(events) == state.thresholds.size
    assert [event.method for event in events] == ["asr"] * state.thresholds.size
    assert [event.stage for event in events] == ["calibration"] * state.thresholds.size
    assert [event.current for event in events] == list(
        range(1, state.thresholds.size + 1)
    )
    assert [event.total for event in events] == [state.thresholds.size] * len(events)
    assert [event.component for event in events] == list(
        range(1, state.thresholds.size + 1)
    )
    np.testing.assert_allclose(
        [event.metric for event in events], state.thresholds, rtol=0.0, atol=1e-12
    )


def test_calibrate_asr_low_memory_matches_full_with_remainder_two(rng):
    """Chunked calibration matches full calibration with a short remainder."""
    sfreq = 250.0
    n_channels = 6
    n_times = 1002
    blocksize = 100
    assert n_times % blocksize == 2
    data = 0.05 * rng.standard_normal((n_channels, n_times))

    full_state, full_diagnostics = calibrate_asr(
        data,
        sfreq,
        cutoff=5.0,
        calibration="manual",
        blocksize=blocksize,
        filter_kind="none",
        max_mem_mb=None,
    )
    state, diagnostics = calibrate_asr(
        data,
        sfreq,
        cutoff=5.0,
        calibration="manual",
        blocksize=blocksize,
        filter_kind="none",
        max_mem_mb=0.001,
    )

    assert isinstance(state, ASRState)
    assert state.M.shape == (n_channels, n_channels)
    assert full_diagnostics["memory_mode"] == "full"
    assert diagnostics["memory_mode"] == "chunked"
    assert diagnostics["used_memory_bound"] is True
    assert (
        diagnostics["estimated_full_cov_bytes"] > diagnostics["peak_cov_buffer_bytes"]
    )
    assert diagnostics["chunk_samples"] == blocksize
    for name in ("M", "T", "thresholds", "calibration_patterns", "cov"):
        np.testing.assert_allclose(getattr(state, name), getattr(full_state, name))
    assert state.rank == full_state.rank


def test_calibrate_asr_rejects_invalid_options():
    """Calibration-specific public options fail with useful errors."""
    cases = [
        ("calibration", {"calibration": "bogus"}, "calibration must be"),
        ("covariance", {"cov_estimator": "bogus"}, "cov_estimator"),
        ("method", {"method": "bogus"}, "method must be"),
        ("blocksize", {"blocksize": 0}, "blocksize"),
    ]
    for _label, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            calibrate_asr(_eeg(), SFREQ, filter_kind="none", **kwargs)


def test_calibrate_asr_riemannian_method():
    """Riemannian method branch uses sqrtm_spd and nonlinear eigenspace."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 2000))
    state, diagnostics = calibrate_asr(
        data,
        SFREQ,
        cutoff=20.0,
        calibration="manual",
        method="riemannian",
        filter_kind="none",
    )
    assert isinstance(state, ASRState)
    assert state.method == "riemannian"
    assert state.riemannian_solver == "nonlinear_eigenspace"
    assert state.M.shape == (4, 4)


def test_calibrate_asr_cutoff_changes_rejection_thresholds():
    """The cutoff controls the fitted component rejection thresholds."""
    data = _eeg()
    aggressive, _ = calibrate_asr(
        data,
        SFREQ,
        cutoff=2.0,
        calibration="manual",
        filter_kind="none",
    )
    permissive, _ = calibrate_asr(
        data,
        SFREQ,
        cutoff=20.0,
        calibration="manual",
        filter_kind="none",
    )

    assert np.all(permissive.thresholds > aggressive.thresholds)


def test_min_clean_fraction_does_not_impose_a_retained_window_quota():
    """MinCleanFraction controls distribution fitting, not window selection."""
    rng = np.random.default_rng(99)
    data = rng.standard_normal((4, 2000))

    def select_five_contiguous_windows(X, starts, win_len, **kwargs):
        del X, win_len, kwargs
        mask = np.zeros(len(starts), dtype=bool)
        mask[5:10] = True
        return mask, np.zeros((len(starts), data.shape[0]))

    with patch(
        "mne_denoise.asr._calibration._select_clean_windows",
        side_effect=select_five_contiguous_windows,
    ):
        _, diagnostics = calibrate_asr(
            data,
            SFREQ,
            cutoff=20.0,
            calibration="auto",
            filter_kind="none",
        )
    assert (
        0
        < diagnostics["n_clean_windows"]
        < (0.25 * diagnostics["n_calibration_windows"])
    )


def test_asr_statistics_filter_is_applied_causally_before_calibration():
    """The default statistics filter is applied causally during calibration."""
    data = _eeg()
    b, a = _design_statistics_filter(SFREQ, "asr")
    zi = np.zeros((data.shape[0], max(len(a), len(b)) - 1))
    filtered, expected_zi = signal.lfilter(b, a, data, axis=1, zi=zi)

    shaped, _ = calibrate_asr(
        data,
        SFREQ,
        cutoff=20.0,
        calibration="manual",
        filter_kind="asr",
    )
    prefiltered, _ = calibrate_asr(
        filtered,
        SFREQ,
        cutoff=20.0,
        calibration="manual",
        filter_kind="none",
    )

    np.testing.assert_allclose(shaped.M, prefiltered.M, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shaped.T, prefiltered.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shaped.filter_zi, expected_zi, rtol=0.0, atol=1e-12)


def test_asr_calibration_is_invariant_to_eeg_unit_scaling():
    """Volts and microvolts produce equivalent fitted ASR geometry."""
    data_uv = _eeg()
    data_v = data_uv * 1e-6
    state_uv, _ = calibrate_asr(
        data_uv,
        SFREQ,
        calibration="manual",
        filter_kind="asr",
    )
    state_v, _ = calibrate_asr(
        data_v,
        SFREQ,
        calibration="manual",
        filter_kind="asr",
    )

    np.testing.assert_allclose(state_v.M * 1e6, state_uv.M, rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(state_v.T * 1e6, state_uv.T, rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(
        state_v.thresholds * 1e6,
        state_uv.thresholds,
        rtol=2e-9,
        atol=1e-10,
    )
