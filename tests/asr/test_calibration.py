import numpy as np
import pytest

from mne_denoise.asr import calibrate_asr
from mne_denoise.asr._types import ASRState

SFREQ = 250.0


def _eeg():
    return np.random.default_rng(42).standard_normal((3, 1000))


def test_calibrate_asr_returns_state_and_diagnostics(synthetic_burst_data):
    """Array-level calibration returns a valid ASR state."""
    data, _, _, sfreq = synthetic_burst_data
    state, diagnostics = calibrate_asr(
        data,
        sfreq,
        cutoff=4.0,
        calibration="auto",
        filter_kind="none",
    )

    assert isinstance(state, ASRState)
    assert state.M.shape == (data.shape[0], data.shape[0])
    assert state.T.shape == (data.shape[0], data.shape[0])
    assert state.thresholds.shape == (data.shape[0],)
    assert diagnostics["clean_window_mask"].ndim == 1
    assert diagnostics["n_clean_windows"] > 0
    assert diagnostics["threshold_mu"].shape == (data.shape[0],)
    assert diagnostics["threshold_sigma"].shape == (data.shape[0],)
    assert diagnostics["threshold_beta"].shape == (data.shape[0],)
    assert diagnostics["threshold_fit_interval"].shape == (data.shape[0], 2)


def test_calibrate_asr_low_memory_handles_remainder_two(rng):
    """Low-memory calibration handles the ASRpy block remainder edge case."""
    sfreq = 250.0
    n_channels = 6
    n_times = 1002
    blocksize = 100
    assert n_times % blocksize == 2
    data = 0.05 * rng.standard_normal((n_channels, n_times))

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
    assert diagnostics["memory_mode"] == "chunked"
    assert diagnostics["used_memory_bound"] is True
    assert (
        diagnostics["estimated_full_cov_bytes"] > diagnostics["peak_cov_buffer_bytes"]
    )
    assert diagnostics["chunk_samples"] == blocksize


def test_calibrate_asr_bad_calibration_raises():
    with pytest.raises(ValueError, match="calibration must be"):
        calibrate_asr(_eeg(), SFREQ, calibration="bogus", filter_kind="none")


def test_calibrate_asr_bad_cov_estimator_raises():
    with pytest.raises(ValueError, match="cov_estimator"):
        calibrate_asr(_eeg(), SFREQ, cov_estimator="bogus", filter_kind="none")


def test_calibrate_asr_bad_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        calibrate_asr(_eeg(), SFREQ, method="bogus", filter_kind="none")


def test_calibrate_asr_bad_blocksize_raises():
    with pytest.raises(ValueError, match="blocksize"):
        calibrate_asr(_eeg(), SFREQ, blocksize=0, filter_kind="none")


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


def test_calibrate_asr_not_enough_clean_windows_raises():
    """All-artifact data should fail the minimum clean window check."""
    rng = np.random.default_rng(99)
    # Create data with massive artifacts everywhere so no windows pass
    data = rng.standard_normal((4, 2000))
    data *= 100.0  # Make everything look like an artifact
    # Add random spikes to break channel correlations
    for ch in range(4):
        spikes = rng.choice(2000, size=500, replace=False)
        data[ch, spikes] += rng.uniform(500, 1000, size=500)
    with pytest.raises(ValueError, match="Not enough clean calibration windows"):
        calibrate_asr(
            data,
            SFREQ,
            cutoff=20.0,
            calibration="auto",
            filter_kind="none",
            ref_tolerances=(-np.inf, 0.01),
        )
