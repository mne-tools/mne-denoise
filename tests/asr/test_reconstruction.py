import numpy as np
import pytest

from mne_denoise.asr import calibrate_asr, process_asr


@pytest.mark.parametrize("method", ["standard", "riemannian", "riemannian_windowed"])
def test_process_asr_reduces_synthetic_bursts(synthetic_burst_data, method):
    """Standard ASR reduces known burst artifact residual variance."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        cutoff=3.0,
        calibration="auto",
        ref_tolerances=(-np.inf, 3.0),
        filter_kind="none",
        method="riemannian" if method.startswith("riemannian") else "standard",
    )
    cleaned, diagnostics = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.66,
        max_dims=0.5,
        method=method,
    )

    assert cleaned.shape == data.shape
    assert diagnostics["n_windows"] > 0
    assert diagnostics["n_components_reconstructed"].sum() > 0
    assert diagnostics["sample_mask"].any()
    assert diagnostics["covariance_geometry"] == method
    assert 0 < diagnostics["max_components_reconstructed"] <= data.shape[0]

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


def test_process_asr_identity_path_emits_no_progress(synthetic_burst_data):
    """The continuous identity path has no reconstruction updates."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        calibration="manual",
        filter_kind="none",
    )
    events = []
    cleaned, diagnostics = process_asr(
        data,
        sfreq,
        state,
        max_dims=0.0,
        callback=events.append,
    )

    assert events == []
    np.testing.assert_array_equal(cleaned, data)
    assert diagnostics["memory_mode"] == "identity"


def test_process_asr_low_memory_matches_full_path(synthetic_burst_data):
    """Low-memory rolling covariance processing matches the full path."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        cutoff=3.0,
        calibration="auto",
        ref_tolerances=(-np.inf, 3.0),
        filter_kind="none",
        max_mem_mb=None,
    )

    full_cleaned, full_diag = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.66,
        max_dims=0.5,
        max_mem_mb=None,
    )
    rolling_cleaned, rolling_diag = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.66,
        max_dims=0.5,
        max_mem_mb=0.001,
    )

    assert full_diag["memory_mode"] == "full"
    assert rolling_diag["memory_mode"] == "rolling"
    assert rolling_diag["used_memory_bound"] is True
    np.testing.assert_allclose(rolling_cleaned, full_cleaned, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(
        rolling_diag["sample_mask"],
        full_diag["sample_mask"],
    )
    np.testing.assert_array_equal(
        rolling_diag["n_components_reconstructed"],
        full_diag["n_components_reconstructed"],
    )


def test_process_asr_scheduling_respects_overlap_lookahead_and_stepsize(
    synthetic_burst_data,
):
    """Processing diagnostics expose contiguous, lookahead-shifted updates."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        method="standard",
        calibration="manual",
        filter_kind="none",
    )

    cleaned, diagnostics = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.5,
        max_dims=0.5,
        lookahead=0.1,
        stepsize=100,
        method="standard",
    )

    assert cleaned.shape == data.shape
    assert diagnostics["window_length_samples"] == int(0.5 * sfreq)
    assert diagnostics["lookahead_samples"] == int(0.1 * sfreq)
    assert diagnostics["stepsize_samples"] == 100
    starts = diagnostics["window_starts"]
    stops = diagnostics["window_stops"]
    assert starts[0] == 0
    assert stops[-1] == data.shape[1]
    np.testing.assert_array_equal(stops[:-1], starts[1:])
    assert np.all(np.diff(starts) > 0)
    assert np.any(diagnostics["n_components_reconstructed"] > 0)
    assert np.all(diagnostics["n_components_reconstructed"] <= data.shape[0])


def test_process_asr_rank_deficient_calibration_is_stable():
    """Regularization keeps reconstruction finite for dependent channels."""
    rng = np.random.default_rng(123)
    source = rng.standard_normal((3, 1200))
    data = np.vstack(
        (
            source[0],
            source[0],
            source[1],
            source[2] + 0.01 * source[1],
        )
    )
    state, calibration_diagnostics = calibrate_asr(
        data,
        250.0,
        calibration="manual",
        filter_kind="none",
        regularization=1e-6,
    )
    cleaned, diagnostics = process_asr(
        data,
        250.0,
        state,
        max_dims=0.5,
    )

    assert 0 < calibration_diagnostics["rank"] <= data.shape[0]
    assert state.rank == calibration_diagnostics["rank"]
    assert calibration_diagnostics["calibration_samples"] == data.shape[1]
    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(state.cov))
    assert np.all(np.isfinite(state.M))
    assert np.all(np.isfinite(state.T))
    assert np.all(np.isfinite(cleaned))
    assert diagnostics["n_windows"] > 0


def test_process_asr_rejects_invalid_scheduling_inputs(synthetic_burst_data):
    """Invalid reconstruction controls fail as one public validation contract."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data, sfreq, method="standard", calibration="manual", filter_kind="none"
    )

    cases = [
        ("method", data, {"method": "bogus"}, "method must be"),
        ("channels", data[:2], {}, "channel count does not match"),
        (
            "short",
            data[:, :10],
            {"window_length": 1.0},
            "Window length",
        ),
        (
            "negative_lookahead",
            data,
            {"lookahead": -0.1},
            "lookahead must be non-negative",
        ),
        (
            "long_lookahead",
            data,
            {"lookahead": 1000.0},
            "lookahead is too long",
        ),
        (
            "zero_stepsize",
            data,
            {"stepsize": 0},
            "stepsize must be at least 1",
        ),
        (
            "large_stepsize",
            data,
            {"stepsize": 1000},
            "stepsize must not exceed window_length",
        ),
    ]
    for _label, candidate, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            process_asr(candidate, sfreq, state, **kwargs)


def test_process_asr_store_matrices(synthetic_burst_data):
    """Test storing reconstruction matrices."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data, sfreq, cutoff=3.0, filter_kind="none", method="riemannian"
    )
    # store_reconstruction_matrices requires lookahead == 0 in standard/riemannian
    clean, diag = process_asr(
        data,
        sfreq,
        state,
        lookahead=0.0,
        store_reconstruction_matrices=True,
        method="riemannian",
    )
    assert clean.shape == data.shape
    assert diag["reconstruction_matrices"].shape == (
        diag["n_windows"],
        data.shape[0],
        data.shape[0],
    )

    # And for riemannian_windowed
    clean2, diag2 = process_asr(
        data,
        sfreq,
        state,
        lookahead=0.0,
        store_reconstruction_matrices=True,
        method="riemannian_windowed",
    )
    assert clean2.shape == data.shape
    assert diag2["reconstruction_matrices"].shape == (
        diag2["n_windows"],
        data.shape[0],
        data.shape[0],
    )
