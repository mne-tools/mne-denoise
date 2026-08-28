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

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


@pytest.mark.parametrize("method", ["standard", "riemannian", "riemannian_windowed"])
def test_process_asr_progress_is_one_ordered_stream_per_backend(
    synthetic_burst_data, method
):
    """Each ASR backend reports completed reconstruction updates once."""
    data, _, _, sfreq = synthetic_burst_data
    calibration_method = "riemannian" if method.startswith("riemannian") else "standard"
    state, _ = calibrate_asr(
        data,
        sfreq,
        cutoff=3.0,
        calibration="manual",
        filter_kind="none",
        method=calibration_method,
    )

    events = []
    cleaned, diagnostics = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        max_dims=0.5,
        lookahead=0.0,
        method=method,
        callback=events.append,
    )
    reference_cleaned, reference_diagnostics = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        max_dims=0.5,
        lookahead=0.0,
        method=method,
    )

    assert len(events) == diagnostics["n_windows"]
    assert [event.method for event in events] == ["asr"] * len(events)
    assert [event.stage for event in events] == ["window"] * len(events)
    assert [event.current for event in events] == list(range(1, len(events) + 1))
    assert [event.total for event in events] == [len(events)] * len(events)
    assert [event.component for event in events] == [None] * len(events)
    np.testing.assert_allclose(
        [event.metric for event in events],
        diagnostics["n_components_reconstructed"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(cleaned, reference_cleaned)
    np.testing.assert_array_equal(
        diagnostics["n_components_reconstructed"],
        reference_diagnostics["n_components_reconstructed"],
    )


def test_process_asr_callback_exception_propagates_unchanged(synthetic_burst_data):
    """A reconstruction callback exception is not caught by ASR."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        cutoff=3.0,
        calibration="manual",
        filter_kind="none",
    )

    class ReconstructionCallbackError(RuntimeError):
        pass

    expected = ReconstructionCallbackError("reconstruction callback failed")

    def callback(event):
        del event
        raise expected

    with pytest.raises(ReconstructionCallbackError) as caught:
        process_asr(
            data,
            sfreq,
            state,
            max_dims=0.5,
            lookahead=0.0,
            callback=callback,
        )
    assert caught.value is expected


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


def test_process_asr_method_validation(synthetic_burst_data):
    """Ensure process_asr raises for invalid inputs."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data, sfreq, method="standard", calibration="manual", filter_kind="none"
    )
    # 1. Invalid method
    with pytest.raises(
        ValueError,
        match="method must be 'standard', 'riemannian', or 'riemannian_windowed'",
    ):
        process_asr(data, sfreq, state, method="bogus")

    # 2. Channel mismatch
    with pytest.raises(ValueError, match="channel count does not match"):
        process_asr(data[:2], sfreq, state)

    # 3. Window length exceeds data length
    with pytest.raises(ValueError, match="Window length"):
        process_asr(data[:, :10], sfreq, state, window_length=1.0)

    # 4. lookahead < 0
    with pytest.raises(ValueError, match="lookahead must be non-negative"):
        process_asr(data, sfreq, state, lookahead=-0.1)

    # 5. lookahead too long
    with pytest.raises(ValueError, match="lookahead is too long"):
        process_asr(data, sfreq, state, lookahead=1000.0)

    # 6. stepsize < 1
    with pytest.raises(ValueError, match="stepsize must be at least 1"):
        process_asr(data, sfreq, state, stepsize=0)

    # 7. stepsize > win_len
    with pytest.raises(ValueError, match="stepsize must not exceed window_length"):
        process_asr(data, sfreq, state, stepsize=1000)

    # 8. max_bad <= 0 (identity pass)
    clean, diag = process_asr(data, sfreq, state, max_dims=0.0)
    assert diag["memory_mode"] == "identity"
    np.testing.assert_array_equal(clean, data)


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
    assert "reconstruction_matrices" in diag

    # And for riemannian_windowed
    clean2, diag2 = process_asr(
        data,
        sfreq,
        state,
        lookahead=0.0,
        store_reconstruction_matrices=True,
        method="riemannian_windowed",
    )
    assert "reconstruction_matrices" in diag2


def test_process_adaptive_chunk() -> None:
    from mne_denoise.asr._reconstruction import _process_adaptive_chunk
    from mne_denoise.asr._types import ASRState

    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))

    state = ASRState(
        M=np.eye(3),
        T=np.eye(3),
        thresholds=np.ones(3),
        calibration_patterns=np.eye(3),
        filter_b=np.array([1.0]),
        filter_a=np.array([1.0]),
        cov=np.eye(3),
        rank=3,
    )
    process_state = {}

    X_clean, diag, next_state = _process_adaptive_chunk(
        X=X,
        sfreq=100.0,
        state=state,
        process_state=process_state,
        window_length=0.5,
        lookahead=None,
        stepsize=None,
        max_dims=2.0,
        store_reconstruction_matrices=True,
        adaptive_variant="psw",
        max_mem_mb=None,
    )

    assert X_clean.shape == (3, 100)
    assert isinstance(diag, dict)
    assert isinstance(next_state, dict)
    assert "reconstruction_matrices" in diag

    X_clean2, diag2, next_state2 = _process_adaptive_chunk(
        X=X,
        sfreq=100.0,
        state=state,
        process_state=process_state,
        window_length=0.5,
        lookahead=0.1,
        stepsize=10,
        max_dims=2.0,
        store_reconstruction_matrices=True,
        adaptive_variant="euclidean",
        max_mem_mb=None,
    )
    assert X_clean2.shape == (3, 100)
