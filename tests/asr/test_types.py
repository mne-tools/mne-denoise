import numpy as np

from mne_denoise.asr._types import ASRState


def test_asr_state_instantiation() -> None:
    """Test that ASRState can be instantiated with its required fields."""
    n_channels = 3
    M = np.eye(n_channels, dtype=np.float64)
    T = np.eye(n_channels, dtype=np.float64) * 2.0
    thresholds = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    calibration_patterns = np.eye(n_channels, dtype=np.float64)
    filter_b = np.array([1.0], dtype=np.float64)
    filter_a = np.array([1.0], dtype=np.float64)
    cov = np.eye(n_channels, dtype=np.float64)
    rank = n_channels

    state = ASRState(
        M=M,
        T=T,
        thresholds=thresholds,
        calibration_patterns=calibration_patterns,
        filter_b=filter_b,
        filter_a=filter_a,
        cov=cov,
        rank=rank,
        method="standard",
    )

    assert state.M.shape == (3, 3)
    assert state.T.shape == (3, 3)
    assert state.thresholds.shape == (3,)
    assert state.calibration_patterns.shape == (3, 3)
    assert state.filter_b.shape == (1,)
    assert state.filter_a.shape == (1,)
    assert state.cov.shape == (3, 3)
    assert state.rank == 3
    assert state.method == "standard"
    assert state.riemannian_solver is None


def test_copy_asr_state() -> None:
    from mne_denoise.asr._types import _copy_asr_state

    n_channels = 3
    state1 = ASRState(
        M=np.eye(n_channels, dtype=np.float64),
        T=np.eye(n_channels, dtype=np.float64),
        thresholds=np.ones(n_channels),
        calibration_patterns=np.eye(n_channels),
        filter_b=np.array([1.0]),
        filter_a=np.array([1.0]),
        cov=np.eye(n_channels),
        rank=n_channels,
    )

    state2 = _copy_asr_state(state1)
    assert state1 is not state2
    assert state1.M is not state2.M
    assert np.allclose(state1.M, state2.M)

    # Mutating state2 should not affect state1
    state2.M[0, 0] = 99.0
    assert state1.M[0, 0] == 1.0


def test_copy_process_state() -> None:
    from mne_denoise.asr._types import _copy_process_state

    pstate1 = {
        "scalar": 42,
        "array": np.array([1.0, 2.0]),
    }

    pstate2 = _copy_process_state(pstate1)
    assert pstate1 is not pstate2
    assert pstate1["scalar"] == pstate2["scalar"]
    assert pstate1["array"] is not pstate2["array"]
    assert np.allclose(pstate1["array"], pstate2["array"])

    # Mutating array in pstate2 should not affect pstate1
    pstate2["array"][0] = 99.0
    assert pstate1["array"][0] == 1.0
