"""Unit tests for the adaptive ASR."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.asr import AdaptiveASR

SFREQ = 250.0


def _epochs(n_epochs=3, n_per=2000):
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=n_epochs * n_per, bursts=6)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    data = X.reshape(8, n_epochs, n_per).transpose(1, 0, 2) * 1e-6
    return mne.EpochsArray(data, info, verbose=False)


def _make_synthetic(
    n_channels: int = 8,
    n_samples: int = 6000,
    sfreq: float = 250.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a deterministic AASR-style synthetic stream.

    Mirrors the lighter end of ``generate_aasr_input.py``: a small 10 Hz +
    6.5 Hz brain background, 5% Gaussian sensor noise, and up to two short
    spatial bursts so the clean-window calibration has contrast to find.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    base = 0.6 * np.sin(2 * np.pi * 10 * t) + 0.15 * np.sin(2 * np.pi * 6.5 * t)

    data = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[c] = (
            base
            + 0.6 * np.sin(2 * np.pi * 10 * t + phase)
            + 0.05 * rng.standard_normal(n_samples)
        )

    # Inject up to two short bursts; skip any that do not fit the trial length.
    burst_samples = max(1, int(0.6 * sfreq))
    for start in (int(4 * sfreq), int(15 * sfreq)):
        if start + burst_samples > n_samples:
            continue
        stop = min(start + burst_samples, n_samples)
        spatial = rng.standard_normal(n_channels)
        spatial /= max(np.linalg.norm(spatial), 1e-12)
        temporal = rng.standard_normal(stop - start)
        data[:, start:stop] += 7.0 * np.outer(spatial, temporal)
    return data


def test_mw_single_window_equals_psp():
    """One window covering all data should reproduce PSP exactly."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=11)

    # mw_window_length longer than the recording -> a single calibration window.
    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=data.shape[1] / sfreq + 1.0,
        verbose=False,
    )
    mw.fit(data)

    psp = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    psp.fit(data)

    assert len(mw.mw_diagnostics_) == 1
    assert mw.mw_diagnostics_[0]["status"] == "passed"
    assert mw.calibration_info_["adaptive_variant"] == "mw"
    # State is byte-for-byte identical: same data, same single calibration.
    assert_allclose(mw.M_, psp.M_, rtol=0.0, atol=1e-12)
    assert_allclose(mw.T_, psp.T_, rtol=0.0, atol=1e-12)
    assert_allclose(mw.thresholds_, psp.thresholds_, rtol=0.0, atol=1e-12)


def test_mw_three_windows_final_state_equals_psp_on_last_window():
    """Final state equals PSP on the last window only (demo Cell 4)."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=22)
    win_s = (data.shape[1] / sfreq) / 3.0  # ~8 s -> 3 windows over 24 s

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=win_s,
        verbose=False,
    )
    mw.fit(data)

    assert len(mw.mw_diagnostics_) == 3
    last_diag = mw.mw_diagnostics_[-1]
    assert last_diag["status"] == "passed"

    # Re-calibrate PSP on the last window alone and compare against MW state.
    last_window = data[:, last_diag["window_start"] : last_diag["window_stop"]]
    psp_last = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    psp_last.fit(last_window)

    assert_allclose(mw.M_, psp_last.M_, rtol=0.0, atol=1e-12)
    assert_allclose(mw.T_, psp_last.T_, rtol=0.0, atol=1e-12)
    assert mw.calibration_info_["mw_n_windows"] == 3
    assert mw.calibration_info_["mw_window_length_s"] == pytest.approx(win_s)


def test_mw_partial_fit_raises_not_implemented():
    """partial_fit is disabled for MW, which re-calibrates per fit() call."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=33)
    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        verbose=False,
    )
    mw.fit(data)

    extra = _make_synthetic(n_samples=2000, sfreq=sfreq, seed=44)
    with pytest.raises(NotImplementedError, match="variant='mw'"):
        mw.partial_fit(extra)


def test_mw_diagnostics_empty_for_psp_psw():
    """mw_diagnostics_ is always defined (empty list) for non-MW variants."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=55)
    for variant in ("psp", "psw"):
        asr = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant=variant, verbose=False)
        asr.fit(data)
        assert asr.mw_diagnostics_ == []


def test_mw_sliding_invalid_mw_mode_raises():
    """mw_mode must be 'final_state' or 'sliding' when variant='mw'."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=66)
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        mw_mode="bogus",
        verbose=False,
    )
    with pytest.raises(ValueError, match="mw_mode must be"):
        asr.fit(data)


def test_mw_sliding_fit_transform_returns_cleaned_shape():
    """Sliding fit_transform returns a finite cleaned array of input shape."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=77)
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        mw_mode="sliding",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.calibration_info_["mw_mode"] == "sliding"
    assert asr.calibration_info_["adaptive_variant"] == "mw"


def test_mw_sliding_diagnostics_one_per_window():
    """Sliding mode records one mw_diagnostics_ entry per processing window."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=88)
    win_s = (data.shape[1] / sfreq) / 3.0  # 3 windows
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=win_s,
        mw_mode="sliding",
        verbose=False,
    )
    asr.fit_transform(data)

    assert len(asr.mw_diagnostics_) == 3
    statuses = [d["status"] for d in asr.mw_diagnostics_]
    assert all(s in ("passed", "skipped_too_short", "failed") for s in statuses)


def test_mw_sliding_single_window_equals_psp_fit_transform():
    """One covering window in sliding mode equals PSP fit_transform.

    Sliding mode calibrates each window on itself and cleans that window with
    the local calibration. With a single window covering all the data this is
    exactly ``AdaptiveASR(variant="psp").fit_transform(data)`` -- calibrate on
    all data, reconstruct all data. PSP is MATLAB-parity-tested, so this
    equality transitively validates the per-window calibrate-and-clean path.
    """
    sfreq = 250.0
    data = _make_synthetic(n_samples=5000, sfreq=sfreq, seed=123)

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=data.shape[1] / sfreq + 1.0,  # one covering window
        mw_mode="sliding",
        verbose=False,
    )
    cleaned_mw = mw.fit_transform(data)

    psp = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    cleaned_psp = psp.fit_transform(data)

    relerr = np.linalg.norm(cleaned_mw - cleaned_psp) / max(
        np.linalg.norm(cleaned_psp), np.finfo(float).eps
    )
    assert relerr < 1e-10, (
        f"sliding single-window output diverged from PSP fit_transform: "
        f"relerr={relerr:.3e}"
    )


def test_mw_sliding_default_final_state_unchanged():
    """Default mw_mode stays 'final_state' (sliding is opt-in).

    Regression guard so the sliding implementation never silently becomes the
    default behavior.
    """
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=99)

    # No mw_mode specified -> default final_state.
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        verbose=False,
    )
    asr.fit(data)

    assert asr.calibration_info_.get("mw_mode", "final_state") != "sliding"
    assert "mw_n_windows" in asr.calibration_info_


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


def _inject_bursts(
    data: np.ndarray,
    n_bursts: int = 8,
    burst_duration_s: float = 0.5,
    amplitude: float = 12.0,
    sfreq: float = 250.0,
    seed: int = 97,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    burst_len = int(round(burst_duration_s * sfreq))
    n_times = data.shape[1]
    starts = np.linspace(burst_len, n_times - burst_len, n_bursts).astype(int)
    contaminated = data.copy()
    scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_times)
        spatial = rng.standard_normal(data.shape[0])
        spatial /= max(np.linalg.norm(spatial), 1e-12)
        temporal = rng.standard_normal(stop - start)
        contaminated[:, start:stop] += amplitude * scale * np.outer(spatial, temporal)
    return contaminated


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


def test_adaptive_window_criterion_on_epochs_populates_rejection():
    epo = _epochs()
    aasr = AdaptiveASR(
        sfreq=SFREQ,
        cutoff=10.0,
        variant="psp",
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    out = aasr.fit_transform(epo)
    assert out.get_data().shape == epo.get_data().shape
    diag = aasr.get_diagnostics()
    assert "rejection_sample_mask" in diag


def test_adaptive_mw_bad_window_length_raises():
    with pytest.raises(ValueError, match="mw_window_length"):
        AdaptiveASR(
            sfreq=SFREQ,
            variant="mw",
            mw_window_length=-1.0,
            verbose=False,
        ).fit(_eeg())


def test_adaptive_mw_bad_mode_raises():
    with pytest.raises(ValueError, match="mw_mode"):
        AdaptiveASR(
            sfreq=SFREQ,
            variant="mw",
            mw_mode="bogus",
            verbose=False,
        ).fit(_eeg())


def test_adaptive_max_dims_zero_is_identity():
    X = _eeg()
    aasr = AdaptiveASR(sfreq=SFREQ, variant="psp", max_dims=0.0, verbose=False)
    cleaned = np.asarray(aasr.fit_transform(X))
    np.testing.assert_allclose(cleaned, X, atol=1e-9)


def test_adaptive_partial_fit_calibration_mask():
    X = _eeg(n_times=8000)
    aasr = AdaptiveASR(sfreq=SFREQ, variant="psp", verbose=False)
    aasr.fit(X[:, :4000])
    with pytest.raises(ValueError, match="calibration_mask must have shape"):
        aasr.partial_fit(X[:, 4000:], calibration_mask=np.ones(10, dtype=bool))
    mask = np.ones(4000, dtype=bool)
    mask[:200] = False
    aasr.partial_fit(X[:, 4000:], calibration_mask=mask)
    assert aasr.calibration_mask_kind_ == "window"


@pytest.mark.parametrize("variant", ["psp", "psw"])
def test_adaptive_partial_fit_short_chunk_is_atomic(variant):
    """An incomplete update segment must not mutate the fitted adaptive state."""
    X = _eeg(n_times=8000)
    aasr = AdaptiveASR(sfreq=SFREQ, variant=variant, verbose=False).fit(X[:, :4000])
    state_before = {
        "M": aasr.M_.copy(),
        "T": aasr.T_.copy(),
        "thresholds": aasr.thresholds_.copy(),
        "learner_M": aasr.adaptive_learner_.M.copy(),
        "learner_W": aasr.adaptive_learner_.W.copy(),
        "learner_Minv": aasr.adaptive_learner_.Minv.copy(),
    }

    with pytest.raises(ValueError, match="requires at least 251 samples"):
        aasr.partial_fit(X[:, 4000:4250])

    np.testing.assert_array_equal(aasr.M_, state_before["M"])
    np.testing.assert_array_equal(aasr.T_, state_before["T"])
    np.testing.assert_array_equal(aasr.thresholds_, state_before["thresholds"])
    np.testing.assert_array_equal(aasr.adaptive_learner_.M, state_before["learner_M"])
    np.testing.assert_array_equal(aasr.adaptive_learner_.W, state_before["learner_W"])
    np.testing.assert_array_equal(
        aasr.adaptive_learner_.Minv, state_before["learner_Minv"]
    )


@pytest.mark.parametrize("variant", ["psp", "psw"])
def test_adaptive_fit_rejects_segment_without_clean_window(variant):
    X = _eeg(n_times=250, bursts=0)
    with pytest.raises(ValueError, match=r"fit\(\) requires at least 251 samples"):
        AdaptiveASR(sfreq=SFREQ, variant=variant, verbose=False).fit(X)


@pytest.mark.parametrize("variant", ["psp", "psw"])
def test_adaptive_partial_fit_threshold_failure_is_atomic(variant, monkeypatch):
    """A downstream fit failure must not commit the candidate learner update."""
    X = _eeg(n_times=8000)
    aasr = AdaptiveASR(sfreq=SFREQ, variant=variant, verbose=False).fit(X[:, :4000])
    state_before = {
        "M": aasr.M_.copy(),
        "T": aasr.T_.copy(),
        "thresholds": aasr.thresholds_.copy(),
        "learner_M": aasr.adaptive_learner_.M.copy(),
        "learner_W": aasr.adaptive_learner_.W.copy(),
        "learner_Minv": aasr.adaptive_learner_.Minv.copy(),
    }

    def _fail_threshold_fit(*args, **kwargs):
        raise RuntimeError("threshold fit failed")

    monkeypatch.setattr(
        "mne_denoise.asr.adaptive._fit_adaptive_thresholds", _fail_threshold_fit
    )
    with pytest.raises(RuntimeError, match="threshold fit failed"):
        aasr.partial_fit(X[:, 4000:])

    np.testing.assert_array_equal(aasr.M_, state_before["M"])
    np.testing.assert_array_equal(aasr.T_, state_before["T"])
    np.testing.assert_array_equal(aasr.thresholds_, state_before["thresholds"])
    np.testing.assert_array_equal(aasr.adaptive_learner_.M, state_before["learner_M"])
    np.testing.assert_array_equal(aasr.adaptive_learner_.W, state_before["learner_W"])
    np.testing.assert_array_equal(
        aasr.adaptive_learner_.Minv, state_before["learner_Minv"]
    )


def test_adaptive_transform_raw_with_window_criterion():
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=8000, bursts=8)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(X * 1e-6, info, verbose=False)
    aasr = AdaptiveASR(
        sfreq=SFREQ,
        cutoff=10.0,
        variant="psp",
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    aasr.fit_transform(raw)
    diag = aasr.get_diagnostics()
    assert "rejection_sample_mask" in diag


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"update_window_length": -1.0}, "update_window_length"),
        ({"calibration_window_length": -1.0}, "clean_window_length"),
        ({"calibration_window_overlap": 1.5}, "clean_window_overlap"),
        ({"ref_max_bad_channels": -0.1}, "clean_max_bad_channels"),
        ({"learning_rate": -1.0}, "learning_rate"),
        ({"tau": -1.0}, "tau must be positive"),
    ],
)
def test_adaptive_validate_param_guards(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        AdaptiveASR(sfreq=SFREQ, variant="psp", verbose=False, **kwargs).fit(_eeg())


@pytest.mark.parametrize("length_s", [5.0, 20.0, 40.0])
@pytest.mark.parametrize("mode", ["final_state", "sliding"])
def test_mw_window_length_parametric(length_s, mode):
    clean = _make_clean_eeg(duration_s=120.0)
    dirty = _inject_bursts(clean, n_bursts=20, sfreq=250.0)
    asr = AdaptiveASR(
        sfreq=250.0,
        cutoff=20.0,
        variant="mw",
        mw_window_length=length_s,
        mw_mode=mode,
        verbose=False,
    )
    if mode == "sliding":
        cleaned = asr.fit_transform(dirty)
    else:
        asr.fit(dirty)
        cleaned = asr.transform(dirty)
    assert cleaned.shape == dirty.shape
    assert np.all(np.isfinite(cleaned))
    # Number of windows should be > 0
    assert len(asr.mw_diagnostics_) > 0


def test_adaptive_asr_psw_updates_and_reduces_bursts(synthetic_burst_data):
    """Adaptive PSW-ASR updates thresholds and suppresses burst residuals."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psw",
        verbose=False,
    )
    asr.fit(data[:, : int(4 * sfreq)])
    initial_T = asr.T_.copy()
    asr.partial_fit(data[:, int(4 * sfreq) : int(8 * sfreq)])

    assert len(asr.adaptive_update_history_) == 2
    assert asr.calibration_info_["event"] == "update"
    assert asr.calibration_info_["adaptive_variant"] == "psw"
    assert not np.allclose(initial_T, asr.T_)

    asr.reset_process_state()
    cleaned = asr.transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.diagnostics_["adaptive_variant"] == "psw"
    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


def test_adaptive_asr_reset_process_state_is_reproducible(synthetic_burst_data):
    """Resetting adaptive process state restores deterministic replay."""
    data, _, _, sfreq = synthetic_burst_data
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psp",
        verbose=False,
    )
    asr.fit(data[:, : int(6 * sfreq)])
    cleaned_first = asr.transform(data)
    asr.reset_process_state()
    cleaned_second = asr.transform(data)
    np.testing.assert_allclose(cleaned_first, cleaned_second, atol=1e-10)


def test_adaptive_asr_low_memory_matches_full_path(synthetic_burst_data):
    """Adaptive ASR honors max_mem_mb without changing reconstruction."""
    data, _, _, sfreq = synthetic_burst_data
    calibration = data[:, : int(6 * sfreq)]

    full = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psw",
        max_mem_mb=None,
        verbose=False,
    )
    low_mem = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psw",
        max_mem_mb=0.001,
        verbose=False,
    )
    full.fit(calibration)
    low_mem.fit(calibration)

    cleaned_full = full.transform(data)
    cleaned_low_mem = low_mem.transform(data)

    assert full.calibration_info_["memory_mode"] == "full"
    assert low_mem.calibration_info_["memory_mode"] == "chunked"
    assert full.diagnostics_["memory_mode"] == "full"
    assert low_mem.diagnostics_["memory_mode"] == "chunked"
    assert low_mem.diagnostics_["used_memory_bound"]
    np.testing.assert_allclose(cleaned_low_mem, cleaned_full, atol=1e-10)


def test_adaptive_asr_mne_raw_preserves_non_picked_channels(synthetic_burst_data):
    """Adaptive ASR cleans EEG picks and preserves non-picked channels."""
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

    asr = AdaptiveASR(cutoff=5.0, variant="psp", verbose=False)
    raw_clean = asr.fit_transform(raw)

    assert isinstance(raw_clean, mne.io.RawArray)
    assert raw_clean.get_data().shape == raw_data.shape
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1", "EOG2"]), eog)
    assert asr.ch_names_ == ch_names[: data.shape[0]]


def test_adaptive_edge_cases():
    mne = pytest.importorskip("mne")
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=1)

    info = mne.create_info([f"EEG{i}" for i in range(8)], sfreq, "eeg")
    evoked = mne.EvokedArray(data, info)
    asr = AdaptiveASR(sfreq=sfreq, variant="psp", verbose=False)

    with pytest.raises(ValueError, match="does not support Evoked"):
        asr.fit(evoked)

    with pytest.raises(ValueError, match="calibration_mask must have shape"):
        asr.fit(data, calibration_mask=np.ones(10, dtype=bool))

    mask = np.ones(data.shape[1], dtype=bool)
    mask[:200] = False
    asr.fit(data, calibration_mask=mask)

    asr.fit(data)

    asr2 = AdaptiveASR(sfreq=sfreq, variant="psp", verbose=False)
    asr2.partial_fit(data)

    with pytest.raises(ValueError, match="does not support Evoked"):
        asr.partial_fit(evoked)

    info_bad = mne.create_info([f"EEG{i}" for i in range(8)], 100.0, "eeg")
    raw_bad = mne.io.RawArray(data, info_bad)
    with pytest.raises(ValueError, match="does not match fitted sfreq"):
        asr.partial_fit(raw_bad)

    epochs = _epochs()
    asr.partial_fit(epochs)

    annot = mne.Annotations(onset=[1.0], duration=[1.0], description=["BAD_TEST"])
    raw_annot = mne.io.RawArray(data, info)
    raw_annot.set_annotations(annot)
    asr3 = AdaptiveASR(
        sfreq=sfreq, variant="psp", reject_by_annotation=True, verbose=False
    )
    asr3.fit(raw_annot)
    asr3.partial_fit(raw_annot)

    with pytest.raises(ValueError, match="does not support Evoked"):
        asr.transform(evoked)

    with pytest.raises(ValueError, match="does not match fitted sfreq"):
        asr.transform(raw_bad)

    asr4 = AdaptiveASR(
        sfreq=sfreq,
        variant="psp",
        reject_by_annotation=True,
        window_criterion=0.3,
        verbose=False,
    )
    asr4.fit(raw_annot)
    asr4.transform(raw_annot)

    asr4_false = AdaptiveASR(
        sfreq=sfreq,
        variant="psp",
        reject_by_annotation=False,
        window_criterion=0.3,
        verbose=False,
    )
    asr4_false.fit(raw_annot)
    asr4_false.transform(raw_annot)

    asr5 = AdaptiveASR(sfreq=sfreq, variant="psp", window_criterion=0.3, verbose=False)
    asr5.fit(epochs)
    asr5.transform(epochs)

    cleaned, diag = asr.transform(data, return_diagnostics=True)
    assert isinstance(diag, dict)

    asr.reset_process_state()

    mw = AdaptiveASR(sfreq=sfreq, variant="mw", mw_mode="sliding", verbose=False)
    with pytest.raises(ValueError, match="does not support Evoked"):
        mw.fit_transform(evoked)

    mw.fit_transform(epochs)
    _, mw_diag = mw.fit_transform(data, return_diagnostics=True)
    assert isinstance(mw_diag, dict)

    asr_epochs = AdaptiveASR(sfreq=sfreq, variant="psp", verbose=False)
    asr_epochs.fit(epochs)
    asr_epochs.transform(epochs)

    mw2 = AdaptiveASR(
        sfreq=sfreq,
        variant="mw",
        mw_mode="sliding",
        reject_by_annotation=True,
        verbose=False,
    )
    mw2.fit_transform(raw_annot)

    mw3 = AdaptiveASR(
        sfreq=sfreq, variant="mw", mw_mode="sliding", blocksize=100000, verbose=False
    )
    with pytest.raises(RuntimeError, match="MW-ASR sliding-mode fit_transform"):
        mw3.fit_transform(data)

    mw4 = AdaptiveASR(sfreq=sfreq, variant="mw", blocksize=100000, verbose=False)
    with pytest.raises(RuntimeError, match="MW-ASR fit.* no usable window"):
        mw4.fit(data)

    import unittest.mock as mock

    with mock.patch.object(
        AdaptiveASR, "_fit_adaptive_state", side_effect=ValueError("mock error")
    ):
        mw5 = AdaptiveASR(sfreq=sfreq, variant="mw", verbose=False)
        with pytest.raises(RuntimeError):
            mw5.fit(data)

    asr_tau = AdaptiveASR(sfreq=sfreq, variant="psp", tau=0.5, verbose=False)
    assert asr_tau._resolved_tau() == 0.5

    asr_tau_none = AdaptiveASR(sfreq=sfreq, variant="psp", tau=None, verbose=False)
    assert asr_tau_none._resolved_tau() == 0.8
