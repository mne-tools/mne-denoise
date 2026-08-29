"""Unit tests for the adaptive ASR."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.asr import AdaptiveASR

SFREQ = 250.0


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


def test_adaptive_summary_identifies_variant_and_fit_state(caplog):
    """AdaptiveASR summaries identify each supported fitted variant."""
    for variant in ("psp", "psw", "mw"):
        data = _make_synthetic(n_samples=4000, seed=101)
        kwargs = {"mw_window_length": 20.0} if variant == "mw" else {}
        estimator = AdaptiveASR(
            sfreq=SFREQ,
            cutoff=20.0,
            variant=variant,
            verbose=False,
            **kwargs,
        )
        with caplog.at_level(logging.INFO, logger="mne_denoise"):
            estimator.fit(data, verbose=True)
        summaries = [
            record
            for record in caplog.records
            if record.message.startswith("AdaptiveASR:")
            and "clean calibration windows=" in record.message
        ]
        assert len(summaries) == 1, variant
        for token in (
            f"variant={variant}",
            "method=",
            "channels=",
            "sfreq=",
            "cutoff=",
            "rank=",
        ):
            assert token in summaries[0].message
        if variant == "mw":
            assert estimator.mw_diagnostics_
        else:
            assert estimator.mw_diagnostics_ == []
        caplog.clear()


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
    """Default final-state mode equals PSP on the last calibration window."""
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
    assert mw.mw_mode == "final_state"
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


def test_adaptive_identity_transform_emits_no_window_progress(synthetic_burst_data):
    """The adaptive identity path has no reconstruction progress units."""
    data, _, _, sfreq = synthetic_burst_data
    asr = AdaptiveASR(
        sfreq=sfreq,
        variant="psp",
        max_dims=0.0,
        verbose=False,
    ).fit(data)
    events = []
    cleaned = asr.transform(data, callback=events.append)

    assert events == []
    np.testing.assert_allclose(cleaned, data)


def test_adaptive_transform_callback_exception_does_not_advance_state(
    synthetic_burst_data,
):
    """A failed callback leaves the adaptive streaming state unchanged."""
    data, _, _, sfreq = synthetic_burst_data
    asr = AdaptiveASR(
        sfreq=sfreq,
        variant="psp",
        lookahead=0.0,
        stepsize=100,
        verbose=False,
    ).fit(data)
    state_before = {
        key: None if value is None else np.asarray(value).copy()
        for key, value in asr.process_state_.items()
        if key != "last_trivial"
    }
    state_before["last_trivial"] = asr.process_state_["last_trivial"]
    sentinel = RuntimeError("adaptive reconstruction callback failed")

    def callback(event):
        del event
        raise sentinel

    with pytest.raises(RuntimeError):
        asr.transform(data, callback=callback)
    for key, expected in state_before.items():
        actual = asr.process_state_[key]
        if expected is None:
            assert actual is None
        elif key == "last_trivial":
            assert actual == expected
        else:
            np.testing.assert_array_equal(actual, expected)


def test_adaptive_fit_transform_combines_calibration_and_window_progress(
    synthetic_burst_data,
):
    """Non-MW adaptive fit_transform orders calibration before reconstruction."""
    data, _, _, sfreq = synthetic_burst_data
    kwargs = {
        "sfreq": sfreq,
        "variant": "psp",
        "max_dims": 0.5,
        "lookahead": 0.0,
        "stepsize": 100,
        "verbose": False,
    }
    events = []
    with_callback = AdaptiveASR(**kwargs)
    cleaned, diagnostics = with_callback.fit_transform(
        data,
        callback=events.append,
        return_diagnostics=True,
    )

    n_components = with_callback.thresholds_.size
    calibration_events = events[:n_components]
    assert [event.method for event in calibration_events] == [
        "adaptive_asr"
    ] * n_components
    assert [event.stage for event in calibration_events] == [
        "calibration"
    ] * n_components
    assert [event.component for event in calibration_events] == list(
        range(1, n_components + 1)
    )
    window_events = events[n_components:]
    assert len(window_events) == diagnostics["n_windows"]
    assert all(event.method == "adaptive_asr" for event in events)
    assert all(event.stage == "window" for event in window_events)
    assert [event.current for event in window_events] == list(
        range(1, len(window_events) + 1)
    )
    assert cleaned.shape == data.shape


@pytest.mark.parametrize("mw_mode", ["final_state", "sliding"])
def test_adaptive_mw_fit_transform_reports_mode_specific_events(mw_mode):
    """MW modes expose outer attempts and final-state transitions distinctly."""
    data = _make_synthetic(n_samples=2250, seed=125)
    asr = AdaptiveASR(
        sfreq=SFREQ,
        variant="mw",
        mw_window_length=4.0,
        blocksize=500,
        mw_mode=mw_mode,
        verbose=False,
    )
    events = []
    cleaned = asr.fit_transform(data, callback=events.append)

    assert cleaned.shape == data.shape
    assert len(asr.mw_diagnostics_) <= 3
    assert all(event.method == "adaptive_asr" for event in events)
    diagnostics_by_window = {
        entry["window_idx"]: entry for entry in asr.mw_diagnostics_
    }
    assert all(
        entry["status"] in ("passed", "skipped_too_short", "failed")
        for entry in asr.mw_diagnostics_
    )
    if mw_mode == "final_state":
        calibration_events = events[:3]
        reconstruction_events = events[3:]
        assert calibration_events
        assert reconstruction_events
        assert all(event.stage == "calibration" for event in calibration_events)
        assert all(event.stage == "window" for event in reconstruction_events)
        assert [event.current for event in calibration_events] == [1, 2, 3]
        assert [event.total for event in calibration_events] == [3, 3, 3]
        assert all(event.component is None for event in calibration_events)
        for event in calibration_events:
            entry = diagnostics_by_window.get(event.current - 1)
            if entry is not None and entry["status"] == "passed":
                assert event.metric == pytest.approx(float(entry["rank"]))
            else:
                assert event.metric is None
    else:
        assert len(events) == 3
        assert all(event.stage == "window" for event in events)
        assert [event.current for event in events] == [1, 2, 3]
        assert [event.total for event in events] == [3, 3, 3]
        assert all(event.component is None for event in events)
        for event in events:
            entry = diagnostics_by_window.get(event.current - 1)
            if entry is not None and entry["status"] == "passed":
                assert event.metric == pytest.approx(float(entry["rank"]))
            else:
                assert event.metric is None


def test_adaptive_mw_fit_reports_failed_window(monkeypatch):
    """A failed MW calibration window is recorded before later windows proceed."""
    data = _make_synthetic(n_samples=3000, seed=126)
    asr = AdaptiveASR(
        sfreq=SFREQ,
        variant="mw",
        mw_window_length=4.0,
        mw_mode="final_state",
        verbose=False,
    )
    original_fit = asr._fit_adaptive_state
    calls = 0

    def fail_first(window, sfreq, callback=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic MW calibration failure")
        return original_fit(window, sfreq, callback=callback)

    monkeypatch.setattr(asr, "_fit_adaptive_state", fail_first)
    events = []
    asr.fit(data, callback=events.append)
    assert events[0].metric is None
    assert asr.mw_diagnostics_[0]["status"] == "failed"


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


def test_adaptive_partial_fit_short_chunk_is_atomic():
    """An incomplete update segment must not mutate the fitted adaptive state."""
    for variant in ("psp", "psw"):
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
        np.testing.assert_array_equal(
            aasr.adaptive_learner_.M, state_before["learner_M"]
        )
        np.testing.assert_array_equal(
            aasr.adaptive_learner_.W, state_before["learner_W"]
        )
        np.testing.assert_array_equal(
            aasr.adaptive_learner_.Minv, state_before["learner_Minv"]
        )


def test_adaptive_fit_rejects_segment_without_clean_window():
    for variant in ("psp", "psw"):
        X = _eeg(n_times=250, bursts=0)
        with pytest.raises(ValueError, match=r"fit\(\) requires at least 251 samples"):
            AdaptiveASR(sfreq=SFREQ, variant=variant, verbose=False).fit(X)


def test_adaptive_partial_fit_threshold_failure_is_atomic(monkeypatch):
    """A downstream fit failure must not commit the candidate learner update."""
    for variant in ("psp", "psw"):
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

        with monkeypatch.context() as patch:
            patch.setattr(
                "mne_denoise.asr.adaptive._fit_adaptive_thresholds",
                _fail_threshold_fit,
            )
            with pytest.raises(RuntimeError, match="threshold fit failed"):
                aasr.partial_fit(X[:, 4000:])

        np.testing.assert_array_equal(aasr.M_, state_before["M"])
        np.testing.assert_array_equal(aasr.T_, state_before["T"])
        np.testing.assert_array_equal(aasr.thresholds_, state_before["thresholds"])
        np.testing.assert_array_equal(
            aasr.adaptive_learner_.M, state_before["learner_M"]
        )
        np.testing.assert_array_equal(
            aasr.adaptive_learner_.W, state_before["learner_W"]
        )
        np.testing.assert_array_equal(
            aasr.adaptive_learner_.Minv, state_before["learner_Minv"]
        )


def test_adaptive_validate_param_guards():
    cases = [
        ("psp", {"update_window_length": -1.0}, "update_window_length"),
        ("psp", {"calibration_window_length": -1.0}, "clean_window_length"),
        ("psp", {"calibration_window_overlap": 1.5}, "clean_window_overlap"),
        ("psp", {"ref_max_bad_channels": -0.1}, "clean_max_bad_channels"),
        ("psp", {"learning_rate": -1.0}, "learning_rate"),
        ("psp", {"tau": -1.0}, "tau must be positive"),
        ("mw", {"mw_window_length": -1.0}, "mw_window_length"),
        ("mw", {"mw_window_length": 8.0, "mw_mode": "bogus"}, "mw_mode"),
    ]
    for variant, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            AdaptiveASR(sfreq=SFREQ, variant=variant, verbose=False, **kwargs).fit(
                _eeg()
            )


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
