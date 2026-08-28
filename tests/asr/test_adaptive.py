"""Unit tests for the adaptive ASR."""

from __future__ import annotations

import inspect
import logging

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


@pytest.mark.parametrize("variant", ["psp", "psw", "mw"])
def test_adaptive_summary_identifies_variant_and_fit_state(variant, caplog):
    """AdaptiveASR INFO includes the variant and fitted operating point."""
    data = _make_synthetic(n_samples=4000, seed=101)
    kwargs = {"mw_window_length": 20.0} if variant == "mw" else {}
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        AdaptiveASR(
            sfreq=SFREQ,
            cutoff=20.0,
            variant=variant,
            verbose=False,
            **kwargs,
        ).fit(data, verbose=True)
    summaries = [
        record
        for record in caplog.records
        if record.message.startswith("AdaptiveASR:")
        and "clean calibration windows=" in record.message
    ]
    assert len(summaries) == 1
    for token in (
        f"variant={variant}",
        "method=",
        "channels=",
        "sfreq=",
        "cutoff=",
        "rank=",
    ):
        assert token in summaries[0].message


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


@pytest.mark.parametrize("variant", ["psp", "psw"])
def test_adaptive_fit_progress_is_numerically_transparent(variant):
    """PSP/PSW fit emits one adaptive calibration event per component."""
    data = _make_synthetic(n_samples=3000, seed=123)
    kwargs = {"sfreq": SFREQ, "variant": variant, "verbose": False}
    events = []
    with_callback = AdaptiveASR(**kwargs)
    with_callback.fit(data, callback=events.append)
    without_callback = AdaptiveASR(**kwargs).fit(data)

    n_components = with_callback.thresholds_.size
    assert len(events) == n_components
    assert [event.method for event in events] == ["adaptive_asr"] * n_components
    assert [event.stage for event in events] == ["calibration"] * n_components
    assert [event.current for event in events] == list(range(1, n_components + 1))
    assert [event.total for event in events] == [n_components] * n_components
    assert [event.component for event in events] == list(range(1, n_components + 1))
    np.testing.assert_allclose(
        [event.metric for event in events],
        with_callback.thresholds_,
        rtol=0.0,
        atol=1e-12,
    )

    for name in ("M_", "T_", "thresholds_", "patterns_"):
        np.testing.assert_allclose(
            getattr(with_callback, name), getattr(without_callback, name)
        )
    assert with_callback.rank_ == without_callback.rank_


def test_adaptive_callback_signature_and_validation():
    """Adaptive public callbacks are runtime parameters, not estimator state."""
    assert "callback" not in inspect.signature(AdaptiveASR.__init__).parameters
    assert "callback" not in inspect.signature(AdaptiveASR.partial_fit).parameters
    for method in (AdaptiveASR.fit, AdaptiveASR.transform, AdaptiveASR.fit_transform):
        parameter = inspect.signature(method).parameters["callback"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    with pytest.raises(TypeError, match="callback must be callable or None"):
        AdaptiveASR(sfreq=SFREQ, variant="psp", verbose=False).fit(
            _make_synthetic(n_samples=1000), callback=1
        )


def test_adaptive_continuous_transform_progress_is_numerically_transparent(
    synthetic_burst_data,
):
    """Adaptive continuous reconstruction reports ordered window progress."""
    data, _, _, sfreq = synthetic_burst_data
    kwargs = {
        "sfreq": sfreq,
        "variant": "psp",
        "max_dims": 0.5,
        "lookahead": 0.0,
        "stepsize": 100,
        "verbose": False,
    }
    with_callback = AdaptiveASR(**kwargs).fit(data)
    without_callback = AdaptiveASR(**kwargs).fit(data)

    events = []
    cleaned, diagnostics = with_callback.transform(
        data,
        callback=events.append,
        return_diagnostics=True,
    )
    reference_cleaned, reference_diagnostics = without_callback.transform(
        data,
        return_diagnostics=True,
    )

    n_windows = diagnostics["n_windows"]
    assert len(events) == n_windows
    assert all(event.method == "adaptive_asr" for event in events)
    assert all(event.stage == "window" for event in events)
    assert [event.current for event in events] == list(range(1, n_windows + 1))
    assert [event.total for event in events] == [n_windows] * n_windows
    assert all(event.component is None for event in events)
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
    for key in ("cov", "carry", "iir", "last_R"):
        actual = with_callback.process_state_[key]
        expected = without_callback.process_state_[key]
        if actual is None:
            assert expected is None
        else:
            np.testing.assert_allclose(actual, expected)
    assert (
        with_callback.process_state_["last_trivial"]
        == without_callback.process_state_["last_trivial"]
    )


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

    with pytest.raises(RuntimeError) as caught:
        asr.transform(data, callback=callback)
    assert caught.value is sentinel
    for key, expected in state_before.items():
        actual = asr.process_state_[key]
        if expected is None:
            assert actual is None
        elif key == "last_trivial":
            assert actual == expected
        else:
            np.testing.assert_array_equal(actual, expected)


def test_adaptive_epoched_transform_reports_outer_epochs_only():
    """Adaptive epoched reconstruction emits no nested window events."""
    mne = pytest.importorskip("mne")
    data = _make_synthetic(n_samples=3000, seed=124)
    n_epochs = 3
    info = mne.create_info(data.shape[0], SFREQ, "eeg")
    epochs = mne.EpochsArray(
        data.reshape(data.shape[0], n_epochs, -1).transpose(1, 0, 2) * 1e-6,
        info,
        verbose=False,
    )
    asr = AdaptiveASR(
        sfreq=SFREQ,
        variant="psp",
        lookahead=0.0,
        stepsize=100,
        verbose=False,
    ).fit(epochs)

    events = []
    _, diagnostics = asr.transform(
        epochs,
        callback=events.append,
        return_diagnostics=True,
    )

    assert len(events) == n_epochs
    assert [event.method for event in events] == ["adaptive_asr"] * n_epochs
    assert [event.stage for event in events] == ["epoch"] * n_epochs
    assert [event.current for event in events] == list(range(1, n_epochs + 1))
    assert [event.total for event in events] == [n_epochs] * n_epochs
    assert all(event.component is None for event in events)
    np.testing.assert_allclose(
        [event.metric for event in events],
        [
            diag["fraction_reconstructed_samples"]
            for diag in diagnostics["epoch_diagnostics"]
        ],
    )


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
    without_callback = AdaptiveASR(**kwargs)
    reference_cleaned, reference_diagnostics = without_callback.fit_transform(
        data,
        return_diagnostics=True,
    )

    n_components = with_callback.thresholds_.size
    assert [event.stage for event in events[:n_components]] == [
        "calibration"
    ] * n_components
    window_events = events[n_components:]
    assert len(window_events) == diagnostics["n_windows"]
    assert all(event.method == "adaptive_asr" for event in events)
    assert all(event.stage == "window" for event in window_events)
    assert [event.current for event in window_events] == list(
        range(1, len(window_events) + 1)
    )
    np.testing.assert_allclose(cleaned, reference_cleaned)
    np.testing.assert_array_equal(
        diagnostics["n_components_reconstructed"],
        reference_diagnostics["n_components_reconstructed"],
    )


@pytest.mark.parametrize("mw_mode", ["final_state", "sliding"])
def test_adaptive_mw_fit_reports_outer_windows_including_skips(mw_mode):
    """MW fit emits calibration progress for every outer attempt."""
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
    asr.fit(data, callback=events.append)

    assert len(events) == 3
    assert all(event.method == "adaptive_asr" for event in events)
    assert all(event.stage == "calibration" for event in events)
    assert [event.current for event in events] == [1, 2, 3]
    assert [event.total for event in events] == [3, 3, 3]
    assert all(event.component is None for event in events)
    diagnostics_by_window = {
        entry["window_idx"]: entry for entry in asr.mw_diagnostics_
    }
    assert [entry["status"] for entry in asr.mw_diagnostics_] == [
        "passed",
        "passed",
    ]
    for event in events[:2]:
        assert event.metric == pytest.approx(
            float(diagnostics_by_window[event.current - 1]["rank"])
        )
    assert events[2].metric is None


def test_adaptive_mw_fit_reports_failed_window_and_propagates_callback(
    monkeypatch,
):
    """MW fit keeps failed-window handling separate from callback failures."""
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

    sentinel = RuntimeError("MW fit callback failed")

    def callback(event):
        del event
        raise sentinel

    fresh = AdaptiveASR(
        sfreq=SFREQ,
        variant="mw",
        mw_window_length=4.0,
        mw_mode="final_state",
        verbose=False,
    )
    calls = 0
    original_fit = fresh._fit_adaptive_state

    def fail_first_again(window, sfreq, callback=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic MW calibration failure")
        return original_fit(window, sfreq, callback=callback)

    monkeypatch.setattr(fresh, "_fit_adaptive_state", fail_first_again)
    with pytest.raises(RuntimeError) as caught:
        fresh.fit(data, callback=callback)
    assert caught.value is sentinel


def test_adaptive_mw_final_state_fit_transform_transitions_to_window_progress():
    """Normal MW fit_transform separates calibration and reconstruction streams."""
    data = _make_synthetic(n_samples=2250, seed=129)
    kwargs = {
        "sfreq": SFREQ,
        "variant": "mw",
        "mw_window_length": 4.0,
        "blocksize": 500,
        "mw_mode": "final_state",
        "verbose": False,
    }
    events = []
    cleaned = AdaptiveASR(**kwargs).fit_transform(data, callback=events.append)

    calibration_events = [event for event in events if event.stage == "calibration"]
    reconstruction_events = [event for event in events if event.stage == "window"]
    assert calibration_events
    assert reconstruction_events
    assert events == calibration_events + reconstruction_events
    assert all(event.method == "adaptive_asr" for event in events)
    assert all(event.component is None for event in calibration_events)
    assert all(event.component is None for event in reconstruction_events)
    assert cleaned.shape == data.shape


def test_adaptive_mw_sliding_fit_transform_progress_is_outer_window_only():
    """MW sliding fit_transform suppresses inner calibration/reconstruction events."""
    data = _make_synthetic(n_samples=2250, seed=127)
    kwargs = {
        "sfreq": SFREQ,
        "variant": "mw",
        "mw_window_length": 4.0,
        "blocksize": 500,
        "mw_mode": "sliding",
        "verbose": False,
    }
    events = []
    with_callback = AdaptiveASR(**kwargs)
    cleaned = with_callback.fit_transform(data, callback=events.append)
    without_callback = AdaptiveASR(**kwargs)
    reference_cleaned = without_callback.fit_transform(data)

    assert len(events) == 3
    assert len(events) == len(with_callback.mw_diagnostics_)
    assert all(event.method == "adaptive_asr" for event in events)
    assert all(event.stage == "window" for event in events)
    assert all(event.component is None for event in events)
    assert [event.current for event in events] == [1, 2, 3]
    assert [event.total for event in events] == [3, 3, 3]
    for event, entry in zip(events, with_callback.mw_diagnostics_):
        if entry["status"] == "passed":
            assert event.metric == pytest.approx(float(entry["rank"]))
        else:
            assert entry["status"] == "skipped_too_short"
            assert event.metric is None
    np.testing.assert_allclose(cleaned, reference_cleaned)
    np.testing.assert_allclose(with_callback.M_, without_callback.M_)
    np.testing.assert_allclose(with_callback.T_, without_callback.T_)
    np.testing.assert_allclose(with_callback.thresholds_, without_callback.thresholds_)


def test_adaptive_mw_sliding_callback_exception_propagates_unchanged(monkeypatch):
    """MW sliding callback errors are not converted into failed windows."""
    data = _make_synthetic(n_samples=3000, seed=128)
    asr = AdaptiveASR(
        sfreq=SFREQ,
        variant="mw",
        mw_window_length=4.0,
        mw_mode="sliding",
        verbose=False,
    )

    def fail_first(window, sfreq, callback=None):
        del window, sfreq, callback
        raise RuntimeError("synthetic MW sliding failure")

    monkeypatch.setattr(asr, "_fit_adaptive_state", fail_first)
    sentinel = RuntimeError("MW sliding callback failed")

    def callback(event):
        del event
        raise sentinel

    with pytest.raises(RuntimeError) as caught:
        asr.fit_transform(data, callback=callback)
    assert caught.value is sentinel
    assert not hasattr(asr, "state_")


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

    with pytest.raises(
        ValueError, match="AdaptiveASR: X has 7 channels; fitted data had 8"
    ):
        asr.transform(data[:-1])

    named_asr = AdaptiveASR(sfreq=sfreq, variant="psp", verbose=False).fit(
        mne.io.RawArray(data, info, verbose=False)
    )
    with pytest.raises(ValueError, match="ASR was fitted with named channels"):
        named_asr.transform(data)
    with pytest.raises(ValueError, match="ASR was fitted with named channels"):
        named_asr.partial_fit(data)

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
