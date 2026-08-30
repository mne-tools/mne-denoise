"""Tests for mne_denoise.asr."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import ASR

SFREQ = 250.0


def test_asr_default_uses_original_spectral_shaping_filter():
    """The ASR default keeps the ASR-specific spectral shaping filter."""
    assert ASR().filter_kind == "asr"


def _epochs(n_epochs=3, n_per=2000):
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=n_epochs * n_per, bursts=6)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    data = X.reshape(8, n_epochs, n_per).transpose(1, 0, 2) * 1e-6
    return mne.EpochsArray(data, info, verbose=False)


def test_asrcore_numpy_qc_and_no_repair_cap(synthetic_burst_data):
    """Estimator path populates diagnostics and max_dims=0 preserves data."""
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=1.0,
        max_dims=0.0,
        filter_kind="none",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    np.testing.assert_allclose(cleaned, data, atol=1e-12)
    assert asr.n_windows_ > 0
    assert asr.n_components_reconstructed_.shape == (asr.n_windows_,)
    assert asr.n_components_reconstructed_.sum() == 0
    assert asr.get_calibration_mask().shape == asr.clean_window_mask_.shape


def test_asr_fit_transform_composes_calibration_and_window_progress(
    synthetic_burst_data,
):
    """ASR.fit_transform emits calibration events before window events."""
    data, _, _, sfreq = synthetic_burst_data
    kwargs = {
        "sfreq": sfreq,
        "cutoff": 3.0,
        "calibration": "manual",
        "filter_kind": "none",
        "max_dims": 0.5,
        "lookahead": 0.0,
        "stepsize": 100,
        "verbose": False,
    }
    events = []
    with_callback = ASR(**kwargs)
    _cleaned, diagnostics = with_callback.fit_transform(
        data,
        callback=events.append,
        return_diagnostics=True,
    )

    n_components = with_callback.thresholds_.size
    calibration_events = events[:n_components]
    window_events = events[n_components:]
    assert [event.method for event in calibration_events] == ["asr"] * n_components
    assert [event.stage for event in calibration_events] == [
        "calibration"
    ] * n_components
    assert len(window_events) == diagnostics["n_windows"]
    assert all(event.method == "asr" for event in window_events)
    assert all(event.stage == "window" for event in window_events)
    assert [event.current for event in window_events] == list(
        range(1, len(window_events) + 1)
    )


def test_asr_epoched_transform_reports_epochs_without_inner_windows(
    synthetic_burst_data,
):
    """Epoched transforms emit one outer event per finished epoch."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    n_epochs = 3
    n_times = data.shape[1] // n_epochs
    info = mne.create_info(data.shape[0], sfreq, "eeg")
    epochs = mne.EpochsArray(
        data.reshape(data.shape[0], n_epochs, n_times).transpose(1, 0, 2),
        info,
        verbose=False,
    )
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        calibration="manual",
        filter_kind="none",
        lookahead=0.0,
        verbose=False,
    ).fit(epochs)

    events = []
    _, diagnostics = asr.transform(
        epochs,
        callback=events.append,
        return_diagnostics=True,
    )

    assert len(events) == n_epochs
    assert [event.method for event in events] == ["asr"] * n_epochs
    assert [event.stage for event in events] == ["epoch"] * n_epochs
    assert [event.current for event in events] == list(range(1, n_epochs + 1))
    assert [event.total for event in events] == [n_epochs] * n_epochs
    assert [event.component for event in events] == [None] * n_epochs
    np.testing.assert_allclose(
        [event.metric for event in events],
        [
            diag["fraction_reconstructed_samples"]
            for diag in diagnostics["epoch_diagnostics"]
        ],
    )


def test_asr_raw_bad_annotations_are_preserved(synthetic_burst_data):
    """Bad annotated Raw spans are excluded from final replacement."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    info = mne.create_info([f"EEG{idx}" for idx in range(data.shape[0])], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[4.0], duration=[0.5], description=["bad_motion"])
    )

    asr = ASR(cutoff=2.0, filter_kind="none", verbose=False)
    raw_clean = asr.fit_transform(raw)
    bad_start = int(4.0 * sfreq)
    bad_stop = int(4.5 * sfreq)

    np.testing.assert_allclose(
        raw_clean.get_data()[:, bad_start:bad_stop],
        raw.get_data()[:, bad_start:bad_stop],
    )


def test_asr_window_criterion_exposes_rejection_samples(synthetic_burst_data):
    """Optional final window rejection exposes retained-sample masks."""
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        filter_kind="none",
        window_criterion=0.25,
        window_criterion_tolerances=(-np.inf, 2.0),
        verbose=False,
    )
    _, diagnostics = asr.fit_transform(data, return_diagnostics=True)

    rejection_mask = asr.get_rejection_mask()
    assert rejection_mask.shape == (data.shape[1],)
    assert rejection_mask.dtype == bool
    assert not np.all(rejection_mask)
    np.testing.assert_array_equal(rejection_mask, diagnostics["rejection_sample_mask"])
    assert diagnostics["fraction_retained_after_window_rejection"] == pytest.approx(
        np.mean(rejection_mask)
    )


def test_asr_riemannian_experimental_backend(synthetic_burst_data):
    """Experimental Riemannian ASR runs end-to-end and suppresses bursts."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        method="riemannian",
        experimental=True,
        filter_kind="none",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.calibration_info_["covariance_geometry"] == "riemannian"
    assert asr.diagnostics_["covariance_geometry"] == "riemannian"
    assert asr.diagnostics_["riemannian_mean_iterations"].shape == (asr.n_windows_,)

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


def test_asr_transform_uses_fitted_state_for_new_data(synthetic_burst_data):
    """A fitted calibration remains fixed while transforming a new stream."""
    data, _, _, sfreq = synthetic_burst_data
    calibration = data[:, : int(6 * sfreq)]
    new_data = data[:, int(6 * sfreq) :]
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        calibration="manual",
        filter_kind="none",
        lookahead=0.0,
        verbose=False,
    ).fit(calibration)
    thresholds = asr.thresholds_.copy()
    cleaned, diagnostics = asr.transform(new_data, return_diagnostics=True)

    assert cleaned.shape == new_data.shape
    assert diagnostics["n_windows"] > 0
    np.testing.assert_array_equal(asr.thresholds_, thresholds)


def test_asr_epochs_round_trip(synthetic_burst_data):
    """Epochs can be calibrated by concatenation and transformed per epoch."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    n_epochs = 3
    epoch_data = np.stack(
        [data[:, idx * 750 : (idx + 1) * 750] for idx in range(n_epochs)]
    )
    info = mne.create_info([f"EEG{idx}" for idx in range(data.shape[0])], sfreq, "eeg")
    epochs = mne.EpochsArray(epoch_data, info, verbose=False)

    asr = ASR(cutoff=4.0, filter_kind="none", verbose=False)
    epochs_clean, diagnostics = asr.fit_transform(epochs, return_diagnostics=True)

    assert epochs_clean.get_data().shape == epoch_data.shape
    assert asr.sample_mask_.shape == (n_epochs, epoch_data.shape[-1])
    assert len(diagnostics["epoch_diagnostics"]) == n_epochs
    assert all(
        epoch_diag["sample_mask"].shape == (epoch_data.shape[-1],)
        for epoch_diag in diagnostics["epoch_diagnostics"]
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


def test_asr_riemannian_windowed_no_experimental_needed():
    cleaned = ASR(
        sfreq=SFREQ, method="riemannian_windowed", verbose=False
    ).fit_transform(_eeg())
    assert cleaned.shape == (8, 8000)


def test_asr_window_criterion_on_epochs_populates_rejection():
    """Epoch processing keeps a separate rejection mask per epoch."""
    epo = _epochs()
    asr = ASR(
        sfreq=SFREQ,
        cutoff=10.0,
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    out, diagnostics = asr.fit_transform(epo, return_diagnostics=True)
    assert out.get_data().shape == epo.get_data().shape
    assert diagnostics["rejection_sample_mask"].shape == (
        len(epo),
        epo.get_data().shape[-1],
    )


def test_asr_rejects_unsupported_calibration_inputs(synthetic_burst_data):
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data

    info = mne.create_info(3, sfreq, "eeg")
    evoked = mne.EvokedArray(data[:3, :100], info)
    with pytest.raises(ValueError, match="does not support Evoked calibration data"):
        ASR(sfreq=sfreq).fit(evoked)

    with pytest.raises(ValueError, match="calibration_mask must have shape"):
        ASR(sfreq=sfreq).fit(
            data, calibration=data, calibration_mask=np.array([True, False])
        )


def test_asr_clears_stale_rejection_state(synthetic_burst_data):
    """Disabling window rejection removes diagnostics from a prior transform."""
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(sfreq=sfreq, window_criterion=0.25, verbose=False).fit(data)
    asr.transform(data)
    assert hasattr(asr, "rejection_sample_mask_")

    asr.window_criterion = None
    asr.transform(data)
    assert not hasattr(asr, "rejection_sample_mask_")


def test_asr_warns_for_unfiltered_projected_input(synthetic_burst_data):
    """ASR reports preprocessing assumptions that affect calibration quality."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    info_warn = mne.create_info(3, sfreq, "eeg")
    with info_warn._unlock():
        info_warn["highpass"] = 0.1

    # MNE Projection must have a valid data dict
    proj = mne.Projection(
        active=False,
        kind=1,
        desc="test",
        data={"data": np.zeros((1, 3)), "col_names": ["1", "2", "3"]},
    )
    raw_warn = mne.io.RawArray(data[:3], info_warn, verbose=False)
    with raw_warn.info._unlock():
        raw_warn.info["projs"] = [proj]

    with pytest.warns(UserWarning) as caught:
        ASR(sfreq=sfreq).fit(raw_warn)
    messages = [str(warning.message) for warning in caught]
    assert any("highpass" in message for message in messages)
    assert any("projectors" in message for message in messages)
