"""Tests for mne_denoise.asr."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise import _mne
from mne_denoise.asr import (
    ASR,
)

SFREQ = 250.0


def test_asr_default_uses_original_spectral_shaping_filter():
    """The estimator defaults to inverse-EEG spectral shaping."""
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


def test_asr_cleaning_is_invariant_to_eeg_unit_scaling(synthetic_burst_data):
    """Equivalent signals in volts and microvolts receive equivalent cleaning."""
    data_uv, _, _, sfreq = synthetic_burst_data
    model_uv = ASR(sfreq=sfreq, cutoff=3.0, calibration="manual", verbose=False)
    clean_uv = model_uv.fit_transform(data_uv)

    data_v = data_uv * 1e-6
    model_v = ASR(sfreq=sfreq, cutoff=3.0, calibration="manual", verbose=False)
    clean_v = model_v.fit_transform(data_v)

    np.testing.assert_allclose(clean_v * 1e6, clean_uv, rtol=2e-7, atol=2e-8)
    np.testing.assert_array_equal(
        model_v.n_components_reconstructed_, model_uv.n_components_reconstructed_
    )


def test_asr_mne_raw_preserves_non_picked_channels(synthetic_burst_data):
    """MNE Raw support cleans EEG picks and preserves non-picked channels."""
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

    asr = ASR(cutoff=3.0, filter_kind="none", verbose=False)
    raw_clean = asr.fit_transform(raw)

    assert isinstance(raw_clean, mne.io.RawArray)
    assert raw_clean.get_data().shape == raw_data.shape
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1", "EOG2"]), eog)
    assert asr.ch_names_ == ch_names[: data.shape[0]]


def test_asr_transform_checks_fitted_channel_layout(synthetic_burst_data):
    """ASR preserves named-fit safety and rejects changed layouts."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    names = [f"EEG{idx}" for idx in range(data.shape[0])]
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=False)
    asr = ASR(cutoff=3.0, filter_kind="none", verbose=False).fit(raw)

    asr.transform(raw)
    with pytest.raises(ValueError, match="ASR was fitted with named channels"):
        asr.transform(data)
    with pytest.raises(ValueError, match="MNE channel names/order differ from fit"):
        asr.transform(raw.copy().reorder_channels(names[::-1]))

    array_asr = ASR(sfreq=sfreq, cutoff=3.0, filter_kind="none", verbose=False).fit(
        data
    )
    with pytest.raises(ValueError, match="ASR: X has 7 channels; fitted data had 8"):
        array_asr.transform(data[:-1])


def test_asr_mne_raw_low_memory_preserves_metadata(synthetic_burst_data):
    """Low-memory Raw processing preserves metadata and non-picked channels."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    eog = np.sin(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq)[None, :]
    raw_data = np.vstack([data, eog])
    ch_names = [f"EEG{idx}" for idx in range(data.shape[0])] + ["EOG1"]
    ch_types = ["eeg"] * data.shape[0] + ["eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    raw.info["bads"] = ["EEG7"]
    raw.set_annotations(mne.Annotations([1.0], [0.25], ["BAD_test"]))

    asr = ASR(
        cutoff=3.0,
        filter_kind="none",
        reject_by_annotation=False,
        max_mem_mb=0.001,
        verbose=False,
    )
    raw_clean = asr.fit_transform(raw)

    assert raw_clean.ch_names == raw.ch_names
    assert raw_clean.info["bads"] == raw.info["bads"]
    assert len(raw_clean.annotations) == len(raw.annotations)
    assert raw_clean.info["sfreq"] == raw.info["sfreq"]
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1"]), eog)
    assert asr.diagnostics_["memory_mode"] == "rolling"


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


def test_asr_to_annotations(synthetic_burst_data):
    """Last-transform repaired windows can be converted to annotations."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(sfreq=sfreq, cutoff=3.0, filter_kind="none", verbose=False)
    asr.fit_transform(data)
    annotations = asr.to_annotations()

    assert isinstance(annotations, mne.Annotations)
    assert len(annotations) >= 1
    assert set(annotations.description) == {"ASR_REPAIR"}


def test_asr_window_criterion_mask_and_annotations(synthetic_burst_data):
    """Optional final window rejection exposes retained-sample masks."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        filter_kind="none",
        window_criterion=0.25,
        window_criterion_tolerances=(-np.inf, 2.0),
        verbose=False,
    )
    asr.fit_transform(data)

    rejection_mask = asr.get_rejection_mask()
    assert rejection_mask.shape == (data.shape[1],)
    assert not np.all(rejection_mask)

    annotations = asr.to_annotations("rejection")
    assert isinstance(annotations, mne.Annotations)
    assert len(annotations) >= 1
    assert set(annotations.description) == {"ASR_REJECT"}


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
    epochs_clean = asr.fit_transform(epochs)

    assert epochs_clean.get_data().shape == epoch_data.shape
    assert asr.sample_mask_.shape == (n_epochs, epoch_data.shape[-1])


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


def test_asr_riemannian_windowed_no_experimental_needed():
    cleaned = ASR(
        sfreq=SFREQ, method="riemannian_windowed", verbose=False
    ).fit_transform(_eeg())
    assert cleaned.shape == (8, 8000)


def test_get_diagnostics_present():
    est = ASR(sfreq=SFREQ, verbose=False)
    est.fit_transform(_eeg())
    assert isinstance(est.get_diagnostics(), dict)


def test_asr_window_criterion_rejection_and_annotations():
    mne = pytest.importorskip("mne")
    asr = ASR(
        sfreq=SFREQ,
        cutoff=10.0,
        calibration="auto",
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    asr.fit_transform(_eeg(bursts=8))
    mask = asr.get_rejection_mask()
    assert mask.dtype == bool
    ann = asr.to_annotations("rejection")
    assert isinstance(ann, mne.Annotations)


def test_standard_low_memory_matches_full():
    X = _eeg()
    full = ASR(sfreq=SFREQ, cutoff=20.0, max_mem_mb=512, verbose=False)
    low = ASR(sfreq=SFREQ, cutoff=20.0, max_mem_mb=1, verbose=False)
    c_full = full.fit_transform(X)
    c_low = low.fit_transform(X)
    np.testing.assert_allclose(c_full, c_low, atol=1e-9)


def test_riemannian_windowed_low_memory_runs():
    X = _eeg()
    asr = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian_windowed",
        max_mem_mb=1,
        verbose=False,
    )
    cleaned = asr.fit_transform(X)
    assert np.all(np.isfinite(cleaned))


def test_asr_window_criterion_on_epochs_populates_rejection():
    epo = _epochs()
    asr = ASR(
        sfreq=SFREQ,
        cutoff=10.0,
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    out = asr.fit_transform(epo)
    assert out.get_data().shape == epo.get_data().shape
    diag = asr.get_diagnostics()
    assert "rejection_sample_mask" in diag
    assert "fraction_retained_after_window_rejection" in diag


def test_riemannian_low_memory_runs():
    asr = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian",
        experimental=True,
        max_mem_mb=1,
        verbose=False,
    )
    cleaned = asr.fit_transform(_eeg())
    assert np.all(np.isfinite(cleaned))


def test_core_edge_cases(synthetic_burst_data, monkeypatch):
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
    # Valid mask should not raise
    ASR(sfreq=sfreq).fit(
        data, calibration=data, calibration_mask=np.ones(data.shape[1], dtype=bool)
    )

    asr = ASR(sfreq=sfreq).fit(data)
    info_diff_sfreq = mne.create_info(8, sfreq + 10.0, "eeg")
    raw_diff_sfreq = mne.io.RawArray(data, info_diff_sfreq, verbose=False)
    with pytest.raises(ValueError, match="does not match fitted sfreq"):
        asr.transform(raw_diff_sfreq)

    cleaned, diag = asr.transform(data, return_diagnostics=True)
    assert isinstance(cleaned, np.ndarray)
    assert isinstance(diag, dict)

    asr_new = ASR(sfreq=sfreq).fit(data)
    assert asr_new.get_diagnostics() == {}

    with pytest.raises(ValueError, match="must be positive"):
        asr._resolve_sfreq(-10.0)

    # First create rejection mask
    asr_rej = ASR(sfreq=sfreq, window_criterion=0.25).fit(data)
    asr_rej.transform(data)
    assert hasattr(asr_rej, "rejection_sample_mask_")
    # Then transform without window criterion (it uses the object's setting, so we disable it)
    asr_rej.window_criterion = None
    asr_rej.transform(data)
    assert not hasattr(asr_rej, "rejection_sample_mask_")

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

    with pytest.warns(UserWarning, match="highpass"):
        with pytest.warns(UserWarning, match="projectors"):
            ASR(sfreq=sfreq).fit(raw_warn)

    # Cover the false branch of highpass < 0.25
    info_ok = mne.create_info(3, sfreq, "eeg")
    with info_ok._unlock():
        info_ok["highpass"] = 1.0
    raw_ok = mne.io.RawArray(data[:3], info_ok, verbose=False)
    ASR(sfreq=sfreq).fit(raw_ok)  # Should not warn

    raw_annot = mne.io.RawArray(
        data[:3], mne.create_info(3, sfreq, "eeg"), verbose=False
    )
    raw_annot.set_annotations(mne.Annotations([0.0], [0.1], ["BAD"]))
    asr_rej_annot = ASR(
        sfreq=sfreq, window_criterion=0.25, reject_by_annotation=True
    ).fit(raw_annot)
    cleaned_raw_rej = asr_rej_annot.transform(raw_annot)
    assert isinstance(cleaned_raw_rej, mne.io.RawArray)

    monkeypatch.setattr(_mne, "mne", None)
    with pytest.raises(ImportError, match="ASR annotations.*MNE-Python"):
        asr.to_annotations("repair")
