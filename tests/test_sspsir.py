"""Tests for the SSP-SIR estimator."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from mne_denoise.sspsir import SSPSIR, compute_sspsir_operator


@pytest.fixture(scope="module")
def eeg_info():
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:24]
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage("standard_1020")
    return info


@pytest.fixture(scope="module")
def tms_epochs(eeg_info):
    """Synthetic TMS-EEG: a clean evoked source + a focal high-frequency burst."""
    rng = np.random.default_rng(1)
    n_ch, n_t, n_ep = 24, 400, 16
    t = (np.arange(n_t) - 100) / 1000.0  # -100..300 ms
    btopo = rng.standard_normal(n_ch)
    btopo /= np.linalg.norm(btopo)
    brain = np.sin(2 * np.pi * 10 * t) * np.exp(-((t - 0.1) ** 2) / 0.01)
    mtopo = np.zeros(n_ch)
    mtopo[[8, 9, 10]] = [1.0, 0.8, 0.6]
    mtopo /= np.linalg.norm(mtopo)
    win = (t >= 0.005) & (t <= 0.050)
    data = np.zeros((n_ep, n_ch, n_t))
    for e in range(n_ep):
        art = np.zeros(n_t)
        art[win] = rng.standard_normal(win.sum()) * 8.0
        data[e] = (
            np.outer(btopo, brain)
            + np.outer(mtopo, art)
            + 0.1 * rng.standard_normal((n_ch, n_t))
        )
    epochs = mne.EpochsArray(data * 1e-6, eeg_info, tmin=t[0], verbose=False)
    return epochs, btopo, brain, win


def test_sspsir_removes_artifact_preserves_source(tms_epochs):
    epochs, btopo, brain, win = tms_epochs
    ss = SSPSIR(n_components=3).fit(epochs)
    cleaned = ss.transform(epochs).get_data()
    before = epochs.get_data()
    # Muscle-window variance on the artifact channels drops sharply.
    assert (
        cleaned[:, [8, 9, 10]][:, :, win].var()
        < 0.1 * before[:, [8, 9, 10]][:, :, win].var()
    )
    # The brain source is preserved outside the artifact window.
    proj = np.einsum("c,ct->t", btopo, cleaned.mean(0))
    corr = np.corrcoef(proj[~win], brain[~win])[0, 1]
    assert corr > 0.95


def test_sspsir_variance_fraction(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=0.9).fit(epochs)
    assert ss.n_components_ >= 1
    assert ss.transform(epochs).get_data().shape == epochs.get_data().shape


def test_sspsir_manual_window(tms_epochs):
    epochs = tms_epochs[0]
    ss = SSPSIR(n_components=2, art_window=(0.005, 0.050)).fit(epochs)
    assert ss.n_components_ == 2
    assert hasattr(ss, "operator_")


def test_sspsir_window_no_overlap_raises(tms_epochs):
    epochs = tms_epochs[0]
    with pytest.raises(ValueError, match="does not overlap"):
        SSPSIR(n_components=2, art_window=(5.0, 9.0)).fit(epochs)


def test_sspsir_runs_on_evoked(tms_epochs):
    evoked = tms_epochs[0].average()
    ss = SSPSIR(n_components=2).fit(evoked)
    out = ss.transform(evoked)
    assert out.data.shape == evoked.data.shape


def test_sspsir_requires_n_components(tms_epochs):
    epochs = tms_epochs[0]
    with pytest.raises(ValueError, match="n_components must be set"):
        SSPSIR().fit(epochs)


def test_sspsir_array_requires_sfreq():
    arr = np.random.default_rng(0).standard_normal((24, 400))
    with pytest.raises(ValueError, match="sampling frequency"):
        SSPSIR(n_components=2).fit(arr)


def test_sspsir_not_fitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        SSPSIR(n_components=2).transform(np.zeros((24, 100)))


def test_compute_sspsir_operator_shape(eeg_info):
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, pos=25.0)
    u = np.linalg.svd(np.random.default_rng(0).standard_normal((24, 24)))[0][:, :2]
    operator = compute_sspsir_operator(leadfield, u, M=20)
    assert operator.shape == (24, 24)


@pytest.fixture(scope="module")
def forward(eeg_info):
    sphere = mne.make_sphere_model(
        r0="auto", head_radius="auto", info=eeg_info, verbose=False
    )
    src = mne.setup_volume_source_space(
        sphere=sphere, pos=25.0, sphere_units="mm", verbose=False
    )
    return mne.make_forward_solution(
        eeg_info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
    )


def test_sspsir_array_with_forward(forward):
    arr = np.random.default_rng(7).standard_normal((24, 400))
    ss = SSPSIR(n_components=2, sfreq=1000.0, forward=forward).fit(arr)
    assert ss.transform(arr).shape == (24, 400)


def test_sspsir_array_without_forward_raises():
    arr = np.random.default_rng(8).standard_normal((24, 400))
    with pytest.raises(ValueError, match="channel positions"):
        SSPSIR(n_components=2, sfreq=1000.0).fit(arr)

