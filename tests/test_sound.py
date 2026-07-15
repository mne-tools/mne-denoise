"""Tests for the SOUND estimator."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from mne_denoise.sound import SOUND, compute_sound
from mne_denoise.sound.core import _ddwiener


@pytest.fixture(scope="module")
def eeg_info():
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:24]
    info = mne.create_info(ch, 500.0, "eeg")
    info.set_montage("standard_1020")
    return info


@pytest.fixture(scope="module")
def noisy_raw(eeg_info):
    """One badly corrupted channel over an otherwise clean dipolar source."""
    rng = np.random.default_rng(0)
    n_ch, n_t = 24, 1200
    topo = rng.standard_normal(n_ch)
    topo /= np.linalg.norm(topo)
    src = np.sin(2 * np.pi * 10 * np.arange(n_t) / 500.0)
    data = np.outer(topo, src) + 0.2 * rng.standard_normal((n_ch, n_t))
    data[7] += 5.0 * rng.standard_normal(n_t)  # corrupt channel 7
    return mne.io.RawArray(data * 1e-6, eeg_info, verbose=False), 7


def test_sound_suppresses_noisy_channel(noisy_raw):
    raw, bad = noisy_raw
    sound = SOUND(n_iter=5, random_state=0).fit(raw)
    cleaned = sound.transform(raw).get_data()
    before = raw.get_data()
    # The corrupted channel gets the largest noise estimate and is suppressed.
    assert int(np.argmax(sound.sigmas_)) == bad
    assert cleaned[bad].var() < before[bad].var()
    # Convergence improves over iterations.
    assert sound.convergence_[-1] < sound.convergence_[0]


def test_sound_runs_on_epochs(eeg_info):
    rng = np.random.default_rng(1)
    data = rng.standard_normal((6, 24, 300)) * 1e-6
    epochs = mne.EpochsArray(data, eeg_info, verbose=False)
    sound = SOUND(n_iter=2, random_state=0).fit(epochs)
    out = sound.transform(epochs)
    assert out.get_data().shape == (6, 24, 300)


def test_sound_array_input_requires_forward(eeg_info):
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((24, 500))
    with pytest.raises(ValueError, match="channel positions"):
        SOUND().fit(arr)


def test_sound_array_input_with_leadfield(eeg_info):
    from mne_denoise._leadfield import make_spherical_leadfield

    leadfield = make_spherical_leadfield(eeg_info, pos=25.0)
    op, sigmas, conv = compute_sound(
        np.random.default_rng(3).standard_normal((24, 600)), leadfield, n_iter=2
    )
    assert op.shape == (24, 24)
    assert sigmas.shape == (24,)


def test_sound_not_fitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        SOUND().transform(np.zeros((24, 100)))


def test_compute_sound_channel_mismatch_raises():
    with pytest.raises(ValueError, match="channels"):
        compute_sound(np.zeros((10, 50)), np.zeros((8, 30)))


def test_compute_sound_too_few_channels_raises():
    with pytest.raises(ValueError, match="at least 3 channels"):
        compute_sound(np.zeros((2, 50)), np.zeros((2, 30)))


def test_ddwiener_flags_noisy_channel():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((8, 400))
    data[3] += 6.0 * rng.standard_normal(400)
    sigmas = _ddwiener(data)
    assert int(np.argmax(sigmas)) == 3


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


def test_sound_array_with_forward(forward):
    arr = np.random.default_rng(5).standard_normal((24, 600))
    sound = SOUND(n_iter=2, forward=forward, random_state=0).fit(arr)
    assert sound.transform(arr).shape == (24, 600)


def test_sound_array_forward_channel_mismatch_raises(forward):
    arr = np.random.default_rng(6).standard_normal((30, 600))
    with pytest.raises(ValueError, match="same number of"):
        SOUND(forward=forward).fit(arr)

