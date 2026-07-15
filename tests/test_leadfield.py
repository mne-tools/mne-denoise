"""Tests for the shared spherical lead-field helper."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from mne_denoise._leadfield import make_spherical_leadfield


@pytest.fixture(scope="module")
def eeg_info():
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:24]
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage("standard_1020")
    return info


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


def test_spherical_leadfield_shape_and_reference(eeg_info):
    leadfield = make_spherical_leadfield(eeg_info, pos=25.0)
    assert leadfield.shape[0] == 24
    assert leadfield.shape[1] > 24  # dense, free-orientation gain
    # Average referenced and spanning the full sensor space (n_ch - 1).
    assert np.allclose(leadfield.mean(axis=0), 0.0, atol=1e-9)
    assert np.linalg.matrix_rank(leadfield) == 23


def test_leadfield_from_forward(eeg_info, forward):
    leadfield = make_spherical_leadfield(eeg_info, forward=forward)
    assert leadfield.shape[0] == 24
    assert np.allclose(leadfield.mean(axis=0), 0.0, atol=1e-9)


def test_leadfield_forward_missing_channels_raises(forward):
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:26]
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="missing channels"):
        make_spherical_leadfield(info, forward=forward)

