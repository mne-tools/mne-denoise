"""Tests for the shared spherical lead-field helper."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from mne_denoise._leadfield import (
    REFERENCE_HEAD,
    SphericalHeadModel,
    fibonacci_sphere,
    make_spherical_leadfield,
    resolve_leadfield,
)


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


def test_fibonacci_sphere_is_unit_and_spread():
    directions = fibonacci_sphere(500)
    assert directions.shape == (500, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    # Quasi-uniform coverage: the centroid sits essentially at the origin.
    assert np.linalg.norm(directions.mean(axis=0)) < 0.01


def test_fibonacci_sphere_rejects_empty():
    with pytest.raises(ValueError, match="n_points must be"):
        fibonacci_sphere(0)


def test_spherical_leadfield_shape_and_reference(eeg_info):
    leadfield = make_spherical_leadfield(eeg_info, n_dipoles=800)
    assert leadfield.shape == (24, 800)
    # Average referenced and spanning the full sensor space (n_ch - 1).
    assert np.allclose(leadfield.mean(axis=0), 0.0, atol=1e-9)
    assert np.linalg.matrix_rank(leadfield) == 23


def test_spherical_leadfield_is_deterministic(eeg_info):
    """The Fibonacci lattice makes the lead field reproducible run-to-run."""
    a = make_spherical_leadfield(eeg_info, n_dipoles=400)
    b = make_spherical_leadfield(eeg_info, n_dipoles=400)
    np.testing.assert_array_equal(a, b)


def test_spherical_leadfield_matches_reference_geometry():
    """Spectrum of L @ L.T matches the reference random-shell construction.

    ``construct_spherical_lead_field.m`` draws 5000 radial dipoles at random on
    a 76 mm shell inside an 88 mm head. Pinning the head radius to 88 mm, the
    deterministic lattice must land inside the spread that the reference's own
    random draws produce.
    """
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:32]
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage("standard_1020")
    sphere = mne.make_sphere_model(
        r0="auto",
        head_radius=0.088,
        info=info,
        relative_radii=(81 / 88, 85 / 88, 1.0),
        sigmas=(0.33, 0.33 / 50, 0.33),
        verbose=False,
    )

    def build(directions):
        src = mne.setup_volume_source_space(
            pos={"rr": directions * 0.076 + sphere["r0"], "nn": directions},
            sphere_units="m",
            verbose=False,
        )
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
        )
        fwd = mne.convert_forward_solution(
            fwd, force_fixed=True, use_cps=False, verbose=False
        )
        gain = np.asarray(fwd["sol"]["data"])
        return gain - gain.mean(axis=0, keepdims=True)

    def normalised_gram(lf):
        gram = lf @ lf.T
        return gram / np.trace(gram)

    def draw(seed):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((5000, 3))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    lattice = normalised_gram(build(fibonacci_sphere(5000)))
    ref_a = normalised_gram(build(draw(0)))
    ref_b = normalised_gram(build(draw(1)))

    def rel(x, y):
        return np.linalg.norm(x - y) / np.linalg.norm(y)

    # The lattice sits closer to a reference draw than two draws are to each
    # other, i.e. within the reference's own sampling noise.
    assert rel(lattice, ref_a) < rel(ref_a, ref_b)


def test_head_model_derives_relative_geometry():
    """Millimetres are the source of truth; relative values are derived."""
    head = SphericalHeadModel()
    np.testing.assert_allclose(head.relative_radii, (81 / 88, 85 / 88, 1.0))
    np.testing.assert_allclose(head.conductivities, (0.33, 0.33 / 50, 0.33))
    assert head.dipole_relative_radius == pytest.approx(76 / 88)
    assert head == REFERENCE_HEAD


def test_head_model_rescales_coherently():
    """Changing the scalp radius keeps every shell proportional."""
    head = SphericalHeadModel(
        inner_skull=81 * 1.1,
        outer_skull=85 * 1.1,
        scalp=88 * 1.1,
        dipole_shell=76 * 1.1,
    )
    np.testing.assert_allclose(head.relative_radii, REFERENCE_HEAD.relative_radii)
    assert head.dipole_relative_radius == pytest.approx(
        REFERENCE_HEAD.dipole_relative_radius
    )


def test_head_model_is_frozen():
    with pytest.raises(Exception):
        REFERENCE_HEAD.scalp = 90.0


def test_head_model_changes_the_lead_field(eeg_info):
    """A different skull conductivity produces a different lead field."""
    default = make_spherical_leadfield(eeg_info, n_dipoles=200)
    conductive_skull = make_spherical_leadfield(
        eeg_info,
        n_dipoles=200,
        head_model=SphericalHeadModel(skull_conductivity_ratio=1.0),
    )
    assert not np.allclose(default, conductive_skull)


def _raw(info):
    """A real Raw, so the copy().pick().info chain is genuinely exercised."""
    return mne.io.RawArray(np.zeros((len(info["ch_names"]), 10)), info, verbose=False)


def test_resolve_leadfield_builds_sphere_without_forward(eeg_info):
    """No forward: falls back to the spherical model, honouring n_dipoles."""
    leadfield = resolve_leadfield(
        inst=_raw(eeg_info),
        ch_names=eeg_info["ch_names"],
        n_channels=24,
        method="SOUND",
        n_dipoles=150,
    )
    assert leadfield.shape == (24, 150)


def test_resolve_leadfield_from_forward(eeg_info, forward):
    """An MNE object plus a forward takes the forward, channel-aligned."""
    leadfield = resolve_leadfield(
        inst=_raw(eeg_info),
        ch_names=eeg_info["ch_names"],
        n_channels=24,
        method="SOUND",
        forward=forward,
    )
    assert leadfield.shape[0] == 24
    assert np.allclose(leadfield.mean(axis=0), 0.0, atol=1e-9)


def test_resolve_leadfield_aligns_forward_channels(eeg_info, forward):
    """Reordering the data's channels reorders the forward's rows to match."""
    names = list(eeg_info["ch_names"])
    shuffled = names[::-1]
    flipped_info = mne.create_info(shuffled, 1000.0, "eeg")
    flipped_info.set_montage("standard_1020")

    straight = resolve_leadfield(
        inst=_raw(eeg_info),
        ch_names=names,
        n_channels=24,
        method="SOUND",
        forward=forward,
    )
    reversed_ = resolve_leadfield(
        inst=_raw(flipped_info),
        ch_names=shuffled,
        n_channels=24,
        method="SOUND",
        forward=forward,
    )
    np.testing.assert_allclose(reversed_, straight[::-1], atol=1e-12)


def test_resolve_leadfield_forward_missing_channels_raises(forward):
    ch = mne.channels.make_standard_montage("standard_1020").ch_names[:26]
    info = mne.create_info(ch, 1000.0, "eeg")
    info.set_montage("standard_1020")
    with pytest.raises(ValueError, match="missing channels"):
        resolve_leadfield(
            inst=_raw(info),
            ch_names=info["ch_names"],
            n_channels=26,
            method="SOUND",
            forward=forward,
        )


def test_resolve_leadfield_array_path_checks_channel_count(forward):
    with pytest.raises(ValueError, match="same number of"):
        resolve_leadfield(
            inst=None,
            ch_names=None,
            n_channels=30,
            method="SOUND",
            forward=forward,
        )


def test_resolve_leadfield_without_positions_names_the_method():
    with pytest.raises(ValueError, match="SOUND needs channel positions"):
        resolve_leadfield(inst=None, ch_names=None, n_channels=24, method="SOUND")


def test_spherical_leadfield_requires_eeg_without_forward():
    info = mne.create_info(["MEG0111", "MEG0121", "MEG0131"], 1000.0, "mag")
    with pytest.raises(ValueError, match="supports EEG channels only"):
        make_spherical_leadfield(info, n_dipoles=10)


@pytest.mark.parametrize("n_dipoles", [0, 1.5, True])
def test_spherical_leadfield_validates_n_dipoles(eeg_info, n_dipoles):
    with pytest.raises(ValueError, match="n_dipoles"):
        make_spherical_leadfield(eeg_info, n_dipoles=n_dipoles)


def test_forward_gain_is_validated(eeg_info):
    bad_forward = {
        "sol": {"data": np.ones(24), "row_names": list(eeg_info["ch_names"])}
    }
    with pytest.raises(ValueError, match="finite 2D gain"):
        resolve_leadfield(
            inst=_raw(eeg_info),
            ch_names=eeg_info["ch_names"],
            n_channels=24,
            method="SOUND",
            forward=bad_forward,
        )

    nonfinite_forward = {
        "sol": {
            "data": np.full((24, 3), np.nan),
            "row_names": list(eeg_info["ch_names"]),
        }
    }
    with pytest.raises(ValueError, match="finite 2D gain"):
        resolve_leadfield(
            inst=None,
            ch_names=None,
            n_channels=24,
            method="SOUND",
            forward=nonfinite_forward,
        )
