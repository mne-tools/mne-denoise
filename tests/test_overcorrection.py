"""Tests for the forward-model overcorrection metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.overcorrection import quantify_overcorrection


@pytest.fixture
def leadfield():
    rng = np.random.default_rng(0)
    lf = rng.standard_normal((16, 40))
    return lf - lf.mean(axis=0, keepdims=True)


def test_identity_filter_is_lossless(leadfield):
    """An identity operator distorts nothing."""
    m = quantify_overcorrection(np.eye(16), leadfield)
    np.testing.assert_allclose(m["amplitude_change"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["correlation"], 1.0, atol=1e-12)
    np.testing.assert_allclose(m["relative_error"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["goodness_of_fit"], 1.0, atol=1e-12)


def test_zero_filter_deletes_everything(leadfield):
    """A null operator removes all signal: amplitude -1, GOF 0."""
    m = quantify_overcorrection(np.zeros((16, 16)), leadfield)
    np.testing.assert_allclose(m["amplitude_change"], -1.0)
    np.testing.assert_allclose(m["relative_error"], 1.0)
    np.testing.assert_allclose(m["goodness_of_fit"], 0.0)
    assert np.all(np.isnan(m["correlation"]))  # undefined for a zero topography


def test_scaling_changes_amplitude_but_not_shape(leadfield):
    """Halving the gain leaves the topography shape (CC) untouched."""
    m = quantify_overcorrection(0.5 * np.eye(16), leadfield)
    np.testing.assert_allclose(m["amplitude_change"], -0.5)
    np.testing.assert_allclose(m["correlation"], 1.0, atol=1e-12)
    np.testing.assert_allclose(m["relative_error"], 0.5)


def test_projection_attenuates_the_projected_direction(leadfield):
    """Projecting out a topography deletes sources aligned with it."""
    direction = leadfield[:, 0] / np.linalg.norm(leadfield[:, 0])
    projector = np.eye(16) - np.outer(direction, direction)
    m = quantify_overcorrection(projector, leadfield)
    # Source 0 lies exactly in the removed direction.
    assert m["goodness_of_fit"][0] < 1e-12
    assert m["amplitude_change"][0] == pytest.approx(-1.0)
    # Other sources survive far better.
    assert np.nanmedian(m["goodness_of_fit"][1:]) > 0.5


def test_metrics_match_their_definitions(leadfield):
    """Reproduce Eqs. (26)-(29) directly for a non-trivial operator."""
    rng = np.random.default_rng(1)
    operator = np.eye(16) + 0.1 * rng.standard_normal((16, 16))
    m = quantify_overcorrection(operator, leadfield)
    after = operator @ leadfield
    norm, norm_after = (
        np.linalg.norm(leadfield, axis=0),
        np.linalg.norm(after, axis=0),
    )
    np.testing.assert_allclose(m["amplitude_change"], (norm_after - norm) / norm)
    np.testing.assert_allclose(
        m["correlation"],
        np.einsum("ij,ij->j", after, leadfield) / (norm_after * norm),
    )
    np.testing.assert_allclose(
        m["relative_error"], np.linalg.norm(after - leadfield, axis=0) / norm
    )
    np.testing.assert_allclose(m["goodness_of_fit"], 1.0 - m["relative_error"] ** 2)


def test_amplifying_filter_gives_negative_goodness_of_fit(leadfield):
    """GOF goes negative when a filter is worse than deleting the source.

    Documented behaviour: ``relative_error > 1`` means the result sits further
    from the truth than a zero topography would.
    """
    m = quantify_overcorrection(-2.0 * np.eye(16), leadfield)
    assert np.all(m["relative_error"] > 1.0)
    assert np.all(m["goodness_of_fit"] < 0.0)
    # Sign flip with growth: shape inverted, amplitude up.
    np.testing.assert_allclose(m["correlation"], -1.0)
    np.testing.assert_allclose(m["amplitude_change"], 1.0)


def test_metrics_emit_no_numpy_warnings(leadfield):
    """Degenerate sources must not trigger divide-by-zero warnings."""
    lf = leadfield.copy()
    lf[:, 2] = 0.0
    with np.errstate(all="raise"):
        metrics = quantify_overcorrection(np.zeros((16, 16)), lf)
    assert np.isnan(metrics["amplitude_change"][2])


def test_degenerate_source_is_nan_not_error(leadfield):
    """A zero-norm topography yields NaN rather than a divide-by-zero."""
    lf = leadfield.copy()
    lf[:, 3] = 0.0
    m = quantify_overcorrection(np.eye(16), lf)
    assert np.isnan(m["amplitude_change"][3])
    assert np.isfinite(m["amplitude_change"][0])


def test_shape_validation():
    with pytest.raises(ValueError, match="must be square"):
        quantify_overcorrection(np.zeros((4, 5)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="channels"):
        quantify_overcorrection(np.eye(4), np.zeros((5, 3)))


def test_works_on_a_fitted_sspsir_operator():
    """End-to-end: the metrics accept the operators this package produces."""
    import mne

    from mne_denoise._leadfield import make_spherical_leadfield
    from mne_denoise.sspsir import compute_sspsir

    names = mne.channels.make_standard_montage("standard_1020").ch_names[:24]
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage("standard_1020")
    lf = make_spherical_leadfield(info, n_dipoles=200)
    topos = np.linalg.svd(lf, full_matrices=False)[0][:, :2]

    m = quantify_overcorrection(compute_sspsir(lf, topos, M=15), lf)
    assert m["correlation"].shape == (200,)
    # Removing two dimensions should preserve most sources but not all.
    assert 0.0 < np.nanmean(m["goodness_of_fit"]) < 1.0
    assert np.nanmean(m["amplitude_change"]) < 0.0  # filtering attenuates
