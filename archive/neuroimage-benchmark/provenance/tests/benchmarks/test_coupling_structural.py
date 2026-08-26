"""Coupling + structural metric tests."""

import numpy as np
import pytest

from mne_denoise.qa import coupling, structural


def test_regress_out_reduces_reference_coupling():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((1, 2000))
    brain = rng.standard_normal((4, 2000))
    data = brain + 2.0 * ref                         # strong reference coupling
    before = coupling.reference_coupling(data, ref, method="max_abs")
    resid = coupling.regress_out(data, ref)
    after = coupling.reference_coupling(resid, ref, method="max_abs")
    assert before > 0.6
    assert after < 0.1
    assert after < before


def test_canonical_correlation_in_unit_interval():
    rng = np.random.default_rng(1)
    d = rng.standard_normal((3, 500)); r = rng.standard_normal((2, 500))
    cc = coupling.reference_coupling(d, r, method="cca")
    assert 0.0 <= cc <= 1.0


def test_hf_band_power_positive():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 2000))
    assert coupling.hf_band_power(x, sfreq=500.0, fmin=20, fmax=100) > 0


def test_event_locked_residual():
    data = np.zeros((2, 1000))
    data[:, 500:520] = 10.0                          # a "blink" burst
    val = coupling.event_locked_residual(data, [510], sfreq=100.0,
                                         tmin=-0.05, tmax=0.05)
    assert val > 5.0


def test_effective_rank_drops_on_rank_reduction():
    rng = np.random.default_rng(3)
    full = rng.standard_normal((8, 4000))
    # rank-2 data: two latent sources mixed into 8 channels
    latent = rng.standard_normal((2, 4000))
    reduced = rng.standard_normal((8, 2)) @ latent
    er_full = structural.effective_rank(full)
    er_red = structural.effective_rank(reduced)
    assert er_full > 5.0
    assert er_red < 3.0
    assert er_red < er_full
    assert structural.effective_rank_change(full, reduced) < 0


def test_covariance_distance_zero_for_identical():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 1000))
    c = (x @ x.T) / x.shape[1]
    assert structural.covariance_distance(c, c) < 1e-6


def test_fraction_helpers():
    assert structural.rejected_window_fraction([True, False, False, True]) == 0.5
    assert structural.calibration_fraction(30, 120) == 0.25
    assert structural.artifact_ic_fraction(["brain", "eye", "muscle", "brain"]) == 0.5
    with pytest.raises(ValueError):
        structural.calibration_fraction(1, 0)
