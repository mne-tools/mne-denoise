"""Preservation-metric tests (hand-checked effect sizes, SME, topography)."""

import numpy as np
import pytest

from mne_denoise.qa import preservation as pres


def test_cohen_dz_hand_calc():
    # diffs [1,2,3] -> mean 2, sd(ddof=1)=1 -> dz = 2.0
    assert pres.cohen_dz([1.0, 2.0, 3.0]) == 2.0
    # paired form
    assert pres.cohen_dz([4, 5, 6], [3, 3, 3]) == 2.0


def test_cohen_dz_zero_variance_raises():
    with pytest.raises(ValueError):
        pres.cohen_dz([2.0, 2.0, 2.0])


def test_analytic_sme_hand_calc():
    amps = np.array([0.0, 2.0, 4.0, 6.0])          # sd(ddof=1)=2.5820..., n=4
    expected = np.std(amps, ddof=1) / 2.0
    assert abs(pres.analytic_sme(amps) - expected) < 1e-12


def test_sme_mean_amplitude_from_epochs():
    rng = np.random.default_rng(0)
    epochs = rng.standard_normal((20, 4, 100))     # trials, ch, times
    times = np.linspace(-0.2, 0.8, 100)
    val = pres.sme_mean_amplitude(epochs, times, 0.0, 0.3)
    assert val > 0


def test_topographic_similarity_sign():
    a = np.array([1.0, -2.0, 3.0, 0.5])
    assert abs(pres.topographic_similarity(a, a) - 1.0) < 1e-12
    assert abs(pres.topographic_similarity(a, -a) + 1.0) < 1e-12


def test_erp_mean_amplitude_window():
    times = np.linspace(0, 1, 11)                  # 0,0.1,...,1.0
    data = np.ones((3, 11)) * 5.0
    assert pres.erp_mean_amplitude(data, times, 0.2, 0.5) == 5.0


def test_erd_ers_and_split_half():
    assert pres.erd_ers([0.5], [1.0]) == -0.5      # 50% desync
    rng = np.random.default_rng(1)
    base = rng.standard_normal(100)
    trials = np.array([base + 0.05 * rng.standard_normal(100) for _ in range(12)])
    assert pres.split_half_reliability(trials) > 0.8
