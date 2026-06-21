"""Ground-truth metric tests, including validity gating and Amari on known mixing."""

import numpy as np
import pytest

from mne_denoise.benchmarks import simulation as sim
from mne_denoise.qa import ground_truth as gt


# -- RRMSE / correlation ----------------------------------------------------
def test_rrmse_zero_on_identical_and_positive_otherwise():
    x = np.random.default_rng(0).standard_normal((4, 100))
    assert gt.rrmse(x, x) == 0.0
    assert gt.rrmse(x + 0.1, x) > 0.0


def test_rrmse_gating():
    with pytest.raises(ValueError):
        gt.rrmse(np.zeros((2, 3)), np.zeros((2, 4)))   # shape mismatch
    with pytest.raises(ValueError):
        gt.rrmse(np.ones(5), np.zeros(5))              # zero-norm target


def test_correlation_with_clean_perfect():
    x = np.random.default_rng(1).standard_normal((3, 200))
    assert gt.correlation_with_clean(x, x) > 0.999


# -- BSS eval ---------------------------------------------------------------
def test_bss_eval_requires_references():
    with pytest.raises(ValueError):
        gt.bss_eval(np.zeros(10), None, 0)


def test_bss_eval_perfect_estimate_high_sdr():
    rng = np.random.default_rng(2)
    S = rng.standard_normal((3, 1000))
    out = gt.bss_eval(S[1], S, target_index=1)
    assert out["sdr"] > 60.0  # near-perfect recovery


# -- Amari (square gating) --------------------------------------------------
def test_amari_zero_for_exact_inverse():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5))
    W = np.linalg.inv(A)
    assert gt.amari_index(W, A) < 1e-8


def test_amari_recovers_known_square_mixing_via_ica():
    from sklearn.decomposition import FastICA

    rep = sim.simulate_replicate(regime="generic", n_channels=6, n_brain=3,
                                 n_artifact=3, n_train=8000, n_test=10, seed=11)
    ica = FastICA(n_components=6, whiten="unit-variance", max_iter=2000,
                  random_state=0)
    ica.fit(rep.X_train.T)
    W = ica.components_                      # channels -> sources
    assert gt.amari_index(W, rep.A) < 0.3


def test_amari_refuses_non_square():
    with pytest.raises(ValueError):
        gt.amari_index(np.zeros((3, 5)), np.zeros((5, 4)))


# -- mixing recovery / contamination ---------------------------------------
def test_mixing_recovery_error_zero_after_alignment():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((6, 4))
    perm = [2, 0, 3, 1]
    A_est = A[:, perm] * np.array([1, -1, 2, -0.5])   # perm + sign + scale
    assert gt.mixing_recovery_error(A_est, A) < 1e-8


def test_contamination_precision_recall():
    true = np.array([1, 1, 0, 0, 1], bool)
    pred = np.array([1, 0, 0, 1, 1], bool)
    out = gt.contamination_precision_recall(pred, true)
    assert out["precision"] == 2 / 3
    assert out["recall"] == 2 / 3
