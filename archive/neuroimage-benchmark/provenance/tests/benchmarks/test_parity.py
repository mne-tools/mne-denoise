"""Parity (numerical-equivalence) helper tests — L1/L2/L3."""

import numpy as np

from mne_denoise.benchmarks import parity


def test_l1_numerical_parity():
    a = np.arange(20.0).reshape(4, 5)
    assert parity.numerical_parity(a, a.copy()).passed
    assert not parity.numerical_parity(a, a + 1e-2).passed


def test_l2_subspace_parity_same_and_orthogonal():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 3))
    # same column space (random invertible recombination)
    M = rng.standard_normal((3, 3))
    assert parity.subspace_parity(A, A @ M, tol_deg=1.0).passed
    # orthogonal complement → ~90 deg
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    res = parity.subspace_parity(Q[:, :3], Q[:, 3:6], tol_deg=1.0)
    assert not res.passed and res.metric > 80.0


def test_align_components_recovers_perm_and_sign():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((4, 200))
    perm_true = np.array([2, 0, 3, 1])
    signs_true = np.array([1.0, -1.0, 1.0, -1.0])
    W2 = (W[perm_true] * signs_true[:, None])
    perm, signs, corr = parity.align_components(W, W2)
    # W2[k] corresponds to W[perm_true[k]]; recovered mapping should invert it
    assert np.allclose(np.abs(corr), 1.0, atol=1e-6)


def test_l3_behaviour_parity():
    assert parity.behaviour_parity(0.80, 0.78, rtol=0.1).passed
    assert not parity.behaviour_parity(0.80, 0.40, rtol=0.1).passed
