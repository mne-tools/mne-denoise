"""SOBI/JADE classic-BSS unmixers (ground-truth arm): separation-quality regression test.

SOBI (Belouchrani 1997, second-order time-lagged) and JADE (Cardoso & Souloumiac 1993,
fourth-order cumulant) both reduce to a real Jacobi joint-diagonaliser; verify they recover
the mixing on a synthetic problem (distinct sinusoids: colored for SOBI, sub-Gaussian for JADE).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from run_ground_truth_arm import _fit_unmix  # noqa: E402
from mne_denoise.qa import ground_truth as gt  # noqa: E402


def test_sobi_jade_separate_synthetic_mixing():
    rng = np.random.default_rng(0)
    n, T = 5, 4000
    t = np.arange(T)
    freqs = [3, 7, 13, 19, 27]
    S = np.stack([np.sin(2 * np.pi * f * t / 200.0) + 0.02 * rng.standard_normal(T) for f in freqs])
    A = rng.standard_normal((n, n))
    X = A @ S
    for meth in ("sobi", "jade"):
        transform, W = _fit_unmix(meth, X, n)
        assert W.shape == (n, n)
        assert transform(X).shape == (n, T)
        assert gt.amari_index(W, A) < 0.1, meth   # good separation (fastica reference ~0.012)


def test_amica_recovers_mixing():
    # AMICA (Palmer et al., GPU/JAX) on super-Gaussian sources; skipped where amica_python absent.
    pytest.importorskip("jax")
    pytest.importorskip("amica_python")
    rng = np.random.default_rng(0)
    n, T = 5, 4000
    S = rng.laplace(size=(n, T))                  # super-Gaussian sources (AMICA's regime)
    A = rng.standard_normal((n, n))
    X = A @ S
    _, W = _fit_unmix("amica", X, n)
    assert W.shape == (n, n)
    assert gt.amari_index(W, A) < 0.2            # recovers the mixing (verified ~0.05 vs fastica 0.06)
