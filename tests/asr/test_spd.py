import numpy as np
import pytest

from mne_denoise.asr._spd import (
    _expm_sym,
    _geometric_median,
    _geometric_median_chunked,
    _invsqrtm_spd,
    _karcher_mean_spd,
    _logm_spd,
    _regularize_spd,
    _riemannian_nonlinear_eigenspace,
    _sqrt_and_eig,
    _sqrtm_spd,
)


def _generate_spd_matrices(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n):
        A = rng.standard_normal((dim, dim))
        mats.append(A @ A.T + np.eye(dim) * 0.1)
    return np.array(mats)


def test_regularize_spd() -> None:
    C = np.array([[1.0, 0.5], [0.5, 0.0]])  # not strictly SPD
    C_reg = _regularize_spd(C, 1e-6)
    w, _ = np.linalg.eigh(C_reg)
    assert np.all(w > 0)


def test_sqrtm_spd_roundtrip() -> None:
    C = _generate_spd_matrices(1, 3)[0]
    S = _sqrtm_spd(C, 1e-6)
    assert np.allclose(S @ S, C)


def test_invsqrtm_spd() -> None:
    C = _generate_spd_matrices(1, 3)[0]
    S_inv = _invsqrtm_spd(C, 1e-6)
    assert np.allclose(S_inv @ C @ S_inv, np.eye(3))


def test_logm_expm_roundtrip() -> None:
    C = _generate_spd_matrices(1, 3)[0]
    L = _logm_spd(C, 1e-6)
    E = _expm_sym(L)
    assert np.allclose(E, C)


def test_geometric_median() -> None:
    covs = _generate_spd_matrices(5, 3)
    median = _geometric_median(covs, max_iter=10)
    assert median.shape == (3, 3)
    w, _ = np.linalg.eigh(median)
    assert np.all(w > 0)


def test_geometric_median_converges_early() -> None:
    covs = np.stack([np.eye(3), np.eye(3)])
    median = _geometric_median(covs, max_iter=10)
    assert np.allclose(median, np.eye(3))


def test_geometric_median_chunked() -> None:
    covs = _generate_spd_matrices(5, 3)

    def iter_factory(chunk_size):
        yield covs

    median = _geometric_median_chunked(
        iter_factory, chunk_blocks=5, n_channels=3, max_iter=10
    )
    assert median.shape == (3, 3)
    w, _ = np.linalg.eigh(median)
    assert np.all(w > 0)


def test_geometric_median_chunked_converges_early() -> None:
    covs = np.stack([np.eye(3), np.eye(3)])

    def iter_factory(chunk_size):
        yield covs

    median = _geometric_median_chunked(
        iter_factory, chunk_blocks=2, n_channels=3, max_iter=10
    )
    assert np.allclose(median, np.eye(3))


def test_karcher_mean_spd() -> None:
    covs = _generate_spd_matrices(3, 3)
    mean_cov, info = _karcher_mean_spd(covs, regularization=1e-6, max_iter=10)
    assert mean_cov.shape == (3, 3)
    assert info["riemannian_mean_iterations"] > 0
    w, _ = np.linalg.eigh(mean_cov)
    assert np.all(w > 0)


def test_karcher_mean_spd_valid_weights() -> None:
    covs = _generate_spd_matrices(2, 3)
    weights = np.array([1.0, 2.0])
    mean_cov, info = _karcher_mean_spd(
        covs, regularization=1e-6, sample_weight=weights, max_iter=10
    )
    assert mean_cov.shape == (3, 3)


def test_sqrt_and_eig() -> None:
    C = _generate_spd_matrices(1, 3)[0]
    M, w, V = _sqrt_and_eig(C, 1e-6)
    assert M.shape == (3, 3)
    assert w.shape == (3,)
    assert V.shape == (3, 3)
    # Check ordered ascending
    assert np.all(np.diff(w) >= 0)


def test_riemannian_nonlinear_eigenspace() -> None:
    L = _generate_spd_matrices(1, 3)[0]
    D, V = _riemannian_nonlinear_eigenspace(L, 1e-6)
    assert D.shape == (3,)
    assert V.shape == (3, 3)
    assert np.all(np.diff(D) >= 0)


def test_riemannian_spd_primitives_roundtrip(rng):
    """SPD log/exp and Karcher mean helpers behave consistently."""
    A = rng.standard_normal((5, 5))
    C = A @ A.T + np.eye(5)
    log_C = _logm_spd(C, 1e-8)
    C_roundtrip = _expm_sym(log_C)
    np.testing.assert_allclose(C_roundtrip, C, rtol=1e-8, atol=1e-8)

    covs = np.stack([C, C], axis=0)
    mean_C, info = _karcher_mean_spd(covs, regularization=1e-8)
    np.testing.assert_allclose(mean_C, C, rtol=1e-8, atol=1e-8)
    assert info["riemannian_mean_converged"]


def test_karcher_mean_spd_validation_guards():
    from mne_denoise.asr._spd import _karcher_mean_spd

    spd = np.stack([np.eye(3), 2.0 * np.eye(3)])
    with pytest.raises(ValueError, match="shape"):
        _karcher_mean_spd(np.eye(3), regularization=1e-8)
    with pytest.raises(ValueError, match="At least one"):
        _karcher_mean_spd(np.empty((0, 3, 3)), regularization=1e-8)
    with pytest.raises(ValueError, match="sample_weight must have shape"):
        _karcher_mean_spd(spd, sample_weight=np.ones(5), regularization=1e-8)
    with pytest.raises(ValueError, match="non-negative"):
        _karcher_mean_spd(spd, sample_weight=np.array([-1.0, 1.0]), regularization=1e-8)
    with pytest.raises(ValueError, match="positive value"):
        _karcher_mean_spd(spd, sample_weight=np.array([0.0, 0.0]), regularization=1e-8)
