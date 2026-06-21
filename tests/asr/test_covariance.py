import numpy as np
import pytest

from mne_denoise.asr._covariance import (
    _aggregate_block_covariances,
    _aggregate_covariances,
    _block_covariances_padded,
    _block_covariances_standard,
    _covariance_chunk_blocks,
    _covariance_stack_bytes,
    _iter_block_covariances_padded,
    _iter_block_covariances_standard,
    _iter_moving_covariances_at,
    _max_mem_bytes,
    _moving_average_padded,
    _process_memory_info,
    _window_covariances,
)


def test_process_memory_info() -> None:
    info = _process_memory_info(
        n_channels=64,
        n_stream_input=1000,
        max_mem_mb=10.0,
        memory_mode="full",
        peak_cov_buffer_bytes=1024,
        chunk_samples=500,
        used_memory_bound=False,
    )
    assert info["memory_mode"] == "full"
    assert info["max_mem_mb"] == 10.0
    assert info["peak_cov_buffer_bytes"] == 1024


def test_max_mem_bytes() -> None:
    assert _max_mem_bytes(None) is None
    assert _max_mem_bytes(1.0) == 1024 * 1024
    with pytest.raises(ValueError, match="non-negative"):
        _max_mem_bytes(-1.0)


def test_covariance_stack_bytes() -> None:
    # 2 items, 3 channels, float64 (8 bytes) = 2 * 9 * 8 = 144
    assert _covariance_stack_bytes(2, 3) == 144


def test_covariance_chunk_blocks() -> None:
    # unlimited
    assert _covariance_chunk_blocks(3, None) > 1000
    # 3 channels = 72 bytes per block. Budget = 1000 // 4 = 250. 250 // 72 = 3
    assert _covariance_chunk_blocks(3, 1000) == 3
    # Low memory fallback
    assert _covariance_chunk_blocks(3, 10) == 1


def test_window_covariances() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    starts = np.array([0, 10, 20])
    covs = _window_covariances(X, starts, win_len=10)
    assert covs.shape == (3, 3, 3)
    # Check symmetric
    assert np.allclose(covs[0], covs[0].T)


def test_iter_moving_covariances_at() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    update_at = np.array([10, 20])
    covs = list(_iter_moving_covariances_at(X, update_at, window_length=10))
    assert len(covs) == 2
    assert covs[0].shape == (3, 3)
    assert np.allclose(covs[0], covs[0].T)


def test_moving_average_padded() -> None:
    X = np.ones((3, 100))
    out, zf = _moving_average_padded(10, X)
    assert out.shape == (3, 100)
    assert zf.shape == (3, 10)
    # After the 10th sample, moving average of 1s should be 1
    assert np.allclose(out[:, 10:], 1.0)


def test_aggregate_block_covariances_full_vs_chunked() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))

    # Standard mode, mean
    cov_full, info_full = _aggregate_block_covariances(
        X, blocksize=10, method="mean", covariance_kind="standard", max_mem_mb=None
    )
    assert info_full["memory_mode"] == "full"

    cov_chunk, info_chunk = _aggregate_block_covariances(
        X, blocksize=10, method="mean", covariance_kind="standard", max_mem_mb=0.0001
    )
    assert info_chunk["memory_mode"] == "chunked"

    assert np.allclose(cov_full, cov_chunk)

    # Padded mode, geometric_median
    cov_full_pad, info_full_pad = _aggregate_block_covariances(
        X,
        blocksize=10,
        method="geometric_median",
        covariance_kind="padded",
        max_mem_mb=None,
    )
    assert info_full_pad["memory_mode"] == "full"

    cov_chunk_pad, info_chunk_pad = _aggregate_block_covariances(
        X,
        blocksize=10,
        method="geometric_median",
        covariance_kind="padded",
        max_mem_mb=0.0001,
    )
    assert info_chunk_pad["memory_mode"] == "chunked"

    assert np.allclose(cov_full_pad, cov_chunk_pad)


def test_aggregate_block_covariances_errors() -> None:
    X = np.ones((3, 100))
    with pytest.raises(ValueError, match="covariance_kind must be"):
        _aggregate_block_covariances(X, 10, "mean", "invalid", None)

    with pytest.raises(ValueError, match="cov_estimator='median' requires full"):
        _aggregate_block_covariances(X, 10, "median", "standard", max_mem_mb=0.0001)


def test_aggregate_covariances_methods() -> None:
    rng = np.random.default_rng(42)
    covs = np.zeros((5, 3, 3))
    for i in range(5):
        A = rng.standard_normal((3, 3))
        covs[i] = A @ A.T

    mean_cov = _aggregate_covariances(covs, "mean")
    med_cov = _aggregate_covariances(covs, "median")
    geom_cov = _aggregate_covariances(covs, "geometric_median")

    assert mean_cov.shape == (3, 3)
    assert med_cov.shape == (3, 3)
    assert geom_cov.shape == (3, 3)


def test_block_covariances_standard_and_padded() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    blocksize = 10

    # Test standard
    covs_std = _block_covariances_standard(X, blocksize)
    assert covs_std.shape == (10, 3, 3)
    assert np.allclose(covs_std[0], (X[:, :10] @ X[:, :10].T) / blocksize)

    # Test padded
    covs_pad = _block_covariances_padded(X, blocksize)
    assert covs_pad.shape == (10, 3, 3)
    assert np.allclose(covs_pad[0], covs_pad[0].T)


def test_iter_block_covariances_chunked() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    blocksize = 10
    chunk_blocks = 3

    # Test standard iterator
    std_chunks = list(_iter_block_covariances_standard(X, blocksize, chunk_blocks))
    assert len(std_chunks) == 4  # (10 blocks) / 3 = 3.33 -> 4 chunks
    assert std_chunks[0].shape == (3, 3, 3)  # First chunk has 3 blocks
    assert std_chunks[-1].shape == (1, 3, 3)  # Last chunk has 1 block

    # Compare with full output
    full_std = _block_covariances_standard(X, blocksize)
    concat_std = np.concatenate(std_chunks, axis=0)
    assert np.allclose(full_std, concat_std)

    # Test padded iterator
    pad_chunks = list(_iter_block_covariances_padded(X, blocksize, chunk_blocks))
    assert len(pad_chunks) == 4
    assert pad_chunks[0].shape == (3, 3, 3)
    assert pad_chunks[-1].shape == (1, 3, 3)

    # Compare with full padded output
    full_pad = _block_covariances_padded(X, blocksize)
    concat_pad = np.concatenate(pad_chunks, axis=0)
    assert np.allclose(full_pad, concat_pad)


def test_adaptive_covariance_sqrt() -> None:
    from mne_denoise.asr._covariance import _adaptive_covariance_sqrt

    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    M, C, eigvals, V, info = _adaptive_covariance_sqrt(
        X, blocksize=10, regularization=1e-5, max_mem_mb=None
    )
    assert M.shape == (3, 3)
    assert C.shape == (3, 3)
    assert eigvals.shape == (3,)
    assert V.shape == (3, 3)
    assert isinstance(info, dict)
    assert np.all(eigvals >= 0)
    assert np.allclose(M, M.T)
    assert np.allclose(M @ M, C)


def test_chunked_moving_covariances() -> None:
    from mne_denoise.asr._covariance import _ChunkedMovingCovariances

    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    update_at = np.array([20, 40, 80])

    iterator = _ChunkedMovingCovariances(
        X=X,
        update_at=update_at,
        window_length=10,
        chunk_samples=25,
        zi=None,
    )

    covs = list(iterator)
    assert len(covs) == 3
    for cov in covs:
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)

    with pytest.raises(RuntimeError, match="Failed to produce"):
        bad_update_at = np.array([20, 120])
        list(_ChunkedMovingCovariances(X, bad_update_at, 10, 25, None))
