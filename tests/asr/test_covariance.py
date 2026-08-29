import numpy as np
import pytest

from mne_denoise.asr._covariance import (
    _aggregate_block_covariances,
    _aggregate_covariances,
    _block_covariances_padded,
    _block_covariances_standard,
    _iter_block_covariances_padded,
    _iter_block_covariances_standard,
    _iter_moving_covariances_at,
    _moving_average_padded,
    _window_covariances,
)


def test_moving_covariance_contract() -> None:
    """Windowed covariance and moving-average paths preserve useful layouts."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    starts = np.array([0, 10, 20])
    covariances = _window_covariances(data, starts, win_len=10)
    assert covariances.shape == (3, 3, 3)
    assert np.allclose(covariances, covariances.swapaxes(-1, -2))

    moving = list(_iter_moving_covariances_at(data, np.array([10, 20]), 10))
    assert len(moving) == 2
    assert all(covariance.shape == (3, 3) for covariance in moving)
    assert all(np.allclose(covariance, covariance.T) for covariance in moving)

    averaged, state = _moving_average_padded(10, np.ones((3, 100)))
    assert averaged.shape == (3, 100)
    assert state.shape == (3, 10)
    assert np.allclose(averaged[:, 10:], 1.0)


def test_aggregate_block_covariances_chunking_contract() -> None:
    """Low-memory block aggregation equals full aggregation for both modes."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    for method, covariance_kind in (
        ("mean", "standard"),
        ("geometric_median", "padded"),
    ):
        full, full_info = _aggregate_block_covariances(
            data, 10, method, covariance_kind, max_mem_mb=None
        )
        chunked, chunk_info = _aggregate_block_covariances(
            data, 10, method, covariance_kind, max_mem_mb=0.0001
        )
        assert full_info["memory_mode"] == "full"
        assert chunk_info["memory_mode"] == "chunked"
        assert np.allclose(full, chunked)


def test_aggregate_covariance_methods_and_validation() -> None:
    """Aggregation methods work and reject unknown modes or incompatible memory use."""
    rng = np.random.default_rng(42)
    covariances = []
    for _ in range(5):
        matrix = rng.standard_normal((3, 3))
        covariances.append(matrix @ matrix.T)
    covariances = np.asarray(covariances)
    for method in ("mean", "median", "geometric_median"):
        covariance = _aggregate_covariances(covariances, method)
        assert covariance.shape == (3, 3)
        assert np.all(np.isfinite(covariance))
        assert np.allclose(covariance, covariance.T)

    with pytest.raises(ValueError, match="covariance_kind must be"):
        _aggregate_block_covariances(np.ones((3, 100)), 10, "mean", "invalid", None)
    with pytest.raises(ValueError, match="cov_estimator='median' requires full"):
        _aggregate_block_covariances(
            np.ones((3, 100)), 10, "median", "standard", max_mem_mb=0.0001
        )


def test_block_covariance_layout_contract() -> None:
    """Standard/padded block iterators match their full materializations."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    blocksize = 10
    standard = _block_covariances_standard(data, blocksize)
    padded = _block_covariances_padded(data, blocksize)
    assert standard.shape == padded.shape == (10, 3, 3)
    assert np.allclose(standard[0], data[:, :10] @ data[:, :10].T / blocksize)
    assert np.allclose(padded, padded.swapaxes(-1, -2))
    for iterator, expected in (
        (_iter_block_covariances_standard(data, blocksize, 3), standard),
        (_iter_block_covariances_padded(data, blocksize, 3), padded),
    ):
        chunks = list(iterator)
        assert [chunk.shape[0] for chunk in chunks] == [3, 3, 3, 1]
        assert np.allclose(np.concatenate(chunks, axis=0), expected)


def test_adaptive_covariance_sqrt() -> None:
    """Adaptive covariance square-root output is symmetric and self-consistent."""
    from mne_denoise.asr._covariance import _adaptive_covariance_sqrt

    rng = np.random.default_rng(42)
    M, C, eigenvalues, V, info = _adaptive_covariance_sqrt(
        rng.standard_normal((3, 100)),
        blocksize=10,
        regularization=1e-5,
        max_mem_mb=None,
    )
    assert M.shape == C.shape == (3, 3)
    assert eigenvalues.shape == (3,)
    assert V.shape == (3, 3)
    assert isinstance(info, dict)
    assert np.all(eigenvalues >= 0)
    assert np.allclose(M, M.T)
    assert np.allclose(M @ M, C)


def test_chunked_moving_covariances() -> None:
    """Chunked moving covariance yields symmetric updates and reports failures."""
    from mne_denoise.asr._covariance import _ChunkedMovingCovariances

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    iterator = _ChunkedMovingCovariances(
        X=data,
        update_at=np.array([20, 40, 80]),
        window_length=10,
        chunk_samples=25,
        zi=None,
    )
    covariances = list(iterator)
    assert len(covariances) == 3
    assert all(covariance.shape == (3, 3) for covariance in covariances)
    assert all(np.allclose(covariance, covariance.T) for covariance in covariances)
    with pytest.raises(RuntimeError, match="Failed to produce"):
        list(_ChunkedMovingCovariances(data, np.array([20, 120]), 10, 25, None))
