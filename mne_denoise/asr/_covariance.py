"""Covariance estimation, aggregation, and memory budgeting for ASR.

This module contains utilities for block and rolling covariance builders
(padded vs standard conventions), robust aggregation (mean, geometric median,
including a memory-bounded chunked path), and helpers that size the covariance
buffers against ``max_mem_mb``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._spd import (
    _geometric_median,
    _geometric_median_chunked,
    _regularize_spd,
    _sqrtm_spd,
)


def _process_memory_info(
    n_channels: int,
    n_stream_input: int,
    max_mem_mb: int | float | None,
    memory_mode: str,
    peak_cov_buffer_bytes: int,
    chunk_samples: int,
    used_memory_bound: bool,
) -> dict[str, Any]:
    """Create process-time covariance memory diagnostics.

    Parameters
    ----------
    n_channels : int
        The number of channels in the data.
    n_stream_input : int
        The number of samples in the stream input.
    max_mem_mb : int, float, or None
        The maximum memory budget in megabytes.
    memory_mode : str
        The active memory strategy (e.g., 'full' or 'chunked').
    peak_cov_buffer_bytes : int
        The peak bytes allocated for the covariance buffer.
    chunk_samples : int
        The size of each processing chunk in samples.
    used_memory_bound : bool
        Whether the memory bound was triggered and utilized.

    Returns
    -------
    diagnostics : dict
        A dictionary containing diagnostic memory usage information.
    """
    return {
        "memory_mode": memory_mode,
        "max_mem_mb": max_mem_mb,
        "used_memory_bound": bool(used_memory_bound),
        "estimated_full_cov_bytes": _covariance_stack_bytes(
            n_stream_input,
            n_channels,
        ),
        "peak_cov_buffer_bytes": int(peak_cov_buffer_bytes),
        "chunk_samples": int(chunk_samples),
    }


def _iter_moving_covariances_at(
    X: np.ndarray,
    update_at: np.ndarray,
    window_length: int,
):
    """Yield trailing moving covariances at specified sample counts.

    Parameters
    ----------
    X : np.ndarray
        The continuous input data of shape (n_channels, n_times).
    update_at : np.ndarray
        A 1D array of strictly increasing sample indices at which to yield
        the current trailing moving average covariance.
    window_length : int
        The length of the moving window in samples.

    Yields
    ------
    cov_sum : np.ndarray
        The instantaneous trailing covariance matrix of shape (n_channels, n_channels)
        normalized by the window length.
    """
    n_channels, _ = X.shape
    cov_sum = np.zeros((n_channels, n_channels), dtype=np.float64)
    current_n = 0
    for target_n in update_at:
        target_n = int(target_n)
        while current_n < target_n:
            sample = X[:, current_n]
            cov_sum += np.outer(sample, sample)
            remove_idx = current_n - window_length
            if remove_idx >= 0:
                old_sample = X[:, remove_idx]
                cov_sum -= np.outer(old_sample, old_sample)
            current_n += 1
        yield cov_sum / window_length


def _moving_average_padded(
    window_length: int,
    X: np.ndarray,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fast, exact moving-average filter to flattened covariance traces.

    Parameters
    ----------
    window_length : int
        The length of the moving window in samples.
    X : np.ndarray
        The input flattened matrix traces of shape (n_features, n_times).
    zi : np.ndarray, optional
        The initial conditions (prior padded samples) of shape
        (n_features, window_length). If None, initializes with zeros.

    Returns
    -------
    out : np.ndarray
        The smoothed trace matrix of shape (n_features, n_times).
    zf : np.ndarray
        The final conditions of shape (n_features, window_length) carrying
        the tail needed to seamlessly filter the next contiguous block.
    """
    if zi is None:
        zi = np.zeros((X.shape[0], window_length), dtype=np.float64)
    Y = np.concatenate([zi, X], axis=1)
    diffs = (Y[:, window_length:] - Y[:, :-window_length]) / window_length
    out = np.cumsum(diffs, axis=1)
    zf_first = Y[:, [-window_length]] - out[:, [-1]] * window_length
    zf_tail = (
        Y[:, -window_length + 1 :] if window_length > 1 else np.empty((X.shape[0], 0))
    )
    zf = np.concatenate([zf_first, zf_tail], axis=1)
    return out, zf


def _window_covariances(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
) -> np.ndarray:
    """Compute mean-centered covariances for a set of specific windows.

    Parameters
    ----------
    X : np.ndarray
        The input data array of shape (n_channels, n_times).
    starts : np.ndarray
        A 1D array of window start indices.
    win_len : int
        The length of each window in samples.

    Returns
    -------
    covariances : np.ndarray
        A 3D array of shape (n_windows, n_channels, n_channels) containing the
        sample covariance of each specified window.
    """
    covariances = np.empty((len(starts), X.shape[0], X.shape[0]), dtype=np.float64)
    for idx, start in enumerate(starts):
        window = X[:, start : start + win_len]
        window = window - window.mean(axis=1, keepdims=True)
        covariances[idx] = (window @ window.T) / max(win_len - 1, 1)
    return covariances


def _max_mem_bytes(max_mem_mb: int | float | None) -> int | None:
    """Convert an optional megabyte memory cap to bytes.

    Parameters
    ----------
    max_mem_mb : int, float, or None
        The maximum memory budget in megabytes.

    Returns
    -------
    max_mem_bytes : int or None
        The maximum memory budget in bytes, or None if uncapped.
    """
    if max_mem_mb is None:
        return None
    max_mem_mb = float(max_mem_mb)
    if not np.isfinite(max_mem_mb) or max_mem_mb < 0:
        raise ValueError("max_mem_mb must be non-negative or None")
    return int(max_mem_mb * 1024 * 1024)


def _covariance_stack_bytes(n_items: int, n_channels: int) -> int:
    """Return the memory allocation size needed for a 3D covariance stack.

    Parameters
    ----------
    n_items : int
        The number of covariance matrices in the stack.
    n_channels : int
        The number of channels (dimension of the square matrices).

    Returns
    -------
    bytes : int
        The estimated memory footprint in bytes.
    """
    return int(n_items * n_channels * n_channels * np.dtype(np.float64).itemsize)


def _covariance_chunk_blocks(n_channels: int, max_mem_bytes: int | None) -> int:
    """Resolve a conservative covariance chunk size under a memory cap.

    Parameters
    ----------
    n_channels : int
        The number of channels.
    max_mem_bytes : int or None
        The maximum memory budget in bytes.

    Returns
    -------
    chunk_blocks : int
        The maximum number of blocks that can be processed simultaneously
        without exceeding the memory budget.
    """
    per_block = _covariance_stack_bytes(1, n_channels)
    if max_mem_bytes is None:
        return np.iinfo(np.int32).max
    budget = max(per_block, max_mem_bytes // 4)
    return max(1, int(budget // per_block))


def _iter_block_covariances_padded(
    X: np.ndarray,
    blocksize: int,
    chunk_blocks: int,
):
    """Yield padded-style trailing-block covariances in memory-safe chunks.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (n_channels, n_times).
    blocksize : int
        The size of each individual block in samples.
    chunk_blocks : int
        The number of blocks to compute and yield per chunk.

    Yields
    ------
    covariances : np.ndarray
        A 3D array of block covariances for the current chunk.
    """
    n_channels, n_times = X.shape
    n_blocks = max(1, int(np.ceil(n_times / blocksize)))
    X_t = X.T
    for start_block in range(0, n_blocks, chunk_blocks):
        stop_block = min(start_block + chunk_blocks, n_blocks)
        block_ids = np.arange(start_block, stop_block, dtype=int)
        covariances = np.zeros(
            (block_ids.size, n_channels, n_channels), dtype=np.float64
        )
        for offset in range(blocksize):
            sample_idx = np.minimum(n_times - 1, offset + block_ids * blocksize)
            samples = X_t[sample_idx]
            covariances += np.einsum("bi,bj->bij", samples, samples, optimize=True)
        yield covariances / blocksize


def _iter_block_covariances_standard(
    X: np.ndarray,
    blocksize: int,
    chunk_blocks: int,
):
    """Yield standard-style contiguous block covariances in memory-safe chunks.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (n_channels, n_times).
    blocksize : int
        The size of each individual block in samples.
    chunk_blocks : int
        The number of blocks to compute and yield per chunk.

    Yields
    ------
    covariances : np.ndarray
        A 3D array of block covariances for the current chunk.
    """
    n_channels, n_times = X.shape
    starts = np.arange(0, n_times, blocksize, dtype=int)
    for start_block in range(0, len(starts), chunk_blocks):
        stop_block = min(start_block + chunk_blocks, len(starts))
        covariances = np.empty(
            (stop_block - start_block, n_channels, n_channels),
            dtype=np.float64,
        )
        for out_idx, sample_start in enumerate(starts[start_block:stop_block]):
            sample_stop = min(sample_start + blocksize, n_times)
            block = X[:, sample_start:sample_stop]
            covariances[out_idx] = (block @ block.T) / blocksize
        yield covariances


def _block_covariances_padded(X: np.ndarray, blocksize: int) -> np.ndarray:
    """Match the padded trailing-block covariance accumulation.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (n_channels, n_times).
    blocksize : int
        The size of each individual block in samples.

    Returns
    -------
    covariances : np.ndarray
        A 3D array of shape (n_blocks, n_channels, n_channels).
    """
    n_channels, n_times = X.shape
    n_blocks = max(1, int(np.ceil(n_times / blocksize)))
    covariances = np.zeros((n_blocks, n_channels, n_channels), dtype=np.float64)
    X_t = X.T
    for offset in range(blocksize):
        sample_idx = np.minimum(
            n_times - 1,
            np.arange(offset, n_times + offset, blocksize, dtype=int),
        )
        samples = X_t[sample_idx]
        covariances += np.einsum("bi,bj->bij", samples, samples)
    return covariances / blocksize


def _block_covariances_standard(X: np.ndarray, blocksize: int) -> np.ndarray:
    """Match the standard contiguous blocks with a full blocksize divisor.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (n_channels, n_times).
    blocksize : int
        The size of each individual block in samples.

    Returns
    -------
    covariances : np.ndarray
        A 3D array of shape (n_blocks, n_channels, n_channels).
    """
    n_channels, n_times = X.shape
    starts = np.arange(0, n_times, blocksize, dtype=int)
    covariances = np.empty((len(starts), n_channels, n_channels), dtype=np.float64)
    for idx, start in enumerate(starts):
        stop = min(start + blocksize, n_times)
        block = X[:, start:stop]
        covariances[idx] = (block @ block.T) / blocksize
    return covariances


def _aggregate_covariances(covariances: np.ndarray, method: str) -> np.ndarray:
    """Aggregate a stack of covariance matrices into a single baseline estimate.

    Parameters
    ----------
    covariances : np.ndarray
        A 3D array of covariance matrices.
    method : str
        The aggregation method, either 'mean', 'median', or 'geometric_median'.

    Returns
    -------
    C : np.ndarray
        The aggregated baseline covariance matrix.
    """
    if method == "mean":
        return np.mean(covariances, axis=0)
    if method == "median":
        return np.median(covariances, axis=0)
    return _geometric_median(covariances)


def _aggregate_block_covariances(
    X: np.ndarray,
    blocksize: int,
    method: str,
    covariance_kind: str,
    max_mem_mb: int | float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate ASR calibration covariances with optional memory bounding.

    Parameters
    ----------
    X : np.ndarray
        The input data array.
    blocksize : int
        The size of the covariance blocks.
    method : str
        The aggregation method (mean, median, or geometric_median).
    covariance_kind : str
        The block generation strategy ('padded' or 'standard').
    max_mem_mb : int, float, or None
        The maximum memory budget for covariance buffers.

    Returns
    -------
    C : np.ndarray
        The aggregated covariance baseline.
    info : dict
        Memory and performance diagnostics.
    """
    n_channels, n_times = X.shape
    if covariance_kind == "standard":
        n_blocks = len(np.arange(0, n_times, blocksize, dtype=int))
        iter_factory = lambda chunk_blocks: _iter_block_covariances_standard(
            X,
            blocksize,
            chunk_blocks,
        )
        full_factory = _block_covariances_standard
    elif covariance_kind == "padded":
        n_blocks = max(1, int(np.ceil(n_times / blocksize)))
        iter_factory = lambda chunk_blocks: _iter_block_covariances_padded(
            X,
            blocksize,
            chunk_blocks,
        )
        full_factory = _block_covariances_padded
    else:
        raise ValueError("covariance_kind must be 'padded' or 'standard'")

    estimated_bytes = _covariance_stack_bytes(n_blocks, n_channels)
    max_bytes = _max_mem_bytes(max_mem_mb)
    use_chunked = max_bytes is not None and estimated_bytes > max_bytes
    if not use_chunked:
        covariances = full_factory(X, blocksize)
        info = {
            "memory_mode": "full",
            "max_mem_mb": max_mem_mb,
            "used_memory_bound": False,
            "estimated_full_cov_bytes": estimated_bytes,
            "peak_cov_buffer_bytes": estimated_bytes,
            "chunk_samples": int(n_times),
            "covariance_chunk_blocks": int(n_blocks),
        }
        return _aggregate_covariances(covariances, method), info

    if method == "median":
        raise ValueError(
            "cov_estimator='median' requires full covariance storage; increase "
            "max_mem_mb or use cov_estimator='geometric_median' or 'mean'."
        )

    chunk_blocks = _covariance_chunk_blocks(n_channels, max_bytes)
    if method == "mean":
        total = np.zeros((n_channels, n_channels), dtype=np.float64)
        count = 0
        for chunk in iter_factory(chunk_blocks):
            total += np.sum(chunk, axis=0)
            count += chunk.shape[0]
        C = total / max(count, 1)
    else:
        C = _geometric_median_chunked(iter_factory, chunk_blocks, n_channels)

    info = {
        "memory_mode": "chunked",
        "max_mem_mb": max_mem_mb,
        "used_memory_bound": True,
        "estimated_full_cov_bytes": estimated_bytes,
        "peak_cov_buffer_bytes": _covariance_stack_bytes(chunk_blocks, n_channels),
        "chunk_samples": int(chunk_blocks * blocksize),
        "covariance_chunk_blocks": int(chunk_blocks),
    }
    return C, info


def _adaptive_covariance_sqrt(
    X: np.ndarray,
    blocksize: int,
    regularization: float,
    max_mem_mb: int | float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute the regularized square root of the block-aggregated covariance.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The input data array.
    blocksize : int
        The number of samples per block for covariance estimation.
    regularization : float
        The regularization parameter added to the diagonal of the covariance.
    max_mem_mb : int | float | None
        The maximum memory in MB to allocate for block covariance storage.

    Returns
    -------
    M : ndarray, shape (n_channels, n_channels)
        The regularized principal square root of the covariance.
    C : ndarray, shape (n_channels, n_channels)
        The regularized covariance matrix.
    eigvals : ndarray, shape (n_channels,)
        The eigenvalues of the square root matrix M.
    V : ndarray, shape (n_channels, n_channels)
        The eigenvectors of M.
    memory_info : dict
        Diagnostic information from the aggregation process.
    """
    C, memory_info = _aggregate_block_covariances(
        X,
        blocksize,
        "geometric_median",
        covariance_kind="padded",
        max_mem_mb=max_mem_mb,
    )
    C = _regularize_spd(C, regularization)
    M = _sqrtm_spd(C, regularization)
    eigvals, V = np.linalg.eigh(M)
    return M, C, eigvals, V, memory_info


class _ChunkedMovingCovariances:
    """Yield moving covariances at update positions without a full time stack.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The input data array.
    update_at : ndarray of int
        The sample indices at which the moving covariance should be yielded.
    window_length : int
        The length of the moving average window in samples.
    chunk_samples : int
        The number of samples to process at a time (block size).
    zi : ndarray | None
        The initial state for the moving average filter.
    """

    def __init__(
        self,
        X: np.ndarray,
        update_at: np.ndarray,
        window_length: int,
        chunk_samples: int,
        zi: np.ndarray | None,
    ) -> None:
        self.X = X
        self.update_at = np.asarray(update_at, dtype=int)
        self.window_length = int(window_length)
        self.chunk_samples = int(chunk_samples)
        self.cov_state = zi.copy() if zi is not None else None

    def __iter__(self):
        n_channels, n_times = self.X.shape
        update_idx = 0
        for start in range(0, n_times, self.chunk_samples):
            stop = min(start + self.chunk_samples, n_times)
            chunk = self.X[:, start:stop]
            outer = np.einsum("it,jt->ijt", chunk, chunk, optimize=True)
            flat = outer.reshape(n_channels * n_channels, stop - start, order="F")
            flat, self.cov_state = _moving_average_padded(
                self.window_length,
                flat,
                zi=self.cov_state,
            )
            while (
                update_idx < self.update_at.size
                and start < self.update_at[update_idx] <= stop
            ):
                local = int(self.update_at[update_idx] - start - 1)
                yield flat[:, local].reshape(n_channels, n_channels, order="F")
                update_idx += 1
        if update_idx != self.update_at.size:
            raise RuntimeError("Failed to produce all adaptive ASR covariance updates")
