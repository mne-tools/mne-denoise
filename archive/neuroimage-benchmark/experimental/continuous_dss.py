"""Stateful continuous DSS for replay and real-time validation.

The implementation keeps the online estimator deliberately separate from the
offline :class:`mne_denoise.dss.DSS` API.  Covariances are updated causally,
the DSS solution is refreshed at a declared interval, component identities are
matched across refreshes, and filter changes are cross-faded at block edges.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin

from ..dss.linear import compute_dss
from ..utils import extract_data_from_mne, reconstruct_mne_object

_EPS = float(np.finfo(np.float64).eps)


class _StreamingBias:
    """Causal state-carrying IIR bias used by :class:`ContinuousDSS`."""

    def __init__(
        self,
        sfreq: float,
        kind: str,
        freq_band: tuple[float, float],
        line_freq: float,
        bandwidth: float,
        order: int,
    ) -> None:
        nyquist = sfreq / 2.0
        if kind == "bandpass":
            low, high = freq_band
        elif kind == "line_noise":
            low, high = line_freq - bandwidth / 2, line_freq + bandwidth / 2
        else:
            raise ValueError("bias must be 'bandpass' or 'line_noise'")
        if not 0 < low < high < nyquist:
            raise ValueError("bias pass band must lie strictly inside Nyquist")
        self.sos = signal.butter(
            order, [low / nyquist, high / nyquist], btype="band", output="sos"
        )
        self.state_: np.ndarray | None = None

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.state_ is None:
            # For filtering along time (axis=1), SciPy requires
            # (n_sections, n_channels, 2) state ordering.
            self.state_ = np.zeros((self.sos.shape[0], block.shape[0], 2))
        filtered, self.state_ = signal.sosfilt(
            self.sos, block, axis=1, zi=self.state_
        )
        return filtered

    def reset(self) -> None:
        self.state_ = None


class _EMACovariance:
    def __init__(self, n_channels: int, forgetting: float) -> None:
        if not 0.0 < forgetting < 1.0:
            raise ValueError("forgetting factors must be in (0, 1)")
        self.n_channels = n_channels
        self.forgetting = forgetting
        self.reset()

    def reset(self) -> None:
        self.covariance_ = np.zeros((self.n_channels, self.n_channels))
        self.mean_ = np.zeros(self.n_channels)
        self.n_updates_ = 0

    def update(self, block: np.ndarray) -> None:
        mean = block.mean(axis=1)
        if self.n_updates_ == 0:
            self.mean_ = mean
        else:
            self.mean_ = self.forgetting * self.mean_ + (
                1.0 - self.forgetting
            ) * mean
        centered = block - self.mean_[:, None]
        block_cov = centered @ centered.T / max(1, block.shape[1])
        if self.n_updates_ == 0:
            self.covariance_ = block_cov
        else:
            self.covariance_ = self.forgetting * self.covariance_ + (
                1.0 - self.forgetting
            ) * block_cov
        self.n_updates_ += 1


class ContinuousDSS(BaseEstimator, TransformerMixin):
    """Causal blockwise DSS with inspectable streaming state.

    Parameters
    ----------
    n_channels : int
        Expected number of homogeneous sensor channels.
    sfreq : float
        Sampling rate in Hz.
    bias : {'bandpass', 'line_noise'}, default='bandpass'
        Causal target operator.
    mode : {'enhance', 'denoise'}, default='enhance'
        Keep or subtract the leading target-aware DSS subspace.
    block_size : int, default=64
        Block size used by :meth:`transform`; :meth:`process_block` accepts any
        positive block length.
    channel_names : sequence of str | None
        Frozen channel order. Reordered block labels are rejected.

    Notes
    -----
    This is an experimental adaptive estimator. Its fitted state changes during
    ``transform``. Benchmarks must therefore compare it with replay-ordered data
    and declare initialization, block size, and information access.
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        *,
        bias: str = "bandpass",
        freq_band: tuple[float, float] = (8.0, 12.0),
        line_freq: float = 60.0,
        bandwidth: float = 2.0,
        filter_order: int = 4,
        n_components: int = 1,
        lambda_baseline: float = 0.995,
        lambda_biased: float = 0.99,
        solve_interval: int = 10,
        warmup_blocks: int = 20,
        block_size: int = 64,
        rank: int | None = None,
        reg: float = 1e-9,
        mode: str = "enhance",
        channel_names: list[str] | tuple[str, ...] | None = None,
        experimental: bool = False,
    ) -> None:
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.bias = bias
        self.freq_band = freq_band
        self.line_freq = line_freq
        self.bandwidth = bandwidth
        self.filter_order = filter_order
        self.n_components = n_components
        self.lambda_baseline = lambda_baseline
        self.lambda_biased = lambda_biased
        self.solve_interval = solve_interval
        self.warmup_blocks = warmup_blocks
        self.block_size = block_size
        self.rank = rank
        self.reg = reg
        self.mode = mode
        self.channel_names = channel_names
        self.experimental = experimental

    def fit(self, X=None, y=None):
        """Reset state and optionally warm it with chronological calibration data."""
        del y
        self._validate_parameters()
        self._initialize_state()
        if X is not None:
            data, names, mne_type = self._extract(X)
            if mne_type == "epochs":
                raise ValueError("ContinuousDSS requires chronological Raw or 2D data")
            self._lock_channel_order(names)
            for start in range(0, data.shape[1], self.block_size):
                self.process_block(data[:, start : start + self.block_size], names)
        return self

    def transform(self, X):
        """Process chronological data in fixed blocks while updating fitted state."""
        if not hasattr(self, "baseline_covariance_"):
            raise RuntimeError("ContinuousDSS is not fitted; call fit() first")
        data, names, mne_type, original, picks = self._extract(X, full=True)
        if mne_type == "epochs":
            raise ValueError("ContinuousDSS cannot infer chronology across Epochs")
        self._lock_channel_order(names)
        pieces = []
        for start in range(0, data.shape[1], self.block_size):
            pieces.append(
                self.process_block(data[:, start : start + self.block_size], names)
            )
        output = np.concatenate(pieces, axis=1) if pieces else data.copy()
        return reconstruct_mne_object(
            output, original, mne_type, picks=picks, verbose=False
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Reset and process ``X`` once in replay order."""
        del y, fit_params
        self.fit()
        return self.transform(X)

    def process_block(
        self, block: np.ndarray, channel_names: list[str] | tuple[str, ...] | None = None
    ) -> np.ndarray:
        """Process one block and return a same-shape output."""
        if not hasattr(self, "baseline_covariance_"):
            raise RuntimeError("ContinuousDSS is not fitted; call fit() first")
        block = np.asarray(block, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != self.n_channels:
            raise ValueError(
                f"block must have shape ({self.n_channels}, n_samples), got {block.shape}"
            )
        if block.shape[1] == 0:
            raise ValueError("blocks must contain at least one sample")
        self._lock_channel_order(channel_names)
        if not np.all(np.isfinite(block)):
            self.failure_counts_["nonfinite_block"] += 1
            self.block_status_.append("rejected_nonfinite")
            return block.copy()

        started = perf_counter()
        self._baseline.update(block)
        biased = self._bias_operator.process(block)
        self._biased.update(biased)
        self.n_blocks_seen_ += 1
        solved = False
        if self.n_blocks_seen_ >= self.warmup_blocks and (
            self.filters_ is None
            or self.n_blocks_seen_ % self.solve_interval == 0
        ):
            solved = self._solve()

        if self.filters_ is None:
            output = block.copy()
            self.block_status_.append("warmup")
        else:
            centered = block - self._baseline.mean_[:, None]
            projection = self.patterns_ @ (self.filters_ @ centered)
            if self.mode == "enhance":
                output = projection + self._baseline.mean_[:, None]
            else:
                output = block - projection
            if solved and self.previous_filters_ is not None:
                old_projection = self.previous_patterns_ @ (
                    self.previous_filters_ @ centered
                )
                old_output = (
                    old_projection + self._baseline.mean_[:, None]
                    if self.mode == "enhance"
                    else block - old_projection
                )
                fade_n = min(block.shape[1], max(2, block.shape[1] // 2))
                fade = np.linspace(0.0, 1.0, fade_n)[None, :]
                output[:, :fade_n] = (1.0 - fade) * old_output[:, :fade_n] + (
                    fade * output[:, :fade_n]
                )
            self.block_status_.append("solved" if solved else "processed")

        self.baseline_covariance_ = self._baseline.covariance_.copy()
        self.biased_covariance_ = self._biased.covariance_.copy()
        self.processing_time_s_.append(perf_counter() - started)
        return output

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a serializable summary of current adaptive state."""
        return {
            "n_blocks_seen": self.n_blocks_seen_,
            "n_solves": self.n_solves_,
            "effective_rank": None if self.filters_ is None else self.filters_.shape[0],
            "failure_counts": dict(self.failure_counts_),
            "mean_processing_time_s": float(np.mean(self.processing_time_s_))
            if self.processing_time_s_
            else 0.0,
            "real_time_factor": float(
                np.sum(self.processing_time_s_)
                / max(self.n_blocks_seen_ * self.block_size / self.sfreq, _EPS)
            ),
            "channel_names": None
            if self.channel_names_ is None
            else list(self.channel_names_),
            "information_access": "causal transform-local adaptation",
        }

    def reset(self):
        """Discard all learned streaming state."""
        return self.fit()

    def _initialize_state(self) -> None:
        self._baseline = _EMACovariance(self.n_channels, self.lambda_baseline)
        self._biased = _EMACovariance(self.n_channels, self.lambda_biased)
        self._bias_operator = _StreamingBias(
            self.sfreq,
            self.bias,
            self.freq_band,
            self.line_freq,
            self.bandwidth,
            self.filter_order,
        )
        self.baseline_covariance_ = self._baseline.covariance_.copy()
        self.biased_covariance_ = self._biased.covariance_.copy()
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.previous_filters_: np.ndarray | None = None
        self.previous_patterns_: np.ndarray | None = None
        self.n_blocks_seen_ = 0
        self.n_solves_ = 0
        self.failure_counts_ = {"nonfinite_block": 0, "eigensolve": 0}
        self.block_status_: list[str] = []
        self.processing_time_s_: list[float] = []
        self.channel_names_ = (
            None if self.channel_names is None else tuple(self.channel_names)
        )

    def _solve(self) -> bool:
        try:
            filters, patterns, values = compute_dss(
                self._baseline.covariance_,
                self._biased.covariance_,
                n_components=self.n_components,
                rank=self.rank,
                reg=self.reg,
            )
        except (ValueError, np.linalg.LinAlgError):
            self.failure_counts_["eigensolve"] += 1
            return False
        if self.filters_ is not None:
            filters, patterns, values = self._match(filters, patterns, values)
            self.previous_filters_ = self.filters_.copy()
            self.previous_patterns_ = self.patterns_.copy()
        self.filters_ = filters
        self.patterns_ = patterns
        self.eigenvalues_ = values
        self.n_solves_ += 1
        return True

    def _match(self, filters, patterns, values):
        count = min(self.filters_.shape[0], filters.shape[0])
        old = self.filters_[:count]
        new = filters[:count]
        old = old / np.maximum(np.linalg.norm(old, axis=1, keepdims=True), _EPS)
        new = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True), _EPS)
        corr = old @ new.T
        rows, cols = linear_sum_assignment(-np.abs(corr))
        reordered_f = filters.copy()
        reordered_p = patterns.copy()
        reordered_v = values.copy()
        reordered_f[rows] = filters[cols]
        reordered_p[:, rows] = patterns[:, cols]
        reordered_v[rows] = values[cols]
        for row, col in zip(rows, cols):
            if corr[row, col] < 0:
                reordered_f[row] *= -1
                reordered_p[:, row] *= -1
        return reordered_f, reordered_p, reordered_v

    def _lock_channel_order(self, names) -> None:
        if names is None:
            return
        names = tuple(names)
        if len(names) != self.n_channels:
            raise ValueError("channel_names length does not match n_channels")
        if self.channel_names_ is None:
            self.channel_names_ = names
        elif names != self.channel_names_:
            self.failure_counts_["channel_order"] = (
                self.failure_counts_.get("channel_order", 0) + 1
            )
            raise ValueError("channel order changed during ContinuousDSS replay")

    def _extract(self, X, full: bool = False):
        data, _, mne_type, original, picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 and mne_type != "epochs":
            raise ValueError("ContinuousDSS requires a 2D channel-by-time input")
        if data.shape[-2 if mne_type == "epochs" else 0] != self.n_channels:
            raise ValueError("input channel count does not match n_channels")
        if full:
            return data, names, mne_type, original, picks
        return data, names, mne_type

    def _validate_parameters(self) -> None:
        if not self.experimental:
            raise ValueError(
                "ContinuousDSS is experimental; pass experimental=True to opt in"
            )
        if self.n_channels < 2 or self.sfreq <= 0:
            raise ValueError("n_channels must be >=2 and sfreq must be positive")
        if self.n_components < 1 or self.n_components > self.n_channels:
            raise ValueError("n_components must be between 1 and n_channels")
        if self.solve_interval < 1 or self.warmup_blocks < 1 or self.block_size < 1:
            raise ValueError("block and update intervals must be positive")
        if self.mode not in {"enhance", "denoise"}:
            raise ValueError("mode must be 'enhance' or 'denoise'")
        if self.channel_names is not None and len(self.channel_names) != self.n_channels:
            raise ValueError("channel_names length does not match n_channels")


__all__ = ["ContinuousDSS"]
