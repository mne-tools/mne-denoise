"""Basic Singular Spectrum Analysis."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from .._logging import logger, verbose
from .._validation import (
    check_channel_first_data,
    check_positive_integer,
    check_positive_real,
)
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._common import (
    _BaseSSATransformer,
    _diagonal_average,
    _resolve_window_length,
    _trajectory_matrix,
)


def ssa_decompose(
    x: np.ndarray,
    window_length: int | None = None,
    *,
    window_seconds: float | None = None,
    sfreq: float | None = None,
    max_window: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decompose a one-dimensional series into Basic SSA components.

    Parameters
    ----------
    x : array-like, shape (n_times,)
        Finite scalar time series.
    window_length : int | None, default=None
        Embedding dimension in samples. If None, choose it automatically.
    window_seconds : float | None, default=None
        Embedding duration in seconds; mutually exclusive with window_length and
        requiring sfreq.
    sfreq : float | None, default=None
        Sampling frequency in Hz, used with window_seconds and automatic selection.
    max_window : int, default=100
        Maximum automatic embedding dimension.

    Returns
    -------
    components : ndarray, shape (n_components, n_times)
        Reconstructed elementary components in decreasing singular-value order.
    info : dict
        Resolved window, trajectory shape, singular values, and numerical rank.

    Notes
    -----
    The trajectory matrix is decomposed by SVD and reconstructed by anti-diagonal
    averaging. :footcite:p:`golyandina_zhigljavsky2013_ssa`.

    References
    ----------
    .. footbibliography::

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If x or the requested embedding is invalid.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    resolved = _resolve_window_length(
        x.size,
        window_length,
        window_seconds=window_seconds,
        sfreq=sfreq,
        max_window=max_window,
    )
    trajectory = _trajectory_matrix(x, resolved)
    left, singular_values, right_t = np.linalg.svd(trajectory, full_matrices=False)
    components = np.stack(
        [
            _diagonal_average(
                singular_values[index] * np.outer(left[:, index], right_t[index])
            )
            for index in range(singular_values.size)
        ]
    )
    tolerance = (
        np.finfo(float).eps
        * max(trajectory.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    return components, {
        "singular_values": singular_values,
        "window_length": resolved,
        "trajectory_shape": trajectory.shape,
        "rank": int(np.count_nonzero(singular_values > tolerance)),
    }


def ssa_w_correlation(components: np.ndarray, window_length: int) -> np.ndarray:
    """Compute weighted correlations between SSA reconstructions.

    Parameters
    ----------
    components : array-like, shape (n_components, n_times)
        Reconstructed components.
    window_length : int
        Embedding dimension used to obtain components.

    Returns
    -------
    correlation : ndarray, shape (n_components, n_components)
        Symmetric weighted-correlation matrix; zero-energy components have zero
        rows and columns.

    Notes
    -----
    The weights are the anti-diagonal multiplicities of the trajectory matrix.
    This is a separability diagnostic, not an artifact-selection rule.
    :footcite:p:`golyandina_zhigljavsky2013_ssa`.

    References
    ----------
    .. footbibliography::
    """
    components = np.asarray(components, dtype=np.float64)
    if components.ndim != 2 or components.shape[1] < 1:
        raise ValueError("components must have shape (n_components, n_times)")
    window_length = check_positive_integer(window_length, name="window_length")
    n_times = components.shape[1]
    n_columns = n_times - window_length + 1
    if window_length > n_columns or window_length < 2:
        raise ValueError("window_length must satisfy 2 <= L <= K")
    indices = np.arange(n_times)
    weights = np.minimum.reduce(
        (
            indices + 1,
            np.full(n_times, window_length),
            np.full(n_times, n_columns),
            n_times - indices,
        )
    ).astype(float)
    weighted = components * np.sqrt(weights)
    gram = weighted @ weighted.T
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    denominator = np.outer(norms, norms)
    correlation = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0,
    )
    return np.clip(correlation, -1.0, 1.0)


def _check_frequency_parameters(
    sfreq: float,
    drop_freq_max: float,
    drop_band: tuple[float, float] | None,
    n_check: int | None,
) -> tuple[float, float, tuple[float, float] | None, int | None]:
    """Validate the package-specific Basic SSA grouping rule."""
    sfreq = check_positive_real(sfreq, name="sfreq")
    if isinstance(drop_freq_max, bool) or not isinstance(drop_freq_max, Real):
        raise TypeError("drop_freq_max must be a finite number")
    drop_freq_max = float(drop_freq_max)
    if not np.isfinite(drop_freq_max):
        raise ValueError("drop_freq_max must be finite")
    nyquist = sfreq / 2.0
    if not 0.0 <= drop_freq_max <= nyquist:
        raise ValueError("drop_freq_max must be between 0 and Nyquist")
    if drop_band is not None:
        if not isinstance(drop_band, tuple) or len(drop_band) != 2:
            raise TypeError("drop_band must be a (low, high) tuple or None")
        low, high = drop_band
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in (low, high)):
            raise TypeError("drop_band bounds must be finite numbers")
        low, high = float(low), float(high)
        if not np.isfinite((low, high)).all():
            raise ValueError("drop_band bounds must be finite numbers")
        if not 0.0 <= low < high <= nyquist:
            raise ValueError("drop_band must satisfy 0 <= low < high <= Nyquist")
        drop_band = (low, high)
    if n_check is not None:
        n_check = check_positive_integer(n_check, name="n_check")
    return sfreq, drop_freq_max, drop_band, n_check


def ssa_clean_channel(
    x: np.ndarray,
    sfreq: float,
    window_length: int | None = None,
    drop_freq_max: float = 3.0,
    drop_band: tuple[float, float] | None = None,
    n_check: int | None = None,
    max_window: int = 100,
    *,
    window_seconds: float | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Clean one channel by frequency-grouping Basic SSA components.

    Parameters
    ----------
    x : array-like, shape (n_times,)
        Finite scalar time series.
    sfreq : float
        Sampling frequency in Hz.
    window_length : int | None, default=None
        Embedding dimension in samples; None selects it automatically.
    drop_freq_max : float, default=3.0
        Upper bound, in Hz, for the dominant-frequency rejection rule when
        drop_band is None.
    drop_band : tuple of float | None, default=None
        Inclusive dominant-frequency interval to reject, in Hz.
    n_check : int | None, default=None
        Number of leading numerical-rank components to inspect; None inspects all.
    max_window : int, default=100
        Maximum automatic embedding dimension.
    window_seconds : float | None, default=None
        Embedding duration in seconds, mutually exclusive with window_length.
    return_info : bool, default=False
        If True, also return decomposition and grouping diagnostics.

    Returns
    -------
    x_clean : ndarray, shape (n_times,)
        Cleaned time series.
    info : dict
        Diagnostics returned only when return_info=True.

    Notes
    -----
    Dominant frequency is the largest real-FFT magnitude bin; DC is included.
    The frequency grouping and thresholds are package heuristics.
    """
    sfreq, drop_freq_max, drop_band, n_check = _check_frequency_parameters(
        sfreq, drop_freq_max, drop_band, n_check
    )
    if not isinstance(return_info, bool):
        raise TypeError("return_info must be a bool")
    components, decomposition = ssa_decompose(
        x,
        window_length,
        window_seconds=window_seconds,
        sfreq=sfreq,
        max_window=max_window,
    )
    x = np.asarray(x, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(components, axis=-1))
    bins = np.fft.rfftfreq(x.size, 1.0 / sfreq)
    dominant_frequencies = bins[np.argmax(spectrum, axis=-1)]
    candidates = np.arange(decomposition["rank"])
    if n_check is not None:
        candidates = candidates[:n_check]
    if drop_band is None:
        selected = candidates[dominant_frequencies[candidates] <= drop_freq_max]
    else:
        selected = candidates[
            (dominant_frequencies[candidates] >= drop_band[0])
            & (dominant_frequencies[candidates] <= drop_band[1])
        ]
    artifact = components[selected].sum(axis=0) if selected.size else np.zeros_like(x)
    cleaned = x - artifact
    info = {
        **decomposition,
        "components": components,
        "dominant_frequencies": dominant_frequencies,
        "dropped_indices": selected,
        "dropped_frequencies": dominant_frequencies[selected],
        "artifact": artifact,
        "frequency_resolution": sfreq / x.size,
    }
    if return_info:
        return cleaned, info
    return cleaned


@verbose
def compute_basic_ssa(
    X: np.ndarray,
    sfreq: float,
    window_length: int | None = None,
    drop_freq_max: float = 3.0,
    drop_band: tuple[float, float] | None = None,
    n_check: int | None = None,
    max_window: int = 100,
    *,
    window_seconds: float | None = None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply frequency-guided Basic SSA independently to each channel.

    Parameters
    ----------
    X : array-like, shape (n_channels, n_times)
        Finite channel-first data. Channels are not mixed.
    sfreq : float
        Sampling frequency in Hz.
    window_length : int | None, default=None
        Embedding dimension in samples; None selects it automatically.
    drop_freq_max : float, default=3.0
        Dominant-frequency upper bound in Hz.
    drop_band : tuple of float | None, default=None
        Inclusive dominant-frequency rejection interval in Hz.
    n_check : int | None, default=None
        Number of leading numerical-rank components to inspect.
    max_window : int, default=100
        Maximum automatic embedding dimension.
    window_seconds : float | None, default=None
        Embedding duration in seconds, mutually exclusive with window_length.
    callback : callable | None, default=None
        Synchronous callback after each channel; return values are ignored and
        callback exceptions propagate.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Independently cleaned channels.
    info : dict
        Per-channel selection diagnostics and the resolved operating point.

    Notes
    -----
    This is repeated univariate SSA, not a multivariate decomposition.
    """
    callback = _validate_callback(callback)
    X = check_channel_first_data(
        X, name="SSA", allow_epochs=False, min_channels=1, min_times=2
    )
    if X.shape[-1] < 3:
        raise ValueError("SSA requires at least 3 time samples")
    cleaned = np.empty_like(X)
    records = []
    for channel_idx, channel in enumerate(X):
        result, info = ssa_clean_channel(
            channel,
            sfreq,
            window_length,
            drop_freq_max,
            drop_band,
            n_check,
            max_window,
            window_seconds=window_seconds,
            return_info=True,
        )
        cleaned[channel_idx] = result
        records.append(info)
        _emit_progress(
            callback,
            method="basic_ssa",
            stage="channel",
            current=channel_idx + 1,
            total=X.shape[0],
            component=None,
            metric=float(len(info["dropped_indices"])),
        )
    info = {
        "method": "basic-frequency",
        "dropped_counts": np.array(
            [len(record["dropped_indices"]) for record in records], dtype=int
        ),
        "dropped_freqs": [record["dropped_frequencies"].tolist() for record in records],
        "dropped_frequencies": [
            record["dropped_frequencies"].copy() for record in records
        ],
        "dropped_indices": [record["dropped_indices"].copy() for record in records],
        "dominant_frequencies": [
            record["dominant_frequencies"].copy() for record in records
        ],
        "singular_values": [record["singular_values"].copy() for record in records],
        "window_length": records[0]["window_length"],
        "frequency_resolution": records[0]["frequency_resolution"],
    }
    logger.info(
        "Basic SSA: window=%d samples, channels=%d, dropped=%d component(s) "
        "(mean %.1f/channel).",
        info["window_length"],
        X.shape[0],
        int(np.sum(info["dropped_counts"])),
        float(np.mean(info["dropped_counts"])),
    )
    return cleaned, info


class SingularSpectrumAnalysis(_BaseSSATransformer):
    """Frequency-guided, channel-wise Basic SSA transformer.

    Parameters
    ----------
    sfreq : float | None, default=None
        Sampling frequency in Hz. NumPy input requires it; MNE input supplies it
        from metadata.
    window_length : int | None, default=None
        Embedding dimension in samples.
    drop_freq_max : float, default=3.0
        Dominant-frequency upper bound in Hz.
    drop_band : tuple of float | None, default=None
        Inclusive dominant-frequency rejection interval in Hz.
    n_check : int | None, default=None
        Number of leading numerical-rank components to inspect.
    max_window : int, default=100
        Maximum automatic embedding dimension.
    verbose : bool, str, int, or None, default=None
        Logging level.
    window_seconds : float | None, default=None
        Embedding duration in seconds, mutually exclusive with window_length.

    Attributes
    ----------
    sfreq_ : float
        Sampling frequency used for fitting.
    n_channels_in_ : int
        Number of fitted data channels.
    ch_names_in_ : tuple of str | None
        Fitted MNE channel names and order, or None for arrays.
    diagnostics_ : dict | list of dict
        Diagnostics from the most recent transform.
    dropped_counts_ : ndarray
        Number of rejected components per channel or epoch and channel.
    dropped_frequencies_ : list
        Dominant frequencies of rejected components.

    See Also
    --------
    LocalSingularSpectrumAnalysis
        Local delay-vector clustering and reconstruction.
    compute_basic_ssa
        One-shot Basic SSA interface.

    Notes
    -----
    The estimator is transductive: fit records the operating point and channel
    layout, while each transform decomposes its input records independently.
    :footcite:p:`golyandina_zhigljavsky2013_ssa`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import SingularSpectrumAnalysis
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> model = SingularSpectrumAnalysis(sfreq=250.0, drop_freq_max=3.0)
    >>> clean = model.fit_transform(data)
    """

    _requires_sfreq = True
    _progress_method = "basic_ssa"

    def __init__(
        self,
        sfreq: float | None = None,
        window_length: int | None = None,
        drop_freq_max: float = 3.0,
        drop_band: tuple[float, float] | None = None,
        n_check: int | None = None,
        max_window: int = 100,
        verbose: bool | str | int | None = None,
        *,
        window_seconds: float | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.window_length = window_length
        self.drop_freq_max = drop_freq_max
        self.drop_band = drop_band
        self.n_check = n_check
        self.max_window = max_window
        self.verbose = verbose
        self.window_seconds = window_seconds

    def _validate_fit_parameters(self, data: np.ndarray, sfreq: float) -> None:
        _check_frequency_parameters(
            sfreq, self.drop_freq_max, self.drop_band, self.n_check
        )
        _resolve_window_length(
            data.shape[-1],
            self.window_length,
            window_seconds=self.window_seconds,
            sfreq=sfreq,
            max_window=self.max_window,
        )

    def _compute_record(
        self,
        data: np.ndarray,
        sfreq: float,
        *,
        callback: _ProgressCallback | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # The estimator owns one aggregate SSA report; suppress the core's
        # standalone summary for each record while retaining its computation.
        return compute_basic_ssa(
            data,
            sfreq,
            self.window_length,
            self.drop_freq_max,
            self.drop_band,
            self.n_check,
            self.max_window,
            window_seconds=self.window_seconds,
            callback=callback,
            verbose="WARNING",
        )

    def _set_diagnostic_attributes(
        self, records: list[dict[str, Any]], *, epoched: bool
    ) -> None:
        if epoched:
            self.dropped_counts_ = np.stack(
                [record["dropped_counts"] for record in records]
            )
            self.dropped_frequencies_ = [
                record["dropped_frequencies"] for record in records
            ]
            mean_count = float(np.mean(self.dropped_counts_))
        else:
            self.dropped_counts_ = records[0]["dropped_counts"]
            self.dropped_frequencies_ = records[0]["dropped_frequencies"]
            mean_count = float(np.mean(self.dropped_counts_))
        window = records[0].get("window_length", self.window_length or "auto")
        logger.info(
            "Basic SSA: window=%s samples, channels=%d, dropped=%d component(s) "
            "(mean %.1f/channel).",
            window,
            self.n_channels_in_,
            int(np.sum(self.dropped_counts_)),
            mean_count,
        )
