"""Basic Singular Spectrum Analysis and dominant-frequency grouping.

Basic SSA embeds a scalar series in a Hankel trajectory matrix, decomposes it
with a singular-value decomposition, and maps every elementary matrix back to
the time domain by anti-diagonal averaging. The reconstructed components are
additive and sum to the input to floating-point precision.

The dominant-frequency rejection rule provided here is an application-specific
grouping strategy. It is not part of the mathematical definition of Basic SSA.

References
----------
.. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for
       Time Series. Springer. https://doi.org/10.1007/978-3-642-34913-3
"""

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
    """Decompose a one-dimensional series into additive Basic SSA components.

    Parameters
    ----------
    x : array-like, shape (n_times,)
        Finite scalar time series.
    window_length : int | None, default=None
        Embedding dimension in samples. It must satisfy
        ``2 <= window_length <= (n_times + 1) // 2``. If None, an automatic
        value is selected.
    window_seconds : float | None, default=None
        Embedding duration in seconds. It is mutually exclusive with
        ``window_length`` and requires ``sfreq``.
    sfreq : float | None, default=None
        Sampling frequency in Hz. It converts ``window_seconds`` to samples and
        sets the automatic window to at most 0.5 seconds.
    max_window : int, default=100
        Maximum embedding dimension used by automatic selection.

    Returns
    -------
    components : ndarray, shape (window_length, n_times)
        Elementary reconstructed series ordered by decreasing singular value.
        Their sum reconstructs ``x`` to floating-point precision.
    info : dict
        Resolved embedding dimension, trajectory-matrix shape, singular values,
        and numerical rank.

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If ``x`` is not a finite one-dimensional series or the requested
        embedding is invalid.

    See Also
    --------
    ssa_w_correlation : Measure weighted component separability.
    ssa_clean_channel : Group and subtract components by dominant frequency.

    Notes
    -----
    Direct SVD of the trajectory matrix is algebraically equivalent to
    eigendecomposition of its lag-covariance matrix, without squaring the
    condition number. Anti-diagonal averaging includes the smaller edge
    multiplicities described for Basic SSA [1]_.

    References
    ----------
    .. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
           Analysis for Time Series. Springer.
           https://doi.org/10.1007/978-3-642-34913-3
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
        Reconstructed time series, typically returned by
        :func:`ssa_decompose`.
    window_length : int
        Embedding dimension used to produce ``components``.

    Returns
    -------
    correlation : ndarray, shape (n_components, n_components)
        Symmetric weighted-correlation matrix. Zero-energy components have a
        zero row and column.

    Raises
    ------
    TypeError
        If ``window_length`` is not an integer.
    ValueError
        If the component array or embedding dimension is invalid.

    Notes
    -----
    The weights equal the anti-diagonal multiplicities of the trajectory
    matrix. Magnitudes near zero indicate stronger separability; magnitudes
    near one indicate that two reconstructed components are strongly mixed.
    W-correlation is a diagnostic, not an artifact-selection rule [1]_.

    References
    ----------
    .. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
           Analysis for Time Series. Springer.
           https://doi.org/10.1007/978-3-642-34913-3
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
    """Clean one channel by grouping Basic SSA components by frequency.

    Parameters
    ----------
    x : array-like, shape (n_times,)
        Finite scalar time series.
    sfreq : float
        Sampling frequency in Hz.
    window_length : int | None, default=None
        Embedding dimension in samples. If None, it is selected automatically.
    drop_freq_max : float, default=3.0
        Reject components whose dominant frequency is at or below this value
        in Hz. Ignored as a selection bound when ``drop_band`` is supplied.
    drop_band : tuple of float | None, default=None
        Inclusive ``(low, high)`` dominant-frequency rejection band in Hz.
    n_check : int | None, default=None
        Restrict selection to this many leading numerical-rank components.
        None examines every numerical-rank component.
    max_window : int, default=100
        Maximum embedding dimension used by automatic window selection.
    window_seconds : float | None, default=None
        Embedding duration in seconds, mutually exclusive with
        ``window_length``.
    return_info : bool, default=False
        If True, also return decomposition and grouping diagnostics.

    Returns
    -------
    x_clean : ndarray, shape (n_times,)
        Cleaned time series.
    info : dict
        Returned only when ``return_info=True``. Contains reconstructed
        components, singular values, dominant frequencies, rejected component
        indices and frequencies, the reconstructed artifact, and the resolved
        embedding information.

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If the time series, frequency bounds, or embedding is invalid.

    See Also
    --------
    compute_basic_ssa : Apply the same rule independently across channels.
    ssa_decompose : Return the complete additive decomposition.

    Notes
    -----
    Dominant frequency is the maximum-magnitude bin of an ``n_times``-point
    real FFT. DC is included and ties select the lower-frequency bin. This
    grouping rule and its thresholds are mne-denoise choices rather than
    defining steps of Basic SSA. A broadband component can therefore be
    classified by a narrow peak, and decisions near a threshold depend on the
    FFT resolution ``sfreq / n_times``.
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
    """Apply frequency-guided Basic SSA independently to every channel.

    Parameters
    ----------
    X : array-like, shape (n_channels, n_times)
        Finite channel-first data. Channels are never mixed.
    sfreq : float
        Sampling frequency in Hz.
    window_length : int | None, default=None
        Embedding dimension in samples. If None, it is selected automatically.
    drop_freq_max : float, default=3.0
        Reject components whose dominant frequency is at or below this value
        in Hz.
    drop_band : tuple of float | None, default=None
        Inclusive ``(low, high)`` dominant-frequency rejection band in Hz.
    n_check : int | None, default=None
        Restrict selection to this many leading numerical-rank components.
        None examines every numerical-rank component.
    max_window : int, default=100
        Maximum embedding dimension used by automatic window selection.
    window_seconds : float | None, default=None
        Embedding duration in seconds, mutually exclusive with
        ``window_length``.
    callback : callable | None, default=None
        Called synchronously after each completed channel with a structured
        ``basic_ssa`` channel progress event. Callback return values are
        ignored and callback exceptions propagate unchanged.
    verbose : bool | str | int | None
        MNE-style logging level. Channel helpers remain silent; this function
        reports one aggregate result at INFO.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Cleaned data with the same shape as ``X``.
    info : dict
        Per-channel component-selection diagnostics and the common resolved
        operating point.

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If ``X``, the frequency bounds, or the embedding is invalid.

    See Also
    --------
    ssa_clean_channel : Canonical single-channel implementation.
    SingularSpectrumAnalysis : MNE/scikit-learn estimator interface.

    Notes
    -----
    This function calls :func:`ssa_clean_channel` independently for every
    channel. It implements repeated univariate SSA, not multivariate SSA.
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
    """Frequency-guided per-channel Basic SSA transformer.

    Parameters
    ----------
    sfreq : float | None, default=None
        Sampling frequency in Hz. NumPy input requires an explicit value. MNE
        input supplies it from metadata and must agree with an explicit value.
    window_length : int | None, default=None
        Embedding dimension in samples. It is mutually exclusive with
        ``window_seconds``. None selects an automatic value.
    drop_freq_max : float, default=3.0
        Reject components whose dominant frequency is at or below this value
        in Hz.
    drop_band : tuple of float | None, default=None
        Inclusive ``(low, high)`` dominant-frequency rejection band in Hz. If
        supplied, it replaces ``drop_freq_max`` as the component-selection
        interval.
    n_check : int | None, default=None
        Restrict selection to this many leading numerical-rank components.
        None examines every numerical-rank component.
    max_window : int, default=100
        Maximum embedding dimension used by automatic window selection.
    verbose : bool | str | int | None, default=None
        MNE-style logging level.
    window_seconds : float | None, default=None
        Embedding duration in seconds, mutually exclusive with
        ``window_length``.

    Attributes
    ----------
    sfreq_ : float
        Validated sampling frequency used during fitting.
    n_channels_in_ : int
        Number of data channels seen during fitting.
    ch_names_in_ : tuple of str | None
        Fitted MNE channel names and order, or None for NumPy input.
    diagnostics_ : dict | list of dict
        Diagnostics from the most recent transformation. Epoched input stores
        one dictionary per epoch.
    dropped_counts_ : ndarray
        Number of rejected components per channel, or per epoch and channel.
    dropped_frequencies_ : list
        Dominant frequencies of rejected components for every channel.

    See Also
    --------
    compute_basic_ssa : Functional interface for channel-first arrays.
    ssa_clean_channel : Canonical single-channel implementation.
    ssa_decompose : Complete additive Basic SSA decomposition.
    mne_denoise.ssa.LocalSingularSpectrumAnalysis : Local clustered SSA.

    Notes
    -----
    The estimator is transductive. ``fit`` validates the operating point and
    records the channel layout; every ``transform`` decomposes the records
    supplied to that call. Changing record or epoch boundaries can therefore
    change the trajectory matrix, Fourier bins, and selected components. The
    additive decomposition follows Basic SSA [1]_; dominant-frequency rejection
    is an application-specific grouping rule.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import SingularSpectrumAnalysis
    >>> sfreq = 100.0
    >>> time = np.arange(500) / sfreq
    >>> data = np.vstack(
    ...     [np.sin(2 * np.pi * 1.0 * time), np.sin(2 * np.pi * 10.0 * time)]
    ... )
    >>> model = SingularSpectrumAnalysis(sfreq=sfreq, drop_freq_max=3.0)
    >>> cleaned = model.fit_transform(data)
    >>> cleaned.shape
    (2, 500)

    References
    ----------
    .. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
           Analysis for Time Series. Springer.
           https://doi.org/10.1007/978-3-642-34913-3
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
