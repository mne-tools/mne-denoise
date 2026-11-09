"""Reference-free BSS-CCA."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._blending import overlap_add_combine
from ._cca import canonical_correlation
from ._data import (
    continuous_to_epochs,
    epochs_to_continuous,
    extract_data_from_mne,
    reconstruct_mne_object,
)
from ._logging import logger, verbose
from ._spatial import (
    apply_spatial_transform,
    fit_mixing_matrix,
)
from ._validation import (
    check_channel_first_data,
    check_channel_layout,
    check_matching_sfreq,
    check_positive_real,
    resolve_sfreq,
)
from .progress import _emit_progress, _validate_callback

__all__ = ["BSSCCA", "compute_bss_cca"]

#: Band over which the spectral slope is fitted, in Hz: muscle power rises and
#: brain power falls with frequency here (Fitzgibbon et al., 2016; Janani et al.,
#: 2018).
_DEFAULT_SLOPE_BAND = (7.0, 75.0)

#: Welch segment length for the spectral slope, in seconds: 1 s Hann segments
#: with 50% overlap, as in the reference implementation of Janani et al. (2018).
_SLOPE_SEGMENT_SECONDS = 1.0

#: Warn when the number of lagged pairs falls below this multiple of the
#: channel count; canonical correlations saturate toward 1 as the ratio drops.
_PAIRS_PER_CHANNEL_WARNING = 10


def _resolve_lag_samples(
    *,
    lag_samples: int | None,
    lag_seconds: float | None,
    sfreq: float | None,
    n_times: int,
) -> int:
    """Resolve one explicit lag declaration to a positive sample count."""
    if lag_samples is not None and lag_seconds is not None:
        raise ValueError("set at most one of lag_samples or lag_seconds")
    if lag_seconds is None:
        if lag_samples is None:
            lag_samples = 1
        if isinstance(lag_samples, bool) or not isinstance(lag_samples, Integral):
            raise TypeError("lag_samples must be a positive integer")
        if lag_samples < 1:
            raise ValueError("lag_samples must be a positive integer")
        resolved = int(lag_samples)
    else:
        if isinstance(lag_seconds, bool) or not isinstance(lag_seconds, Real):
            raise TypeError("lag_seconds must be a finite number")
        lag_seconds = float(lag_seconds)
        if not np.isfinite(lag_seconds):
            raise TypeError("lag_seconds must be a finite number")
        if lag_seconds <= 0:
            raise ValueError("lag_seconds must be positive")
        if sfreq is None:
            raise ValueError("sfreq is required when lag_seconds is used")
        sfreq = check_positive_real(sfreq, name="sfreq")
        resolved = int(np.floor(lag_seconds * sfreq + 0.5))
        if resolved < 1:
            raise ValueError(
                f"lag_seconds={lag_seconds} resolves to less than one sample "
                f"at sfreq={sfreq}"
            )
    if resolved >= n_times:
        raise ValueError(
            f"lag ({resolved} samples) leaves no paired samples for n_times={n_times}"
        )
    return resolved


def _check_selection(
    n_remove: int | None,
    rho_threshold: float | None,
    slope_threshold: float | None = None,
    slope_band: tuple[float, float] = _DEFAULT_SLOPE_BAND,
) -> tuple[int | None, float | None, float | None, tuple[float, float] | None]:
    """Validate the selection rule; only rho_threshold and slope_threshold combine."""
    thresholds = rho_threshold is not None or slope_threshold is not None
    if (n_remove is None) != thresholds:
        raise ValueError(
            "set exactly one of n_remove, rho_threshold or slope_threshold "
            "(rho_threshold and slope_threshold may be combined); there is no "
            "universally valid default (De Clercq et al. describe "
            "autocorrelation thresholding as unvalidated future work, and "
            "select a component count instead)"
        )
    if n_remove is not None:
        if isinstance(n_remove, bool) or not isinstance(n_remove, Integral):
            raise TypeError("n_remove must be a non-negative integer")
        if n_remove < 0:
            raise ValueError("n_remove must be a non-negative integer")
        return int(n_remove), None, None, None
    if slope_threshold is None:
        slope_band = None
    else:
        if (
            isinstance(slope_threshold, bool)
            or not isinstance(slope_threshold, Real)
            or not np.isfinite(slope_threshold)
        ):
            raise TypeError("slope_threshold must be a finite number")
        slope_threshold = float(slope_threshold)
        try:
            fmin, fmax = (float(freq) for freq in slope_band)
        except (TypeError, ValueError):
            raise TypeError("slope_band must be a pair (fmin, fmax) in Hz") from None
        if not 0.0 < fmin < fmax:
            raise ValueError(
                f"slope_band must satisfy 0 < fmin < fmax, got {slope_band}"
            )
        slope_band = (fmin, fmax)
        if rho_threshold is None:
            return None, None, slope_threshold, slope_band
    if isinstance(rho_threshold, bool) or not isinstance(rho_threshold, Real):
        raise TypeError("rho_threshold must be a finite number")
    rho_threshold = float(rho_threshold)
    if not np.isfinite(rho_threshold):
        raise TypeError("rho_threshold must be a finite number")
    if not 0.0 <= rho_threshold <= 1.0:
        raise ValueError("rho_threshold must be between 0 and 1")
    return None, rho_threshold, slope_threshold, slope_band


def _lagged_pairs(X: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return current and lagged CCA views without epoch-boundary pairs."""
    if X.ndim == 2:
        current = X[:, lag_samples:].T
        past = X[:, :-lag_samples].T
    else:
        n_channels = X.shape[1]
        current = np.transpose(X[:, :, lag_samples:], (0, 2, 1)).reshape(-1, n_channels)
        past = np.transpose(X[:, :, :-lag_samples], (0, 2, 1)).reshape(-1, n_channels)
    return np.ascontiguousarray(current), np.ascontiguousarray(past)


def _select_components(
    correlations: np.ndarray,
    *,
    n_remove: int | None,
    rho_threshold: float | None,
    reject: str = "low",
    threshold_on: str = "rho",
) -> np.ndarray:
    """Return the component mask selected by the requested rule."""
    n_components = correlations.size
    if n_remove is not None:
        if n_remove > n_components:
            raise ValueError(
                f"n_remove={n_remove} exceeds the fitted CCA rank {n_components}"
            )
        keep = np.ones(n_components, dtype=bool)
        if n_remove:
            if reject == "low":
                keep[n_components - n_remove :] = False
            else:
                keep[:n_remove] = False
        return keep

    # ``rsq`` compares against the SQUARED correlation. Squaring the data
    # rather than square-rooting the threshold keeps the comparison exact at
    # the endpoints.
    metric = correlations if threshold_on == "rho" else correlations**2
    keep = metric >= rho_threshold if reject == "low" else metric <= rho_threshold
    if not keep.any():
        bound = "reaches" if reject == "low" else "falls below"
        logger.warning(
            "BSS-CCA: no component %s rho_threshold=%.4f (%s range %.4f-%.4f); "
            "every component will be removed. Adjust the threshold or use "
            "n_remove.",
            bound,
            rho_threshold,
            threshold_on,
            float(metric.min()) if n_components else float("nan"),
            float(metric.max()) if n_components else float("nan"),
        )
    return keep


def _component_slopes(
    sources: np.ndarray,
    data_shape: tuple[int, ...],
    *,
    sfreq: float,
    slope_band: tuple[float, float],
) -> np.ndarray:
    """Return the log-log spectral slope of each component inside slope_band.

    As in the reference implementation of Janani et al. (2018): Welch spectra
    with 1 s Hann segments and 50% overlap, then a least-squares line on log
    power against log frequency. Epoched sources average their spectra across
    epochs, so no segment spans an epoch boundary.
    """
    if len(data_shape) == 3:
        n_epochs, _n_channels, n_times = data_shape
        sources = continuous_to_epochs(sources, (n_epochs, sources.shape[0], n_times))
    nperseg = min(sources.shape[-1], int(round(sfreq * _SLOPE_SEGMENT_SECONDS)))
    freqs, psd = welch(sources, fs=sfreq, nperseg=nperseg, axis=-1)
    band = (freqs >= slope_band[0]) & (freqs <= slope_band[1])
    if band.sum() < 2:
        raise ValueError(
            f"slope_band={slope_band} contains {band.sum()} Welch frequency "
            f"bin(s) at sfreq={sfreq} for {sources.shape[-1]}-sample series; "
            f"widen the band or supply longer blocks or epochs"
        )
    if psd.ndim == 3:
        psd = psd.mean(axis=0)
    return np.polyfit(np.log(freqs[band]), np.log(psd[:, band]).T, 1)[0]


def _learn_operator(
    X: np.ndarray,
    *,
    lag_samples: int,
    n_remove: int | None,
    rho_threshold: float | None,
    reject: str,
    threshold_on: str,
    bound: tuple[int, int, int, int],
    slope_threshold: float | None = None,
    slope_band: tuple[float, float] | None = None,
    sfreq: float | None = None,
) -> dict[str, Any]:
    """Learn one channel-space BSS-CCA operator and diagnostics."""
    ext_start, ext_end, own_start, own_end = bound
    data = X[..., ext_start:ext_end]
    current, past = _lagged_pairs(data, lag_samples)
    n_pairs, n_channels = current.shape
    if n_pairs < 2:
        raise ValueError(
            f"BSS-CCA requires at least two lagged pairs, got {n_pairs}; "
            f"reduce the lag or supply more samples"
        )
    if n_pairs <= n_channels:
        raise ValueError(
            f"BSS-CCA requires more lagged pairs than channels, got "
            f"{n_pairs} pairs for {n_channels} channels; every canonical "
            f"correlation would saturate at 1 and nothing would be removed"
        )
    if n_pairs < _PAIRS_PER_CHANNEL_WARNING * n_channels:
        logger.warning(
            "BSS-CCA: only %d lagged pairs for %d channels; canonical "
            "correlations are biased upward when samples are scarce.",
            n_pairs,
            n_channels,
        )

    filters_x, filters_y, correlations, _u, _v = canonical_correlation(current, past)
    n_components = filters_x.shape[1]
    if n_components == 0:
        raise ValueError("BSS-CCA found zero-rank current or lagged data")

    unmixing = filters_x.T
    continuous = epochs_to_continuous(data)
    training_mean = continuous.mean(axis=1, keepdims=True)
    centered = continuous - training_mean
    sources = unmixing @ centered
    mixing = fit_mixing_matrix(centered, sources)

    keep = np.ones(correlations.size, dtype=bool)
    if n_remove is not None or rho_threshold is not None:
        keep = _select_components(
            correlations,
            n_remove=n_remove,
            rho_threshold=rho_threshold,
            reject=reject,
            threshold_on=threshold_on,
        )
    slopes = None
    if slope_threshold is not None:
        slopes = _component_slopes(
            sources, data.shape, sfreq=sfreq, slope_band=slope_band
        )
        if keep.any() and not (keep & (slopes <= slope_threshold)).any():
            logger.warning(
                "BSS-CCA: every remaining component has a spectral slope above "
                "slope_threshold=%.4g; every component will be removed. Adjust "
                "the threshold or the band.",
                slope_threshold,
            )
        keep &= slopes <= slope_threshold
    cleaning = mixing @ (keep[:, np.newaxis] * unmixing)

    return {
        "cleaning_matrix": cleaning,
        "filters": unmixing,
        "patterns": mixing,
        "correlations": correlations,
        "autocorrelations": _signed_autocorrelations(current, past, filters_x),
        "filter_asymmetry": _filter_asymmetry(filters_x, filters_y),
        "kept_mask": keep,
        "slopes": slopes,
        "training_mean": training_mean,
        "input_rank": n_components,
        "n_pairs": n_pairs,
        "span": (ext_start, ext_end),
        "own_span": (own_start, own_end),
    }


def _signed_autocorrelations(
    current: np.ndarray, past: np.ndarray, filters_x: np.ndarray
) -> np.ndarray:
    """Return signed lagged autocorrelations for canonical components."""
    zc = current @ filters_x
    zp = past @ filters_x
    zc = zc - zc.mean(axis=0, keepdims=True)
    zp = zp - zp.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(zc, axis=0) * np.linalg.norm(zp, axis=0)
    norm[norm == 0.0] = 1.0
    return np.einsum("ij,ij->j", zc, zp) / norm


def _filter_asymmetry(filters_x: np.ndarray, filters_y: np.ndarray) -> np.ndarray:
    """Return the signed-distance diagnostic between canonical filters."""

    def _unit(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    a = _unit(filters_x)
    b = _unit(filters_y)
    signs = np.sign(np.einsum("ij,ij->j", a, b))
    signs[signs == 0.0] = 1.0
    return np.linalg.norm(a - b * signs, axis=0)


def _segment_bounds(
    n_times: int, *, n_block: int, hop: int
) -> list[tuple[int, int, int, int]]:
    """Return extended and owned half-open spans for each block."""
    if n_times <= n_block:
        return [(0, n_times, 0, n_times)]

    starts = list(range(0, n_times - n_block + 1, hop))
    if starts[-1] + n_block < n_times:
        starts.append(n_times - n_block)

    # Neighbouring blocks hand over in the middle of their overlap, so each
    # block's extensions are symmetric and the raised-cosine blend fades one
    # block out while the next fades in, with no jump in the weights.
    handovers = [
        (start + previous + n_block) // 2
        for previous, start in zip(starts[:-1], starts[1:])
    ]
    own_starts = [0, *handovers]
    own_ends = [*handovers, n_times]
    return [
        (start, start + n_block, own_start, own_end)
        for start, own_start, own_end in zip(starts, own_starts, own_ends)
    ]


@verbose
def compute_bss_cca(
    X: np.ndarray,
    *,
    lag_samples: int | None = None,
    lag_seconds: float | None = None,
    sfreq: float | None = None,
    n_remove: int | None = None,
    rho_threshold: float | None = None,
    reject: str = "low",
    threshold_on: str = "rho",
    slope_threshold: float | None = None,
    slope_band: tuple[float, float] = _DEFAULT_SLOPE_BAND,
    segment_len: float | None = None,
    overlap: float = 0.0,
    preserve_mean: bool = True,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Learn and apply reference-free BSS-CCA to channel-first data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Continuous or epoched channel-first data.
    lag_samples : int or None, default=None
        Positive lag in samples. Defaults to one sample unless lag_seconds is set.
    lag_seconds : float or None, default=None
        Positive lag in seconds; requires sfreq and is mutually exclusive with
        lag_samples.
    sfreq : float or None, default=None
        Sampling frequency, required for lag_seconds and segment_len.
    n_remove : int or None, default=None
        Number of components removed from the end selected by reject.
    rho_threshold : float or None, default=None
        Correlation threshold used instead of n_remove. It can be combined with
        slope_threshold; n_remove cannot be combined with either threshold.
    reject : {"low", "high"}, default="low"
        End of the correlation spectrum treated as artifactual. slope_threshold
        requires "low".
    threshold_on : {"rho", "rsq"}, default="rho"
        Scale for rho_threshold.
    slope_threshold : float or None, default=None
        Remove components whose log-log power spectrum within slope_band has a
        slope above this value. Requires sfreq. Together with rho_threshold, a
        component is removed when either rule flags it. There is no default; the
        user guide lists the published values and their conditions.
    slope_band : tuple of float, default=(7.0, 75.0)
        Frequency band ``(fmin, fmax)`` in Hz over which the slope is fitted; used
        only with slope_threshold. The default is the band of the published
        method (see Notes).
    segment_len : float or None, default=None
        Continuous-data block length in seconds. None uses one operator.
    overlap : float, default=0.0
        Fraction shared by neighboring continuous-data blocks.
    preserve_mean : bool, default=True
        Add the fitted channel mean after cleaning.
    callback : callable or None, default=None
        Synchronous block-progress callback in segmented mode.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    X_clean : ndarray
        Cleaned data with the same shape as X.
    info : dict
        Operators, resolved settings, and component diagnostics.

    Notes
    -----
    CCA is computed between the signal and a lagged copy. With reject="low",
    components with the lowest lagged correlation are removed; reject="high" removes
    the highest. Segmented mode is continuous-only and cannot span epoch boundaries
    :footcite:p:`declercq2006_bss_cca,vergult2007_bss_cca,hotelling1936_cca`.
    Vergult et al. cleaned contiguous 10 s epochs independently; the overlap and
    the raised-cosine cross-fade between neighbouring blocks, which hand over in
    the middle of their overlap, are additions of this package.

    slope_threshold removes components whose spectrum between 7 and 75 Hz is
    flatter than brain activity, the muscle signature shown on recordings under
    neuromuscular blockade
    :footcite:p:`fitzgibbon2016_emg_paralysis,janani2018_bss_cca_slope`.
    The slope is estimated as in the reference implementation of Janani et al.
    (1 s Welch segments, least squares on log power against log frequency); it is
    the exponent ``a`` of ``P(f) ∝ f**a``. Janani et al. combine a correlation
    stage and a slope stage, which rho_threshold and slope_threshold reproduce
    together. Their published values depend on the decomposition, montage,
    sampling rate and lag, so neither threshold has a default; the user guide
    lists them with the conditions they were measured in. Remove line noise
    first: it flattens spectra inside slope_band.

    See Also
    --------
    BSSCCA
        Estimator that learns operators in fit and reuses them in transform.

    References
    ----------
    .. footbibliography::
    """
    callback = _validate_callback(callback)
    X = check_channel_first_data(X, name="BSS-CCA")
    if not isinstance(preserve_mean, bool):
        raise TypeError("preserve_mean must be a bool")
    n_remove, rho_threshold, slope_threshold, slope_band = _check_selection(
        n_remove, rho_threshold, slope_threshold, slope_band
    )
    if reject not in ("low", "high"):
        raise ValueError(f"reject must be 'low' or 'high', got {reject!r}")
    if threshold_on not in ("rho", "rsq"):
        raise ValueError(f"threshold_on must be 'rho' or 'rsq', got {threshold_on!r}")
    if slope_threshold is not None:
        if reject != "low":
            raise ValueError(
                "reject='high' cannot be used with slope_threshold: the slope "
                "rule removes muscle-like components, which sit at the low end "
                "of the correlation spectrum"
            )
        if sfreq is None:
            raise ValueError("sfreq is required when slope_threshold is used")
        sfreq = check_positive_real(sfreq, name="sfreq")
        if slope_band[1] > sfreq / 2.0:
            raise ValueError(
                f"slope_band upper edge {slope_band[1]} Hz exceeds the Nyquist "
                f"frequency {sfreq / 2.0} Hz"
            )
    lag = _resolve_lag_samples(
        lag_samples=lag_samples,
        lag_seconds=lag_seconds,
        sfreq=sfreq,
        n_times=X.shape[-1],
    )
    n_block, hop = _resolve_blocking(
        segment_len=segment_len,
        overlap=overlap,
        sfreq=sfreq,
        n_times=X.shape[-1],
    )
    if n_block is not None and X.ndim == 3:
        raise ValueError(
            "segment_len is only supported for 2-D continuous data; epoched "
            "input is already segmented, so call once per epoch set with "
            "segment_len=None"
        )

    n_times = X.shape[-1]
    bounds = (
        [(0, n_times, 0, n_times)]
        if n_block is None
        else _segment_bounds(n_times, n_block=n_block, hop=hop)
    )
    operators = []
    for block_idx, bound in enumerate(bounds):
        operator = _learn_operator(
            X,
            lag_samples=lag,
            n_remove=n_remove,
            rho_threshold=rho_threshold,
            reject=reject,
            threshold_on=threshold_on,
            bound=bound,
            slope_threshold=slope_threshold,
            slope_band=slope_band,
            sfreq=sfreq,
        )
        operators.append(operator)
        if segment_len is not None:
            _emit_progress(
                callback,
                method="bss_cca",
                stage="block",
                current=block_idx + 1,
                total=len(bounds),
                component=None,
                metric=float(np.mean(operator["correlations"])),
            )
    cleaned = _apply_operators(X, operators, preserve_mean=preserve_mean)

    # Per-block quantities are reported as tuples only when blocking is in
    # effect, so the common single-operator case stays flat.
    blocked = len(operators) > 1 or segment_len is not None
    kept = [operator["kept_mask"] for operator in operators]

    def per_block(values: list[Any]) -> Any:
        return tuple(values) if blocked else values[0]

    info = {
        "cleaning_matrix": per_block([op["cleaning_matrix"] for op in operators]),
        "filters": per_block([op["filters"] for op in operators]),
        "patterns": per_block([op["patterns"] for op in operators]),
        "correlations": per_block([op["correlations"] for op in operators]),
        "autocorrelations": per_block([op["autocorrelations"] for op in operators]),
        "filter_asymmetry": per_block([op["filter_asymmetry"] for op in operators]),
        "kept_mask": per_block(kept),
        "slopes": None
        if slope_threshold is None
        else per_block([op["slopes"] for op in operators]),
        "training_mean": per_block([op["training_mean"] for op in operators]),
        "input_rank": per_block([int(mask.size) for mask in kept]),
        "n_kept": per_block([int(mask.sum()) for mask in kept]),
        "n_removed": per_block([int(mask.size - mask.sum()) for mask in kept]),
        "spans": tuple(op["span"] for op in operators),
        "n_blocks": len(operators),
        "lag_samples": lag,
        "lag_seconds": None if sfreq is None else lag / float(sfreq),
        "sfreq": None if sfreq is None else float(sfreq),
        "segment_len": segment_len,
        "overlap": float(overlap),
        "preserve_mean": preserve_mean,
        "n_channels": int(X.shape[-2]),
        "reject": reject,
        "threshold_on": threshold_on,
        "n_remove": n_remove,
        "rho_threshold": rho_threshold,
        "slope_threshold": slope_threshold,
        "slope_band": slope_band,
        # Minimal records needed to re-apply the fitted operators; consumed by
        # BSSCCA.fit so the estimator never re-derives them.
        "operators": tuple(
            {
                key: op[key]
                for key in ("cleaning_matrix", "training_mean", "span", "own_span")
            }
            for op in operators
        ),
    }
    if n_remove is not None:
        selection = f"reject={reject}, n_remove={n_remove}"
    elif rho_threshold is None:
        selection = ""
    else:
        selection = (
            f"reject={reject}, threshold_on={threshold_on}, "
            f"rho_threshold={rho_threshold:.4g}"
        )
    if slope_threshold is not None:
        slope = (
            f"slope_threshold={slope_threshold:.4g} over "
            f"{slope_band[0]:g}-{slope_band[1]:g} Hz"
        )
        selection = f"{selection} or {slope}" if selection else slope
    logger.info(
        "BSS-CCA: lag=%d sample(s), %d block(s), %s, removed %s of %s components.",
        lag,
        len(operators),
        selection,
        info["n_removed"],
        info["input_rank"],
    )
    return cleaned, info


def _resolve_blocking(
    *,
    segment_len: float | None,
    overlap: float,
    sfreq: float | None,
    n_times: int,
) -> tuple[int | None, int | None]:
    """Resolve ``segment_len``/``overlap`` to a block length and hop."""
    if isinstance(overlap, bool) or not isinstance(overlap, Real):
        raise TypeError("overlap must be a finite number")
    overlap = float(overlap)
    if not np.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be finite and in [0, 1)")
    if segment_len is None:
        return None, None
    segment_len = check_positive_real(segment_len, name="segment_len")
    if sfreq is None:
        raise ValueError("sfreq is required when segment_len is used")
    sfreq = check_positive_real(sfreq, name="sfreq")
    n_block = int(np.floor(segment_len * sfreq + 0.5))
    if n_block < 2:
        raise ValueError(
            f"segment_len={segment_len} resolves to {n_block} samples at "
            f"sfreq={sfreq}; use a longer block"
        )
    if n_block >= n_times:
        logger.debug(
            "BSS-CCA: segment_len covers the whole recording; learning one operator."
        )
    hop = max(1, n_block - int(np.floor(overlap * n_block + 0.5)))
    return n_block, hop


def _apply_operators(
    X: np.ndarray,
    operators: list[dict[str, Any]],
    *,
    preserve_mean: bool,
) -> np.ndarray:
    """Apply one operator or blend block-wise operators."""
    continuous = epochs_to_continuous(X)

    if len(operators) == 1:
        operator = operators[0]
        cleaned = apply_spatial_transform(
            operator["cleaning_matrix"], continuous - operator["training_mean"]
        )
        if preserve_mean:
            cleaned = cleaned + operator["training_mean"]
        return continuous_to_epochs(cleaned, X.shape)

    n_channels, n_times = continuous.shape
    chunks = []
    for operator in operators:
        ext_start, ext_end = operator["span"]
        own_start, own_end = operator["own_span"]
        block = apply_spatial_transform(
            operator["cleaning_matrix"],
            continuous[:, ext_start:ext_end] - operator["training_mean"],
        )
        if preserve_mean:
            block = block + operator["training_mean"]
        chunks.append(
            {
                "data": block,
                "ext_start": ext_start,
                "ext_end": ext_end,
                "start": own_start,
                "end": own_end,
            }
        )
    return overlap_add_combine((n_channels, n_times), chunks)


class BSSCCA(BaseEstimator, TransformerMixin):
    """Reference-free BSS-CCA estimator.

    The estimator learns fixed channel-space operators from the fitted data and
    reuses them during transform.

    Parameters
    ----------
    lag_samples : int or None, default=None
        Positive lag in samples.
    lag_seconds : float or None, default=None
        Positive lag in seconds; NumPy input then requires sfreq.
    sfreq : float or None, default=None
        Sampling frequency for NumPy data.
    n_remove : int or None, default=None
        Number of components to remove.
    rho_threshold : float or None, default=None
        Correlation threshold used instead of n_remove; it can be combined with
        slope_threshold.
    reject : {"low", "high"}, default="low"
        End of the correlation spectrum treated as artifactual. slope_threshold
        requires "low".
    threshold_on : {"rho", "rsq"}, default="rho"
        Scale for rho_threshold.
    slope_threshold : float or None, default=None
        Remove components whose log-log spectral slope within slope_band is above
        this value. Requires a sampling frequency. No default; see
        :func:`compute_bss_cca` for the published values.
    slope_band : tuple of float, default=(7.0, 75.0)
        Frequency band ``(fmin, fmax)`` in Hz over which the slope is fitted; used
        only with slope_threshold.
    segment_len : float or None, default=None
        Continuous-data block length in seconds.
    overlap : float, default=0.0
        Fraction shared by neighboring blocks.
    preserve_mean : bool, default=True
        Add the fitted channel mean after cleaning.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    cleaning_matrix_ : ndarray or tuple of ndarray
        Fitted channel-space operator(s).
    filters_ : ndarray
        Canonical filters ordered by decreasing correlation.
    patterns_ : ndarray
        Least-squares channel patterns.
    correlations_ : ndarray
        Canonical correlations.
    autocorrelations_ : ndarray
        Signed lagged autocorrelations.
    filter_asymmetry_ : ndarray
        Canonical-filter asymmetry diagnostic.
    kept_mask_ : ndarray of bool
        Retained component mask.
    slopes_ : ndarray, tuple of ndarray, or None
        Spectral slope of each component when slope_threshold is used, else None.
    training_mean_ : ndarray
        Fitted channel mean.
    spans_ : tuple
        Block spans when segmented.
    lag_samples_ : int
        Resolved lag in samples.

    See Also
    --------
    mne_denoise.icanclean.ICanClean
        Reference-based CCA cleaning using physical or derived reference signals.
    compute_bss_cca
        One-shot functional interface for array data.

    Notes
    -----
    With segment_len set, transform requires the same number of samples used during
    fit. MNE Raw, Epochs, and Evoked inputs preserve their container type and
    channel metadata :footcite:p:`declercq2006_bss_cca,vergult2007_bss_cca`.
    Channels listed in ``info["bads"]`` at fit time are left out of the
    decomposition and pass through transform unchanged.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.bss_cca import BSSCCA
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> model = BSSCCA(sfreq=250.0, lag_samples=1, n_remove=1)
    >>> clean = model.fit_transform(data)
    """

    def __init__(
        self,
        *,
        lag_samples: int | None = None,
        lag_seconds: float | None = None,
        sfreq: float | None = None,
        n_remove: int | None = None,
        rho_threshold: float | None = None,
        reject: str = "low",
        threshold_on: str = "rho",
        slope_threshold: float | None = None,
        slope_band: tuple[float, float] = _DEFAULT_SLOPE_BAND,
        segment_len: float | None = None,
        overlap: float = 0.0,
        preserve_mean: bool = True,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.lag_samples = lag_samples
        self.lag_seconds = lag_seconds
        self.sfreq = sfreq
        self.n_remove = n_remove
        self.rho_threshold = rho_threshold
        self.reject = reject
        self.threshold_on = threshold_on
        self.slope_threshold = slope_threshold
        self.slope_band = slope_band
        self.segment_len = segment_len
        self.overlap = overlap
        self.preserve_mean = preserve_mean
        self.verbose = verbose

    @verbose
    def fit(
        self,
        X: Any,
        y=None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> BSSCCA:
        """Fit the BSS-CCA operators.

        Parameters
        ----------
        X : array-like or MNE Raw, Epochs, or Evoked
            Data used to learn the operators.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous progress callback in segmented mode.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        BSSCCA
            The fitted estimator.
        """
        del y
        callback = _validate_callback(callback)
        data, data_sfreq, _mne_type, _orig, _picks, names = extract_data_from_mne(
            X, auto_pick=True, exclude_bads=True
        )
        sfreq = resolve_sfreq(self.sfreq, data_sfreq, required=False)
        _cleaned, info = compute_bss_cca(
            data,
            lag_samples=self.lag_samples,
            lag_seconds=self.lag_seconds,
            sfreq=sfreq,
            n_remove=self.n_remove,
            rho_threshold=self.rho_threshold,
            reject=self.reject,
            threshold_on=self.threshold_on,
            slope_threshold=self.slope_threshold,
            slope_band=self.slope_band,
            segment_len=self.segment_len,
            overlap=self.overlap,
            preserve_mean=self.preserve_mean,
            callback=callback,
        )
        self.cleaning_matrix_ = info["cleaning_matrix"]
        self.filters_ = info["filters"]
        self.patterns_ = info["patterns"]
        self.correlations_ = info["correlations"]
        self.autocorrelations_ = info["autocorrelations"]
        self.filter_asymmetry_ = info["filter_asymmetry"]
        self.kept_mask_ = info["kept_mask"]
        self.slopes_ = info["slopes"]
        self.training_mean_ = info["training_mean"]
        self.input_rank_ = info["input_rank"]
        self.n_kept_ = info["n_kept"]
        self.n_removed_ = info["n_removed"]
        self.spans_ = info["spans"]
        self.n_blocks_ = info["n_blocks"]
        self.lag_samples_ = info["lag_samples"]
        self.sfreq_ = info["sfreq"]
        self.n_channels_in_ = info["n_channels"]
        self.n_times_in_ = int(np.asarray(data).shape[-1])
        self.feature_names_in_ = None if names is None else tuple(names)
        self._operators = list(info["operators"])
        return self

    @verbose
    def transform(
        self,
        X: Any,
        y=None,
        *,
        verbose: bool | str | int | None = None,
    ) -> Any:
        """Apply the fitted BSS-CCA operators.

        Parameters
        ----------
        X : array-like or MNE Raw, Epochs, or Evoked
            Data with the fitted channel layout.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            A cleaned copy.
        """
        del y
        check_is_fitted(self, ("cleaning_matrix_", "training_mean_"))
        data, data_sfreq, mne_type, orig_inst, picks, names = extract_data_from_mne(
            X, ch_names=list(self.feature_names_in_) if self.feature_names_in_ else None
        )
        transform_sfreq = resolve_sfreq(self.sfreq, data_sfreq, required=False)
        check_matching_sfreq(transform_sfreq, self.sfreq_, name="BSS-CCA")
        data = check_channel_first_data(data, name="BSS-CCA")
        check_channel_layout(
            "BSS-CCA",
            n_channels=data.shape[-2],
            fitted_n_channels=self.n_channels_in_,
            ch_names=None if names is None else tuple(names),
            fitted_ch_names=self.feature_names_in_,
        )
        if self.n_blocks_ > 1 and data.shape[-1] != self.n_times_in_:
            raise ValueError(
                f"a block-wise operator is tied to the timeline it was learned "
                f"on: expected {self.n_times_in_} samples, got {data.shape[-1]}"
            )
        cleaned = _apply_operators(
            data, self._operators, preserve_mean=self.preserve_mean
        )
        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

    @verbose
    def fit_transform(
        self,
        X: Any,
        y=None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
        **fit_params,
    ) -> Any:
        """Fit BSS-CCA and apply the fitted operators to X.

        Parameters
        ----------
        X : array-like or MNE Raw, Epochs, or Evoked
            Data to fit and transform.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous progress callback in segmented mode.
        verbose : bool, str, int, or None, default=None
            Logging level.
        **fit_params : dict
            Reserved for scikit-learn compatibility.

        Returns
        -------
        same type as X
            A cleaned copy.
        """
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unexpected}")
        callback = _validate_callback(callback)
        return self.fit(X, y, callback=callback).transform(X)
