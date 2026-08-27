"""Core reference-free BSS-CCA algorithm and estimator.

This module contains:

1. ``compute_bss_cca``: the canonical array implementation of BSS-CCA [1]_.
2. ``BSSCCA``: the scikit-learn estimator, compatible with MNE-Python objects
   or channel-first NumPy arrays.

BSS-CCA solves canonical correlation analysis between the multichannel signal
:math:`x(t)` and a delayed copy :math:`y(t) = x(t - 1)` of itself [1]_. The
resulting components are ordered by decreasing lagged correlation. Muscle
activity resembles temporally white noise and therefore concentrates in the
**lowest** components, which are dropped before the signal is projected back to
the sensors.

The method assumes band-limited input. Both source papers band-pass filter
before decomposition (0.3-35 Hz plus a notch in [2]_) and use an average
reference montage; see :ref:`the user guide <bss_cca>` for why that matters.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
       Van Huffel, S. (2006). Canonical correlation analysis applied to remove
       muscle artifacts from the electroencephalogram. IEEE Transactions on
       Biomedical Engineering, 53(12), 2583-2587.
       https://doi.org/10.1109/TBME.2006.879459
.. [2] Vergult, A., De Clercq, W., Palmini, A., Vanrumste, B., Dupont, P.,
       Van Huffel, S., & Van Paesschen, W. (2007). Improving the interpretation
       of ictal scalp EEG: BSS-CCA algorithm for muscle artifact removal.
       Epilepsia, 48(5), 950-958.
       https://doi.org/10.1111/j.1528-1167.2007.01031.x
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
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

__all__ = ["BSSCCA", "compute_bss_cca"]

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
    n_remove: int | None, rho_threshold: float | None
) -> tuple[int | None, float | None]:
    """Validate that exactly one selection rule is requested."""
    if (n_remove is None) == (rho_threshold is None):
        raise ValueError(
            "set exactly one of n_remove or rho_threshold; there is no "
            "universally valid default (De Clercq et al. describe "
            "autocorrelation thresholding as unvalidated future work, and "
            "select a component count instead)"
        )
    if n_remove is not None:
        if isinstance(n_remove, bool) or not isinstance(n_remove, Integral):
            raise TypeError("n_remove must be a non-negative integer")
        if n_remove < 0:
            raise ValueError("n_remove must be a non-negative integer")
        return int(n_remove), None
    if isinstance(rho_threshold, bool) or not isinstance(rho_threshold, Real):
        raise TypeError("rho_threshold must be a finite number")
    rho_threshold = float(rho_threshold)
    if not np.isfinite(rho_threshold):
        raise TypeError("rho_threshold must be a finite number")
    if not 0.0 <= rho_threshold <= 1.0:
        raise ValueError("rho_threshold must be between 0 and 1")
    return None, rho_threshold


def _lagged_pairs(X: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return current/past CCA views without wrap or epoch-boundary pairs.

    For 3-D input the pairs are formed inside each epoch and then stacked, so
    no pair ever spans an epoch boundary.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) | (n_epochs, n_channels, n_times)
        Channel-first data.
    lag_samples : int
        Positive lag.

    Returns
    -------
    current : ndarray, shape (n_pairs, n_channels)
        Samples ``t``.
    past : ndarray, shape (n_pairs, n_channels)
        Samples ``t - lag_samples``, aligned row-wise with ``current``.
    """
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
    """Choose which canonical components to retain.

    ``correlations`` is descending, so low-autocorrelation components are at the
    tail and high-autocorrelation components at the head.

    ``reject`` selects which end is artifactual:

    ``'low'``
        Drop the tail. Broadband, temporally incoherent sources -- muscle/EMG --
        have low lag-1 autocorrelation, which is the regime De Clercq et al.
        [1]_ target and the package's original behaviour.
    ``'high'``
        Drop the head. Strongly autocorrelated sources -- slow drift and
        movement artifact -- sit at the top of the spectrum, the opposite end
        from muscle.

    ``n_remove`` drops that many components from the chosen end.
    """
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


def _learn_operator(
    X: np.ndarray,
    *,
    lag_samples: int,
    n_remove: int | None,
    rho_threshold: float | None,
    reject: str,
    threshold_on: str,
    bound: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Learn one BSS-CCA channel-space operator and its diagnostics.

    ``bound`` is ``(ext_start, ext_end, own_start, own_end)``: the operator is
    fitted on ``[ext_start, ext_end)`` and is solely responsible for
    ``[own_start, own_end)``. The two coincide unless blocks overlap.

    The back-projection is obtained by least squares against the data rather
    than by inverting the canonical filters. The two agree exactly when the
    data is full rank, but only the least-squares form is correct when it is
    not; see the warning in :mod:`mne_denoise._cca`.
    """
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

    keep = _select_components(
        correlations,
        n_remove=n_remove,
        rho_threshold=rho_threshold,
        reject=reject,
        threshold_on=threshold_on,
    )
    cleaning = mixing @ (keep[:, np.newaxis] * unmixing)

    return {
        "cleaning_matrix": cleaning,
        "filters": unmixing,
        "patterns": mixing,
        "correlations": correlations,
        "autocorrelations": _signed_autocorrelations(current, past, filters_x),
        "filter_asymmetry": _filter_asymmetry(filters_x, filters_y),
        "kept_mask": keep,
        "training_mean": training_mean,
        "input_rank": n_components,
        "n_pairs": n_pairs,
        "span": (ext_start, ext_end),
        "own_span": (own_start, own_end),
    }


def _signed_autocorrelations(
    current: np.ndarray, past: np.ndarray, filters_x: np.ndarray
) -> np.ndarray:
    """Signed lag-1 autocorrelation of each canonical component.

    The canonical correlation is non-negative by construction, so a component
    dominated by near-Nyquist energy is *anti*-correlated at the lag yet ranks
    highly. Applying the same filter to both views recovers the signed value.
    """
    zc = current @ filters_x
    zp = past @ filters_x
    zc = zc - zc.mean(axis=0, keepdims=True)
    zp = zp - zp.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(zc, axis=0) * np.linalg.norm(zp, axis=0)
    norm[norm == 0.0] = 1.0
    return np.einsum("ij,ij->j", zc, zp) / norm


def _filter_asymmetry(filters_x: np.ndarray, filters_y: np.ndarray) -> np.ndarray:
    """Distance between the two canonical filters of each component.

    Reading the canonical correlation as an autocorrelation presumes the two
    views share a filter. This returns ``0`` when they do and grows to ``2``
    when they are opposed, giving a per-component validity check.
    """

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
    """Return ``(ext_start, ext_end, own_start, own_end)`` per block.

    ``ext`` is the range a block is fitted on; ``own`` is the range it is
    solely responsible for. With ``hop == n_block`` the two coincide and the
    blocks tile the recording exactly, matching the contiguous 10 s scheme
    of [2]_.
    """
    if n_times <= n_block:
        return [(0, n_times, 0, n_times)]

    starts = list(range(0, n_times - n_block + 1, hop))
    if starts[-1] + n_block < n_times:
        starts.append(n_times - n_block)

    bounds = []
    own_start = 0
    for index, start in enumerate(starts):
        own_end = n_times if index == len(starts) - 1 else min(start + hop, n_times)
        own_end = max(own_end, own_start)
        bounds.append((start, start + n_block, own_start, own_end))
        own_start = own_end
    return bounds


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
    segment_len: float | None = None,
    overlap: float = 0.0,
    preserve_mean: bool = True,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Learn and apply reference-free BSS-CCA to a channel-first array.

    This is the canonical implementation of the algorithm of [1]_. It applies
    the learned operator to the same data used to estimate it; use
    :class:`BSSCCA` to fit and transform separate data.

    Exactly one of ``n_remove`` or ``rho_threshold`` must be supplied. ``[1]_``
    selects a component count and describes threshold-based selection as
    unvalidated future work, so no default is assumed on your behalf.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) | (n_epochs, n_channels, n_times)
        Continuous or epoched channel-first data.
    lag_samples : int | None, default=None
        Positive lag in samples. ``None`` uses the paper's value of ``1``
        unless ``lag_seconds`` is given.
    lag_seconds : float | None, default=None
        Positive lag in physical time. Requires ``sfreq``. Mutually exclusive
        with ``lag_samples``.
    sfreq : float | None, default=None
        Sampling frequency, required by ``lag_seconds`` and ``segment_len``.
    n_remove : int | None, default=None
        Number of components to remove from the end selected by ``reject``,
        the operating knob used in [1]_.
    rho_threshold : float | None, default=None
        Retain components whose canonical correlation is on the side of this
        value selected by ``reject`` -- at least it for ``'low'``, at most it
        for ``'high'``.
    reject : {'low', 'high'}, default='low'
        Which end of the autocorrelation spectrum is artifactual. ``'low'``
        drops the least autocorrelated components, where muscle concentrates
        [1]_. ``'high'`` drops the most autocorrelated components, where slow
        drift and movement artifact concentrate.
    threshold_on : {'rho', 'rsq'}, default='rho'
        Scale on which ``rho_threshold`` is expressed: the canonical correlation
        itself, or its square. Ignored when ``n_remove`` is used.
    segment_len : float | None, default=None
        Block length in seconds. ``None`` learns one operator for all data.
        A value fits an independent operator per block, as in the contiguous
        10 s scheme of [2]_. Blocks never span an epoch boundary.
    overlap : float, default=0.0
        Fraction of ``segment_len`` shared between consecutive blocks. ``0``
        reproduces the paper's contiguous blocks; a positive value blends
        neighbouring blocks and is a package extension.
    preserve_mean : bool, default=True
        Add the fitted channel mean back after cleaning. Equation (7) of [1]_
        reconstructs mean-free data; restoring the mean keeps the output on
        the same offset as the input.
    verbose : bool | str | int | None, default=None
        MNE-style logging level.

    Returns
    -------
    X_clean : ndarray
        Cleaned data with the same shape as ``X``.
    info : dict
        Fitted operators, component diagnostics, and the resolved operating
        point. Per-block entries are tuples ordered by block.

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If ``X``, the lag, the selection rule, or the blocking is invalid, or
        if there are not more lagged pairs than channels.

    See Also
    --------
    BSSCCA : Estimator interface with leakage-safe fit/transform.
    mne_denoise.icanclean.compute_icanclean : Reference-based CCA cleaning.

    Notes
    -----
    Canonical correlations are non-negative, so a component dominated by
    near-Nyquist energy is anti-correlated at the lag yet ranks near the top.
    Band-limit the input as both source papers do, and check
    ``info['autocorrelations']`` for negative entries.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.bss_cca import compute_bss_cca
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(2500) / 250.0
    >>> brain = np.sin(2 * np.pi * 10 * t) * rng.standard_normal((8, 1))
    >>> observed = brain + 0.5 * rng.standard_normal((8, t.size))
    >>> cleaned, info = compute_bss_cca(observed, n_remove=4)
    >>> cleaned.shape
    (8, 2500)

    References
    ----------
    .. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
           Van Huffel, S. (2006). Canonical correlation analysis applied to
           remove muscle artifacts from the electroencephalogram. IEEE
           Transactions on Biomedical Engineering, 53(12), 2583-2587.
           https://doi.org/10.1109/TBME.2006.879459
    .. [2] Vergult, A., De Clercq, W., Palmini, A., Vanrumste, B., Dupont, P.,
           Van Huffel, S., & Van Paesschen, W. (2007). Improving the
           interpretation of ictal scalp EEG: BSS-CCA algorithm for muscle
           artifact removal. Epilepsia, 48(5), 950-958.
           https://doi.org/10.1111/j.1528-1167.2007.01031.x
    """
    X = check_channel_first_data(X, name="BSS-CCA")
    if not isinstance(preserve_mean, bool):
        raise TypeError("preserve_mean must be a bool")
    n_remove, rho_threshold = _check_selection(n_remove, rho_threshold)
    if reject not in ("low", "high"):
        raise ValueError(f"reject must be 'low' or 'high', got {reject!r}")
    if threshold_on not in ("rho", "rsq"):
        raise ValueError(f"threshold_on must be 'rho' or 'rsq', got {threshold_on!r}")
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
    operators = [
        _learn_operator(
            X,
            lag_samples=lag,
            n_remove=n_remove,
            rho_threshold=rho_threshold,
            reject=reject,
            threshold_on=threshold_on,
            bound=bound,
        )
        for bound in bounds
    ]
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
    else:
        selection = (
            f"reject={reject}, threshold_on={threshold_on}, "
            f"rho_threshold={rho_threshold:.4g}"
        )
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
    """Apply one global operator, or blend per-block operators."""
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
    """Reference-free BSS-CCA artifact-attenuation estimator.

    Implements the blind source separation by canonical correlation analysis of
    De Clercq et al. [1]_, solving CCA between the recording and a lagged copy
    of itself and dropping the lowest-correlation components in which muscle
    activity concentrates.

    ``fit`` learns the channel mean and one or more fixed channel-space
    operators; ``transform`` applies them without refitting, so a sample gets
    the same result whether it is transformed alone, in a temporal chunk, or
    among other epochs.

    With ``segment_len`` set, the fitted operator is *piecewise in time*: block
    ``k`` is applied to the samples block ``k`` was learned on. ``transform``
    therefore requires input with the same number of samples as ``fit`` saw.

    Parameters
    ----------
    lag_samples : int | None, default=None
        Positive lag in samples. ``None`` uses the paper's value of ``1``
        unless ``lag_seconds`` is given.
    lag_seconds : float | None, default=None
        Positive lag in physical time. MNE inputs supply their own sampling
        frequency; NumPy inputs require ``sfreq``.
    sfreq : float | None, default=None
        Sampling frequency for NumPy data. A value supplied alongside an MNE
        input must agree with ``info['sfreq']``.
    n_remove : int | None, default=None
        Number of components to remove from the end selected by ``reject``.
    rho_threshold : float | None, default=None
        Retain components whose canonical correlation is on the side of this
        value selected by ``reject`` -- at least it for ``'low'``, at most it
        for ``'high'``. Exactly one of ``n_remove`` or ``rho_threshold`` is
        required.
    reject : {'low', 'high'}, default='low'
        Which end of the autocorrelation spectrum is artifactual. ``'low'``
        drops the least autocorrelated components, where muscle concentrates
        [1]_. ``'high'`` drops the most autocorrelated components, where slow
        drift and movement artifact concentrate.
    threshold_on : {'rho', 'rsq'}, default='rho'
        Scale on which ``rho_threshold`` is expressed: the canonical correlation
        itself, or its square. Ignored when ``n_remove`` is used.
    segment_len : float | None, default=None
        Block length in seconds. ``None`` learns one operator for all data.
    overlap : float, default=0.0
        Fraction of ``segment_len`` shared between consecutive blocks.
    preserve_mean : bool, default=True
        Add the fitted channel mean back after cleaning.
    verbose : bool | str | int | None, default=None
        MNE-style logging level.

    Attributes
    ----------
    cleaning_matrix_ : ndarray, shape (n_channels, n_channels)
        Channel-space operator, applied to mean-centered data. A tuple of
        matrices when ``segment_len`` is set.
    filters_ : ndarray, shape (n_components, n_channels)
        Canonical filters, rows ordered by decreasing correlation.
    patterns_ : ndarray, shape (n_channels, n_components)
        Least-squares mixing matrix; columns are sensor patterns.
    correlations_ : ndarray, shape (n_components,)
        Non-negative canonical correlations in descending order.
    autocorrelations_ : ndarray, shape (n_components,)
        Signed lag-1 autocorrelation of each component.
    filter_asymmetry_ : ndarray, shape (n_components,)
        Distance between the two canonical filters of each component.
    kept_mask_ : ndarray of bool, shape (n_components,)
        Components retained in the reconstruction.
    training_mean_ : ndarray, shape (n_channels, 1)
        Channel mean learned during ``fit``.
    input_rank_ : int
        Number of canonical components, below ``n_channels_in_`` when the
        training data is rank deficient.
    n_kept_, n_removed_ : int
        Component counts.
    n_channels_in_ : int
        Channels seen during ``fit``.
    feature_names_in_ : tuple of str | None
        Channel names when fitted on an MNE object.

    See Also
    --------
    compute_bss_cca : Canonical array implementation used by ``fit``.

    References
    ----------
    .. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
           Van Huffel, S. (2006). Canonical correlation analysis applied to
           remove muscle artifacts from the electroencephalogram. IEEE
           Transactions on Biomedical Engineering, 53(12), 2583-2587.
           https://doi.org/10.1109/TBME.2006.879459
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
        verbose: bool | str | int | None = None,
    ) -> BSSCCA:
        """Learn the BSS-CCA operators.

        Parameters
        ----------
        X : array-like | mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked
            Data used to learn the operators.
        y : None
            Ignored. Included for scikit-learn compatibility.

        Returns
        -------
        self : BSSCCA
            Fitted estimator.
        """
        del y
        data, data_sfreq, _mne_type, _orig, _picks, names = extract_data_from_mne(
            X, auto_pick=True
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
            segment_len=self.segment_len,
            overlap=self.overlap,
            preserve_mean=self.preserve_mean,
        )
        self.cleaning_matrix_ = info["cleaning_matrix"]
        self.filters_ = info["filters"]
        self.patterns_ = info["patterns"]
        self.correlations_ = info["correlations"]
        self.autocorrelations_ = info["autocorrelations"]
        self.filter_asymmetry_ = info["filter_asymmetry"]
        self.kept_mask_ = info["kept_mask"]
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
        """Apply the fitted operators to new data.

        Parameters
        ----------
        X : array-like | mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked
            Data with the channel layout seen during ``fit``.
        y : None
            Ignored. Included for scikit-learn compatibility.

        Returns
        -------
        X_clean : same type as X
            A copy with the selected data channels replaced.
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
        verbose: bool | str | int | None = None,
        **fit_params,
    ) -> Any:
        """Fit on ``X`` and apply the fitted operators to ``X``.

        Parameters
        ----------
        X : array-like | mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked
            Data to fit and transform.
        y : None
            Ignored. Included for scikit-learn compatibility.
        **fit_params : dict
            Reserved for scikit-learn compatibility.

        Returns
        -------
        X_clean : same type as X
            Cleaned data.
        """
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unexpected}")
        return self.fit(X, y).transform(X)
