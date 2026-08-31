"""iCanClean algorithms."""

# Patent notice:
# A public U.S. patent application has been filed for the iCanClean method:
# US20230363718A1, "Removing latent noise components from data signals"
# (Application 18/245,496). Patent applications, and any resulting patents,
# may affect commercial use.

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg as la
from scipy.signal import sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

from . import _mne
from ._cca import canonical_correlation
from ._data import extract_data_from_mne, reconstruct_mne_object
from ._filtering import design_butter_sos
from ._logging import logger, verbose
from .progress import _emit_progress, _ProgressCallback, _validate_callback

__all__ = ["ICanClean", "compute_icanclean", "null_r2_threshold"]

#: Default number of circular-shift surrogates for ``threshold='null'``. 20 is
#: the floor at which the default alpha's quantile is even defined; 100 gives
#: a materially more stable estimate at a still-cheap cost per window.
_NULL_N_SURROGATE = 100
#: Default family-wise false-rejection rate for ``threshold='null'``.
_NULL_ALPHA = 0.05
#: Smallest circular shift, as a fraction of the window, when building the null.
#: Shifts near zero leave the blocks nearly aligned and inflate the threshold.
_NULL_MIN_SHIFT = 0.1


def _filter_channels(
    data: np.ndarray,
    filter_spec: tuple[str, float | tuple[float, float]] | None,
    sfreq: float,
) -> np.ndarray:
    """Filter along the last axis with a zero-phase 4th-order Butterworth."""
    if filter_spec is None:
        return data
    btype, freqs = filter_spec
    sos = design_butter_sos(4, freqs, btype, sfreq)
    return sosfiltfilt(sos, data, axis=-1)


def _r2_from_projections(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Squared correlation of each column pair of two projected CCA bases."""
    U_zm = U - U.mean(axis=0, keepdims=True)
    V_zm = V - V.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(U_zm**2, axis=0)) * np.sqrt(np.sum(V_zm**2, axis=0))
    denom[denom == 0] = 1.0
    R = np.sum(U_zm * V_zm, axis=0) / denom
    return np.clip(R**2, 0.0, 1.0).astype(np.float64)


def null_r2_threshold(
    X_cca: np.ndarray,
    Y_cca: np.ndarray,
    *,
    alpha: float = _NULL_ALPHA,
    n_surrogate: int = _NULL_N_SURROGATE,
    random_state: int | np.random.Generator | None = None,
) -> float:
    r"""Estimate a circular-shift null threshold for squared CCA correlations.

    Parameters
    ----------
    X_cca : ndarray, shape (n_times, n_primary)
        Primary CCA block.
    Y_cca : ndarray, shape (n_times, n_reference)
        Reference CCA block.
    alpha : float, default=0.05
        Upper-tail probability used for the null quantile.
    n_surrogate : int, default=100
        Number of circular-shift surrogates.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for shift offsets.

    Returns
    -------
    float
        Quantile of the maximum surrogate squared canonical correlation.

    See Also
    --------
    ICanClean
        Estimator that applies the threshold within its cleaning workflow.

    Notes
    -----
    Circular shifts preserve within-channel temporal structure while disrupting
    alignment between the two blocks. The threshold addresses finite-sample shared
    correlation; it does not identify whether shared variance is artifact. This is
    package functionality around the published iCanClean workflow, not a claim
    about its original core method.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.icanclean import null_r2_threshold
    >>> rng = np.random.default_rng(0)
    >>> X_cca = rng.standard_normal((1000, 4))
    >>> Y_cca = rng.standard_normal((1000, 2))
    >>> threshold = null_r2_threshold(X_cca, Y_cca, n_surrogate=20, random_state=0)
    """
    rng = np.random.default_rng(random_state)
    n_times = X_cca.shape[0]
    lo = max(1, int(_NULL_MIN_SHIFT * n_times))
    hi = n_times - lo
    if hi <= lo:
        # The guard band leaves no room on a very short window. Any nonzero
        # shift still decorrelates the blocks; falling back to one fixed
        # shift for every surrogate would collapse the quantile to a single
        # sample instead of estimating one.
        lo, hi = 1, n_times
    maxima = np.empty(n_surrogate, dtype=np.float64)
    for i in range(n_surrogate):
        shift = int(rng.integers(lo, hi))
        try:
            _, _, R_null, _, _ = canonical_correlation(
                X_cca, np.roll(Y_cca, shift, axis=0)
            )
        except Exception:  # noqa: BLE001 - a failed surrogate must not kill the pass
            maxima[i] = 1.0
            continue
        maxima[i] = float((R_null**2).max()) if R_null.size else 1.0
    return float(np.quantile(maxima, 1.0 - alpha))


def _compute_icanclean_impl(
    X_primary: np.ndarray,
    X_ref: np.ndarray,
    sfreq: float,
    mode: str = "sliding",
    clean_with: str = "X",
    segment_len: float = 2.0,
    overlap: float = 0.0,
    threshold: float | str = 0.7,
    max_reject_fraction: float = 0.5,
    reref_primary: bool | str = False,
    reref_ref: bool | str = False,
    stats_segment_len: float | None = None,
    null_random_state: int | None = None,
    callback: _ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute one continuous NumPy iCanClean pass."""
    if mode == "hybrid":
        raise ValueError(
            "compute_icanclean supports only single-pass 'global', "
            "'sliding', or 'calibrated' modes; use ICanClean(..., "
            "mode='hybrid') for two-pass orchestration"
        )
    _validate_icanclean_config(
        mode=mode,
        clean_with=clean_with,
        overlap=overlap,
        threshold=threshold,
        max_reject_fraction=max_reject_fraction,
        reref_primary=reref_primary,
        reref_ref=reref_ref,
        segment_len=segment_len,
        stats_segment_len=stats_segment_len,
        global_threshold=None,
        global_clean_with=None,
        global_max_reject_fraction=None,
    )

    X_primary = np.asarray(X_primary, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)

    if X_primary.ndim != 2:
        raise ValueError(
            f"X_primary must be 2D with shape (n_primary, n_times), got {X_primary.shape}"
        )
    if X_ref.ndim != 2:
        raise ValueError(
            f"X_ref must be 2D with shape (n_ref, n_times), got {X_ref.shape}"
        )
    if X_primary.shape[1] != X_ref.shape[1]:
        raise ValueError(
            "X_primary and X_ref must have the same number of time samples, "
            f"got {X_primary.shape[1]} and {X_ref.shape[1]}"
        )
    if X_primary.shape[0] == 0 or X_ref.shape[0] == 0:
        raise ValueError("X_primary and X_ref must both contain at least one channel")

    use_windows = mode in ("sliding", "calibrated")
    n_times = X_primary.shape[1]

    if use_windows:
        win_samples = int(segment_len * sfreq)
        step_samples = max(1, int(round(win_samples * (1 - overlap))))

        if win_samples > n_times:
            raise ValueError(
                f"Window length ({win_samples} samples = {segment_len}s) "
                f"exceeds data length ({n_times} samples)."
            )

        starts = list(np.arange(0, n_times - win_samples + 1, step_samples))
        last_possible = n_times - win_samples
        if starts and starts[-1] < last_possible:
            starts.append(last_possible)
    else:
        win_samples = n_times
        starts = [0]

    cleaned_primary = np.zeros_like(X_primary)
    weights = np.zeros(n_times, dtype=np.float64)

    all_corr: list[np.ndarray] = []
    all_n_removed: list[int] = []
    all_removed_idx: list[np.ndarray] = []
    all_filters: list[np.ndarray] = []
    all_patterns: list[np.ndarray] = []
    running_r2: list[float] = []
    # Recorded so a zero removal is never ambiguous. Without these,
    # "0 components removed" is indistinguishable from "the threshold was
    # above every achievable R^2", which is how a whole benchmark arm came
    # to be read as a measurement.
    window_thresholds: list[float] = []
    window_max_r2: list[float] = []

    if (
        stats_segment_len is not None
        and stats_segment_len > segment_len
        and mode == "sliding"
    ):
        stats_win_samples = int(stats_segment_len * sfreq)
    else:
        stats_win_samples = None

    if mode == "calibrated":
        X_global = X_primary.T
        Y_global = X_ref.T
        X_global_cca = _apply_reref(X_global, reref_primary)
        Y_global_cca = _apply_reref(Y_global, reref_ref)

        try:
            A_global, B_global, R_global, U_global, V_global = canonical_correlation(
                X_global_cca, Y_global_cca
            )
        except Exception as exc:
            raise RuntimeError("CCA failed for calibrated global pass") from exc
        if R_global.size == 0:
            raise ValueError(
                "CCA returned 0 components for calibrated global pass; "
                "check the rank/variance of the primary and reference channels"
            )

        Z_global = _select_basis(U_global, V_global, clean_with)

        X_global_mc = X_global - X_global.mean(axis=0, keepdims=True)
        Z_global_mc = Z_global - Z_global.mean(axis=0, keepdims=True)
        beta_global, *_ = la.lstsq(Z_global_mc, X_global_mc, lapack_driver="gelsy")
        n_global_comp = R_global.size

    for window_idx, start in enumerate(starts):
        end = min(start + win_samples, n_times)
        actual_len = end - start

        if stats_win_samples is not None:
            extra = stats_win_samples - actual_len
            extra_pre = extra // 2
            extra_post = extra - extra_pre
            s_start = start - extra_pre
            s_end = end + extra_post
            if s_start < 0:
                s_end = min(n_times, s_end - s_start)
                s_start = 0
            if s_end > n_times:
                s_start = max(0, s_start - (s_end - n_times))
                s_end = n_times
            inner_offset = start - s_start
        else:
            s_start, s_end = start, end
            inner_offset = 0

        X_orig = X_primary[:, s_start:s_end].T
        Y_orig = X_ref[:, s_start:s_end].T

        X_cca = _apply_reref(X_orig, reref_primary)
        Y_cca = _apply_reref(Y_orig, reref_ref)

        if mode == "calibrated":
            X_cca_mc = X_cca - X_cca.mean(axis=0, keepdims=True)
            Y_cca_mc = Y_cca - Y_cca.mean(axis=0, keepdims=True)
            U = X_cca_mc @ A_global
            V = Y_cca_mc @ B_global
            r2 = _r2_from_projections(U, V)
            A = A_global
            B = B_global
        else:
            try:
                A, B, R, U, V = canonical_correlation(X_cca, Y_cca)
            except Exception as exc:
                raise RuntimeError(f"CCA failed for window {start}:{end}") from exc
            if R.size == 0:
                raise ValueError(
                    f"CCA returned 0 components for window {start}:{end}; "
                    "check the rank/variance of the primary and reference channels"
                )
            r2 = (R**2).astype(np.float64)
        running_r2.extend(r2.tolist())

        if threshold == "auto":
            if len(running_r2) > 10:
                thr = float(np.percentile(running_r2, 95))
            else:
                thr = 0.95
        elif threshold == "null":
            # Recomputed per window: the null depends on this window's sample
            # count and channel counts, which is the whole point.
            thr = null_r2_threshold(X_cca, Y_cca, random_state=null_random_state)
        else:
            thr = float(threshold)

        window_thresholds.append(thr)
        window_max_r2.append(float(r2.max()) if r2.size else float("nan"))

        bad_mask = r2 >= thr

        max_bad = (
            0
            if max_reject_fraction == 0
            else max(1, int(max_reject_fraction * len(r2)))
        )
        if bad_mask.sum() > max_bad:
            order = np.argsort(r2)[::-1]
            bad_mask[:] = False
            if max_bad > 0:
                bad_mask[order[:max_bad]] = True

        bad_idx = np.where(bad_mask)[0]

        all_corr.append(r2)
        all_n_removed.append(int(bad_idx.size))
        all_removed_idx.append(bad_idx)
        all_filters.append(A)
        all_patterns.append(B)
        logger.debug(
            "iCanClean window %d/%d: threshold=%.3f, max R^2=%.3f, "
            "removed=%d component(s).",
            len(all_corr),
            len(starts),
            thr,
            window_max_r2[-1],
            bad_idx.size,
        )

        if bad_idx.size > 0:
            if mode == "calibrated":
                noise_sources = _select_basis(U, V, clean_with, bad_idx)
                if clean_with in ("X", "Y"):
                    beta = beta_global[bad_idx, :]
                else:
                    beta_idx = np.concatenate((bad_idx, bad_idx + n_global_comp))
                    beta = beta_global[beta_idx, :]

                X_clean_win = X_orig - noise_sources @ beta
                cleaned_primary[:, start:end] += X_clean_win.T
            else:
                noise_sources = _select_basis(U, V, clean_with, bad_idx)

                X_mc = X_orig - X_orig.mean(axis=0, keepdims=True)
                Z_mc = noise_sources - noise_sources.mean(axis=0, keepdims=True)

                beta, *_ = la.lstsq(Z_mc, X_mc, lapack_driver="gelsy")
                X_clean_full = X_orig - Z_mc @ beta
                X_clean_win = X_clean_full[inner_offset : inner_offset + actual_len]
                cleaned_primary[:, start:end] += X_clean_win.T
        else:
            X_inner = X_orig[inner_offset : inner_offset + actual_len]
            cleaned_primary[:, start:end] += X_inner.T

        weights[start:end] += 1.0

        if mode in ("sliding", "calibrated"):
            _emit_progress(
                callback,
                method="icanclean",
                stage="window",
                current=window_idx + 1,
                total=len(starts),
                component=None,
                metric=float(bad_idx.size),
            )

    mask = weights > 0
    cleaned_primary[:, mask] /= weights[mask]
    if not mask.all():
        cleaned_primary[:, ~mask] = X_primary[:, ~mask]

    qc = {
        "correlations_": _pad_ragged(all_corr),
        "n_removed_": np.array(all_n_removed, dtype=int),
        "removed_idx_": all_removed_idx,
        "filters_": all_filters,
        "patterns_": all_patterns,
        "n_windows_": len(starts),
        "thresholds_": np.array(window_thresholds, dtype=float),
        "max_r2_": np.array(window_max_r2, dtype=float),
        "samples_per_variable_": float(
            win_samples / max(1, X_primary.shape[0] + X_ref.shape[0])
        ),
    }

    return cleaned_primary.astype(np.float64), qc


def _log_icanclean_summary(
    mode: str,
    qc: dict[str, Any],
    *,
    n_primary: int,
    n_ref: int,
    n_epochs: int | None = None,
) -> None:
    """Emit one aggregate iCanClean report from already-computed QC state."""
    epoch_prefix = f"{n_epochs} epochs, " if n_epochs is not None else ""
    if mode == "hybrid":
        global_removed = np.asarray(qc["global_n_removed_"], dtype=int)
        sliding_removed = np.asarray(qc["sliding_n_removed_"], dtype=int)
        n_sliding = sliding_removed.size
        pct_sliding = (
            100.0 * np.count_nonzero(sliding_removed > 0) / n_sliding
            if n_sliding
            else 0.0
        )
        mean_sliding = float(np.mean(sliding_removed)) if n_sliding else 0.0
        logger.info(
            "iCanClean: mode=hybrid, %sglobal removed=%d component(s); "
            "sliding=%d windows, %d primary / %d reference channels, "
            "%.1f%% windows had removals, mean removed=%.1f component(s).",
            epoch_prefix,
            int(np.sum(global_removed)),
            n_sliding,
            n_primary,
            n_ref,
            pct_sliding,
            mean_sliding,
        )
        return

    removed = np.asarray(qc["n_removed_"], dtype=int)
    n_windows = removed.size
    pct_windows = (
        100.0 * np.count_nonzero(removed > 0) / n_windows if n_windows else 0.0
    )
    mean_removed = float(np.mean(removed)) if n_windows else 0.0
    logger.info(
        "iCanClean: mode=%s, %s%d windows, %d primary / %d reference channels, "
        "%.1f%% windows had removals, mean removed=%.1f component(s).",
        mode,
        epoch_prefix,
        n_windows,
        n_primary,
        n_ref,
        pct_windows,
        mean_removed,
    )


@verbose
def compute_icanclean(
    X_primary: np.ndarray,
    X_ref: np.ndarray,
    sfreq: float,
    mode: str = "sliding",
    clean_with: str = "X",
    segment_len: float = 2.0,
    overlap: float = 0.0,
    threshold: float | str = 0.7,
    max_reject_fraction: float = 0.5,
    reref_primary: bool | str = False,
    reref_ref: bool | str = False,
    stats_segment_len: float | None = None,
    null_random_state: int | None = None,
    verbose: bool | str | int | None = None,
    callback=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Compute one iCanClean pass on continuous NumPy arrays.

    Parameters
    ----------
    X_primary : ndarray, shape (n_primary, n_times)
        Primary channels to clean.
    X_ref : ndarray, shape (n_reference, n_times)
        Reference channels.
    sfreq : float
        Sampling frequency in Hz.
    mode : {"sliding", "global", "calibrated"}, default="sliding"
        CCA fitting and cleaning mode.
    clean_with : {"X", "Y", "both"}, default="X"
        Canonical basis used for artifact regression.
    segment_len : float, default=2.0
        Cleaning-window length in seconds for windowed modes.
    overlap : float, default=0.0
        Fractional window overlap.
    threshold : float or {"auto", "null"}, default=0.7
        Squared-correlation rejection threshold.
    max_reject_fraction : float, default=0.5
        Maximum fraction of components removed per window.
    reref_primary : bool or str, default=False
        Average-reference option for the primary CCA block.
    reref_ref : bool or str, default=False
        Average-reference option for the reference CCA block.
    stats_segment_len : float or None, default=None
        Optional broader statistics window for sliding mode.
    null_random_state : int or None, default=None
        Seed for threshold="null".
    verbose : bool, str, int, or None, default=None
        Logging level.
    callback : callable or None, default=None
        Synchronous callback for completed window operations.

    Returns
    -------
    X_primary_clean : ndarray, shape (n_primary, n_times)
        Cleaned primary data.
    qc : dict
        Quality-control arrays and resolved settings.

    Notes
    -----
    The function is array-only and transductive: the operator is estimated from the
    same recording it cleans. Use ICanClean for MNE containers and estimator
    lifecycle semantics. A high canonical correlation indicates shared variance,
    not artifact identity.

    References
    ----------
    :footcite:p:`downey_ferris2022_icanclean,downey_ferris2023_icanclean_phantom,gonsisko2023_icanclean_ica`

    .. footbibliography::
    """
    callback = _validate_callback(callback)
    cleaned, qc = _compute_icanclean_impl(
        X_primary,
        X_ref,
        sfreq,
        mode=mode,
        clean_with=clean_with,
        segment_len=segment_len,
        overlap=overlap,
        threshold=threshold,
        max_reject_fraction=max_reject_fraction,
        reref_primary=reref_primary,
        reref_ref=reref_ref,
        stats_segment_len=stats_segment_len,
        null_random_state=null_random_state,
        callback=callback,
    )
    _log_icanclean_summary(
        mode,
        qc,
        n_primary=np.asarray(X_primary).shape[0],
        n_ref=np.asarray(X_ref).shape[0],
    )
    return cleaned, qc


class ICanClean(BaseEstimator, TransformerMixin):
    r"""Reference-based CCA artifact-removal estimator.

    ICanClean compares primary channels with physical or derived reference channels
    and removes selected shared canonical components. Cleaning is estimated during
    transform; fit is a compatibility no-op.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    ref_channels : list of str, list of int, or None, default=None
        Reference channels. Required unless pseudo_ref=True.
    primary_channels : list of str, list of int, or None, default=None
        Primary channels; by default all channels not in ref_channels.
    mode : {"sliding", "global", "calibrated", "hybrid"}, default="sliding"
        CCA fitting and cleaning mode.
    clean_with : {"X", "Y", "both"}, default="X"
        Canonical basis used for artifact regression.
    segment_len : float, default=2.0
        Cleaning-window length in seconds.
    overlap : float, default=0.0
        Fractional overlap between windows.
    threshold : float or {"auto", "null"}, default=0.7
        Squared-correlation rejection threshold.
    max_reject_fraction : float, default=0.5
        Maximum fraction of components removed per window.
    reref_primary : bool or str, default=False
        Average-reference option for primary channels used in CCA.
    reref_ref : bool or str, default=False
        Average-reference option for reference channels used in CCA.
    stats_segment_len : float or None, default=None
        Broader statistics window for supported sliding modes.
    filter_ref : tuple or None, default=None
        Optional zero-phase Butterworth specification applied to reference data.
    pseudo_ref : bool, default=False
        Build the reference block from filtered primary data.
    null_random_state : int or None, default=None
        Seed for threshold="null" surrogates.
    global_threshold : float, str, or None, default=None
        Threshold for the global pass in hybrid mode.
    global_clean_with : {"X", "Y", "both"} or None, default=None
        Basis for the global pass in hybrid mode.
    global_max_reject_fraction : float or None, default=None
        Removal cap for the global pass in hybrid mode.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    correlations_ : ndarray
        Squared canonical correlations by window.
    n_removed_ : ndarray
        Number of removed components by window.
    removed_idx_ : list of ndarray
        Removed component indices by window.
    filters_, patterns_ : list of ndarray
        CCA filters and patterns by window.
    n_windows_ : int
        Number of processed windows.
    primary_channels_, ref_channels_ : list
        Fitted channel selections.

    See Also
    --------
    mne_denoise.bss_cca.BSSCCA
        Reference-free CCA using a lagged copy of the primary signal.
    compute_icanclean
        One-shot functional interface for continuous array data.
    null_r2_threshold
        Package circular-shift surrogate threshold for squared canonical
        correlations; this is an extension around the published method.

    Notes
    -----
    NumPy input is channel-first; MNE Raw, Epochs, and Evoked inputs are supported
    and returned as the same container type. A high shared correlation is not, by
    itself, evidence that a component is artifact.

    References
    ----------
    :footcite:p:`downey_ferris2022_icanclean,downey_ferris2023_icanclean_phantom,gonsisko2023_icanclean_ica`

    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.icanclean import ICanClean
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> model = ICanClean(sfreq=250.0, ref_channels=[6, 7])
    >>> clean = model.fit_transform(data)
    """

    def __init__(
        self,
        sfreq: float,
        ref_channels: list[str] | list[int] | None = None,
        primary_channels: list[str] | list[int] | None = None,
        mode: str = "sliding",
        clean_with: str = "X",
        segment_len: float = 2.0,
        overlap: float = 0.0,
        threshold: float | str = 0.7,
        max_reject_fraction: float = 0.5,
        reref_primary: bool | str = False,
        reref_ref: bool | str = False,
        stats_segment_len: float | None = None,
        filter_ref: tuple | None = None,
        pseudo_ref: bool = False,
        null_random_state: int | None = None,
        global_threshold: float | str | None = None,
        global_clean_with: str | None = None,
        global_max_reject_fraction: float | None = None,
        verbose: bool | str | int | None = None,
    ):
        # In pseudo-reference mode the reference block is derived from the
        # primary channels themselves, so there are no reference channels for
        # the caller to name.
        if ref_channels is None and not pseudo_ref:
            raise ValueError("ref_channels must be provided explicitly")
        if pseudo_ref and ref_channels is not None:
            raise ValueError(
                "pseudo_ref=True builds its own reference block from the "
                "primary channels; ref_channels is not used in this mode "
                "and would be silently ignored, so it must be left as None."
            )
        _validate_filter_ref(filter_ref, sfreq)
        if pseudo_ref and filter_ref is None:
            raise ValueError(
                "pseudo_ref=True requires filter_ref. Without a filter the "
                "reference block is identical to the primary block, every "
                "canonical correlation is 1.0, and the whole signal is removed."
            )
        _validate_icanclean_config(
            mode=mode,
            clean_with=clean_with,
            overlap=overlap,
            threshold=threshold,
            max_reject_fraction=max_reject_fraction,
            reref_primary=reref_primary,
            reref_ref=reref_ref,
            segment_len=segment_len,
            stats_segment_len=stats_segment_len,
            global_threshold=global_threshold,
            global_clean_with=global_clean_with,
            global_max_reject_fraction=global_max_reject_fraction,
        )

        self.sfreq = float(sfreq)
        self.ref_channels = ref_channels
        self.primary_channels = primary_channels
        self.mode = mode
        self.clean_with = clean_with
        self.segment_len = segment_len
        self.overlap = overlap
        self.threshold = threshold
        self.max_reject_fraction = max_reject_fraction
        self.reref_primary = reref_primary
        self.reref_ref = reref_ref
        self.stats_segment_len = stats_segment_len
        self.filter_ref = filter_ref
        self.pseudo_ref = pseudo_ref
        self.null_random_state = null_random_state
        self.global_threshold = global_threshold
        self.global_clean_with = global_clean_with
        self.global_max_reject_fraction = global_max_reject_fraction
        self.verbose = verbose

    @verbose
    def fit(
        self,
        X: Any,
        y=None,
        *,
        verbose: bool | str | int | None = None,
    ) -> ICanClean:
        """Return self without performing cleaning.

        Cleaning is estimated during transform because the estimator operates on
        record-specific reference blocks and windows.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Input data; not transformed by this method.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        ICanClean
            The estimator.
        """
        return self

    @verbose
    def transform(
        self,
        X: Any,
        y=None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> Any:
        """Apply iCanClean to the input.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data to clean. NumPy input is channel-first.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous callback for completed continuous windows.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            Cleaned data in a copy of the input container or array layout.
        """
        callback = _validate_callback(callback)
        self._reset_qc_attrs()

        data, sfreq_data, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, auto_pick=False
        )
        sfreq = sfreq_data if sfreq_data is not None else self.sfreq
        channel_data = data[0] if mne_type == "epochs" else data
        primary_idx, ref_idx = self._resolve_channels(channel_data, orig_inst)
        data, ref_idx, n_orig = self._build_reference_block(
            data, sfreq, primary_idx, ref_idx
        )

        if mne_type == "epochs":
            cleaned = self._transform_epochs(data, sfreq, primary_idx, ref_idx)
        else:
            cleaned = self._clean_continuous(
                data, sfreq, primary_idx, ref_idx, callback=callback
            )

        summary_qc = {"n_removed_": self.n_removed_}
        if self.mode == "hybrid":
            summary_qc.update(
                {
                    "global_n_removed_": self.global_n_removed_,
                    "sliding_n_removed_": self.sliding_n_removed_,
                }
            )
        _log_icanclean_summary(
            self.mode,
            summary_qc,
            n_primary=primary_idx.size,
            n_ref=ref_idx.size,
            n_epochs=data.shape[0] if mne_type == "epochs" else None,
        )

        if n_orig is not None:
            # Drop the pseudo-reference rows appended above so the output has
            # the same channel set as the input.
            cleaned = cleaned[..., :n_orig, :]

        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

    def _build_reference_block(
        self,
        data: np.ndarray,
        sfreq: float,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Build the reference block used by the CCA pass."""
        if not self.pseudo_ref and self.filter_ref is None:
            return data, ref_idx, None

        # Continuous data is (n_channels, n_times); epochs are
        # (n_epochs, n_channels, n_times). Filtering always runs on the last
        # axis, but the channel axis moves.
        ch_axis = data.ndim - 2
        n_orig = data.shape[ch_axis]

        if self.pseudo_ref:
            pseudo = _filter_channels(
                np.take(data, primary_idx, axis=ch_axis).copy(), self.filter_ref, sfreq
            )
            data = np.concatenate([data, pseudo], axis=ch_axis)
            return data, np.arange(n_orig, data.shape[ch_axis], dtype=int), n_orig

        data = data.copy()
        filtered = _filter_channels(
            np.take(data, ref_idx, axis=ch_axis), self.filter_ref, sfreq
        )
        if ch_axis == 0:
            data[ref_idx, :] = filtered
        else:
            data[:, ref_idx, :] = filtered
        return data, ref_idx, None

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
        """Return fit(X).transform(X).

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data to clean.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous callback for completed continuous windows.
        verbose : bool, str, int, or None, default=None
            Logging level.
        **fit_params : dict
            Unexpected fit parameters raise TypeError.

        Returns
        -------
        same type as X
            Cleaned data.
        """
        callback = _validate_callback(callback)
        self.fit(X, y)
        return self.transform(X, callback=callback)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _reset_qc_attrs(self) -> None:
        """Clear QC attributes from a previous transform call."""
        for attr in (
            "correlations_",
            "n_removed_",
            "removed_idx_",
            "filters_",
            "patterns_",
            "n_windows_",
            "epoch_window_counts_",
            "epoch_window_slices_",
            "global_correlations_",
            "global_n_removed_",
            "global_removed_idx_",
            "global_filters_",
            "global_patterns_",
            "global_epoch_window_slices_",
            "sliding_correlations_",
            "sliding_n_removed_",
            "sliding_removed_idx_",
            "sliding_filters_",
            "sliding_patterns_",
            "sliding_epoch_window_slices_",
            # Previously omitted: these are written by the hybrid path
            # (_clean_continuous copies every qc key onto self) but were
            # absent from this tuple, so stale hybrid counters survived a
            # re-fit in a different mode.
            "global_n_windows_",
            "sliding_n_windows_",
            "thresholds_",
            "max_r2_",
            "samples_per_variable_",
            "global_thresholds_",
            "global_max_r2_",
            "global_samples_per_variable_",
            "sliding_thresholds_",
            "sliding_max_r2_",
            "sliding_samples_per_variable_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _transform_epochs(
        self,
        data: np.ndarray,
        sfreq: float,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
    ) -> np.ndarray:
        """Clean each epoch independently and aggregate QC state."""
        cleaned_epochs = np.empty_like(data)
        epoch_corrs: list[np.ndarray] = []
        epoch_n_removed: list[int] = []
        epoch_removed_idx: list[np.ndarray] = []
        epoch_filters: list[np.ndarray] = []
        epoch_patterns: list[np.ndarray] = []
        epoch_window_counts: list[int] = []
        epoch_window_slices: list[slice] = []
        epoch_global_corrs: list[np.ndarray] = []
        epoch_global_n_removed: list[int] = []
        epoch_global_removed_idx: list[np.ndarray] = []
        epoch_global_filters: list[np.ndarray] = []
        epoch_global_patterns: list[np.ndarray] = []
        global_epoch_window_slices: list[slice] = []
        epoch_sliding_corrs: list[np.ndarray] = []
        epoch_sliding_n_removed: list[int] = []
        epoch_sliding_removed_idx: list[np.ndarray] = []
        epoch_sliding_filters: list[np.ndarray] = []
        epoch_sliding_patterns: list[np.ndarray] = []
        sliding_epoch_window_slices: list[slice] = []

        epoch_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self._clean_epoch)(epoch, sfreq, primary_idx, ref_idx)
            for epoch in data
        )

        for i, (cleaned_epoch, qc) in enumerate(epoch_results):
            cleaned_epochs[i] = cleaned_epoch
            start = len(epoch_corrs)
            stop = start + qc["n_windows_"]
            epoch_window_slices.append(slice(start, stop))
            epoch_corrs.extend(
                [qc["correlations_"][j].copy() for j in range(qc["n_windows_"])]
            )
            epoch_n_removed.extend(qc["n_removed_"].tolist())
            epoch_removed_idx.extend([idx.copy() for idx in qc["removed_idx_"]])
            epoch_filters.extend(qc["filters_"])
            epoch_patterns.extend(qc["patterns_"])
            epoch_window_counts.append(qc["n_windows_"])

            if self.mode == "hybrid":
                global_start = len(epoch_global_corrs)
                global_stop = global_start + len(qc["global_n_removed_"])
                global_epoch_window_slices.append(slice(global_start, global_stop))
                epoch_global_corrs.extend(
                    [
                        qc["global_correlations_"][j].copy()
                        for j in range(qc["global_correlations_"].shape[0])
                    ]
                )
                epoch_global_n_removed.extend(qc["global_n_removed_"].tolist())
                epoch_global_removed_idx.extend(
                    [idx.copy() for idx in qc["global_removed_idx_"]]
                )
                epoch_global_filters.extend(qc["global_filters_"])
                epoch_global_patterns.extend(qc["global_patterns_"])

                sliding_start = len(epoch_sliding_corrs)
                sliding_stop = sliding_start + len(qc["sliding_n_removed_"])
                sliding_epoch_window_slices.append(slice(sliding_start, sliding_stop))
                epoch_sliding_corrs.extend(
                    [
                        qc["sliding_correlations_"][j].copy()
                        for j in range(qc["sliding_correlations_"].shape[0])
                    ]
                )
                epoch_sliding_n_removed.extend(qc["sliding_n_removed_"].tolist())
                epoch_sliding_removed_idx.extend(
                    [idx.copy() for idx in qc["sliding_removed_idx_"]]
                )
                epoch_sliding_filters.extend(qc["sliding_filters_"])
                epoch_sliding_patterns.extend(qc["sliding_patterns_"])

        self.correlations_ = _pad_ragged(epoch_corrs)
        self.n_removed_ = np.array(epoch_n_removed, dtype=int)
        self.removed_idx_ = epoch_removed_idx
        self.filters_ = epoch_filters
        self.patterns_ = epoch_patterns
        self.n_windows_ = len(epoch_corrs)
        self.epoch_window_counts_ = epoch_window_counts
        self.epoch_window_slices_ = tuple(epoch_window_slices)

        if self.mode == "hybrid":
            self.global_correlations_ = _pad_ragged(epoch_global_corrs)
            self.global_n_removed_ = np.array(epoch_global_n_removed, dtype=int)
            self.global_removed_idx_ = epoch_global_removed_idx
            self.global_filters_ = epoch_global_filters
            self.global_patterns_ = epoch_global_patterns
            self.global_epoch_window_slices_ = tuple(global_epoch_window_slices)
            self.sliding_correlations_ = _pad_ragged(epoch_sliding_corrs)
            self.sliding_n_removed_ = np.array(epoch_sliding_n_removed, dtype=int)
            self.sliding_removed_idx_ = epoch_sliding_removed_idx
            self.sliding_filters_ = epoch_sliding_filters
            self.sliding_patterns_ = epoch_sliding_patterns
            self.sliding_epoch_window_slices_ = tuple(sliding_epoch_window_slices)

        return cleaned_epochs

    def _clean_epoch(
        self,
        data: np.ndarray,
        sfreq: float,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Clean one epoch without mutating estimator state."""
        cleaned_primary, qc = self._compute_continuous_cleaning(
            data, sfreq, primary_idx, ref_idx
        )
        data_out = data.copy()
        data_out[primary_idx, :] = cleaned_primary
        return data_out, qc

    def _resolve_channels(
        self, data: np.ndarray, orig_inst: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve primary and reference channel indices.

        Returns
        -------
        primary_idx : ndarray of int
        ref_idx : ndarray of int
        """
        n_channels = data.shape[0]
        ch_names = None
        if orig_inst is not None and hasattr(orig_inst, "ch_names"):
            _mne.require_mne("iCanClean MNE channel handling")
            ch_names = list(orig_inst.ch_names)

        if self.ref_channels is None:
            # Only reachable with pseudo_ref=True, which the constructor
            # enforces. The reference block does not exist yet; it is built
            # from the primary channels in _build_reference_block.
            if self.primary_channels is not None:
                if ch_names is not None and isinstance(self.primary_channels[0], str):
                    primary_idx = _mne.mne.pick_channels(
                        ch_names,
                        include=list(self.primary_channels),
                        ordered=True,
                    )
                else:
                    primary_idx = np.asarray(self.primary_channels, dtype=int)
            else:
                primary_idx = np.arange(n_channels, dtype=int)
            if ch_names is not None:
                self.primary_channels_ = [ch_names[i] for i in primary_idx]
                self.ref_channels_ = None
            return primary_idx, np.array([], dtype=int)

        if ch_names is not None and isinstance(self.ref_channels[0], str):
            ref_idx = _mne.mne.pick_channels(
                ch_names,
                include=list(self.ref_channels),
                ordered=True,
            )
            ref_names = list(self.ref_channels)
        else:
            ref_idx = np.asarray(self.ref_channels, dtype=int)
            ref_names = (
                [ch_names[idx] for idx in ref_idx] if ch_names is not None else None
            )

        if self.primary_channels is not None:
            if ch_names is not None and isinstance(self.primary_channels[0], str):
                primary_idx = _mne.mne.pick_channels(
                    ch_names,
                    include=list(self.primary_channels),
                    ordered=True,
                )
                primary_names = list(self.primary_channels)
            else:
                primary_idx = np.asarray(self.primary_channels, dtype=int)
                primary_names = (
                    [ch_names[idx] for idx in primary_idx]
                    if ch_names is not None
                    else None
                )
        else:
            all_idx = set(range(n_channels))
            primary_idx = np.array(
                sorted(all_idx - set(ref_idx.tolist())),
                dtype=int,
            )
            primary_names = (
                [ch_names[idx] for idx in primary_idx] if ch_names is not None else None
            )

        if ch_names is not None:
            self.primary_channels_ = primary_names
            self.ref_channels_ = ref_names

        return primary_idx, ref_idx

    def _clean_continuous(
        self,
        data: np.ndarray,
        sfreq: float,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
        callback: _ProgressCallback | None = None,
    ) -> np.ndarray:
        """Orchestrate continuous CCA cleaning for the configured mode."""
        cleaned_primary, qc = self._compute_continuous_cleaning(
            data, sfreq, primary_idx, ref_idx, callback=callback
        )
        for key, value in qc.items():
            setattr(self, key, value)

        data_out = data.copy()
        data_out[primary_idx, :] = cleaned_primary
        return data_out

    def _compute_continuous_cleaning(
        self,
        data: np.ndarray,
        sfreq: float,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
        callback: _ProgressCallback | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute cleaned primary channels and QC without mutating state."""
        if self.mode == "hybrid":
            cleaned_after_global, qc_global = _compute_icanclean_impl(
                data[primary_idx, :],
                data[ref_idx, :],
                sfreq=sfreq,
                mode="global",
                clean_with=self.global_clean_with,
                segment_len=self.segment_len,
                overlap=self.overlap,
                threshold=self.global_threshold,
                max_reject_fraction=self.global_max_reject_fraction,
                reref_primary=self.reref_primary,
                reref_ref=self.reref_ref,
                stats_segment_len=None,
                null_random_state=self.null_random_state,
                callback=None,
            )
            cleaned_primary, qc = _compute_icanclean_impl(
                cleaned_after_global,
                data[ref_idx, :],
                sfreq=sfreq,
                mode="sliding",
                clean_with=self.clean_with,
                segment_len=self.segment_len,
                overlap=self.overlap,
                threshold=self.threshold,
                max_reject_fraction=self.max_reject_fraction,
                reref_primary=self.reref_primary,
                reref_ref=self.reref_ref,
                stats_segment_len=self.stats_segment_len,
                null_random_state=self.null_random_state,
                callback=callback,
            )
            qc["global_correlations_"] = qc_global["correlations_"]
            qc["global_n_removed_"] = qc_global["n_removed_"]
            qc["global_removed_idx_"] = qc_global["removed_idx_"]
            qc["global_filters_"] = qc_global["filters_"]
            qc["global_patterns_"] = qc_global["patterns_"]
            qc["global_n_windows_"] = qc_global["n_windows_"]
        else:
            cleaned_primary, qc = _compute_icanclean_impl(
                data[primary_idx, :],
                data[ref_idx, :],
                sfreq=sfreq,
                mode=self.mode,
                clean_with=self.clean_with,
                segment_len=self.segment_len,
                overlap=self.overlap,
                threshold=self.threshold,
                max_reject_fraction=self.max_reject_fraction,
                reref_primary=self.reref_primary,
                reref_ref=self.reref_ref,
                stats_segment_len=self.stats_segment_len,
                null_random_state=self.null_random_state,
                callback=callback,
            )
        if self.mode == "hybrid":
            qc["sliding_correlations_"] = qc["correlations_"].copy()
            qc["sliding_n_removed_"] = qc["n_removed_"].copy()
            qc["sliding_removed_idx_"] = [idx.copy() for idx in qc["removed_idx_"]]
            qc["sliding_filters_"] = list(qc["filters_"])
            qc["sliding_patterns_"] = list(qc["patterns_"])
            qc["sliding_n_windows_"] = qc["n_windows_"]
        return cleaned_primary, qc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_threshold(value: Any, name: str) -> None:
    """Accept 'auto', 'null', or a float in [0, 1].

    A value above 1 makes the estimator a silent pass-through, since no
    :math:`R^2` can exceed it; a value below 0 silently flags every
    component; a numeric string such as ``"0.5"`` would silently compare
    against floats without conversion. None of these raise on their own, so
    this check exists to turn them into an explicit error at construction.
    """
    if value in ("auto", "null"):
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, 'auto', or 'null'") from exc
    if isinstance(value, str):
        raise ValueError(
            f"{name} must be a float, 'auto', or 'null', not the string "
            f"{value!r}; pass {numeric} instead"
        )
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"{name} is a squared canonical correlation and must lie in "
            f"[0, 1], got {numeric}. Values above 1 make the estimator a "
            f"no-op; values below 0 flag every component."
        )


def _validate_icanclean_config(
    mode: str,
    clean_with: str,
    overlap: float,
    threshold: float | str,
    max_reject_fraction: float,
    reref_primary: bool | str,
    reref_ref: bool | str,
    segment_len: float,
    stats_segment_len: float | None,
    global_threshold: float | str | None,
    global_clean_with: str | None,
    global_max_reject_fraction: float | None,
) -> None:
    """Validate shared iCanClean configuration parameters."""
    if mode not in ("sliding", "global", "calibrated", "hybrid"):
        raise ValueError(
            f"mode must be 'sliding', 'global', 'calibrated', or 'hybrid', got {mode!r}"
        )
    if clean_with not in ("X", "Y", "both"):
        raise ValueError(f"clean_with must be 'X', 'Y', or 'both', got {clean_with!r}")
    if not (0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    if not (0 <= max_reject_fraction <= 1):
        raise ValueError(
            f"max_reject_fraction must be in [0, 1], got {max_reject_fraction}"
        )
    _validate_threshold(threshold, "threshold")
    if reref_primary not in (False, True, "fullrank", "loserank"):
        raise ValueError(
            "reref_primary must be False, True, 'fullrank', or "
            f"'loserank', got {reref_primary!r}"
        )
    if reref_ref not in (False, True, "fullrank", "loserank"):
        raise ValueError(
            "reref_ref must be False, True, 'fullrank', or "
            f"'loserank', got {reref_ref!r}"
        )
    if stats_segment_len is not None:
        if stats_segment_len <= 0:
            raise ValueError("stats_segment_len must be positive")
        if stats_segment_len < segment_len:
            raise ValueError(
                "stats_segment_len must be greater than or equal to segment_len"
            )
        if mode in ("global", "calibrated"):
            raise ValueError(
                "stats_segment_len is only supported in 'sliding' and 'hybrid' modes"
            )
    has_global_params = any(
        value is not None
        for value in (
            global_threshold,
            global_clean_with,
            global_max_reject_fraction,
        )
    )
    if mode == "hybrid":
        if not has_global_params or any(
            value is None
            for value in (
                global_threshold,
                global_clean_with,
                global_max_reject_fraction,
            )
        ):
            raise ValueError(
                "mode='hybrid' requires global_threshold, "
                "global_clean_with, and global_max_reject_fraction"
            )
        if global_clean_with not in ("X", "Y", "both"):
            raise ValueError(
                "global_clean_with must be 'X', 'Y', or 'both', "
                f"got {global_clean_with!r}"
            )
        if not (0 <= global_max_reject_fraction <= 1):
            raise ValueError(
                "global_max_reject_fraction must be in [0, 1], got "
                f"{global_max_reject_fraction}"
            )
        _validate_threshold(global_threshold, "global_threshold")
    elif has_global_params:
        raise ValueError(
            "global_threshold, global_clean_with, and "
            "global_max_reject_fraction are only supported when "
            "mode='hybrid'"
        )


#: btypes accepted by ``filter_ref`` -- scipy's own names, not a shorthand.
_FILTER_REF_BTYPES = ("bandpass", "bandstop", "highpass", "lowpass")


def _validate_filter_ref(filter_ref: tuple | None, sfreq: float) -> None:
    """Check a ``filter_ref`` spec against ``sfreq`` before any data is touched."""
    if filter_ref is None:
        return
    if not isinstance(filter_ref, (tuple, list)) or len(filter_ref) != 2:
        raise ValueError(
            f"filter_ref must be a (btype, freqs) pair, got {filter_ref!r}"
        )
    kind, freqs = filter_ref
    if kind not in _FILTER_REF_BTYPES:
        raise ValueError(
            f"filter_ref btype must be one of {sorted(_FILTER_REF_BTYPES)}, got {kind!r}"
        )
    if kind in ("bandstop", "bandpass"):
        if not isinstance(freqs, (tuple, list)) or len(freqs) != 2:
            raise ValueError(f"filter_ref {kind!r} needs a (low, high) pair")
        if not 0 < freqs[0] < freqs[1]:
            raise ValueError(f"filter_ref {kind!r} needs 0 < low < high, got {freqs!r}")
        max_freq = freqs[1]
    else:
        if not np.isscalar(freqs):
            raise ValueError(f"filter_ref {kind!r} needs a single frequency")
        if freqs <= 0:
            raise ValueError(
                f"filter_ref {kind!r} needs a positive frequency, got {freqs!r}"
            )
        max_freq = freqs
    nyquist = sfreq / 2.0
    if max_freq >= nyquist:
        raise ValueError(
            f"filter_ref {filter_ref!r} exceeds Nyquist ({nyquist} Hz) for sfreq={sfreq}"
        )


def _select_basis(
    U: np.ndarray,
    V: np.ndarray,
    clean_with: str,
    idx: np.ndarray | None = None,
) -> np.ndarray:
    """Select the requested canonical basis from U, V, or both."""
    if idx is None:
        U_sel = U
        V_sel = V
    else:
        U_sel = U[:, idx]
        V_sel = V[:, idx]

    if clean_with == "X":
        return U_sel
    if clean_with == "Y":
        return V_sel
    return np.concatenate((U_sel, V_sel), axis=1)


def _apply_reref(data: np.ndarray, reref: bool | str) -> np.ndarray:
    """Apply average re-referencing across channels."""
    if reref is False:
        return data
    n_ch = data.shape[1]
    if reref is True or reref == "fullrank":
        # eye(n) - ones(n)/(n+1)
        ref = np.eye(n_ch) - np.ones((n_ch, n_ch)) / (n_ch + 1)
    elif reref == "loserank":
        # eye(n) - ones(n)/n — standard average reference
        ref = np.eye(n_ch) - np.ones((n_ch, n_ch)) / n_ch
    else:
        raise ValueError(
            f"reref must be False, True, 'fullrank', or 'loserank', got {reref!r}"
        )
    # ref is symmetric, so data @ ref == data @ ref.T
    return data @ ref


def _pad_ragged(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack a list of possibly different-length 1-D arrays into a 2-D array.

    Shorter rows are padded with NaN. Returns shape ``(n_rows, max_len)``.
    """
    if not arrays or all(a.size == 0 for a in arrays):
        return np.empty((len(arrays), 0), dtype=np.float64)
    max_len = max(a.size for a in arrays)
    out = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        if a.size > 0:
            out[i, : a.size] = a
    return out
