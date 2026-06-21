"""Core iCanClean algorithm and estimator.

This module contains:
1. ``compute_icanclean``: The core iCanClean implementation for continuous
   NumPy arrays.
2. ``ICanClean``: The Scikit-learn estimator compatible with MNE-Python
   objects or NumPy arrays.

iCanClean removes latent artifact subspaces shared by primary channels and
reference channels [1]_ [2]_ [3]_. The core procedure is:

1. Compute canonical variates shared by the primary and reference recordings [5]_.
2. Score those variates by squared canonical correlation.
3. Select the artifact-dominated variates.
4. Project the selected variates back to the primary channels.
5. Subtract the projected artifact activity from the original primary signal.

.. note:: A public U.S. patent application has been filed for the
          iCanClean method: US20230363718A1, "Removing latent noise components
          from data signals" (Application 18/245,496). Patent applications, and
          any resulting patents, may affect commercial use. Consult lawyer if
          necessary.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.
.. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
       Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
       23(19), 8214. https://doi.org/10.3390/s23198214
.. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
       Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
       https://doi.org/10.3390/s23020928
.. [4] Nordin, A. D., Hairston, W. D., & Ferris, D. P. (2018). Dual-electrode
       motion artifact cancellation for mobile electroencephalography.
       Journal of Neural Engineering, 15(5), 056024.
       https://doi.org/10.1088/1741-2552/aad7d7
.. [5] Hotelling, H. (1936). Relations between two sets of variates.
       Biometrika, 28(3/4), 321-377.
"""

# Patent notice:
# A public U.S. patent application has been filed for the iCanClean method:
# US20230363718A1, "Removing latent noise components from data signals"
# (Application 18/245,496). Patent applications, and any resulting patents,
# may affect commercial use.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg as la
from sklearn.base import BaseEstimator, TransformerMixin

from .._logging import set_log_level_from_verbose
from ..utils import extract_data_from_mne, reconstruct_mne_object
from ._cca import canonical_correlation

# Optional MNE support
try:
    import mne
except ImportError:
    mne = None

logger = logging.getLogger(__name__)


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
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Compute one iCanClean pass on continuous NumPy arrays.

    This implements the core array-based iCanClean algorithm for one
    continuous primary/reference recording pair. It returns cleaned primary
    channels together with per-pass quality-control outputs.

    The supported single-pass modes differ in where the CCA decomposition is
    estimated and how that decomposition is reused:

    ``mode='global'``
        1. Run CCA once on the full recording.
        2. Threshold the resulting :math:`R^2` values.
        3. Build the selected noise basis from ``U``, ``V``, or both.
        4. Regress that basis onto the full primary recording and subtract it.

    ``mode='sliding'``
        1. Split the recording into overlapping clean windows.
        2. Run a fresh CCA inside each window, optionally using a broader
           stats window when ``stats_segment_len`` is larger than
           ``segment_len``.
        3. Threshold the window-local :math:`R^2` values.
        4. Regress the selected window-local basis onto that window and
           combine cleaned windows with overlap-add.

    ``mode='calibrated'``
        1. Run CCA once on the full recording to obtain a fixed global
           decomposition.
        2. Fit one global least-squares map from the selected global
           canonical basis back to the primary channels.
        3. For each clean window, project the local data through the fixed
           global decomposition, score components with window-local
           correlations, and subtract only the window-local active part of
           the globally calibrated basis.

    Sliding-window cleaning is currently executed sequentially. A future
    optimization could parallelize fixed-threshold sliding windows, but that
    would need to preserve overlap-add semantics and the current sequential
    behavior of ``threshold='auto'``.

    Parameters
    ----------
    X_primary : ndarray, shape (n_primary, n_times)
        Primary channels to clean.
    X_ref : ndarray, shape (n_ref, n_times)
        Reference channels that capture artifact activity.
    sfreq : float
        Sampling frequency in Hz.
    mode : {'sliding', 'global', 'calibrated'}, default='sliding'
        Cleaning mode for this single pass. ``'global'`` runs one pass over
        the full recording, ``'sliding'`` uses overlapping windows, and
        ``'calibrated'`` uses one global CCA calibration followed by
        per-window scoring and subtraction with the fixed global basis.
    clean_with : {'X', 'Y', 'both'}, default='X'
        Canonical variates used as the noise basis.
    segment_len : float, default=2.0
        Sliding clean-window duration in seconds.
    overlap : float, default=0.0
        Fractional overlap between consecutive windows in [0, 1).
    threshold : float | 'auto', default=0.7
        :math:`R^2` threshold for component rejection.
    max_reject_fraction : float, default=0.5
        Maximum fraction of canonical components removed per window.
    reref_primary : bool | str, default=False
        Average re-reference mode applied to primary channels for CCA only.
    reref_ref : bool | str, default=False
        Average re-reference mode applied to reference channels for CCA only.
    stats_segment_len : float | None, default=None
        Broader stats-window duration in seconds for sliding mode.
    verbose : bool, default=True
        Whether to log progress information.

    Returns
    -------
    X_primary_clean : ndarray, shape (n_primary, n_times)
        Cleaned primary channels.
    qc : dict
        Per-pass quality-control outputs with the same top-level fields used by
        :class:`ICanClean`: ``correlations_``, ``n_removed_``, ``removed_idx_``,
        ``filters_``, ``patterns_``, and ``n_windows_``.

    See Also
    --------
    ICanClean : Estimator interface for MNE-Python objects and NumPy arrays.
    canonical_correlation : Core CCA solver used by iCanClean.

    Notes
    -----
    .. note:: A public U.S. patent application has been filed for the
              iCanClean method: US20230363718A1, "Removing latent noise
              components from data signals" (Application 18/245,496).
              Patent applications, and any resulting patents, may affect
              commercial use. Consult lawyer if necessary.

    When ``threshold='auto'``, the rejection threshold is the 95th percentile
    of the running :math:`R^2` distribution after at least 10 values have been
    accumulated. Before that point, a conservative threshold of 0.95 is used.

    Sliding-window cleaning is executed sequentially. A future optimization
    could parallelize fixed-threshold sliding windows, but that would need to
    preserve overlap-add semantics and the current sequential behavior of
    ``threshold='auto'``.
    """
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

    for start in starts:
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

            U_zm = U - U.mean(axis=0, keepdims=True)
            V_zm = V - V.mean(axis=0, keepdims=True)
            denom = np.sqrt(np.sum(U_zm**2, axis=0)) * np.sqrt(np.sum(V_zm**2, axis=0))
            denom[denom == 0] = 1.0
            R = np.sum(U_zm * V_zm, axis=0) / denom
            r2 = np.clip(R**2, 0.0, 1.0).astype(np.float64)
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
        else:
            thr = float(threshold)

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
    }

    if verbose:
        total_removed = qc["n_removed_"].sum()
        pct_windows = (
            (qc["n_removed_"] > 0).sum() / qc["n_windows_"] * 100
            if qc["n_windows_"] > 0
            else 0
        )
        logger.info(
            "ICanClean: %d windows, %.1f%% had removals, "
            "%.1f components removed on average",
            qc["n_windows_"],
            pct_windows,
            total_removed / max(qc["n_windows_"], 1),
        )

    return cleaned_primary.astype(np.float64), qc


class ICanClean(BaseEstimator, TransformerMixin):
    r"""ICanClean Transformer for reference-based artifact removal.

    Implements the iCanClean algorithm [1]_ [2]_ [3]_ using canonical
    correlation analysis (CCA) between primary channels and reference
    channels to identify and remove artifact-dominated subspaces.

    The estimator supports four operating modes:

    ``mode='global'``
        Estimate one CCA decomposition on the full recording and subtract the
        selected artifact basis once.

    ``mode='sliding'``
        Estimate a fresh CCA decomposition in each clean window and combine the
        cleaned windows with overlap-add.

    ``mode='calibrated'``
        Estimate one global CCA decomposition, then reuse that fixed basis for
        window-local scoring and subtraction.

    ``mode='hybrid'``
        Run an explicit global cleaning pass first, then run the standard
        sliding-window cleaner on the globally cleaned output.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    ref_channels : list of str | list of int
        Explicit reference noise channels. For MNE objects, provide channel
        names or integer channel indices. For NumPy arrays, provide integer
        channel indices.
    primary_channels : list of str | list of int | None, default=None
        Explicit primary (scalp) channels to clean. If ``None``, all channels
        not listed in ``ref_channels`` are used.
    mode : {'sliding', 'global', 'calibrated', 'hybrid'}, default='sliding'
        Cleaning mode. Use ``'global'`` for a single full-recording pass,
        ``'sliding'`` for the standard windowed cleaner, ``'calibrated'``
        for a global CCA calibration with local window scoring, or
        ``'hybrid'`` to run an explicit global pass followed by the
        sliding pass.
    clean_with : {'X', 'Y', 'both'}, default='X'
        Canonical variates used as the noise basis. ``'X'`` uses the
        data-side variates ``U``, ``'Y'`` uses the reference-side
        variates ``V``, and ``'both'`` concatenates both sets.
    segment_len : float, default=2.0
        Sliding window duration in seconds (the "clean window").
    overlap : float, default=0.0
        Overlap between consecutive windows as a fraction in [0, 1).
    threshold : float | 'auto', default=0.7
        :math:`R^2` threshold for component rejection.
        If ``'auto'``, uses an adaptive threshold based on the 95th percentile
        of the running correlation distribution.
    max_reject_fraction : float, default=0.5
        Safety cap: at most this fraction of canonical components can be
        removed per window.
    reref_primary : bool | str, default=False
        Apply average re-referencing to primary channels *for CCA only*
        (the original data is used for cleaning). ``True`` or ``'fullrank'``
        uses a full-rank average reference that preserves rank; ``'loserank'``
        uses a standard average reference that reduces rank by one.
    reref_ref : bool | str, default=False
        Same as ``reref_primary`` but for reference channels.
    stats_segment_len : float | None, default=None
        Duration (seconds) of the broader "stats window" for CCA
        computation. If ``None`` or equal to ``segment_len``, the same
        window is used for CCA and cleaning. When larger, CCA is computed
        on the broader window but only the inner ``segment_len`` portion
        is cleaned. This is only valid for ``'sliding'`` and ``'hybrid'``
        modes and must be greater than or equal to ``segment_len``.
    global_threshold : float | 'auto' | None, default=None
        Threshold for the explicit global pass in ``mode='hybrid'``.
    global_clean_with : {'X', 'Y', 'both'} | None, default=None
        Noise basis for the explicit global pass in ``mode='hybrid'``.
    global_max_reject_fraction : float | None, default=None
        Reject cap for the explicit global pass in ``mode='hybrid'``.
    verbose : bool, default=True
        Whether to log progress information.

    Attributes
    ----------
    correlations_ : ndarray, shape (n_windows, d)
        Squared canonical correlations per window.
        ``d = min(rank(primary), rank(reference))``.
    n_removed_ : ndarray, shape (n_windows,)
        Number of components removed per window.
    removed_idx_ : list of ndarray
        Indices of rejected components per window.
    filters_ : list of ndarray
        CCA coefficient matrices ``A`` (primary) per window.
    patterns_ : list of ndarray
        CCA coefficient matrices ``B`` (reference) per window.
    n_windows_ : int
        Total number of windows processed.
    primary_channels_ : list of str
        Primary channel names used during cleaning.
    ref_channels_ : list of str
        Reference channel names used during cleaning.

    See Also
    --------
    mne_denoise.DSS : Denoising Source Separation.
    mne_denoise.ZapLine : DSS-based line noise removal.

    Notes
    -----
    .. note:: A public U.S. patent application has been filed for the
              iCanClean method: US20230363718A1, "Removing latent noise
              components from data signals" (Application 18/245,496).
              Patent applications, and any resulting patents, may affect
              commercial use. Consult lawyer if necessary.

    When ``threshold='auto'``, the adaptive threshold is computed as the
    95th percentile of all :math:`R^2` values accumulated so far. For the
    first 10 windows (insufficient statistics), a conservative default of
    0.95 is used.

    The ``max_reject_fraction`` parameter prevents the algorithm from
    removing too many components in a single window, which would distort
    the signal. This is especially important for short windows or noisy
    reference sensors.

    Examples
    --------
    Basic usage with MNE Raw object (explicit reference channels):

    >>> from mne_denoise.icanclean import ICanClean
    >>> icanclean = ICanClean(
    ...     sfreq=raw.info["sfreq"],
    ...     ref_channels=["EOG1", "EOG2", "EMG1"],
    ... )
    >>> raw_clean = icanclean.fit_transform(raw)

    Dual-layer EEG with explicit channel names:

    >>> icanclean = ICanClean(
    ...     sfreq=256.0,
    ...     primary_channels=["1-EEG0", "1-EEG1", "1-EEG2"],
    ...     ref_channels=["2-NSE0", "2-NSE1"],
    ...     segment_len=2.0,
    ...     overlap=0.5,
    ...     threshold=0.85,
    ... )
    >>> raw_clean = icanclean.fit_transform(raw)
    >>> print(f"Removed {icanclean.n_removed_.mean():.1f} components on average")

    NumPy array interface:

    >>> import numpy as np
    >>> primary = np.random.randn(32, 5000)  # (n_primary, n_times)
    >>> reference = np.random.randn(4, 5000)  # (n_ref, n_times)
    >>> data = np.vstack([primary, reference])
    >>> icanclean = ICanClean(
    ...     sfreq=250.0,
    ...     ref_channels=list(range(32, 36)),  # last 4 channels
    ... )
    >>> cleaned = icanclean.fit_transform(data)

    References
    ----------
    .. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
           How to Remove Artifacts using Reference Noise Recordings.
           arXiv:2201.11798.
    .. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
           Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
           23(19), 8214. https://doi.org/10.3390/s23198214
    .. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
           Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
           https://doi.org/10.3390/s23020928
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
        global_threshold: float | str | None = None,
        global_clean_with: str | None = None,
        global_max_reject_fraction: float | None = None,
        verbose: bool | str | int | None = None,
    ):
        if ref_channels is None:
            raise ValueError("ref_channels must be provided explicitly")
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
        self.global_threshold = global_threshold
        self.global_clean_with = global_clean_with
        self.global_max_reject_fraction = global_max_reject_fraction
        self.verbose = verbose
        set_log_level_from_verbose(self.verbose)

    def fit(self, X: Any, y=None) -> ICanClean:
        """Fit is a no-op; included for sklearn compatibility.

        The actual computation happens in :meth:`transform` since ICanClean
        operates on a sliding-window basis and does not learn a single global
        decomposition.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data (unused aside from validation).
        y : None
            Ignored.

        Returns
        -------
        self : ICanClean
        """
        set_log_level_from_verbose(self.verbose)
        return self

    def transform(self, X: Any, y=None) -> Any:
        """Apply ICanClean artifact removal.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data to clean. Accepted formats:

            - MNE ``Raw``: channels are resolved by name.
            - MNE ``Epochs``: each epoch is cleaned individually.
            - ndarray, shape ``(n_channels, n_times)``: channel indices in
              ``ref_channels`` / ``primary_channels`` are used directly.

        y : None
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data in the same format as the input.
        """
        set_log_level_from_verbose(self.verbose)
        self._reset_qc_attrs()

        data, sfreq_data, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, auto_pick=False
        )
        sfreq = sfreq_data if sfreq_data is not None else self.sfreq
        channel_data = data[0] if mne_type == "epochs" else data
        primary_idx, ref_idx = self._resolve_channels(channel_data, orig_inst)

        if mne_type == "epochs":
            cleaned = self._transform_epochs(data, sfreq, primary_idx, ref_idx)
        else:
            cleaned = self._clean_continuous(data, sfreq, primary_idx, ref_idx)

        return reconstruct_mne_object(
            cleaned, orig_inst, mne_type, picks=picks, verbose=False
        )

    def fit_transform(self, X: Any, y=None, **fit_params) -> Any:
        """Fit and apply ICanClean in one step.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data.
        y : None
            Ignored.
        **fit_params
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data.
        """
        return self.fit(X, y).transform(X)

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
        if mne is not None and orig_inst is not None and hasattr(orig_inst, "ch_names"):
            ch_names = list(orig_inst.ch_names)

        if ch_names is not None and isinstance(self.ref_channels[0], str):
            ref_idx = np.array(
                [ch_names.index(ch) for ch in self.ref_channels],
                dtype=int,
            )
            ref_names = list(self.ref_channels)
        else:
            ref_idx = np.asarray(self.ref_channels, dtype=int)
            ref_names = (
                [ch_names[idx] for idx in ref_idx] if ch_names is not None else None
            )

        if self.primary_channels is not None:
            if ch_names is not None and isinstance(self.primary_channels[0], str):
                primary_idx = np.array(
                    [ch_names.index(ch) for ch in self.primary_channels],
                    dtype=int,
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
    ) -> np.ndarray:
        """Orchestrate CCA cleaning based on mode.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        sfreq : float
        primary_idx : ndarray of int
        ref_idx : ndarray of int

        Returns
        -------
        data_out : ndarray, shape (n_channels, n_times)
        """
        cleaned_primary, qc = self._compute_continuous_cleaning(
            data, sfreq, primary_idx, ref_idx
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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute cleaned primary channels and QC without mutating state."""
        if self.mode == "hybrid":
            cleaned_after_global, qc_global = compute_icanclean(
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
                verbose=self.verbose,
            )
            cleaned_primary, qc = compute_icanclean(
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
                verbose=self.verbose,
            )
            qc["global_correlations_"] = qc_global["correlations_"]
            qc["global_n_removed_"] = qc_global["n_removed_"]
            qc["global_removed_idx_"] = qc_global["removed_idx_"]
            qc["global_filters_"] = qc_global["filters_"]
            qc["global_patterns_"] = qc_global["patterns_"]
            qc["global_n_windows_"] = qc_global["n_windows_"]
        else:
            cleaned_primary, qc = compute_icanclean(
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
                verbose=self.verbose,
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
    if threshold != "auto":
        try:
            float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold must be a float or 'auto'") from exc
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
        if global_threshold != "auto":
            try:
                float(global_threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError("global_threshold must be a float or 'auto'") from exc
    elif has_global_params:
        raise ValueError(
            "global_threshold, global_clean_with, and "
            "global_max_reject_fraction are only supported when "
            "mode='hybrid'"
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
    """Apply average re-reference across channels for CCA input.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Windowed data used for the CCA decomposition.
    reref : bool or str
        ``False``: no re-referencing.
        ``True`` or ``'fullrank'``: full-rank average re-reference,
        implemented as :math:`I - 11^T / (n + 1)`.
        ``'loserank'``: standard average re-reference,
        implemented as :math:`I - 11^T / n`.

    Returns
    -------
    data_reref : ndarray, shape (n_samples, n_channels)
        Re-referenced data used only for CCA estimation.
    """
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
