"""SOUND: automatic, robust noise suppression for EEG/MEG.

Implementation of the SOUND algorithm (Source-estimate-Utilizing
Noise-discarding) from Mutanen et al. (2018) [1]_, translated from the reference
MATLAB ``Sound-Demo-Package`` by Tuomas Mutanen. SOUND iteratively estimates the
noise amplitude of every channel by predicting it from all the other channels
through a forward model (a leave-one-channel-out minimum-norm estimate), then
applies a Wiener filter that suppresses channel-specific noise while preserving
signal that is consistent with the forward model. The covariance-form
iteration follows the later derivation by Mutanen et al. (2022) [2]_.
SOUND is also derived within a unified spatial-filtering framework in a broader
review of TMS-EEG artifact-removal methods [3]_.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J.
       (2018). Automatic and robust noise suppression in EEG and MEG: The
       SOUND algorithm. NeuroImage, 166, 135-151.
.. [2] Mutanen, T. P., Metsomaa, J., Makkonen, M., Varone, G., Marzetti, L.,
       & Ilmoniemi, R. J. (2022). Source-based artifact-rejection techniques
       for TMS-EEG. Journal of Neuroscience Methods, 382, 109693.
.. [3] Hernandez-Pavon, J. C., Kugiumtzis, D., Zrenner, C., Kimiskidis, V. K.,
       & Metsomaa, J. (2022). Removing artifacts from TMS-evoked EEG: A methods
       review and a unifying theoretical framework. Journal of Neuroscience
       Methods, 376, 109591.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._data import epochs_to_continuous, extract_data_from_mne, reconstruct_mne_object
from ._leadfield import _validate_leadfield, resolve_leadfield
from ._logging import logger, verbose
from ._validation import check_channel_layout, check_option, check_positive_real
from .progress import _emit_progress, _ProgressCallback, _validate_callback

__all__ = ["SOUND", "compute_sound", "compute_sound_ref_best"]


def _noise_level(noise_filter: np.ndarray, cov: np.ndarray, n_times: int) -> float:
    """Noise amplitude of one channel from the data covariance.

    Mutanen et al. (2022), Eqs. (35)-(36): the residual of predicting a channel
    from the others is ``w_N.T @ Y``, so its RMS is ``sqrt(w_N.T @ Cov(Y) @
    w_N)``. Evaluating it this way means the time samples are touched once, when
    the covariance is formed, rather than on every channel of every iteration --
    the paper notes this "may speed up the computational time by several orders
    of magnitude compared to sample-wise updates". It is an exact identity, not
    an approximation.
    """
    return float(np.sqrt(noise_filter @ cov @ noise_filter / n_times))


def _ddwiener(data: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
    """Data-driven Wiener noise-amplitude estimate (initial SOUND estimate).

    Predicts each channel from all the others using the data covariance and
    returns the residual RMS per channel. ``cov`` may be passed when the
    caller has already formed ``data @ data.T``; it is the one place the time
    samples are touched, and on long recordings it dominates the call.

    A channel that its neighbours predict exactly -- a flat or duplicated
    channel -- yields a zero estimate, which SOUND cannot divide by. Those
    entries are replaced with the mean of the usable ones, so downstream
    whitening treats them as merely average rather than infinitely reliable.
    The reference ``DDWiener.m`` has no such guard; it is the one place this
    port deliberately departs from it.
    """
    n_channels, n_times = data.shape
    if cov is None:
        cov = data @ data.T
    gamma = np.mean(np.diag(cov))
    if not gamma > 0:
        raise ValueError(
            "All channels have a zero noise estimate; SOUND cannot whiten the "
            "data. This usually means the input is constant or all-zero."
        )
    sigmas = np.empty(n_channels)
    eye = np.eye(n_channels - 1)
    for i in range(n_channels):
        others = np.concatenate([np.arange(i), np.arange(i + 1, n_channels)])
        weights = np.linalg.solve(
            cov[np.ix_(others, others)] + gamma * eye, cov[others, i]
        )
        noise_filter = np.zeros(n_channels)
        noise_filter[i] = 1.0
        noise_filter[others] = -weights
        sigmas[i] = _noise_level(noise_filter, cov, n_times)

    positive = sigmas > 0
    if not positive.any():
        raise ValueError(
            "All channels have a zero noise estimate; SOUND cannot whiten the "
            "data. This usually means the input is constant or all-zero."
        )
    return np.where(positive, sigmas, sigmas[positive].mean())


def _validate_sound_inputs(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    lambda_: float,
    n_iter: int,
    min_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce the inputs shared by both public SOUND solvers."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("data must contain only finite values.")
    leadfield = _validate_leadfield(leadfield)
    if (
        isinstance(lambda_, (bool, np.bool_))
        or not isinstance(lambda_, Real)
        or not np.isfinite(lambda_)
        or lambda_ < 0
    ):
        raise ValueError(
            f"lambda_ must be a finite non-negative number, got {lambda_!r}."
        )
    if (
        isinstance(n_iter, (bool, np.bool_))
        or not isinstance(n_iter, Integral)
        or n_iter < 1
    ):
        raise ValueError(f"n_iter must be a positive integer, got {n_iter!r}.")

    n_channels, n_times = data.shape
    if leadfield.shape[0] != n_channels:
        raise ValueError(
            f"Lead field has {leadfield.shape[0]} channels but data has {n_channels}."
        )
    if n_channels < min_channels:
        suffix = " (one is dropped as the reference)" if min_channels == 4 else ""
        raise ValueError(f"SOUND requires at least {min_channels} channels{suffix}.")
    if n_times == 0:
        raise ValueError("data must contain at least one sample.")
    if not np.any(data):
        raise ValueError(
            "All channels have a zero noise estimate because the data are "
            "all-zero; SOUND cannot be fitted."
        )
    return data, leadfield


def _estimate_sigmas(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    lambda_: float,
    n_iter: int,
    tol: float | None,
    random_state,
    callback: _ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SOUND's iteration and return ``(sigmas, convergence, llt)``.

    The shared core of :func:`compute_sound` and
    :func:`compute_sound_ref_best`, which differ only in the operator they
    build from the converged ``sigmas``. Returning ``llt`` lets the caller
    reuse it rather than re-forming ``leadfield @ leadfield.T``.

    Callers are expected to have validated shapes; only ``tol`` is checked
    here, since both entry points accept it unchanged.
    """
    n_channels, n_times = data.shape
    if tol is not None:
        try:
            tol = check_positive_real(tol, name="tol")
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"tol must be positive and finite or None, got {tol!r}."
            ) from err

    rng = np.random.default_rng(random_state)
    llt = leadfield @ leadfield.T  # (n_channels, n_channels); spans the column space
    # Time samples enter only here; the iterations below are covariance-only
    # (Mutanen et al. 2022, Eqs. 35-36). DDWiener needs the same covariance,
    # so it is formed once and shared.
    cov = data @ data.T

    sigmas = _ddwiener(data, cov=cov)
    eye = np.eye(n_channels - 1)
    convergence = []

    for _ in range(n_iter):
        sigmas_old = sigmas.copy()
        for i in rng.permutation(n_channels):
            others = np.concatenate([np.arange(i), np.arange(i + 1, n_channels)])
            w = 1.0 / sigmas[others]
            # WLLW = diag(w) @ LLt[o,o] @ diag(w)
            g = (w[:, None] * llt[np.ix_(others, others)]) * w[None, :]
            reg = lambda_ * np.trace(g) / (n_channels - 1)
            b = llt[i, others] * w  # (L @ WL.T)[i]
            try:
                m = np.linalg.solve(g + reg * eye, b)
            except np.linalg.LinAlgError as err:
                raise ValueError(
                    "SOUND encountered a singular lead-field system; provide a "
                    "positive lambda_ or a better-conditioned forward model."
                ) from err
            noise_filter = np.zeros(n_channels)
            noise_filter[i] = 1.0
            noise_filter[others] = -(m * w)
            sigmas[i] = _noise_level(noise_filter, cov, n_times)
        relative_change = np.max(np.abs(sigmas_old - sigmas) / sigmas_old)
        convergence.append(relative_change)
        _emit_progress(
            callback,
            method="sound",
            stage="iteration",
            current=len(convergence),
            total=n_iter,
            metric=float(relative_change),
        )
        logger.debug(
            "SOUND sigma iteration %d/%d: max relative change %.2e.",
            len(convergence),
            n_iter,
            relative_change,
        )
        if tol is not None and convergence[-1] < tol:
            break
    return sigmas, np.asarray(convergence), llt


@verbose
def compute_sound(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    lambda_: float = 0.1,
    n_iter: int = 5,
    tol: float | None = None,
    random_state=None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the SOUND cleaning operator from data and a lead field.

    This is the core solver: it assumes ``data`` and ``leadfield`` are
    already in a common reference. :class:`SOUND` handles the reference
    bookkeeping around it.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Sensor data, in the same reference as ``leadfield``.
    leadfield : ndarray, shape (n_channels, n_sources)
        Lead-field matrix, in the same reference as ``data``.
    lambda_ : float
        Regularisation parameter, the heuristic tuning scalar ``lambda_0`` of
        ``lambda = lambda_0 * trace(L~ L~.T) / C`` (Mutanen et al. 2022,
        Eq. 18). Larger values suppress noise more aggressively at the cost of
        distorting the signal. Default 0.1, as in TESA. Mutanen et al. (2022)
        [2]_ suggest ``lambda_0 = 1 / SNR`` as a starting point.
    n_iter : int
        Maximum number of SOUND iterations. Default 5.
    tol : float | None
        Convergence tolerance on the maximum relative change of ``sigmas``. If
        None (default), exactly ``n_iter`` iterations are run, matching
        ``SOUND.m`` and TESA. Set e.g. ``tol=0.01`` for the stopping rule used
        on real data in Mutanen et al. (2018) [1]_, which iterates "until the
        relative change in all the channels is less than 1%"; the algorithm
        statement in Mutanen et al. (2022) [2]_ combines both criteria.
    random_state : int | np.random.Generator | None
        Seed/generator controlling the random channel-update order.
    callback : callable | None
        Called synchronously after each completed SOUND sigma iteration with a
        progress event having ``method="sound"`` and ``stage="iteration"``.
        Callback return values are ignored and callback exceptions propagate
        unchanged. The event metric is the maximum relative sigma change.
    verbose : bool | str | int | None
        MNE-style logging level. The final convergence report is emitted at
        INFO and sigma iterations at DEBUG.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        The linear SOUND cleaning operator (``cleaned = operator @ data``).
    sigmas : ndarray, shape (n_channels,)
        Estimated per-channel noise amplitudes.
    convergence : ndarray, shape (n_iter_run,)
        Maximum relative change of ``sigmas`` at each iteration actually run.
        Shorter than ``n_iter`` if ``tol`` triggered an early stop.

    References
    ----------
    .. [1] Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J.
           (2018). Automatic and robust noise suppression in EEG and MEG: The
           SOUND algorithm. NeuroImage, 166, 135-151.
    .. [2] Mutanen, T. P., Metsomaa, J., Makkonen, M., Varone, G., Marzetti,
           L., & Ilmoniemi, R. J. (2022). Source-based artifact-rejection
           techniques for TMS-EEG. Journal of Neuroscience Methods, 382,
           109693.
    """
    callback = _validate_callback(callback)
    data, leadfield = _validate_sound_inputs(
        data, leadfield, lambda_=lambda_, n_iter=n_iter, min_channels=3
    )
    n_channels = data.shape[0]

    sigmas, convergence, llt = _estimate_sigmas(
        data,
        leadfield,
        lambda_=lambda_,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        callback=callback,
    )

    # Final cleaning operator from the converged noise estimate.
    w = 1.0 / sigmas
    g = (w[:, None] * llt) * w[None, :]
    reg = lambda_ * np.trace(g) / n_channels
    try:
        inv = np.linalg.inv(g + reg * np.eye(n_channels))
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "SOUND could not construct the final cleaning operator; provide a "
            "positive lambda_ or a better-conditioned forward model."
        ) from err
    operator = ((llt * w[None, :]) @ inv) * w[None, :]
    logger.info(
        "SOUND: %d iteration(s), channels=%d, sources=%d, "
        "final max relative sigma change %.2e, reference=average",
        convergence.size,
        n_channels,
        leadfield.shape[1],
        convergence[-1] if convergence.size else float("nan"),
    )
    return operator, sigmas, convergence


@verbose
def compute_sound_ref_best(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    lambda_: float = 0.1,
    n_iter: int = 5,
    tol: float | None = None,
    random_state=None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SOUND with the reference bookkeeping of ``tesa_sound`` [1]_.

    Runs :func:`compute_sound` against a single-channel reference chosen for
    minimum noise, then returns the montage to an average reference. See Notes
    for why the reference matters to the algorithm.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Sensor data.
    leadfield : ndarray, shape (n_channels, n_sources)
        Lead-field matrix in the same channel order as ``data``.
    lambda_ : float
        Regularisation parameter. Default 0.1.
    n_iter : int
        Maximum number of SOUND iterations. Default 5.
    tol : float | None
        Convergence tolerance; see :func:`compute_sound`.
    random_state : int | np.random.Generator | None
        Seed/generator controlling the random channel-update order.
    callback : callable | None
        Called synchronously after each completed SOUND sigma iteration with a
        progress event having ``method="sound"`` and ``stage="iteration"``.
        Callback return values are ignored and callback exceptions propagate
        unchanged. The event metric is the maximum relative sigma change.
    verbose : bool | str | int | None
        MNE-style logging level. The final convergence report is emitted at
        INFO and sigma iterations at DEBUG.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        Cleaning operator mapping the input data to average-referenced,
        SOUND-corrected data.
    sigmas : ndarray, shape (n_channels - 1,)
        Estimated noise amplitudes for the retained channels.
    convergence : ndarray, shape (n_iter,)
        Maximum relative change of ``sigmas`` at each iteration.
    best_channel : int
        Index of the reference channel chosen by DDWiener.

    Notes
    -----
    **Why the reference matters.** SOUND whitens with ``diag(1 / sigma)``, which
    is only a whitener if the sensor noise covariance is diagonal -- that is, if
    the channels' noise is mutually independent. Re-referencing destroys that
    independence, because every channel then shares a term with every other.

    Under an average reference, ``y_i - mean(y)`` mixes a fraction of all the
    noise into all the channels, and the resulting covariance has no structure
    the whitener can absorb. Under a *single-channel* reference ``y_i - y_k``
    the damage is far more tractable: the covariance becomes ``diag(sigma^2) +
    sigma_k^2 * ones``, so the entire violation is one common term of size
    ``sigma_k^2``, shared equally by every pair. It cannot be removed, but it
    can be minimised -- which is why the reference is the channel DDWiener
    estimates to be *least* noisy, rather than an anatomically motivated one.
    Channel ``k`` itself becomes identically zero once referenced to itself, so
    it carries no information and is dropped from the estimation; its noise
    estimate would be zero and the whitener would divide by it.

    **Getting the montage back.** Dropping a channel and referencing to another
    would leave the output in a reference nobody asked for, one channel short.
    The reconstruction step avoids that by going through the source estimate:
    sources are estimated from the referenced data with the referenced lead
    field, then projected back out through an *average-referenced* lead field.
    The output is therefore average referenced, and channel ``k`` reappears --
    predicted from the sources rather than measured.

    **Why it is still one matrix.** Every step is linear, so the chain
    (reference, drop, whiten, regularised inverse, forward-project) collapses
    to a single operator::

        operator = L_avg @ L_ref.T @ W @ inv(W @ L_ref @ L_ref.T @ W + reg * I)
                   @ W @ R

    with ``W = diag(1 / sigma)`` and ``R`` the ``(n - 1, n)`` matrix that
    applies the reference and removes channel ``k``. Written this way the
    intermediates are ``(n, n - 1)`` and never reach ``n_sources``, so cost
    scales with channels rather than with the size of the source space.

    Unlike the plain :func:`compute_sound` operator, this one is **not
    symmetric**: it enters in one reference and leaves in another.

    References
    ----------
    .. [1] Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J.
           (2018). Automatic and robust noise suppression in EEG and MEG: The
           SOUND algorithm. NeuroImage, 166, 135-151.
    """
    callback = _validate_callback(callback)
    data, leadfield = _validate_sound_inputs(
        data, leadfield, lambda_=lambda_, n_iter=n_iter, min_channels=4
    )
    n_channels = data.shape[0]

    best = int(np.argmin(_ddwiener(data)))
    keep = np.array([i for i in range(n_channels) if i != best])
    data_ref = (data - data[best])[keep]
    lf_ref = (leadfield - leadfield[best])[keep]
    lf_avg = leadfield - leadfield.mean(axis=0, keepdims=True)

    sigmas, convergence, llt_ref = _estimate_sigmas(
        data_ref,
        lf_ref,
        lambda_=lambda_,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        callback=callback,
    )

    # Fold ref_best + drop + MNE + average-referenced reconstruction into one
    # operator. All products stay in channel space: (n, n-1) rather than
    # (n, n_sources).
    w = 1.0 / sigmas
    g = (w[:, None] * llt_ref) * w[None, :]
    reg = lambda_ * np.trace(g) / (n_channels - 1)
    # R applies ref_best and drops the reference channel: (n - 1, n).
    r = np.eye(n_channels)[keep] - np.eye(n_channels)[best][None, :]
    cross = lf_avg @ lf_ref.T  # (n, n - 1)
    try:
        reconstruction = np.linalg.solve(
            g + reg * np.eye(n_channels - 1), w[:, None] * r
        )
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "SOUND could not construct the final cleaning operator; provide a "
            "positive lambda_ or a better-conditioned forward model."
        ) from err
    operator = cross * w[None, :] @ reconstruction
    logger.info(
        "SOUND: %d iteration(s), channels=%d, sources=%d, "
        "final max relative sigma change %.2e, reference=best channel %d",
        convergence.size,
        n_channels,
        leadfield.shape[1],
        convergence[-1] if convergence.size else float("nan"),
        best,
    )
    return operator, sigmas, convergence, best


class SOUND(BaseEstimator, TransformerMixin):
    """Automatic, robust noise suppression for EEG/MEG (SOUND).

    SOUND estimates the noise level of every sensor and applies a forward-model
    Wiener filter that suppresses channel-specific noise while preserving
    field-consistent brain signal. With no individualised forward model, a
    three-layer spherical lead field is built from the montage.

    Parameters
    ----------
    lambda_ : float
        Regularisation parameter. Default 0.1.
    n_iter : int
        Maximum number of SOUND iterations. Default 5.
    tol : float | None
        Convergence tolerance on the maximum relative change of ``sigmas_``. If
        None (default), exactly ``n_iter`` iterations run, matching TESA. Set
        ``tol=0.01`` to reproduce the 1% stopping rule Mutanen et al. (2018)
        [1]_ use on real data.
    forward : mne.Forward | None
        Optional pre-computed forward solution. If None, a spherical lead field
        is built from the data's montage.
    reference : {'best', 'average'}
        Reference handling. ``'best'`` (default) follows ``tesa_sound``:
        re-reference to the least noisy channel, drop it from the estimation,
        and reconstruct the full average-referenced montage from the source
        estimate. This keeps SOUND's diagonal-noise assumption intact.
        ``'average'`` runs the plain ``SOUND.m`` solver on all channels and
        assumes the input is already average referenced.
    sigma_source : {'evoked', 'trials'}
        For epoched input, whether to estimate the per-channel noise from the
        trial average (``'evoked'``, the default, as in ``tesa_sound``) or from
        all trials concatenated in time. Only the *relative* pattern of
        ``sigmas_`` affects the operator -- a global rescaling cancels -- so
        this matters mainly when artifacts are trial-locked. Ignored for
        continuous input.
    n_dipoles : int
        Number of dipoles for the spherical lead field.
    random_state : int | np.random.Generator | None
        Controls the random channel-update order for reproducibility.
    verbose : bool | str | int | None, default=None
        MNE-style logging level. The fitted SOUND summary is emitted at INFO;
        sigma iterations are emitted at DEBUG.

    Attributes
    ----------
    leadfield_ : ndarray, shape (n_channels, n_dipoles)
        The average-referenced lead field used.
    operator_ : ndarray, shape (n_channels, n_channels)
        The fitted linear cleaning operator.
    sigmas_ : ndarray
        Estimated per-channel noise amplitudes. With ``reference='best'`` this
        has ``n_channels - 1`` entries, for the channels other than
        ``best_channel_``.
    best_channel_ : int | None
        Index of the reference channel chosen by DDWiener; None when
        ``reference='average'``.
    convergence_ : ndarray
        Maximum relative change of ``sigmas_`` per iteration actually run.

    References
    ----------
    .. [1] Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J.
           (2018). Automatic and robust noise suppression in EEG and MEG: The
           SOUND algorithm. NeuroImage, 166, 135-151.
    """

    def __init__(
        self,
        *,
        lambda_: float = 0.1,
        n_iter: int = 5,
        tol: float | None = None,
        forward=None,
        reference: str = "best",
        sigma_source: str = "evoked",
        n_dipoles: int = 5000,
        random_state=None,
        verbose: bool | str | int | None = None,
    ):
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.tol = tol
        self.forward = forward
        self.reference = reference
        self.sigma_source = sigma_source
        self.n_dipoles = n_dipoles
        self.random_state = random_state
        self.verbose = verbose

    def _fit_data(self, data):
        """Reduce input to the (n_channels, n_times) matrix sigmas come from.

        Epoched input offers two readings of "the noise": averaging first
        (``sigma_source='evoked'``) measures what survives averaging, so the
        estimate tracks the evoked response; concatenating trials
        (``'trials'``) measures single-trial noise, which is larger and less
        sensitive to trial count.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            return data
        if self.sigma_source == "evoked":
            return data.mean(axis=0)
        return epochs_to_continuous(data)

    def _warn_if_not_average_referenced(self, data, orig_inst):
        """Warn when ``reference='average'`` meets data that is not.

        Two independent signals. ``custom_ref_applied`` is authoritative:
        MNE sets it when a *non*-average reference is applied and clears it
        when an average one is (see ``mne/_fiff/reference.py``), so a true
        flag is a statement that the montage is referenced to something else.
        The common-mode check is the fallback for data that reached us
        without that history -- a plain array, or a montage referenced
        outside MNE.
        """
        if self.reference != "average":
            return
        custom_ref = orig_inst is not None and bool(
            orig_inst.info.get("custom_ref_applied", False)
        )
        offset = np.abs(data.mean(axis=0)).max() / (np.abs(data).max() or 1.0)
        if custom_ref or offset > 1e-6:
            warnings.warn(
                "reference='average' assumes average-referenced input, but "
                "this data does not appear to be average referenced. The lead "
                "field is average referenced, so a mismatched data reference "
                "will bias the result. Apply an average reference first, or "
                "use reference='best'.",
                RuntimeWarning,
                stacklevel=3,
            )

    @verbose
    def fit(
        self,
        X,
        y=None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ):
        """Estimate the SOUND cleaning operator from ``X``.

        Parameters
        ----------
        callback : callable | None
            Called synchronously after each completed SOUND sigma iteration
            with a progress event whose ``method`` is ``"sound"`` and whose
            ``stage`` is ``"iteration"``. Callback return values are ignored
            and callback exceptions propagate unchanged. The event metric is
            the maximum relative sigma change.
        """
        callback = _validate_callback(callback)
        check_option(self.reference, name="reference", allowed=("best", "average"))
        check_option(
            self.sigma_source, name="sigma_source", allowed=("evoked", "trials")
        )
        data, _, _, orig_inst, _, ch_names = extract_data_from_mne(X)
        n_channels = data.shape[-2]  # (..., n_channels, n_times)
        self.leadfield_ = resolve_leadfield(
            inst=orig_inst,
            ch_names=ch_names,
            n_channels=n_channels,
            method="SOUND",
            forward=self.forward,
            n_dipoles=self.n_dipoles,
        )
        self._mne_ch_names_ = ch_names

        fit_data = self._fit_data(data)
        self._warn_if_not_average_referenced(fit_data, orig_inst)

        if self.reference == "best":
            (
                self.operator_,
                self.sigmas_,
                self.convergence_,
                self.best_channel_,
            ) = compute_sound_ref_best(
                fit_data,
                self.leadfield_,
                lambda_=self.lambda_,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
                callback=callback,
            )
        else:
            self.operator_, self.sigmas_, self.convergence_ = compute_sound(
                fit_data,
                self.leadfield_,
                lambda_=self.lambda_,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
                callback=callback,
            )
            self.best_channel_ = None
        return self

    @verbose
    def fit_transform(
        self,
        X,
        y=None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ):
        """Fit SOUND and apply the fitted operator to ``X``.

        Parameters
        ----------
        callback : callable | None
            Called synchronously after each completed SOUND sigma iteration
            during fitting. Callback return values are ignored and callback
            exceptions propagate unchanged.
        verbose : bool | str | int | None
            MNE-style logging level for the composed fit and transform call.
        """
        callback = _validate_callback(callback)
        return self.fit(X, y=y, callback=callback).transform(X)

    @verbose
    def transform(
        self,
        X,
        *,
        verbose: bool | str | int | None = None,
    ):
        """Apply the fitted SOUND operator to ``X``."""
        check_is_fitted(self, attributes=["operator_"])
        data, _, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, ch_names=getattr(self, "_mne_ch_names_", None)
        )
        check_channel_layout(
            "SOUND",
            n_channels=data.shape[-2],
            fitted_n_channels=self.operator_.shape[1],
            ch_names=ch_names,
            fitted_ch_names=getattr(self, "_mne_ch_names_", None),
        )
        # matmul broadcasts the (n_channels, n_channels) operator over any
        # leading epoch axis, so 2D and 3D share one BLAS-backed path.
        cleaned = self.operator_ @ data
        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)
