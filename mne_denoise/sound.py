"""SOUND source-informed denoising."""

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
    """Estimate one channel noise level from covariance."""
    return float(np.sqrt(noise_filter @ cov @ noise_filter / n_times))


def _ddwiener(data: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
    """Estimate channel noise amplitudes with data-driven Wiener prediction."""
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
    """Validate SOUND data and lead-field inputs."""
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
    """Run SOUND iterations and return noise levels, convergence, and lead-field covariance."""
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
    """Compute the SOUND cleaning operator.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Sensor data in the same reference as leadfield.
    leadfield : ndarray, shape (n_channels, n_sources)
        Lead-field matrix in the same reference as data.
    lambda_ : float, default=0.1
        Non-negative regularization scale.
    n_iter : int, default=5
        Maximum number of iterations.
    tol : float or None, default=None
        Stop when the maximum relative noise-level change is below this value.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for channel-update order.
    callback : callable or None, default=None
        Synchronous callback after each iteration.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        Linear operator such that cleaned = operator @ data.
    sigmas : ndarray, shape (n_channels,)
        Estimated channel noise amplitudes.
    convergence : ndarray, shape (n_iter_run,)
        Maximum relative noise-level change per iteration.

    Notes
    -----
    The input data and lead field must use a common reference
    :footcite:p:`mutanen2018_sound,mutanen2022_source_artifact`.

    References
    ----------
    .. footbibliography::
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
    """Compute SOUND with a selected single-channel reference.

    The function excludes the least-noisy reference channel during estimation and
    returns an average-referenced full-channel operator.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Sensor data.
    leadfield : ndarray, shape (n_channels, n_sources)
        Lead-field matrix in the same channel order and reference as data.
    lambda_ : float, default=0.1
        Non-negative regularization scale.
    n_iter : int, default=5
        Maximum number of iterations.
    tol : float or None, default=None
        Convergence tolerance.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for channel-update order.
    callback : callable or None, default=None
        Synchronous callback after each iteration.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        Average-referenced cleaning operator.
    sigmas : ndarray, shape (n_channels - 1,)
        Noise amplitudes for channels other than best_channel.
    convergence : ndarray, shape (n_iter_run,)
        Maximum relative noise-level change per iteration.
    best_channel : int
        Index of the selected reference channel.

    Notes
    -----
    The reference channel is selected from the data-driven Wiener noise estimates;
    its index is returned so the reference choice remains explicit.
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
    """SOUND estimator for source-informed noise suppression.

    SOUND estimates channel noise levels and fits a forward-model-based linear
    operator. For compatible EEG MNE input with a montage, a spherical lead field
    is built when no forward solution is supplied. MEG or mixed-channel MNE input
    and NumPy input require an explicit forward solution.

    Parameters
    ----------
    lambda_ : float, default=0.1
        Non-negative regularization scale.
    n_iter : int, default=5
        Maximum number of iterations.
    tol : float or None, default=None
        Convergence tolerance; None runs n_iter iterations.
    forward : mne.Forward or None, default=None
        Optional explicit forward solution. For compatible EEG MNE input with a
        montage, None uses a spherical fallback; MEG or mixed-channel MNE input
        and NumPy input require an explicit forward.
    reference : {"best", "average"}, default="best"
        Reference handling. "best" selects a low-noise single-channel reference
        and reconstructs an average-referenced output; "average" uses all channels
        and assumes average-referenced input.
    sigma_source : {"evoked", "trials"}, default="evoked"
        For epoched data, estimate noise from the trial average or concatenated
        trials.
    n_dipoles : int, default=5000
        Number of dipoles for the spherical lead field.
    random_state : int, numpy.random.Generator, or None, default=None
        Random state for channel-update order.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    leadfield_ : ndarray
        Lead field used during fitting.
    operator_ : ndarray
        Fitted channel-space cleaning operator.
    sigmas_ : ndarray
        Estimated channel noise amplitudes.
    best_channel_ : int or None
        Selected reference channel, or None for reference="average".
    convergence_ : ndarray
        Relative noise-level change by iteration.

    See Also
    --------
    mne_denoise.sns.SNS
        Spatial-redundancy sensor-noise suppression without a lead field.
    compute_sound
        All-channel SOUND functional interface.
    compute_sound_ref_best
        Best-reference SOUND functional interface.

    Notes
    -----
    NumPy input is (n_channels, n_times) or (n_epochs, n_channels, n_times).
    MNE Raw, Epochs, and Evoked inputs are supported and returned without
    mutation :footcite:p:`mutanen2018_sound,mutanen2022_source_artifact`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    A preloaded MNE ``Raw`` object with a compatible EEG montage can use the
    spherical fallback lead field:

    .. code-block:: python

       from mne_denoise.sound import SOUND

       model = SOUND(reference="best")
       clean = model.fit_transform(raw)
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
        """Reduce fitting data to channel-by-time form according to sigma_source."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            return data
        if self.sigma_source == "evoked":
            return data.mean(axis=0)
        return epochs_to_continuous(data)

    def _warn_if_not_average_referenced(self, data, orig_inst):
        """Warn when reference="average" does not match the input metadata or common mode."""
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
        """Fit the SOUND cleaning operator.

        Parameters
        ----------
        X : ndarray, Raw, Epochs, or Evoked
            Data used to estimate channel noise and the operator.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous iteration callback.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        SOUND
            The fitted estimator.
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
        """Fit SOUND and transform the input.

        Parameters
        ----------
        X : ndarray, Raw, Epochs, or Evoked
            Data to fit and transform.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous fitting callback.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            Cleaned data.
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
        """Apply the fitted SOUND operator.

        Parameters
        ----------
        X : ndarray, Raw, Epochs, or Evoked
            Data with the fitted channel layout.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            Cleaned data; MNE inputs are copied and NumPy inputs are not mutated.
        """
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
