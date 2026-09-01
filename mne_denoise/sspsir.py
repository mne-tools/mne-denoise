"""SSP-SIR source-informed reconstruction."""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _mne
from ._data import extract_data_from_mne, reconstruct_mne_object
from ._leadfield import _validate_leadfield, resolve_leadfield
from ._logging import logger, verbose
from ._validation import (
    check_channel_layout,
    check_matching_sfreq,
    check_option,
    check_positive_real,
)

__all__ = ["SSPSIR", "compute_sir", "compute_sspsir"]

#: 10-90% transition width (s) of the crossfade around a user artifact window.
_SMOOTH_LENGTH = 0.010


def _truncated_svd(
    matrix: np.ndarray, M: int, what: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return leading singular triplets up to the requested numerical rank."""
    if isinstance(M, (bool, np.bool_)) or not isinstance(M, Integral) or M < 1:
        raise ValueError(f"M must be a positive integer, got {M!r}.")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    requested_rank = int(M)
    tol = s[0] * max(matrix.shape) * np.finfo(s.dtype).eps if s.size else 0.0
    numerical_rank = int(np.count_nonzero(s > tol))
    if numerical_rank == 0:
        raise ValueError(f"Cannot reconstruct from {what}: its numerical rank is zero.")
    if numerical_rank < requested_rank:
        warnings.warn(
            f"M={M} exceeds the numerical rank ({numerical_rank}) of {what}; "
            f"using M={numerical_rank} instead.",
            RuntimeWarning,
            stacklevel=3,
        )
    rank = min(requested_rank, numerical_rank)
    return u[:, :rank], s[:rank], vt[:rank]


def _artifact_subspace(
    svd_input: np.ndarray, n_components
) -> tuple[np.ndarray, int, np.ndarray]:
    """Estimate the high-frequency artifact subspace and component count."""
    svd_input = np.asarray(svd_input, dtype=float)
    if svd_input.ndim != 2 or 0 in svd_input.shape:
        raise ValueError(
            f"svd_input must be a non-empty 2D array, got shape {svd_input.shape}."
        )
    if not np.isfinite(svd_input).all():
        raise ValueError("svd_input must contain only finite values.")
    if isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, Real):
        raise ValueError(
            "n_components must be a positive integer or a variance fraction in (0, 1)."
        )

    is_count = isinstance(n_components, Integral)
    if is_count:
        n_pc = int(n_components)
        if n_pc < 1:
            raise ValueError(
                f"n_components must be a positive integer, got {n_components}."
            )
    elif not 0.0 < float(n_components) < 1.0:
        raise ValueError(
            "A floating-point n_components must be a variance fraction in "
            f"(0, 1), got {n_components}."
        )

    u, s, _ = np.linalg.svd(svd_input, full_matrices=False)
    if s[0] == 0.0:
        raise ValueError("Cannot estimate an artifact subspace from all-zero data.")
    if is_count:
        if n_pc > u.shape[1]:
            raise ValueError(
                f"n_components={n_pc} exceeds the {u.shape[1]} available "
                "artifact dimensions."
            )
    else:
        power = (s / s[0]) ** 2
        n_pc = int(
            np.searchsorted(np.cumsum(power), float(n_components) * power.sum()) + 1
        )
    if n_pc >= svd_input.shape[0]:
        raise ValueError(
            "n_components must leave at least one channel dimension outside "
            "the artifact subspace."
        )
    return u[:, :n_pc], n_pc, s


def compute_sspsir(
    leadfield: np.ndarray, artifact_topographies: np.ndarray, M: int
) -> np.ndarray:
    """Build the projected SSP-SIR reconstruction operator.

    Parameters
    ----------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead field.
    artifact_topographies : ndarray, shape (n_channels, n_components)
        Orthonormal artifact subspace.
    M : int
        Source-informed reconstruction rank.

    Returns
    -------
    ndarray, shape (n_channels, n_channels)
        Artifact-suppressing operator.

    See Also
    --------
    SSPSIR
        Estimator that fits the artifact subspace and applies the operator.
    compute_sir
        Unprojected source-informed reconstruction operator.

    Notes
    -----
    This operator implements the projected SSP-SIR reconstruction
    :footcite:p:`mutanen2016_sspsir`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.sspsir import compute_sspsir
    >>> rng = np.random.default_rng(0)
    >>> leadfield = rng.standard_normal((8, 16))
    >>> leadfield -= leadfield.mean(axis=0, keepdims=True)
    >>> artifact_topographies = rng.standard_normal((8, 2))
    >>> artifact_topographies -= artifact_topographies.mean(axis=0, keepdims=True)
    >>> artifact_topographies = np.linalg.qr(artifact_topographies)[0][:, :2]
    >>> operator = compute_sspsir(leadfield, artifact_topographies, M=3)
    """
    leadfield = _validate_leadfield(leadfield)
    artifact_topographies = np.asarray(artifact_topographies, dtype=float)
    if artifact_topographies.ndim != 2:
        raise ValueError(
            "artifact_topographies must be 2D, got shape "
            f"{artifact_topographies.shape}."
        )
    if not np.isfinite(artifact_topographies).all():
        raise ValueError("artifact_topographies must contain only finite values.")
    n_channels = leadfield.shape[0]
    if artifact_topographies.shape[0] != n_channels:
        raise ValueError(
            f"Lead field has {n_channels} channels but the artifact "
            f"topographies have {artifact_topographies.shape[0]}."
        )
    n_artifact = artifact_topographies.shape[1]
    if not 1 <= n_artifact < n_channels:
        raise ValueError(
            "artifact_topographies must contain between 1 and n_channels - 1 "
            "components."
        )
    gram = artifact_topographies.T @ artifact_topographies
    if not np.allclose(gram, np.eye(n_artifact), rtol=1e-7, atol=1e-9):
        raise ValueError("artifact_topographies must have orthonormal columns.")
    proj = np.eye(n_channels) - artifact_topographies @ artifact_topographies.T
    pl = proj @ leadfield
    u, s, vt = _truncated_svd(pl, M, "the projected lead field")
    return leadfield @ (vt.T / s) @ u.T @ proj


def compute_sir(leadfield: np.ndarray, M: int) -> np.ndarray:
    """Build the unprojected source-informed reconstruction operator.

    Parameters
    ----------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead field.
    M : int
        Source-informed reconstruction rank.

    Returns
    -------
    ndarray, shape (n_channels, n_channels)
        Rank-truncated source-informed operator.

    See Also
    --------
    SSPSIR
        Estimator that fits the artifact subspace and applies source-informed
        reconstruction.
    compute_sspsir
        Projected source-informed reconstruction operator.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.sspsir import compute_sir
    >>> rng = np.random.default_rng(0)
    >>> leadfield = rng.standard_normal((8, 16))
    >>> leadfield -= leadfield.mean(axis=0, keepdims=True)
    >>> operator = compute_sir(leadfield, M=3)
    """
    leadfield = _validate_leadfield(leadfield)
    u, _, _ = _truncated_svd(leadfield, M, "the lead field")
    return u @ u.T


def _as_mne_projections(topographies: np.ndarray, ch_names: list[str]) -> list:
    """Wrap artifact topographies as MNE projection objects."""
    _mne.require_mne("SSP-SIR artifact projections")

    return [
        _mne.mne.Projection(
            data={
                "data": topographies[:, i : i + 1].T,
                "col_names": list(ch_names),
                "row_names": None,
                "ncol": len(ch_names),
                "nrow": 1,
            },
            active=False,
            desc=f"SSP-SIR artifact {i + 1}",
            kind=1,  # FIFFV_PROJ_ITEM_FIELD
            explained_var=None,
        )
        for i in range(topographies.shape[1])
    ]


class SSPSIR(BaseEstimator, TransformerMixin):
    """Source-informed signal-space projection for TMS-evoked muscle artifact removal.

    SSP-SIR projects out an artifact subspace, reconstructs through a lead field,
    and blends the projected and unprojected reconstructions around the artifact
    window.

    Parameters
    ----------
    n_components : int or float
        Number of artifact components, or high-frequency variance fraction in (0, 1).
    forward : mne.Forward or None, default=None
        Optional explicit forward solution. For compatible EEG MNE input with a
        montage, None uses a spherical fallback; MEG or mixed-channel MNE input
        and NumPy input require an explicit forward.
    art_window : tuple of float or None, default=None
        Artifact interval (tmin, tmax) in seconds; None derives a high-frequency
        envelope from the epoch.
    blend : {"auto", "constant"}, default="auto"
        Crossfade rule.
    high_pass : float, default=100.0
        High-pass cutoff in Hz for artifact-subspace estimation.
    M : int or None, default=None
        Source-informed reconstruction rank; None uses data rank minus artifact rank.
    smooth_length : float, default=0.010
        Crossfade transition width in seconds.
    sfreq : float or None, default=None
        Sampling frequency for NumPy input.
    n_dipoles : int, default=5000
        Dipoles for a generated spherical EEG lead field.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    leadfield_ : ndarray
        Lead field used during fitting.
    artifact_topographies_ : ndarray
        Fitted artifact subspace.
    operator_ : ndarray
        Projected reconstruction operator.
    operator_orig_ : ndarray
        Unprojected reconstruction operator.
    kernel_ : ndarray
        Temporal crossfade weights.
    n_components_ : int
        Effective artifact-component count.
    M_ : int
        Effective reconstruction rank.
    singular_values_ : ndarray
        Singular values used for artifact-subspace selection.
    projs_ : list of mne.Projection
        Artifact directions when fitted on named MNE data.

    See Also
    --------
    compute_sspsir
        Construct the projected source-informed reconstruction operator.
    compute_sir
        Construct the unprojected source-informed reconstruction operator.
    mne_denoise.sound.SOUND
        Another forward-model-based denoising method with a different noise model.

    Notes
    -----
    NumPy input uses (n_channels, n_times) or (n_epochs, n_channels, n_times).
    MNE Raw, Epochs, and Evoked inputs are supported and returned without mutation.
    The artifact-window reconstruction is time-locked to the fitted data
    :footcite:p:`mutanen2016_sspsir,mutanen2022_source_artifact,mutanen2024_sspsir_simulation,hernandez_pavon2022_tms_review`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    A preloaded MNE ``Epochs`` object with a compatible EEG montage and a
    sampling rate whose Nyquist frequency is above the configured high-pass
    cutoff can use the spherical fallback:

    .. code-block:: python

       from mne_denoise.sspsir import SSPSIR

       model = SSPSIR(
           n_components=3,
           art_window=(0.005, 0.050),
       )
       clean = model.fit_transform(epochs)
    """

    def __init__(
        self,
        *,
        n_components=None,
        forward=None,
        art_window=None,
        blend: str = "auto",
        high_pass: float = 100.0,
        M=None,
        smooth_length: float = _SMOOTH_LENGTH,
        sfreq=None,
        n_dipoles: int = 5000,
        verbose: bool | str | int | None = None,
    ):
        self.n_components = n_components
        self.forward = forward
        self.art_window = art_window
        self.blend = blend
        self.high_pass = high_pass
        self.M = M
        self.smooth_length = smooth_length
        self.sfreq = sfreq
        self.n_dipoles = n_dipoles
        self.verbose = verbose

    def _resolve_sfreq(self, sfreq):
        sfreq = sfreq if sfreq is not None else self.sfreq
        if sfreq is None:
            raise ValueError(
                "SSP-SIR needs a sampling frequency: pass an MNE object or set sfreq."
            )
        if isinstance(sfreq, (bool, np.bool_)) or not isinstance(sfreq, Real):
            raise ValueError(f"sfreq must be a positive finite number, got {sfreq!r}.")
        sfreq = float(sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0.0:
            raise ValueError(f"sfreq must be a positive finite number, got {sfreq!r}.")
        if not 0.0 < float(self.high_pass) < sfreq / 2.0:
            raise ValueError(
                f"high_pass ({self.high_pass} Hz) must be between 0 and the "
                f"Nyquist frequency ({sfreq / 2.0} Hz) for sfreq={sfreq} Hz."
            )
        return sfreq

    def _svd_input(self, evoked, sfreq, times):
        """Filter the evoked data and construct the artifact-subspace input and crossfade."""
        b, a = butter(2, self.high_pass / (sfreq / 2.0), btype="high")
        data_high = filtfilt(b, a, evoked, axis=1)

        if self.art_window is not None:
            tmin, tmax = self.art_window
            mask = (times >= tmin) & (times <= tmax)
            if not mask.any():
                raise ValueError("art_window does not overlap the data time range.")
            # Smooth step function around the artifact window.
            slope = 4.0 / float(self.smooth_length)
            kernel = expit(slope * (times - tmin)) - expit(slope * (times - tmax))
            return data_high[:, mask], kernel

        # Weight by a 50 ms sliding RMS of high-frequency power.
        win = max(1, int(round(sfreq / 1000.0 * 50.0)))
        power = uniform_filter1d(data_high**2, size=win, axis=1, mode="nearest")
        kernel = power.mean(axis=0)
        kernel = np.sqrt(kernel / kernel.max()) if kernel.max() > 0 else kernel
        return kernel[None, :] * data_high, kernel

    @verbose
    def fit(
        self,
        X,
        y=None,
        *,
        verbose: bool | str | int | None = None,
    ):
        """Fit the SSP-SIR artifact subspace and reconstruction operators.

        Parameters
        ----------
        X : ndarray, Raw, Epochs, or Evoked
            EEG data used for fitting. NumPy input is channel-first.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        SSPSIR
            The fitted estimator.
        """
        if self.n_components is None:
            raise ValueError(
                "n_components must be set (number of artifact PCs to remove, or a "
                "variance fraction in (0, 1))."
            )
        check_option(self.blend, name="blend", allowed=("auto", "constant"))
        check_positive_real(self.high_pass, name="high_pass")
        check_positive_real(self.smooth_length, name="smooth_length")
        if isinstance(self.n_dipoles, (bool, np.bool_)) or not isinstance(
            self.n_dipoles, Integral
        ):
            raise ValueError(
                f"n_dipoles must be a positive integer, got {self.n_dipoles!r}."
            )
        if self.n_dipoles < 1:
            raise ValueError(
                f"n_dipoles must be a positive integer, got {self.n_dipoles!r}."
            )
        if self.M is not None and (
            isinstance(self.M, (bool, np.bool_))
            or not isinstance(self.M, Integral)
            or self.M < 1
        ):
            raise ValueError(f"M must be a positive integer or None, got {self.M!r}.")
        if self.art_window is not None:
            if not isinstance(self.art_window, tuple) or len(self.art_window) != 2:
                raise ValueError("art_window must be a (tmin, tmax) tuple or None.")
            tmin, tmax = self.art_window
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                for value in (tmin, tmax)
            ):
                raise ValueError("art_window values must be finite numbers.")
            if tmin >= tmax:
                raise ValueError(
                    f"art_window must satisfy tmin < tmax, got {self.art_window}."
                )
        data, sfreq, _, orig_inst, _, ch_names = extract_data_from_mne(X)
        sfreq = self._resolve_sfreq(sfreq)
        times = getattr(orig_inst, "times", None)

        # Average reference and an evoked (trial-averaged) view for the subspace.
        evoked = data.mean(axis=0) if data.ndim == 3 else np.asarray(data, float)
        evoked = evoked - evoked.mean(axis=0, keepdims=True)
        n_channels = evoked.shape[0]
        if times is None:
            times = np.arange(evoked.shape[1], dtype=float) / sfreq
        else:
            times = np.asarray(times, dtype=float)

        svd_input, kernel = self._svd_input(evoked, sfreq, times)
        (
            self.artifact_topographies_,
            self.n_components_,
            self.singular_values_,
        ) = _artifact_subspace(svd_input, self.n_components)

        self.leadfield_ = resolve_leadfield(
            inst=orig_inst,
            ch_names=ch_names,
            n_channels=n_channels,
            method="SSP-SIR",
            forward=self.forward,
            n_dipoles=self.n_dipoles,
        )

        data_rank = int(np.linalg.matrix_rank(evoked))
        M = self.M if self.M is not None else max(1, data_rank - self.n_components_)
        self.operator_ = compute_sspsir(self.leadfield_, self.artifact_topographies_, M)
        self.M_ = int(np.linalg.matrix_rank(self.operator_))
        self.operator_orig_ = compute_sir(self.leadfield_, self.M_)
        self.kernel_ = (
            np.ones(evoked.shape[1])
            if self.blend == "constant"
            else np.clip(kernel, 0.0, 1.0)
        )
        self.sfreq_ = sfreq
        self.times_ = times.copy()
        self._mne_ch_names_ = ch_names
        self.projs_ = (
            _as_mne_projections(self.artifact_topographies_, ch_names)
            if ch_names is not None
            else []
        )
        logger.info(
            "SSP-SIR: channels=%d, removed %d artifact component(s), "
            "SIR truncation M=%d (data rank %d), blend=%s",
            n_channels,
            self.n_components_,
            self.M_,
            data_rank,
            self.blend,
        )
        return self

    @verbose
    def transform(
        self,
        X,
        *,
        verbose: bool | str | int | None = None,
    ):
        """Apply the fitted SSP-SIR operators.

        Parameters
        ----------
        X : ndarray, Raw, Epochs, or Evoked
            Data with the fitted channel layout and, for time-varying blending, time axis.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            Reconstructed data; non-fitted MNE channels are preserved.
        """
        check_is_fitted(self, attributes=["operator_", "operator_orig_", "kernel_"])
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, ch_names=getattr(self, "_mne_ch_names_", None)
        )
        check_channel_layout(
            "SSPSIR",
            n_channels=data.shape[-2],
            fitted_n_channels=self.operator_.shape[1],
            ch_names=ch_names,
            fitted_ch_names=getattr(self, "_mne_ch_names_", None),
        )
        n_times = data.shape[-1]
        if self.blend == "constant":
            # This branch is spatially and temporally invariant.
            cleaned = self.operator_ @ data
            return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)
        if n_times != self.kernel_.size:
            raise ValueError(
                f"SSPSIR was fitted on {self.kernel_.size} time points but got "
                f"{n_times}. The crossfade kernel is time-locked to the fitted "
                "data; refit on this data, or use blend='constant' to apply the "
                "projection uniformly over time."
            )
        check_matching_sfreq(
            sfreq,
            self.sfreq_,
            name="SSPSIR",
            rtol=0.0,
            atol=1e-12,
        )
        times = getattr(orig_inst, "times", None)
        if times is not None and not np.allclose(
            np.asarray(times), self.times_, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "SSPSIR transform data have a different time axis from the fitted "
                "data. Refit on this data, or use blend='constant'."
            )
        # matmul broadcasts the (n_channels, n_channels) operators over any
        # leading epoch axis, so 2D and 3D share one BLAS-backed path.
        suppressed = self.operator_ @ data
        original = self.operator_orig_ @ data
        cleaned = self.kernel_ * suppressed + (1.0 - self.kernel_) * original
        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)
