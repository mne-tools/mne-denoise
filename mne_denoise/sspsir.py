"""SSP-SIR: suppress TMS-evoked muscle artifacts from EEG.

Implements signal-space projection--source-informed reconstruction (SSP-SIR)
[1]_. The method projects out the high-variance muscle-artifact subspace and
reconstructs the brain signal lost to that projection through a forward model.
Temporal blending can restrict the suppression to the artifact window [2]_.
Broader reviews discuss SSP-SIR as a source-based method [3]_ and within a
unified framework for TMS-EEG artifact removal [4]_.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Mutanen, T. P., Kukkonen, M., Nieminen, J. O., Stenroos, M., Sarvas,
       J., & Ilmoniemi, R. J. (2016). Recovering TMS-evoked EEG responses
       masked by muscle artifacts. NeuroImage, 139, 157-166.
.. [2] Mutanen, T. P., Ilmoniemi, I., Atti, I., Metsomaa, J., & Ilmoniemi,
       R. J. (2024). A simulation study: comparing independent component
       analysis and signal-space projection - source-informed reconstruction
       for rejecting muscle artifacts evoked by transcranial magnetic
       stimulation. Frontiers in Human Neuroscience, 18, 1324958.
.. [3] Mutanen, T. P., Metsomaa, J., Makkonen, M., Varone, G., Marzetti, L.,
       & Ilmoniemi, R. J. (2022). Source-based artifact-rejection techniques
       for TMS-EEG. Journal of Neuroscience Methods, 382, 109693.
.. [4] Hernandez-Pavon, J. C., Kugiumtzis, D., Zrenner, C., Kimiskidis, V. K.,
       & Metsomaa, J. (2022). Removing artifacts from TMS-evoked EEG: A methods
       review and a unifying theoretical framework. Journal of Neuroscience
       Methods, 376, 109591.
"""

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
    """Return the leading singular triplets, limited to numerical rank."""
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
    """Left singular vectors of ``svd_input`` and the number of artifact PCs.

    ``n_components`` is either an int (number of PCs) or a float in (0, 1)
    (cumulative high-frequency variance fraction).
    """
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
    """Build the artifact-suppressing SSP-SIR operator (``cleaned = C @ data``).

    This is the projected branch of SSP-SIR: project the artifact subspace out,
    then reconstruct through the forward model. In the full method it is
    crossfaded against :func:`compute_sir`; see :class:`SSPSIR`.

    Parameters
    ----------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead field.
    artifact_topographies : ndarray, shape (n_channels, n_components)
        Orthonormal artifact subspace (left singular vectors).
    M : int
        Truncation dimension of the source-informed reconstruction.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        The artifact-suppressing SSP-SIR operator.
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
    """Build the source-informed reconstruction operator without projection.

    A rank-``M`` reconstruction of the data through the forward model, with no
    artifact subspace removed. It is
    *not* the identity -- it restricts the data to the ``M`` leading
    lead-field topographies -- so the crossfade in :class:`SSPSIR` applies the
    same rank truncation inside and outside the artifact window.

    Parameters
    ----------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead field.
    M : int
        Truncation dimension of the source-informed reconstruction.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        The unprojected source-informed reconstruction operator.
    """
    leadfield = _validate_leadfield(leadfield)
    u, _, _ = _truncated_svd(leadfield, M, "the lead field")
    return u @ u.T


def _as_mne_projections(topographies: np.ndarray, ch_names: list[str]) -> list:
    """Wrap artifact topographies as :class:`mne.Projection` objects.

    Exposing them in MNE's own container means the directions SSP-SIR removed
    can be inspected and plotted with the standard tooling, e.g.
    ``mne.viz.plot_projs_topomap(ssp.projs_, raw.info)`` -- worth doing for a
    method whose main failure mode is removing too much.
    """
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
    """Suppress TMS-evoked muscle artifacts from EEG (SSP-SIR).

    SSP-SIR projects out the high-variance muscle-artifact subspace and then
    reconstructs the brain signal lost to that projection through a forward
    model. With no individualised forward model, a three-layer spherical lead
    field is built from the montage. Conceptually this is signal-space
    projection followed by the source-informed reconstruction that MNE-Python
    also exposes as ``proj="reconstruct"``.

    As in the published method, the artifact-suppressed reconstruction is
    crossfaded in time against a reconstruction of the *unprojected* data
    (:func:`compute_sir`), so that the artifact subspace is removed
    only where the muscle artifact lives. Without this crossfade the projection
    also removes brain signal from the baseline and from late TEP components.
    Restricting suppression to the artifact window is also what the method's
    authors do in practice (Mutanen et al., 2024) [2]_.

    The numerical core has reference parity and synthetic validation, but the
    estimator has not yet been independently validated on a public real
    TMS-EEG dataset. Treat real-data use as experimental.

    Parameters
    ----------
    n_components : int | float
        Number of artifact principal components to remove (int), or the
        cumulative high-frequency variance fraction to cover (float in (0, 1)).
        See Notes for the exact criterion.
    forward : mne.Forward | None
        Optional forward solution; if None a spherical lead field is built.
    art_window : tuple of float | None
        ``(tmin, tmax)`` in seconds delimiting the muscle artifact used to
        estimate the artifact subspace. If None, the subspace is estimated
        automatically from high-frequency power across the whole epoch.
    blend : {'auto', 'constant'}
        How to crossfade the projected and unprojected reconstructions.
        ``'auto'`` (default) matches the kernel to the subspace mode: the
        sliding-RMS high-frequency envelope when ``art_window`` is None, or a
        smooth window around ``art_window`` otherwise. ``'constant'`` applies
        the projection uniformly over time.
    high_pass : float
        High-pass cutoff (Hz) used to isolate the muscle artifact. Default 100.
    M : int | None
        Source-informed reconstruction truncation dimension. If None, uses
        ``rank(data) - n_components``.
    smooth_length : float
        10-90% transition width (seconds) of the ``art_window`` crossfade.
        Default 0.010.
    sfreq : float | None
        Sampling frequency, required only for plain-array input.
    n_dipoles : int
        Number of dipoles for the spherical lead field.
    verbose : bool | str | int | None, default=None
        MNE-style logging level. The fitted SSP-SIR summary is emitted at
        INFO; numerical reconstruction helpers remain silent.

    Attributes
    ----------
    leadfield_ : ndarray
        The average-referenced lead field used.
    artifact_topographies_ : ndarray, shape (n_channels, n_components)
        The removed artifact subspace.
    operator_ : ndarray, shape (n_channels, n_channels)
        The fitted artifact-suppressing operator.
    operator_orig_ : ndarray, shape (n_channels, n_channels)
        The unprojected source-informed reconstruction operator.
    kernel_ : ndarray, shape (n_times,)
        Crossfade weights in [0, 1]; 1 where the artifact subspace is fully
        removed. All ones when ``blend='constant'``.
    n_components_ : int
        Number of artifact components removed.
    M_ : int
        Effective source-informed reconstruction rank used. This can be lower
        than the requested ``M`` when the projected lead field has lower
        numerical rank.
    singular_values_ : ndarray
        Singular values of the high-frequency data the subspace was estimated
        from. Inspect these to choose ``n_components`` by the spectrum elbow,
        which is what Mutanen et al. (2016) actually recommend [1]_.
    projs_ : list of mne.Projection
        ``artifact_topographies_`` wrapped as MNE projections, so the removed
        directions can be plotted with ``mne.viz.plot_projs_topomap``. Empty
        when fitted on a plain array, which carries no channel names.

    Notes
    -----
    With a float ``n_components`` this keeps the components covering that
    fraction of *variance*, ``sum(s[:k]**2) / sum(s**2)`` -- the standard PCA
    criterion. Choosing by the elbow of ``singular_values_`` is what Mutanen
    et al. (2016) actually recommend [1]_; pass an explicit int to do so.

    Established implementations expose the same three modes under different
    names -- automatic and windowed subspace estimation, and applying the
    projection across the whole epoch rather than only around the artifact --
    so ``art_window`` and ``blend`` translate directly. One difference is
    substantive rather than nominal: a float ``n_components`` selects a
    variance fraction here, whereas the equivalent data-driven option
    elsewhere thresholds ``sum(s[:k])**2 / sum(s)**2``, a squared
    nuclear-norm fraction, and so keeps more components for the same nominal
    percentage -- 7 against 3 at 90% on a typical spectrum. Pass an explicit
    int when a particular count is required.

    For broader treatments of SSP-SIR as a source-based method and as part of
    the general TMS-EEG artifact-removal landscape, see Mutanen et al. (2022)
    [3]_ and Hernandez-Pavon et al. (2022) [4]_, respectively.

    References
    ----------
    .. [1] Mutanen, T. P., Kukkonen, M., Nieminen, J. O., Stenroos, M.,
           Sarvas, J., & Ilmoniemi, R. J. (2016). Recovering TMS-evoked EEG
           responses masked by muscle artifacts. NeuroImage, 139, 157-166.
    .. [2] Mutanen, T. P., Ilmoniemi, I., Atti, I., Metsomaa, J., & Ilmoniemi,
           R. J. (2024). A simulation study: comparing independent component
           analysis and signal-space projection - source-informed
           reconstruction for rejecting muscle artifacts evoked by
           transcranial magnetic stimulation. Frontiers in Human Neuroscience,
           18, 1324958.
    .. [3] Mutanen, T. P., Metsomaa, J., Makkonen, M., Varone, G., Marzetti,
           L., & Ilmoniemi, R. J. (2022). Source-based artifact-rejection
           techniques for TMS-EEG. Journal of Neuroscience Methods, 382,
           109693.
    .. [4] Hernandez-Pavon, J. C., Kugiumtzis, D., Zrenner, C., Kimiskidis,
           V. K., & Metsomaa, J. (2022). Removing artifacts from TMS-evoked
           EEG: A methods review and a unifying theoretical framework. Journal
           of Neuroscience Methods, 376, 109591.
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
        """High-pass the evoked; return the SVD input and the crossfade kernel.

        Notes
        -----
        The high-pass is a second-order Butterworth applied with
        :func:`scipy.signal.filtfilt` rather than
        :func:`mne.filter.filter_data`, because the artifact subspace is
        defined through exactly this filter in the published method. The two
        differ only in how far they extend the signal past the epoch edges
        (scipy pads 9 samples by odd reflection, MNE considerably more) --
        neither extension is more correct, since the true signal outside the
        epoch is unknown. That moves the filtered trace by ~1e-2 relative at
        the boundaries and ~3e-12 in the interior. The artifact subspaces
        that result differ by under 0.15 degrees, so the choice does not
        matter to the outcome -- but keeping the published filter preserves
        an exact reference check that would otherwise be lost.
        """
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
        """Estimate the SSP-SIR cleaning operators from ``X``."""
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
        """Apply the fitted SSP-SIR operators to ``X``."""
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
