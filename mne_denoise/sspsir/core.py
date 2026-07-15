"""SSP-SIR: suppress TMS-evoked muscle artifacts from EEG.

Reimplementation of SSP-SIR (Signal-Space Projection -- Source-Informed
Reconstruction) from Mutanen et al. (2016), translated from the reference MATLAB
in EEGLAB's TESA toolbox. SSP-SIR removes the high-variance muscle-artifact
subspace with a signal-space projection and then recovers the brain signal lost
to that projection by reconstructing it through a forward model (the
source-informed reconstruction step).

References
----------
Mutanen, T. P., Kukkonen, M., Nieminen, J. O., Stenroos, M., Sarvas, J., &
Ilmoniemi, R. J. (2016). Recovering TMS-evoked EEG responses masked by muscle
artifacts. NeuroImage, 139, 157-166.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin

from .._leadfield import make_spherical_leadfield
from ..utils import extract_data_from_mne, reconstruct_mne_object


def _truncated_pinv(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Rank-truncated pseudoinverse (keep the largest ``rank`` singular values)."""
    u, s, vt = np.linalg.svd(matrix)
    s_inv = np.zeros_like(s)
    rank = max(1, min(int(rank), len(s)))
    s_inv[:rank] = 1.0 / s[:rank]
    return (vt.T * s_inv) @ u.T


def _artifact_subspace(svd_input: np.ndarray, n_components) -> tuple[np.ndarray, int]:
    """Left singular vectors of ``svd_input`` and the number of artifact PCs.

    ``n_components`` is either an int (number of PCs) or a float in (0, 1)
    (cumulative high-frequency variance fraction).
    """
    u, s, _ = np.linalg.svd(svd_input, full_matrices=False)
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        cum = np.cumsum(s**2) / np.sum(s**2)
        n_pc = int(np.searchsorted(cum, n_components) + 1)
    else:
        n_pc = int(n_components)
    n_pc = max(1, min(n_pc, u.shape[1]))
    return u[:, :n_pc], n_pc


def compute_sspsir_operator(
    leadfield: np.ndarray, artifact_topographies: np.ndarray, M: int
) -> np.ndarray:
    """Build the SSP-SIR cleaning operator ``C`` (``cleaned = C @ data``).

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
        The SSP-SIR cleaning operator.
    """
    n_channels = leadfield.shape[0]
    proj = np.eye(n_channels) - artifact_topographies @ artifact_topographies.T
    pl = proj @ leadfield
    tau_inv = _truncated_pinv(pl @ pl.T, M)
    return leadfield @ pl.T @ tau_inv @ proj


class SSPSIR(BaseEstimator, TransformerMixin):
    """Suppress TMS-evoked muscle artifacts from EEG (SSP-SIR).

    SSP-SIR projects out the high-variance muscle-artifact subspace and then
    reconstructs the brain signal lost to that projection through a forward
    model. With no individualised forward model, a three-layer spherical lead
    field is built from the montage. Conceptually this is signal-space
    projection followed by the source-informed reconstruction that MNE-Python
    also exposes as ``proj="reconstruct"``.

    Parameters
    ----------
    n_components : int | float
        Number of artifact principal components to remove (int), or the
        cumulative high-frequency variance fraction to cover (float in (0, 1)).
    forward : mne.Forward | None
        Optional forward solution; if None a spherical lead field is built.
    art_window : tuple of float | None
        ``(tmin, tmax)`` in seconds delimiting the muscle artifact used to
        estimate the artifact subspace. If None, the subspace is estimated
        automatically from high-frequency power across the whole epoch.
    high_pass : float
        High-pass cutoff (Hz) used to isolate the muscle artifact. Default 100.
    M : int | None
        Source-informed reconstruction truncation dimension. If None, uses
        ``rank(data) - n_components``.
    sfreq : float | None
        Sampling frequency, required only for plain-array input.
    pos : float
        Source-grid spacing (mm) for the spherical lead field.

    Attributes
    ----------
    leadfield_ : ndarray
        The average-referenced lead field used.
    artifact_topographies_ : ndarray, shape (n_channels, n_components)
        The removed artifact subspace.
    operator_ : ndarray, shape (n_channels, n_channels)
        The fitted SSP-SIR cleaning operator.
    n_components_ : int
        Number of artifact components removed.

    References
    ----------
    Mutanen et al. (2016), NeuroImage 139, 157-166.
    """

    def __init__(
        self,
        *,
        n_components=None,
        forward=None,
        art_window=None,
        high_pass: float = 100.0,
        M=None,
        sfreq=None,
        pos: float = 15.0,
    ):
        self.n_components = n_components
        self.forward = forward
        self.art_window = art_window
        self.high_pass = high_pass
        self.M = M
        self.sfreq = sfreq
        self.pos = pos

    def _resolve_sfreq(self, sfreq):
        sfreq = sfreq if sfreq is not None else self.sfreq
        if sfreq is None:
            raise ValueError(
                "SSP-SIR needs a sampling frequency: pass an MNE object or set sfreq."
            )
        return float(sfreq)

    def _svd_input(self, evoked, sfreq, times):
        """High-pass the evoked and build the matrix whose SVD gives the subspace."""
        b, a = butter(2, self.high_pass / (sfreq / 2.0), btype="high")
        data_high = filtfilt(b, a, evoked, axis=1)
        if self.art_window is not None:
            tmin, tmax = self.art_window
            if times is None:
                times = np.arange(evoked.shape[1]) / sfreq
            mask = (times >= tmin) & (times <= tmax)
            if not mask.any():
                raise ValueError("art_window does not overlap the data time range.")
            return data_high[:, mask]
        # Automatic: weight by a 50 ms sliding RMS of high-frequency power.
        win = max(1, int(round(sfreq / 1000.0 * 50.0)))
        power = uniform_filter1d(data_high**2, size=win, axis=1, mode="nearest")
        kernel = power.mean(axis=0)
        kernel = np.sqrt(kernel / kernel.max()) if kernel.max() > 0 else kernel
        return kernel[None, :] * data_high

    def fit(self, X, y=None):
        """Estimate the SSP-SIR cleaning operator from ``X``."""
        if self.n_components is None:
            raise ValueError(
                "n_components must be set (number of artifact PCs to remove, or a "
                "variance fraction in (0, 1))."
            )
        data, sfreq, _, orig_inst, _, ch_names = extract_data_from_mne(X)
        sfreq = self._resolve_sfreq(sfreq)
        times = getattr(orig_inst, "times", None)

        # Average reference and an evoked (trial-averaged) view for the subspace.
        evoked = data.mean(axis=0) if data.ndim == 3 else np.asarray(data, float)
        evoked = evoked - evoked.mean(axis=0, keepdims=True)

        if orig_inst is not None:
            info = orig_inst.copy().pick(ch_names).info
            self.leadfield_ = make_spherical_leadfield(
                info, forward=self.forward, pos=self.pos
            )
        elif self.forward is not None:
            gain = np.asarray(self.forward["sol"]["data"], dtype=float)
            self.leadfield_ = gain - gain.mean(axis=0, keepdims=True)
        else:
            raise ValueError(
                "SSP-SIR needs channel positions: pass an MNE object with a "
                "montage, or provide a `forward` for array input."
            )

        svd_input = self._svd_input(evoked, sfreq, times)
        self.artifact_topographies_, self.n_components_ = _artifact_subspace(
            svd_input, self.n_components
        )
        rank = np.linalg.matrix_rank(evoked)
        M = self.M if self.M is not None else rank - self.n_components_
        self.M_ = max(1, int(M))
        self.operator_ = compute_sspsir_operator(
            self.leadfield_, self.artifact_topographies_, self.M_
        )
        self._mne_ch_names_ = ch_names
        return self

    def transform(self, X):
        """Apply the fitted SSP-SIR operator to ``X``."""
        if not hasattr(self, "operator_"):
            raise RuntimeError("SSPSIR is not fitted. Call fit() first.")
        data, _, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X, ch_names=getattr(self, "_mne_ch_names_", None)
        )
        if data.ndim == 3:
            cleaned = np.einsum("ij,ejt->eit", self.operator_, data)
        else:
            cleaned = self.operator_ @ data
        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

