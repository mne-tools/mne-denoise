"""SOUND: automatic, robust noise suppression for EEG/MEG.

Reimplementation of the SOUND algorithm (Source-estimate-Utilizing
Noise-discarding) from Mutanen et al. (2018), translated from the reference
MATLAB ``Sound-Demo-Package`` by Tuomas Mutanen. SOUND iteratively estimates the
noise amplitude of every channel by predicting it from all the other channels
through a forward model (a leave-one-channel-out minimum-norm estimate), then
applies a Wiener filter that suppresses channel-specific noise while preserving
signal that is consistent with the forward model.

References
----------
Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J. (2018).
Automatic and robust noise suppression in EEG and MEG: The SOUND algorithm.
NeuroImage, 166, 135-151.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .._leadfield import make_spherical_leadfield
from ..utils import extract_data_from_mne, reconstruct_mne_object


def _to_2d(data: np.ndarray) -> np.ndarray:
    """Collapse (n_epochs, n_channels, n_times) to continuous (n_channels, n_times)."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 3:
        return np.transpose(data, (1, 0, 2)).reshape(data.shape[1], -1)
    return data


def _ddwiener(data: np.ndarray) -> np.ndarray:
    """Data-driven Wiener noise-amplitude estimate (initial SOUND estimate).

    Predicts each channel from all the others using the data covariance and
    returns the residual RMS per channel.
    """
    n_channels, n_times = data.shape
    cov = data @ data.T
    gamma = np.mean(np.diag(cov))
    residual_sq = np.empty(n_channels)
    eye = np.eye(n_channels - 1)
    for i in range(n_channels):
        others = np.concatenate([np.arange(i), np.arange(i + 1, n_channels)])
        coef = np.linalg.solve(cov[np.ix_(others, others)] + gamma * eye, data[others])
        pred = cov[i, others] @ coef
        residual_sq[i] = np.sum((data[i] - pred) ** 2)
    return np.sqrt(residual_sq) / np.sqrt(n_times)


def compute_sound(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    lambda_: float = 0.1,
    n_iter: int = 5,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the SOUND cleaning operator from data and a lead field.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Continuous (average-referenceable) sensor data.
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead-field matrix.
    lambda_ : float
        Regularisation parameter (Tikhonov, relative to the lead-field trace).
        Default 0.1, as in the TESA implementation.
    n_iter : int
        Number of SOUND iterations. Default 5.
    random_state : int | np.random.Generator | None
        Seed/generator controlling the random channel-update order.

    Returns
    -------
    operator : ndarray, shape (n_channels, n_channels)
        The linear SOUND cleaning operator (``cleaned = operator @ data``).
    sigmas : ndarray, shape (n_channels,)
        Estimated per-channel noise amplitudes.
    convergence : ndarray, shape (n_iter,)
        Maximum relative change of ``sigmas`` at each iteration.
    """
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    n_channels, n_times = data.shape
    if leadfield.shape[0] != n_channels:
        raise ValueError(
            f"Lead field has {leadfield.shape[0]} channels but data has {n_channels}."
        )
    if n_channels < 3:
        raise ValueError("SOUND requires at least 3 channels.")

    rng = np.random.default_rng(random_state)
    llt = leadfield @ leadfield.T  # (n_channels, n_channels); spans the column space

    sigmas = _ddwiener(data)
    sigmas = np.where(sigmas > 0, sigmas, np.mean(sigmas[sigmas > 0]) or 1.0)
    eye = np.eye(n_channels - 1)
    convergence = np.empty(n_iter)

    for k in range(n_iter):
        sigmas_old = sigmas.copy()
        for i in rng.permutation(n_channels):
            others = np.concatenate([np.arange(i), np.arange(i + 1, n_channels)])
            w = 1.0 / sigmas[others]
            # WLLW = diag(w) @ LLt[o,o] @ diag(w)
            g = (w[:, None] * llt[np.ix_(others, others)]) * w[None, :]
            reg = lambda_ * np.trace(g) / (n_channels - 1)
            b = llt[i, others] * w  # (L @ WL.T)[i]
            m = np.linalg.solve(g + reg * eye, b)
            pred = (m * w) @ data[others]
            sigmas[i] = np.sqrt(np.sum((pred - data[i]) ** 2)) / np.sqrt(n_times)
        convergence[k] = np.max(np.abs(sigmas_old - sigmas) / sigmas_old)

    # Final cleaning operator from the converged noise estimate.
    w = 1.0 / sigmas
    g = (w[:, None] * llt) * w[None, :]
    reg = lambda_ * np.trace(g) / n_channels
    inv = np.linalg.inv(g + reg * np.eye(n_channels))
    operator = ((llt * w[None, :]) @ inv) * w[None, :]
    return operator, sigmas, convergence


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
        Number of SOUND iterations. Default 5.
    forward : mne.Forward | None
        Optional pre-computed forward solution. If None, a spherical lead field
        is built from the data's montage.
    pos : float
        Source-grid spacing (mm) for the spherical lead field. Default 15.0.
    random_state : int | np.random.Generator | None
        Controls the random channel-update order for reproducibility.

    Attributes
    ----------
    leadfield_ : ndarray, shape (n_channels, n_sources)
        The average-referenced lead field used.
    operator_ : ndarray, shape (n_channels, n_channels)
        The fitted linear cleaning operator.
    sigmas_ : ndarray, shape (n_channels,)
        Estimated per-channel noise amplitudes.
    convergence_ : ndarray, shape (n_iter,)
        Maximum relative change of ``sigmas_`` per iteration.

    References
    ----------
    Mutanen et al. (2018), NeuroImage 166, 135-151.
    """

    def __init__(
        self,
        *,
        lambda_: float = 0.1,
        n_iter: int = 5,
        forward=None,
        pos: float = 15.0,
        random_state=None,
    ):
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.forward = forward
        self.pos = pos
        self.random_state = random_state

    def _get_leadfield(self, orig_inst, ch_names, n_channels):
        if orig_inst is not None:
            info = orig_inst.copy().pick(ch_names).info
            return make_spherical_leadfield(info, forward=self.forward, pos=self.pos)
        if self.forward is not None:
            gain = np.asarray(self.forward["sol"]["data"], dtype=float)
            if gain.shape[0] != n_channels:
                raise ValueError(
                    "For array input, the forward must have the same number of "
                    f"channels as the data ({gain.shape[0]} vs {n_channels})."
                )
            return gain - gain.mean(axis=0, keepdims=True)
        raise ValueError(
            "SOUND needs channel positions: pass an MNE object with a montage, "
            "or provide a `forward` for array input."
        )

    def fit(self, X, y=None):
        """Estimate the SOUND cleaning operator from ``X``."""
        data, _, _, orig_inst, _, ch_names = extract_data_from_mne(X)
        n_channels = data.shape[1] if data.ndim == 3 else data.shape[0]
        self.leadfield_ = self._get_leadfield(orig_inst, ch_names, n_channels)
        self._mne_ch_names_ = ch_names
        self.operator_, self.sigmas_, self.convergence_ = compute_sound(
            _to_2d(data),
            self.leadfield_,
            lambda_=self.lambda_,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        return self

    def transform(self, X):
        """Apply the fitted SOUND operator to ``X``."""
        if not hasattr(self, "operator_"):
            raise RuntimeError("SOUND is not fitted. Call fit() first.")
        data, _, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X, ch_names=getattr(self, "_mne_ch_names_", None)
        )
        if data.ndim == 3:
            cleaned = np.einsum("ij,ejt->eit", self.operator_, data)
        else:
            cleaned = self.operator_ @ data
        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

