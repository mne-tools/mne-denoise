"""Literal transliterations of the SOUND / SSP-SIR reference MATLAB.

These functions are deliberately *not* written the way the package writes them.
They follow the reference sources statement by statement, including the
source-space detour (``x = WL' * (...)``; ``y_solved = LFM * x``) that
``mne_denoise`` folds into channel-space products, and the two separate
truncated pseudo-inverses that ``tesa_sspsir`` computes. That structural
difference is the point: agreement between the two is evidence about the
algebra, not about a shared implementation.

Sources
-------
``SOUND.m``, ``DDWiener.m``
    Tuomas Mutanen, *Sound-Demo-Package*
    https://github.com/tuomasmutanen/Sound-Demo-Package
``tesa_sound.m`` (reference handling), ``tesa_sspsir.m`` (``SSP_SIR``)
    EEGLAB TESA toolbox, https://github.com/nigelrogasch/TESA

Used by ``test_sound_sspsir_parity.py``, which runs these in CI without MATLAB
and additionally checks them against a live MATLAB Engine when one is present.
"""

from __future__ import annotations

import numpy as np


def ddwiener(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``DDWiener.m``: data-driven Wiener estimate and noise amplitudes."""
    cov = data @ data.T
    gamma = np.mean(np.diag(cov))
    n_channels = data.shape[0]
    y_solved = np.empty_like(data)
    for i in range(n_channels):
        idiff = np.array([j for j in range(n_channels) if j != i])
        y_solved[i] = cov[i, idiff] @ np.linalg.solve(
            cov[np.ix_(idiff, idiff)] + gamma * np.eye(n_channels - 1), data[idiff]
        )
    residual = data - y_solved
    sigmas = np.sqrt(np.diag(residual @ residual.T)) / np.sqrt(data.shape[1])
    return y_solved, sigmas


def sound(
    data: np.ndarray,
    leadfield: np.ndarray,
    n_iter: int,
    lambda0: float,
    orders: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``SOUND.m``.

    ``orders`` supplies the per-iteration channel visiting order in place of
    MATLAB's ``randperm``, so the stochastic update order can be matched
    against the package's ``random_state``.
    """
    n_channels, n_times = data.shape
    _, sigmas = ddwiener(data)
    convergence = np.empty(n_iter)

    for k in range(n_iter):
        sigmas_old = sigmas.copy()
        for i in orders[k]:
            chan = np.array([j for j in range(n_channels) if j != i])
            w = np.diag(1.0 / sigmas)
            wl = w[np.ix_(chan, chan)] @ leadfield[chan, :]
            wllw = wl @ wl.T
            lam = lambda0 * np.trace(wllw) / (n_channels - 1)
            x = wl.T @ np.linalg.solve(
                wllw + lam * np.eye(n_channels - 1),
                w[np.ix_(chan, chan)] @ data[chan, :],
            )
            y_solved = leadfield @ x
            sigmas[i] = np.sqrt(
                (y_solved[i] - data[i]) @ (y_solved[i] - data[i])
            ) / np.sqrt(n_times)
        convergence[k] = np.max(np.abs(sigmas_old - sigmas) / sigmas_old)

    w = np.diag(1.0 / sigmas)
    wl = w @ leadfield
    wllw = wl @ wl.T
    lam = lambda0 * np.trace(wllw) / n_channels
    x = wl.T @ np.linalg.solve(wllw + lam * np.eye(n_channels), w @ data)
    return leadfield @ x, sigmas, convergence


def tesa_sound(
    data: np.ndarray,
    leadfield: np.ndarray,
    n_iter: int,
    lambda0: float,
    orders: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """``tesa_sound.m``: ref_best, drop the reference, reconstruct average-ref."""
    n_channels = data.shape[0]
    _, sigmas0 = ddwiener(data)
    best = int(np.argmin(sigmas0))
    chans = np.array([i for i in range(n_channels) if i != best])

    # ref_best.m applied to both data and lead field.
    data_tmp = (data - data[best])[chans]
    lf_tmp = (leadfield - leadfield[best])[chans]
    # ref_ave.m for the reconstruction lead field.
    lf_mean = leadfield - leadfield.mean(axis=0, keepdims=True)

    _, sigmas, _ = sound(data_tmp, lf_tmp, n_iter, lambda0, orders)

    w = np.diag(1.0 / sigmas)
    wl = w @ lf_tmp
    wllw = wl @ wl.T
    lam = lambda0 * np.trace(wllw) / (n_channels - 1)
    x = wl.T @ np.linalg.solve(wllw + lam * np.eye(n_channels - 1), w @ data_tmp)
    return lf_mean @ x, sigmas, best


def _truncated_pinv(matrix: np.ndarray, M: int) -> np.ndarray:
    """``[U,S,V] = svd(tau_proj); S_inv(1:M,1:M) = ...; tau_inv = V*S_inv*U'``."""
    u, s, vh = np.linalg.svd(matrix)
    s_inv = np.zeros_like(s)
    s_inv[:M] = 1.0 / s[:M]
    return (vh.T * s_inv) @ u.T


def ssp_sir(
    data: np.ndarray,
    leadfield: np.ndarray,
    artifact_topographies: np.ndarray,
    M: int,
    filt_ker: np.ndarray | None = None,
) -> np.ndarray:
    """``SSP_SIR`` from ``tesa_sspsir.m``.

    ``filt_ker`` is the crossfade kernel; None reproduces
    ``artScale='manualConstant'`` (no crossfade).
    """
    n_channels = data.shape[0]
    proj = np.eye(n_channels) - artifact_topographies @ artifact_topographies.T
    data_clean = proj @ data

    pl = proj @ leadfield
    suppr_data_sir = leadfield @ pl.T @ _truncated_pinv(pl @ pl.T, M) @ data_clean

    llt = leadfield @ leadfield.T
    orig_data_sir = llt @ _truncated_pinv(llt, M) @ data

    if filt_ker is None:
        return suppr_data_sir
    return filt_ker * suppr_data_sir + orig_data_sir - filt_ker * orig_data_sir


def automatic_filt_ker(data_high: np.ndarray, sfreq: float) -> np.ndarray:
    """``artScale='automatic'`` kernel: 50 ms sliding RMS of high-frequency power."""
    tmp = data_high**2
    x_scal = int(round(sfreq / 1000.0 * 50.0))
    kernel = np.array(
        [np.convolve(row, np.ones(x_scal), mode="same") / x_scal for row in tmp]
    )
    kernel = kernel.sum(axis=0) / tmp.shape[0]
    kernel = kernel / kernel.max()
    return np.sqrt(kernel)


def manual_filt_ker(
    times: np.ndarray, tmin: float, tmax: float, smooth_length: float
) -> np.ndarray:
    """``artScale='manual'`` kernel: ``dsigmf(time, [4/L t1 4/L t2])``."""

    def sigmf(x, slope, centre):
        return 1.0 / (1.0 + np.exp(-slope * (x - centre)))

    slope = 4.0 / smooth_length
    return sigmf(times, slope, tmin) - sigmf(times, slope, tmax)
