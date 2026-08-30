"""Filter design and streaming-edge helpers for ASR and Adaptive ASR.

This module provides internal signal processing utilities for ASR variants:
1. The inverse-EEG spectral shaping filter from the original ASR algorithm,
   plus an optional lightweight high-pass statistics filter.
2. Boundary padding helpers (lookahead tail and carry reflection) used by
   streaming/chunked ASR processors to mitigate edge artifacts.
3. An 8th-order Yule-Walker IIR pre-emphasis filter used strictly by the
   Adaptive ASR (AASR) variants for similarity-matching updates.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import toeplitz


def _design_statistics_filter(
    sfreq: float,
    filter_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Design the requested ASR statistics filter.

    Parameters
    ----------
    sfreq : float
        The sampling frequency of the data.
    filter_kind : str
        The type of filter to design. Can be "asr", "highpass", or "none".

    Returns
    -------
    b : np.ndarray
        Numerator coefficients of the filter.
    a : np.ndarray
        Denominator coefficients of the filter.
    """
    if filter_kind == "none":
        return np.array([1.0]), np.array([1.0])
    if filter_kind not in ("asr", "highpass"):
        raise ValueError("filter_kind must be 'none', 'asr', or 'highpass'")
    if filter_kind == "asr":
        return _design_asr_filter(sfreq)
    cutoff = min(0.5, sfreq * 0.1)
    if cutoff >= sfreq / 2:
        return np.array([1.0]), np.array([1.0])
    return signal.butter(2, cutoff, btype="highpass", fs=sfreq)


def _apply_statistics_filter(
    X: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Apply the statistics filter using zero-phase (filtfilt) offline filtering.

    Parameters
    ----------
    X : np.ndarray
        The input signal array of shape (n_channels, n_samples).
    b : np.ndarray
        Numerator coefficients of the filter.
    a : np.ndarray
        Denominator coefficients of the filter.

    Returns
    -------
    X_filtered : np.ndarray
        The filtered signal array.
    """
    if b.size == 1 and a.size == 1:
        return X.copy()
    padlen = 3 * max(len(a), len(b))
    if X.shape[1] <= padlen:
        return signal.lfilter(b, a, X, axis=1)
    return signal.filtfilt(b, a, X, axis=1)


def _append_streaming_tail(X: np.ndarray, n_tail: int) -> np.ndarray:
    """Append the reflected lookahead tail used by the standard streaming ASR algorithm.

    Parameters
    ----------
    X : np.ndarray
        The input signal array.
    n_tail : int
        The number of samples to append.

    Returns
    -------
    X_padded : np.ndarray
        The signal array with the reflected tail appended.
    """
    if n_tail == 0:
        return X.copy()
    if X.shape[1] <= n_tail:
        raise ValueError("Data must contain more samples than the lookahead tail")
    tail = 2.0 * X[:, [-1]] - X[:, -2 : -n_tail - 2 : -1]
    return np.concatenate([X, tail], axis=1)


def _prepend_streaming_carry(X: np.ndarray, n_carry: int) -> np.ndarray:
    """Prepend the reflected initial carry used by the standard streaming ASR algorithm.

    Parameters
    ----------
    X : np.ndarray
        The input signal array.
    n_carry : int
        The number of samples to prepend.

    Returns
    -------
    X_padded : np.ndarray
        The signal array with the reflected carry prepended.
    """
    if n_carry == 0:
        return X.copy()
    if X.shape[1] <= n_carry:
        raise ValueError("Data must contain more samples than the lookahead carry")
    carry = 2.0 * X[:, [0]] - X[:, n_carry:0:-1]
    return np.concatenate([carry, X], axis=1)


def _apply_statistics_filter_streaming(
    X: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Apply the statistics filter in the causal direction used by ASR.

    Parameters
    ----------
    X : np.ndarray
        The input signal array.
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.

    Returns
    -------
    X_filtered : np.ndarray
        The causally filtered signal.
    """
    if b.size == 1 and a.size == 1:
        return X.copy()
    return signal.lfilter(b, a, X, axis=1)


def _polystab(a: np.ndarray) -> np.ndarray:
    """Stabilize a polynomial by reflecting roots inside the unit circle.

    Parameters
    ----------
    a : np.ndarray
        Polynomial coefficients.

    Returns
    -------
    b : np.ndarray
        Stabilized polynomial coefficients.
    """
    roots = np.roots(a)
    keep = roots != 0
    reflect = 0.5 * (np.sign(np.abs(roots[keep]) - 1.0) + 1.0)
    roots[keep] = (1.0 - reflect) * roots[keep] + reflect / np.conj(roots[keep])
    nz = np.flatnonzero(a != 0)
    b = a[nz[0]] * np.poly(roots)
    if not np.any(np.imag(a)):
        b = np.real(b)
    return np.asarray(b, dtype=np.float64)


def _numf(h: np.ndarray, a: np.ndarray, nb: int) -> np.ndarray:
    """Least-squares FIR numerator matching impulse response ``h`` given ``a``.

    Parameters
    ----------
    h : np.ndarray
        Desired impulse response.
    a : np.ndarray
        Denominator coefficients.
    nb : int
        Numerator order.

    Returns
    -------
    b : np.ndarray
        Estimated numerator coefficients.
    """
    nh = int(np.max(h.size))
    impulse = np.zeros(nh, dtype=np.float64)
    impulse[0] = 1.0
    impr = signal.lfilter(np.array([1.0]), a, impulse)
    rhs = np.concatenate(([1.0], np.zeros(nb, dtype=np.float64)))
    return np.linalg.lstsq(toeplitz(impr, rhs), h.T, rcond=None)[0].T


def _denf(R: np.ndarray, na: int) -> np.ndarray:
    """Least-squares AR denominator from autocorrelation ``R``.

    Parameters
    ----------
    R : np.ndarray
        Autocorrelation sequence.
    na : int
        Denominator order.

    Returns
    -------
    a : np.ndarray
        Estimated denominator coefficients.
    """
    nr = int(np.max(np.size(R)))
    Rm = toeplitz(R[na : nr - 1], R[na:0:-1])
    rhs = -R[na + 1 : nr]
    return np.concatenate(([1.0], np.linalg.lstsq(Rm, rhs.T, rcond=None)[0].T))


def _yulewalk(
    order: int, F: np.ndarray, M: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Yule-Walker IIR design from a piecewise-linear magnitude template.

    Parameters
    ----------
    order : int
        Filter order.
    F : np.ndarray
        Frequency points (0 to 1).
    M : np.ndarray
        Magnitude response at the frequency points.

    Returns
    -------
    B : np.ndarray
        Numerator coefficients.
    A : np.ndarray
        Denominator coefficients.
    """
    F = np.asarray(F, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    npt = 513
    lap = int(np.trunc((npt - 1) / 25))
    Ht = np.zeros(npt, dtype=np.float64)
    Ht[0] = M[0]
    df = np.diff(F)

    nb = 0
    for idx in range(F.size - 1):
        if df[idx] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.trunc(F[idx + 1] * npt)) - 1
        j = np.arange(nb, ne + 1)
        inc = 0.0 if ne == nb else (j - nb) / (ne - nb)
        Ht[nb : ne + 1] = inc * M[idx + 1] + (1.0 - inc) * M[idx]
        nb = ne + 1

    Ht = np.concatenate([Ht, Ht[-2:0:-1]])
    n = Ht.size
    n2 = int(np.trunc((n + 1) / 2))
    nr = 4 * order
    nt = np.arange(nr, dtype=np.float64)

    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / max(nr - 1, 1)))

    Rwindow = np.concatenate(
        (
            np.array([0.5], dtype=np.float64),
            np.ones(max(n2 - 1, 0), dtype=np.float64),
            np.zeros(max(n - n2, 0), dtype=np.float64),
        )
    )
    A = _polystab(_denf(R, order))
    Qh = _numf(np.concatenate(([R[0] / 2.0], R[1:nr])), A, order)

    _, Ss = signal.freqz(Qh, A, worN=n, whole=True)
    Ss = np.maximum(2.0 * np.real(Ss), np.finfo(float).eps)
    hh = np.fft.ifft(np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss)))))
    B = np.real(_numf(hh[:nr], A, order))
    return np.asarray(B, dtype=np.float64), np.asarray(A, dtype=np.float64)


def _design_asr_filter(sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Design the original ASR inverse-EEG pre-emphasis filter.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be positive for the ASR statistics filter")
    freq_hz = np.array(
        [
            0.0,
            2.0,
            3.0,
            13.0,
            16.0,
            40.0,
            min(80.0, (sfreq / 2.0) - 1.0),
            sfreq / 2.0,
        ],
        dtype=np.float64,
    )
    mags = np.array([3.0, 0.75, 0.33, 0.33, 1.0, 1.0, 3.0, 3.0], dtype=np.float64)
    if sfreq < 80.0:
        # Omit the 40-Hz breakpoint and its otherwise duplicated final gain
        # when Nyquist is below 40 Hz.
        freq_hz = np.delete(freq_hz, -3)
        mags = mags[:-1]
    freqs = freq_hz * 2.0 / sfreq
    return _yulewalk(8, freqs, mags)


def _lfilter_channels(
    X: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a causal IIR filter channel-wise and return final conditions.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The input data array.
    b : ndarray
        Numerator filter coefficients.
    a : ndarray
        Denominator filter coefficients.
    zi : ndarray | None
        Initial conditions for the filter delay line, shape (n_channels, order).
        If None, initial conditions are assumed to be zero.

    Returns
    -------
    out : ndarray, shape (n_channels, n_times)
        The filtered output data.
    zf : ndarray, shape (n_channels, order)
        The final conditions of the filter delay line.
    """
    X = np.asarray(X, dtype=np.float64)
    order = max(len(a), len(b)) - 1
    if order <= 0:
        empty = np.empty((X.shape[0], 0), dtype=np.float64)
        return X.copy(), empty
    if zi is None:
        zi = np.zeros((X.shape[0], order), dtype=np.float64)
    else:
        zi = np.asarray(zi, dtype=np.float64)
        if zi.shape != (X.shape[0], order):
            raise ValueError(
                "Filter state shape does not match data: "
                f"{zi.shape} vs {(X.shape[0], order)}"
            )

    out = np.empty_like(X, dtype=np.float64)
    zf = np.empty_like(zi, dtype=np.float64)
    for ch_idx in range(X.shape[0]):
        out[ch_idx], zf[ch_idx] = signal.lfilter(b, a, X[ch_idx], zi=zi[ch_idx])
    return out, zf
