"""Spectrum interpolation for power-line noise removal.

Implements the FFT-based spectrum-interpolation method of Leske & Dalal (2019).
The power-line frequency and its harmonics are removed by replacing the
*amplitude* of the spectrum inside a narrow band around each line frequency
with the mean amplitude of neighbouring frequency bins, while the original
phase is preserved. The cleaned signal is obtained by an inverse transform.

Unlike a notch filter, this leaves the phase spectrum untouched and only edits
a thin amplitude band, so broadband activity around the line frequency is
largely preserved.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG and
       MEG data via spectrum interpolation. NeuroImage, 189, 763-776.
       https://doi.org/10.1016/j.neuroimage.2019.01.026
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .. import _mne
from .._data import extract_data_from_mne, reconstruct_mne_object
from .._logging import logger, verbose
from .._validation import (
    check_matching_sfreq,
    check_positive_real,
    resolve_sfreq,
)


def interpolate_spectrum(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    *,
    bandwidth: float = 1.0,
    neighbour_width: float = 2.0,
) -> np.ndarray:
    """Remove line noise from 2D data by amplitude spectrum interpolation.

    For each target frequency, the amplitude of the FFT bins inside a band of
    half-width ``bandwidth`` is replaced by the mean amplitude of the
    neighbouring reference bands. The original phase is kept, following
    Leske & Dalal (2019) [1]_.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input time series. Each channel is processed independently.
    sfreq : float
        Sampling frequency in Hz.
    freqs : array-like of float
        Target frequencies (e.g. the line frequency and its harmonics) in Hz.
    bandwidth : float
        Half-width in Hz of the band that is interpolated around each target
        frequency. For example, ``bandwidth=1`` replaces 49--51 Hz around a
        50 Hz target. Default 1.0.
    neighbour_width : float
        Width in Hz of the reference band used on each side of the interpolated
        band to estimate the replacement amplitude. Default 2.0.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_times)
        Line-noise-reduced time series, in the same units as ``data``.

    References
    ----------
    .. [1] Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG
           and MEG data via spectrum interpolation. NeuroImage, 189, 763-776.
    """
    if np.iscomplexobj(data):
        raise ValueError("data must be real-valued")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_channels, n_times), got {data.ndim}D")
    if data.shape[1] == 0:
        raise ValueError("data must contain at least one time sample")

    sfreq = check_positive_real(sfreq, name="sfreq")
    bandwidth = check_positive_real(bandwidth, name="bandwidth")
    neighbour_width = check_positive_real(neighbour_width, name="neighbour_width")

    target_freqs = np.asarray(freqs, dtype=float).reshape(-1)
    if not np.all(np.isfinite(target_freqs)) or np.any(target_freqs <= 0):
        raise ValueError("freqs must contain positive, finite frequencies")

    n_times = data.shape[1]
    nyquist = sfreq / 2.0
    target_freqs = target_freqs[target_freqs < nyquist]
    if target_freqs.size == 0:
        return data.copy()

    spectrum = np.fft.rfft(data, axis=1)
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    new_mag = magnitude.copy()

    for f in target_freqs:
        if not (0.0 < f < nyquist):
            continue

        band = (fft_freqs >= f - bandwidth) & (fft_freqs <= f + bandwidth)
        if not np.any(band):
            # Band narrower than the frequency resolution: snap to nearest bin.
            nearest = int(np.argmin(np.abs(fft_freqs - f)))
            band = np.zeros_like(band)
            band[nearest] = True

        left = (fft_freqs >= f - bandwidth - neighbour_width) & (
            fft_freqs < f - bandwidth
        )
        right = (fft_freqs > f + bandwidth) & (
            fft_freqs <= f + bandwidth + neighbour_width
        )
        neighbours = left | right
        if not np.any(neighbours):
            # No usable neighbours; leave this frequency untouched.
            continue

        replacement = new_mag[:, neighbours].mean(axis=1, keepdims=True)
        new_mag[:, band] = replacement

    spectrum_clean = new_mag * np.exp(1j * phase)
    return np.fft.irfft(spectrum_clean, n=n_times, axis=1)


class SpectrumInterpolation(BaseEstimator, TransformerMixin):
    """Remove power-line noise by amplitude spectrum interpolation.

    Frequency-domain line-noise remover following Leske & Dalal (2019) [1]_.
    The amplitude of a thin band around the line frequency (and its harmonics)
    is replaced by the mean amplitude of neighbouring bins, while the phase is
    preserved.


    Parameters
    ----------
    sfreq : float, optional
        Sampling frequency in Hz. Required for NumPy-array inputs; for MNE
        objects it is read from ``info['sfreq']`` and overrides this value.
    line_freq : float | array-like of float
        Power-line frequency in Hz (e.g. 50 or 60). A sequence of explicit
        frequencies may be given instead, in which case they are used directly.
        Default 50.0.
    n_harmonics : int, optional
        Number of harmonics of ``line_freq`` to remove (including the
        fundamental). If None, all harmonics below the Nyquist frequency are
        removed. Ignored when ``line_freq`` is a sequence.
    bandwidth : float
        Half-width in Hz of the interpolated band around each frequency.
        For example, ``bandwidth=1`` replaces 49--51 Hz around a 50 Hz target.
        Default 1.0.
    neighbour_width : float
        Width in Hz of the reference band on each side used to estimate the
        replacement amplitude. Default 2.0.

    Attributes
    ----------
    sfreq_ : float
        Sampling frequency used during the fit.
    freqs_ : ndarray
        Resolved target frequencies (line frequency and harmonics).

    Examples
    --------
    >>> from mne_denoise.spectrum_interpolation import SpectrumInterpolation
    >>> si = SpectrumInterpolation(sfreq=1000.0, line_freq=60.0)
    >>> clean = si.fit_transform(data)  # doctest: +SKIP

    Notes
    -----
    This FFT-based method is best suited to continuous recordings or long data
    segments with stationary line noise. Short epochs can exhibit edge effects,
    especially when their duration does not contain complete cycles of the
    targeted frequencies. Inspect the result when processing short epochs.

    References
    ----------
    .. [1] Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG
           and MEG data via spectrum interpolation. NeuroImage, 189, 763-776.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        line_freq: float | ArrayLike = 50.0,
        n_harmonics: int | None = None,
        bandwidth: float = 1.0,
        neighbour_width: float = 2.0,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.line_freq = line_freq
        self.n_harmonics = n_harmonics
        self.bandwidth = bandwidth
        self.neighbour_width = neighbour_width
        self.verbose = verbose

    def _target_freqs(self, sfreq: float) -> np.ndarray:
        nyquist = sfreq / 2.0
        check_positive_real(self.bandwidth, name="bandwidth")
        check_positive_real(self.neighbour_width, name="neighbour_width")

        if np.asarray(self.line_freq).ndim == 0:
            if self.n_harmonics is not None and (
                isinstance(self.n_harmonics, bool)
                or not isinstance(self.n_harmonics, Integral)
                or self.n_harmonics < 1
            ):
                raise ValueError("n_harmonics must be a positive integer or None")
            base = float(self.line_freq)
            if not np.isfinite(base) or base <= 0:
                raise ValueError("line_freq must contain positive, finite frequencies")
            if base >= nyquist:
                raise ValueError("line_freq must be below the Nyquist frequency")
            max_h = int(np.ceil(nyquist / base)) - 1
            n_h = max_h if self.n_harmonics is None else min(self.n_harmonics, max_h)
            candidates = [base * (k + 1) for k in range(n_h)]
        else:
            candidates = np.asarray(self.line_freq, dtype=float).reshape(-1)
            if candidates.size == 0:
                raise ValueError("line_freq must contain at least one frequency")
            if not np.all(np.isfinite(candidates)) or np.any(candidates <= 0):
                raise ValueError("line_freq must contain positive, finite frequencies")
            if np.any(candidates >= nyquist):
                raise ValueError("line_freq must be below the Nyquist frequency")
        return np.unique(np.asarray(candidates, dtype=float))

    def _apply(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.ndim == 2:
            return interpolate_spectrum(
                data,
                self.sfreq_,
                self.freqs_,
                bandwidth=self.bandwidth,
                neighbour_width=self.neighbour_width,
            )
        if data.ndim == 3:
            flat = data.reshape(-1, data.shape[-1])
            return interpolate_spectrum(
                flat,
                self.sfreq_,
                self.freqs_,
                bandwidth=self.bandwidth,
                neighbour_width=self.neighbour_width,
            ).reshape(data.shape)
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")

    @verbose
    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> SpectrumInterpolation:
        """Resolve the sampling rate and target frequencies.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Data to clean. Only metadata (sampling frequency) is read here.
        y : None
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        self : SpectrumInterpolation
            The fitted estimator.
        """
        _, mne_sfreq, mne_type, _, _, _ = extract_data_from_mne(X, auto_pick=False)
        is_mne = mne_type != "array"
        sfreq = resolve_sfreq(
            self.sfreq,
            mne_sfreq,
            context="a NumPy array input",
        )
        if not is_mne:
            data = np.asarray(X)
            if data.ndim not in (2, 3):
                raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
        self.sfreq_ = sfreq
        self.freqs_ = self._target_freqs(sfreq)
        logger.info(
            "Spectrum interpolation: target frequencies=%s Hz, targets=%d, "
            "bandwidth=%.3g Hz, neighbour width=%.3g Hz.",
            np.array2string(self.freqs_, precision=4, separator=", "),
            self.freqs_.size,
            self.bandwidth,
            self.neighbour_width,
        )
        return self

    @verbose
    def transform(
        self,
        X: Any,
        *,
        verbose: bool | str | int | None = None,
    ) -> Any:
        """Apply spectrum interpolation to ``X``.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Data to clean. For MNE objects every data channel is processed and
            non-data channels are returned unchanged.

        Returns
        -------
        out : Raw | Epochs | Evoked | ndarray
            Cleaned data, of the same type and shape as ``X``.
        """
        check_is_fitted(self, attributes=["sfreq_", "freqs_"])

        _, _, mne_type, _, _, _ = extract_data_from_mne(X, auto_pick=False)
        if mne_type != "array":
            _mne.require_mne("SpectrumInterpolation MNE input support")
            data_picks = _mne.mne.pick_types(
                X.info,
                meg=True,
                ref_meg=False,
                eeg=True,
                seeg=True,
                ecog=True,
                dbs=True,
                fnirs=True,
                csd=True,
                exclude=(),
            )
            if data_picks.size == 0:
                return X.copy()

            data, sfreq, mne_type, orig_inst, picks, _ = extract_data_from_mne(
                X,
                ch_names=[X.ch_names[pick] for pick in data_picks],
                auto_pick=False,
            )
            check_matching_sfreq(sfreq, self.sfreq_, name="SpectrumInterpolation")
            cleaned = self._apply(data)
            return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

        return self._apply(np.asarray(X, dtype=float))

    @verbose
    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        *,
        verbose: bool | str | int | None = None,
        **fit_params: Any,
    ) -> Any:
        """Fit then transform ``X`` in one step.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Data to clean.
        y : None
            Ignored; present for scikit-learn API compatibility.
        **fit_params : dict
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        out : Raw | Epochs | Evoked | ndarray
            Cleaned data, of the same type and shape as ``X``.
        """
        return self.fit(X, y).transform(X)
