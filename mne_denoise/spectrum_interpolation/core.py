"""Spectrum interpolation for power-line noise removal.

Implements the spectrum-interpolation method of Leske & Dalal (2019), a
frequency-domain translation of FieldTrip's ``ft_preproc_dftfilter`` in its
``'neighbour'`` (spectrum interpolation) mode. The power-line frequency and its
harmonics are removed by replacing the *amplitude* of the spectrum inside a
narrow band around each line frequency with an interpolation of the amplitudes
in neighbouring frequency bins, while the original phase is preserved. The
cleaned signal is obtained by an inverse transform.

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

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Optional MNE support
try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw

    _HAS_MNE = True
except ImportError:
    mne = None
    _HAS_MNE = False

from ..utils import reconstruct_mne_object


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
    width ``bandwidth`` (centred on the frequency) is replaced by a linear
    interpolation between the mean amplitudes of the neighbouring reference
    bands on either side. The original phase is kept, following Leske & Dalal
    (2019) [1]_.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input time series. Each channel is processed independently.
    sfreq : float
        Sampling frequency in Hz.
    freqs : array-like of float
        Target frequencies (e.g. the line frequency and its harmonics) in Hz.
    bandwidth : float
        Full width in Hz of the band that is interpolated around each target
        frequency. Default 1.0.
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
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_channels, n_times), got {data.ndim}D")

    n_times = data.shape[1]
    nyquist = sfreq / 2.0
    half_bw = bandwidth / 2.0

    spectrum = np.fft.rfft(data, axis=1)
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    orig_mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    new_mag = orig_mag.copy()

    for f in np.atleast_1d(freqs).astype(float):
        if not (0.0 < f < nyquist):
            continue

        band = (fft_freqs >= f - half_bw) & (fft_freqs <= f + half_bw)
        if not np.any(band):
            # Band narrower than the frequency resolution: snap to nearest bin.
            nearest = int(np.argmin(np.abs(fft_freqs - f)))
            band = np.zeros_like(band)
            band[nearest] = True

        left = (fft_freqs >= f - half_bw - neighbour_width) & (fft_freqs < f - half_bw)
        right = (fft_freqs > f + half_bw) & (fft_freqs <= f + half_bw + neighbour_width)

        left_amp = orig_mag[:, left].mean(axis=1) if np.any(left) else None
        right_amp = orig_mag[:, right].mean(axis=1) if np.any(right) else None
        band_f = fft_freqs[band]

        if left_amp is not None and right_amp is not None:
            # The reference bands sit strictly below/above the band, so the
            # left mean frequency is always lower than the right one.
            left_f = fft_freqs[left].mean()
            right_f = fft_freqs[right].mean()
            weight = (band_f - left_f) / (right_f - left_f)
            replacement = (
                left_amp[:, None] * (1.0 - weight[None, :])
                + right_amp[:, None] * weight[None, :]
            )
        elif left_amp is not None:
            replacement = np.repeat(left_amp[:, None], band_f.size, axis=1)
        elif right_amp is not None:
            replacement = np.repeat(right_amp[:, None], band_f.size, axis=1)
        else:
            # No usable neighbours; leave this frequency untouched.
            continue

        new_mag[:, band] = replacement

    spectrum_clean = new_mag * np.exp(1j * phase)
    return np.fft.irfft(spectrum_clean, n=n_times, axis=1)


class SpectrumInterpolation(BaseEstimator, TransformerMixin):
    """Remove power-line noise by amplitude spectrum interpolation.

    Frequency-domain line-noise remover following Leske & Dalal (2019) [1]_, a
    translation of FieldTrip's ``ft_preproc_dftfilter`` spectrum-interpolation
    mode. The amplitude of a thin band around the line frequency (and its
    harmonics) is replaced by an interpolation of neighbouring amplitudes, while
    the phase is preserved.

    The estimator follows the scikit-learn ``fit`` / ``transform`` API and
    accepts MNE ``Raw``, ``Epochs`` and ``Evoked`` objects as well as plain
    NumPy arrays. Being a per-channel temporal operation, it is applied to every
    data channel regardless of sensor type; non-data channels (e.g. stim, misc)
    are passed through unchanged.

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
        fundamental). If None, all harmonics up to the Nyquist frequency are
        removed. Ignored when ``line_freq`` is a sequence.
    bandwidth : float
        Full width in Hz of the interpolated band around each frequency.
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

    References
    ----------
    .. [1] Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG
           and MEG data via spectrum interpolation. NeuroImage, 189, 763-776.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        line_freq: float | Any = 50.0,
        n_harmonics: int | None = None,
        bandwidth: float = 1.0,
        neighbour_width: float = 2.0,
    ) -> None:
        self.sfreq = sfreq
        self.line_freq = line_freq
        self.n_harmonics = n_harmonics
        self.bandwidth = bandwidth
        self.neighbour_width = neighbour_width

    def _resolve_sfreq(self, X: Any) -> float:
        if _HAS_MNE and isinstance(X, BaseRaw | BaseEpochs | Evoked):
            return float(X.info["sfreq"])
        if self.sfreq is None:
            raise ValueError("sfreq must be provided when fitting on a NumPy array.")
        return float(self.sfreq)

    def _target_freqs(self, sfreq: float) -> np.ndarray:
        nyquist = sfreq / 2.0
        if np.isscalar(self.line_freq):
            base = float(self.line_freq)
            max_h = int(np.floor(nyquist / base)) if base > 0 else 0
            n_h = max_h if self.n_harmonics is None else min(self.n_harmonics, max_h)
            candidates = [base * (k + 1) for k in range(n_h)]
        else:
            candidates = list(np.atleast_1d(self.line_freq).astype(float))
        freqs = sorted({f for f in candidates if 0.0 < f < nyquist})
        return np.asarray(freqs, dtype=float)

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
            out = np.empty_like(data)
            for i in range(data.shape[0]):
                out[i] = interpolate_spectrum(
                    data[i],
                    self.sfreq_,
                    self.freqs_,
                    bandwidth=self.bandwidth,
                    neighbour_width=self.neighbour_width,
                )
            return out
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")

    def fit(self, X: Any, y: Any = None) -> SpectrumInterpolation:
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
        sfreq = self._resolve_sfreq(X)
        self.sfreq_ = sfreq
        self.freqs_ = self._target_freqs(sfreq)
        return self

    def transform(self, X: Any) -> Any:
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
        if not hasattr(self, "freqs_"):
            raise RuntimeError("SpectrumInterpolation not fitted. Call fit() first.")

        if _HAS_MNE and isinstance(X, BaseRaw | BaseEpochs | Evoked):
            if isinstance(X, BaseEpochs):
                mne_type = "epochs"
            elif isinstance(X, Evoked):
                mne_type = "evoked"
            else:
                mne_type = "raw"

            data_picks = mne.pick_types(
                X.info,
                meg=True,
                eeg=True,
                seeg=True,
                ecog=True,
                dbs=True,
                fnirs=True,
                exclude=[],
            )
            if data_picks.size == 0:
                data_picks = np.arange(len(X.ch_names))

            data = X.get_data(picks=data_picks)
            cleaned = self._apply(data)
            return reconstruct_mne_object(
                cleaned, X, mne_type, picks=data_picks, verbose=False
            )

        return self._apply(np.asarray(X, dtype=float))

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
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
        return self.fit(X).transform(X)

