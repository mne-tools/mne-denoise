"""Spectrum interpolation for line-noise removal."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _mne
from ._data import extract_data_from_mne, reconstruct_mne_object
from ._logging import logger, verbose
from ._validation import (
    check_matching_sfreq,
    check_positive_real,
    resolve_sfreq,
)

__all__ = ["SpectrumInterpolation", "interpolate_spectrum"]


def interpolate_spectrum(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    *,
    bandwidth: float = 1.0,
    neighbour_width: float = 2.0,
) -> np.ndarray:
    """Interpolate line-noise amplitudes in a 2-D signal spectrum.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Real-valued channel-first data.
    sfreq : float
        Sampling frequency in Hz.
    freqs : array-like
        Target frequencies in Hz.
    bandwidth : float, default=1.0
        Half-width of each replaced target band in Hz.
    neighbour_width : float, default=2.0
        Width of the neighboring reference bands in Hz.

    Returns
    -------
    ndarray, shape (n_channels, n_times)
        Cleaned data with the same shape and units as data.

    See Also
    --------
    SpectrumInterpolation
        Estimator that resolves target frequencies and preserves MNE containers.

    Notes
    -----
    Amplitudes in target bins are replaced using neighboring amplitudes while the
    original FFT phase is retained :footcite:p:`leske_dalal2019_spectrum`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.spectrum_interpolation import interpolate_spectrum
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> clean = interpolate_spectrum(data, sfreq=250.0, freqs=np.array([60.0, 120.0]))
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
    """Line-noise remover based on amplitude spectrum interpolation.

    Parameters
    ----------
    sfreq : float or None, default=None
        Sampling frequency in Hz; inferred from MNE metadata when available.
    line_freq : float or array-like, default=50.0
        Fundamental frequency or explicit target frequencies in Hz.
    n_harmonics : int or None, default=None
        Number of harmonics when line_freq is scalar; None uses all harmonics below
        Nyquist.
    bandwidth : float, default=1.0
        Half-width of each replaced band in Hz.
    neighbour_width : float, default=2.0
        Width of each neighboring reference band in Hz.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    sfreq_ : float
        Sampling frequency used during fit.
    freqs_ : ndarray
        Resolved target frequencies.

    See Also
    --------
    interpolate_spectrum
        One-shot array interface.
    mne_denoise.zapline.ZapLine
        DSS-based spatial line-noise removal.

    Notes
    -----
    NumPy input uses 2-D or 3-D channel-first layouts; 3-D records are processed
    independently. MNE data channels are processed while non-data channels and
    container metadata are preserved. Short segments may have limited spectral
    resolution :footcite:p:`leske_dalal2019_spectrum`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.spectrum_interpolation import SpectrumInterpolation
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> model = SpectrumInterpolation(sfreq=250.0, line_freq=60.0, n_harmonics=2)
    >>> clean = model.fit_transform(data)
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
        """Resolve the sampling frequency and target frequencies.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Input whose metadata or shape is inspected.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        SpectrumInterpolation
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
        """Apply spectrum interpolation.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data to clean.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            Cleaned data with the same shape.
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
        """Fit spectrum interpolation and transform X.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data to clean.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.
        **fit_params : dict
            Ignored for scikit-learn compatibility.

        Returns
        -------
        same type as X
            Cleaned data.
        """
        return self.fit(X, y).transform(X)
