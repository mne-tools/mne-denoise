"""Narrowband DSS variant.

Frequency-targeted DSS for extracting oscillatory components.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np

from ..._logging import logger, verbose
from ...progress import _emit_progress, _ProgressCallback, _validate_callback
from ..denoisers.spectral import BandpassBias
from ..linear import DSS


def narrowband_dss(
    sfreq: float,
    freq: float,
    *,
    bandwidth: float = 2.0,
    n_components: int | None = None,
    **dss_kws,
) -> DSS:
    """Create a DSS configured for a specific frequency band.

    Returns a pre-configured DSS object that extracts components with
    maximum power in the specified frequency band.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Target center frequency in Hz.
    bandwidth : float
        Bandwidth of the bandpass filter in Hz. Default 2.0.
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    dss : DSS
        A DSS object configured with a BandpassBias.

    Examples
    --------
    >>> # Extract 10 Hz (alpha) components
    >>> dss = narrowband_dss(sfreq=250, freq=10)
    >>> dss.fit(data)
    >>> alpha_sources = dss.transform(data)
    """
    low = freq - bandwidth / 2
    high = freq + bandwidth / 2
    bias = BandpassBias(freq_band=(low, high), sfreq=sfreq)
    return DSS(bias=bias, n_components=n_components, **dss_kws)


@verbose
def narrowband_scan(
    data: np.ndarray,
    sfreq: float,
    *,
    freq_range: tuple[float, float] = (1, 40),
    freq_step: float = 1.0,
    bandwidth: float = 2.0,
    n_components: int = 1,
    callback: _ProgressCallback | None = None,
    verbose: bool | str | int | None = None,
    **dss_kws,
) -> tuple[DSS, np.ndarray, np.ndarray]:
    """Scan frequencies to find optimal narrowband DSS components.

    Sweeps through a frequency range, computing DSS at each frequency.
    Returns the fitted DSS at the best frequency, along with the full
    eigenvalue spectrum for visualization.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    freq_range : tuple of float
        (min_freq, max_freq) range to scan. Default (1, 40).
    freq_step : float
        Frequency step size in Hz. Default 1.0.
    bandwidth : float
        Bandwidth of bandpass filter at each frequency. Default 2.0.
    n_components : int
        Number of DSS components to compute at each frequency. Default 1.
    callback : callable | None
        Called synchronously after each attempted candidate frequency with a
        ProgressEvent. Successful candidates report their leading DSS
        eigenvalue in ``metric``; failed candidates report ``metric=None``.
        Callback return values are ignored and callback exceptions propagate
        unchanged. Events use ``method="narrowband_scan"`` and
        ``stage="frequency"``.
    verbose : bool | str | int | None
        MNE-style logging level. Per-frequency DSS fits are reported through
        this scan's aggregate result.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    best_dss : DSS
        Fitted DSS at the frequency with highest eigenvalue.
    frequencies : ndarray, shape (n_freqs,)
        Frequencies that were scanned.
    eigenvalues : ndarray, shape (n_freqs,)
        First eigenvalue at each frequency.

    Examples
    --------
    >>> # Find dominant alpha frequency
    >>> best_dss, freqs, eigs = narrowband_scan(data, sfreq=250, freq_range=(7, 14))
    >>> print(f"Peak alpha at {freqs[np.argmax(eigs)]:.1f} Hz")
    >>> alpha_sources = best_dss.transform(data)

    >>> # Plot eigenvalue spectrum
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freqs, eigs)
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("DSS Eigenvalue")
    """
    callback = _validate_callback(callback)
    data = np.asarray(data)

    if dss_kws.get("adaptive", False):
        raise ValueError(
            "narrowband_scan does not support adaptive=True. The scan ranks "
            "candidate frequencies by a single global eigenvalue, so it would "
            "silently ignore the segmentation. Scan first to locate the "
            "frequency, then clean with adaptive=True at that frequency:\n"
            "    best, freqs, eigs = narrowband_scan(data, sfreq=sfreq)\n"
            "    peak = freqs[np.argmax(eigs)]\n"
            "    dss = narrowband_dss(sfreq=sfreq, freq=peak, adaptive=True)\n"
            "    cleaned = dss.fit_transform(data)"
        )

    nyquist = sfreq / 2
    min_freq, max_freq = freq_range

    # Validate frequency range
    min_freq = max(min_freq, 0.5)
    max_freq = min(max_freq, nyquist * 0.9)

    frequencies = np.arange(min_freq, max_freq + freq_step, freq_step)
    n_freqs = len(frequencies)
    eigenvalues = np.zeros(n_freqs)

    best_eigenvalue = -np.inf
    best_dss = None
    best_index = None

    for i, freq in enumerate(frequencies):
        metric = None
        try:
            candidate_kws = dict(dss_kws)
            # The scan owns the aggregate result; candidate DSS instances
            # are hidden so each tested frequency does not emit INFO.
            candidate_kws["verbose"] = "WARNING"
            dss = narrowband_dss(
                sfreq=sfreq,
                freq=freq,
                bandwidth=bandwidth,
                n_components=n_components,
                **candidate_kws,
            )
            dss.fit(data, verbose="WARNING")
            eigenvalues[i] = dss.eigenvalues_[0]
            metric = float(eigenvalues[i])

            logger.debug(
                "Narrowband DSS scan %d/%d: %.3g Hz eigenvalue=%.6g.",
                i + 1,
                n_freqs,
                freq,
                eigenvalues[i],
            )

            if eigenvalues[i] > best_eigenvalue:
                best_eigenvalue = eigenvalues[i]
                best_dss = dss
                best_index = i

        except Exception as exc:
            # Skip problematic frequencies
            logger.debug(
                "Narrowband DSS scan %d/%d: %.3g Hz failed: %s.",
                i + 1,
                n_freqs,
                freq,
                exc,
            )

        _emit_progress(
            callback,
            method="narrowband_scan",
            stage="frequency",
            current=i + 1,
            total=n_freqs,
            component=None,
            metric=metric,
        )

    if best_dss is None:
        raise RuntimeError("Failed to fit DSS at any frequency")

    logger.info(
        "Narrowband DSS scan: %.3g-%.3g Hz in %.3g Hz steps, "
        "bandwidth=%.3g Hz, best=%.3g Hz (eigenvalue=%.6g).",
        min_freq,
        max_freq,
        freq_step,
        bandwidth,
        frequencies[best_index],
        best_eigenvalue,
    )

    return best_dss, frequencies, eigenvalues
