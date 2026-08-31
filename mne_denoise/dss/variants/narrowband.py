"""Narrowband DSS variants."""

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
    """Create a DSS estimator with a bandpass bias.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Center frequency in Hz.
    bandwidth : float, default=2.0
        Bandpass width in Hz.
    n_components : int or None, default=None
        Number of DSS components.
    **dss_kws
        Additional keyword arguments for :class:`~mne_denoise.dss.DSS`.

    Returns
    -------
    DSS
        Configured estimator.
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
    """Scan candidate frequencies with narrowband DSS.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Channel-first input data.
    sfreq : float
        Sampling frequency in Hz.
    freq_range : tuple of float, default=(1, 40)
        Candidate frequency range in Hz; it is clipped to the implementation's
        valid range.
    freq_step : float, default=1.0
        Candidate spacing in Hz.
    bandwidth : float, default=2.0
        Bandpass width in Hz.
    n_components : int, default=1
        Components fitted at each candidate.
    callback : callable or None, default=None
        Synchronous callback after each attempted candidate.
    verbose : bool, str, int, or None, default=None
        Logging level.
    **dss_kws
        Additional keyword arguments for :class:`~mne_denoise.dss.DSS`.

    Returns
    -------
    best_dss : DSS
        Fitted DSS at the highest-scoring candidate.
    frequencies : ndarray, shape (n_freqs,)
        Candidate frequencies.
    scores : ndarray, shape (n_freqs,)
        Leading DSS eigenvalue for each candidate. Failed candidates have score
        zero and the scan continues.
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
