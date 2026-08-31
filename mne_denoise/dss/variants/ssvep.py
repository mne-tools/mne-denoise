"""SSVEP DSS convenience wrappers."""

from __future__ import annotations

from ..denoisers.periodic import CombFilterBias
from ..linear import DSS


def ssvep_dss(
    sfreq: float,
    stim_freq: float,
    *,
    n_harmonics: int = 3,
    n_components: int | None = None,
    **dss_kws,
) -> DSS:
    """Create a DSS estimator configured with :class:`CombFilterBias`.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    stim_freq : float
        Fundamental stimulus frequency in Hz.
    n_harmonics : int, default=3
        Number of harmonics.
    n_components : int or None, default=None
        Number of DSS components.
    **dss_kws
        Additional keyword arguments for :class:`~mne_denoise.dss.DSS`.

    Returns
    -------
    DSS
        Configured estimator.
    """
    bias = CombFilterBias(
        fundamental_freq=stim_freq,
        sfreq=sfreq,
        n_harmonics=n_harmonics,
    )
    return DSS(bias=bias, n_components=n_components, **dss_kws)
