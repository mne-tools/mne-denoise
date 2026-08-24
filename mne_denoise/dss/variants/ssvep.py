"""SSVEP DSS variant.

Convenience wrapper for extracting Steady-State Visually Evoked Potentials.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

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
    """Create a DSS configured for SSVEP extraction.

    Returns a pre-configured DSS object that can be fit on data to extract
    components locked to a stimulus frequency and its harmonics.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    stim_freq : float
        SSVEP stimulus frequency in Hz.
    n_harmonics : int
        Number of harmonics to include in the comb filter. Default 3.
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS` (e.g. `reg`, `normalize_input`).

    Returns
    -------
    dss : DSS
        A DSS object configured with a CombFilterBias. Call `.fit(data)` to
        compute spatial filters, then `.transform(data)` to extract sources.

    Examples
    --------
    >>> # Create SSVEP DSS for 12 Hz stimulation
    >>> dss = ssvep_dss(sfreq=250, stim_freq=12)
    >>> dss.fit(epochs)
    >>> ssvep_sources = dss.transform(epochs)

    >>> # Retain the leading SSVEP component in sensor space
    >>> dss = ssvep_dss(
    ...     sfreq=250,
    ...     stim_freq=12,
    ...     n_components=3,
    ...     n_select=1,
    ...     component_action="retain",
    ... )
    >>> dss.fit(epochs)
    >>> denoised_epochs = dss.transform(epochs)
    """
    bias = CombFilterBias(
        fundamental_freq=stim_freq,
        sfreq=sfreq,
        n_harmonics=n_harmonics,
    )
    return DSS(bias=bias, n_components=n_components, **dss_kws)
