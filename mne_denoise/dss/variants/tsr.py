"""Lag-averaging and smoothing DSS convenience constructors.

These bias-side temporal operators extract autocorrelated signals, slow waves,
and DC shifts. They do not implement the data-side lag augmentation of true
time-shift DSS [1]_; use the separate experimental ``TimeShiftDSS`` estimator
for that data-side operation.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] de Cheveigné, A. (2010). Time-shift denoising source separation.
       Journal of Neuroscience Methods, 189(1), 113-120.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..denoisers.temporal import LagAveragingBias, SmoothingBias
from ..linear import DSS


def lag_averaging_dss(
    shifts: int | np.ndarray = 10,
    *,
    method: str = "autocorrelation",
    n_components: int | None = None,
    **dss_kws,
) -> DSS:
    """Create a DSS configured with a bias-side lag average.

    Returns a pre-configured DSS object that extracts components with
    high lag predictability. The resulting filter is spatial DSS; this helper
    does not construct data-side lag-augmented filters.

    Parameters
    ----------
    shifts : int or array-like
        If int, use lags from 1 to shifts.
        If array, use specified lag values in samples.
        Default 10.
    method : str
        Method for constructing bias:
        - 'autocorrelation': Average of shifted versions (default)
        - 'prediction': Weighted average (closer lags weighted more)
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    dss : DSS
        A DSS object configured with a :class:`LagAveragingBias`.

    Examples
    --------
    >>> # Extract temporally predictable components
    >>> dss = lag_averaging_dss(shifts=20)
    >>> dss.fit(data)
    >>> slow_sources = dss.transform(data)

    >>> # Use specific lags
    >>> dss = lag_averaging_dss(shifts=np.array([1, 2, 5, 10, 20]))
    >>> dss.fit(data)
    """
    bias = LagAveragingBias(shifts=shifts, method=method)
    return DSS(bias=bias, n_components=n_components, **dss_kws)


def time_shift_dss(
    shifts: int | np.ndarray = 10,
    *,
    method: str = "autocorrelation",
    n_components: int | None = None,
    **dss_kws,
) -> DSS:
    """Construct lag-averaging DSS through its deprecated compatibility name.

    The released helper configures bias-side lag averaging rather than true
    time-shift DSS. Its 0.x numerical behavior is retained under the canonical
    name while this compatibility wrapper emits ``FutureWarning``.
    """
    warnings.warn(
        "'time_shift_dss' is deprecated and will be removed in mne-denoise "
        "1.0; use 'lag_averaging_dss' instead. For true data-side lag "
        "augmentation, use the separate experimental 'TimeShiftDSS' "
        "estimator.",
        FutureWarning,
        stacklevel=2,
    )
    return lag_averaging_dss(
        shifts=shifts,
        method=method,
        n_components=n_components,
        **dss_kws,
    )


def smooth_dss(
    window: int = 10,
    *,
    n_components: int | None = None,
    **dss_kws,
) -> DSS:
    """Create a DSS configured for temporally smooth sources.

    Returns a pre-configured DSS that extracts components with
    low-frequency temporal structure.

    Parameters
    ----------
    window : int
        Smoothing window size in samples. Default 10.
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    dss : DSS
        A DSS object configured with a SmoothingBias.

    Examples
    --------
    >>> # Extract slow components
    >>> dss = smooth_dss(window=20)
    >>> dss.fit(data)
    >>> slow_sources = dss.transform(data)
    """
    bias = SmoothingBias(window=window)
    return DSS(bias=bias, n_components=n_components, **dss_kws)
