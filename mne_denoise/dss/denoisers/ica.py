"""ICA-based nonlinearities for DSS (FastICA equivalence).

This module implements nonlinear denoising functions $f(s)$ that correspond to
maximizing non-Gaussianity (negentropy, kurtosis), effectively making DSS
equivalent to FastICA or RobustICA.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent
       component analysis. IEEE Trans. Neural Netw., 10(3), 626-634.
"""

from __future__ import annotations

import numpy as np

from .base import NonlinearDenoiser


class TanhMaskDenoiser(NonlinearDenoiser):
    r"""Tanh mask denoiser (Standard FastICA nonlinearity).

    Implements the hyperbolic tangent nonlinearity used widely in ICA for super-Gaussian
    source extraction. It is robust to outliers compared to kurtosis ($s^3$).

    Formula:
        $s_{new} = \\tanh(\\alpha \\cdot s)$

    Parameters
    ----------
    alpha : float
        Scaling factor controlling the saturation slope. Default 1.0.
    normalize : bool
        If True, normalizes the source to unit variance before applying tanh, then
        rescales back. This ensures $\\alpha=1$ has a consistent meaning. Default True.

    Examples
    --------
    >>> # Use for robust ICA
    >>> from mne_denoise.dss.denoisers import TanhMaskDenoiser, beta_tanh
    >>> denoiser = TanhMaskDenoiser()
    >>> dss = IterativeDSS(denoiser=denoiser, beta=beta_tanh)

    References
    ----------
    Särelä & Valpola (2005). Section 4.2.2 "BETTER ESTIMATE FOR THE SIGNAL VARIANCE"
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        normalize: bool = True,
    ) -> None:
        self.alpha = alpha
        self.normalize = normalize

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the scaled hyperbolic-tangent nonlinearity.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series. Normalization, when enabled, is computed over
            the supplied values.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            ``tanh(alpha * source)`` with optional standard-deviation scaling.
        """
        if self.normalize:
            std = np.std(source)
            if std > 1e-15:
                source_scaled = source / std
                denoised = np.tanh(self.alpha * source_scaled)
                return denoised * std
            else:
                return source

        return np.tanh(self.alpha * source)


class RobustTanhDenoiser(NonlinearDenoiser):
    r"""Robust tanh denoiser (FastICA / RobustICA formulation).

    Implements:
        $s_{new} = s - \\tanh(\\alpha \\cdot s)$

    This form is often used in deflationary FastICA schemas (like `pow3`) where strictly
    structure relates to optimizing specific cost functions (like negentropy).

    Parameters
    ----------
    alpha : float
        Scaling factor. Default 1.0.

    Examples
    --------
    >>> # Use for robust ICA
    >>> from mne_denoise.dss.denoisers import RobustTanhDenoiser, beta_tanh
    >>> denoiser = RobustTanhDenoiser()
    >>> dss = IterativeDSS(denoiser=denoiser, beta=beta_tanh)

    References
    ----------
    Särelä & Valpola (2005). Section 4.2.2 "BETTER ESTIMATE FOR THE SIGNAL VARIANCE"
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the robust hyperbolic-tangent nonlinearity.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            ``source - tanh(alpha * source)``.
        """
        return source - np.tanh(self.alpha * source)


class GaussDenoiser(NonlinearDenoiser):
    r"""Gaussian nonlinearity (FastICA 'gauss').

    Implements:
        $s_{new} = s \\cdot \\exp(-a s^2 / 2)$

    This nonlinearity is robust and works well for super-Gaussian distributions but
    is also capable of separating sub-Gaussian sources depending on the sign of kurtosis.
    It corresponds to the derivative of the Gaussian function.

    Parameters
    ----------
    a : float
        Parameter 'a_1' in FastICA literature. Default 1.0.

    Examples
    --------
    >>> # Use for robust ICA
    >>> from mne_denoise.dss.denoisers import GaussDenoiser, beta_gauss
    >>> denoiser = GaussDenoiser()
    >>> dss = IterativeDSS(denoiser=denoiser, beta=beta_gauss)

    References
    ----------
    Särelä & Valpola (2005). Section 4.2.2 "BETTER ESTIMATE FOR THE SIGNAL VARIANCE"
    """

    def __init__(self, a: float = 1.0) -> None:
        self.a = a

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the Gaussian FastICA nonlinearity.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            ``source * exp(-a * source**2 / 2)``.
        """
        s2 = source**2
        return source * np.exp(-self.a * s2 / 2)


class SkewDenoiser(NonlinearDenoiser):
    """Skewness nonlinearity (FastICA 'skew').

    Implements:
        $s_{new} = s^2$

    Used for extracting sources with asymmetric probability distributions.
    This maximizes skewness rather than kurtosis.

    Examples
    --------
    >>> # Use for robust ICA
    >>> from mne_denoise.dss.denoisers import SkewDenoiser, beta_gauss
    >>> denoiser = SkewDenoiser()
    >>> dss = IterativeDSS(denoiser=denoiser, beta=beta_gauss)

    References
    ----------
    Särelä & Valpola (2005). Section 4.2.2 "BETTER ESTIMATE FOR THE SIGNAL VARIANCE"
    """

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the squared-source skewness nonlinearity.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            Element-wise squared source.
        """
        return source**2


class KurtosisDenoiser(NonlinearDenoiser):
    """Kurtosis maximization denoiser.

    Can wrap different nonlinearities ('tanh', 'cube', 'gauss') to maximize non-Gaussianity.
    Included for checking various ICA contrasts.

    Parameters
    ----------
    nonlinearity : {'tanh', 'cube', 'gauss'}
        The function $g(s)$ to use. 'cube' ($s^3$) is the classic kurtosis maximization.
    alpha : float
        Scaling parameter.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import KurtosisDenoiser
    >>> denoiser = KurtosisDenoiser(nonlinearity="cube")
    >>> denoised = denoiser.denoise(source)

    References
    ----------
    Särelä & Valpola (2005). Section 4.2.1 "KURTOSIS-BASED ICA"
    """

    def __init__(
        self,
        nonlinearity: str = "tanh",
        alpha: float = 1.0,
    ) -> None:
        if nonlinearity not in ("tanh", "cube", "gauss"):
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        self.nonlinearity = nonlinearity
        self.alpha = alpha

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the configured ICA contrast nonlinearity.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            Transformed source using ``tanh``, ``cube``, or ``gauss``.
        """
        if self.nonlinearity == "tanh":
            return np.tanh(self.alpha * source)
        elif self.nonlinearity == "cube":
            return source**3
        else:  # self.nonlinearity == 'gauss' (validated in __init__)
            return source * np.exp(-0.5 * (self.alpha * source) ** 2)


class SmoothTanhDenoiser(NonlinearDenoiser):
    r"""Smoothed tanh denoiser.

    Applies temporal smoothing before the tanh nonlinearity. This can help
    extract sources with both temporal structure and non-Gaussian statistics.

    Formula:
        $s_{smoothed} = \\text{lowpass}(s)$
        $s_{new} = \\tanh(\\alpha \\cdot s_{smoothed})$

    Parameters
    ----------
    alpha : float
        Scaling factor for tanh. Default 1.0.
    window : int
        Smoothing window size in samples. Default 10.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import SmoothTanhDenoiser
    >>> denoiser = SmoothTanhDenoiser(window=20)
    >>> dss = IterativeDSS(denoiser=denoiser)
    """

    def __init__(self, alpha: float = 1.0, window: int = 10) -> None:
        self.alpha = alpha
        self.window = max(3, window)

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Smooth a source and apply the scaled tanh nonlinearity.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series. Smoothing is applied along the time axis.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            Smoothed and nonlinearly transformed source.
        """
        from scipy.ndimage import uniform_filter1d

        # Smooth the source
        smoothed = uniform_filter1d(source, size=self.window, mode="reflect")

        # Apply tanh to smoothed signal
        return np.tanh(self.alpha * smoothed)


# =============================================================================
# Helper functions for beta (Newton step)
# =============================================================================


def beta_tanh(source: np.ndarray) -> float:
    r"""Compute beta for Tanh denoiser (FastICA Newton step).

    Parameters
    ----------
    source : ndarray
        Source samples used to estimate the expectation.

    Returns
    -------
    beta : float
        Estimated Newton-step coefficient,
        ``-mean(1 - tanh(source)**2)``.

    Notes
    -----
    Formula: $\\beta = -E[1 - \\tanh^2(s)]$.
    Legacy: `beta_tanh.m`
    """
    return -np.mean(1 - np.tanh(source) ** 2)


def beta_pow3(source: np.ndarray) -> float:
    r"""Compute beta for Cubic ($s^3$) denoiser.

    Parameters
    ----------
    source : ndarray
        Unused source array accepted for a common beta-function interface.

    Returns
    -------
    beta : float
        The constant ``-3`` used for the cubic nonlinearity under the
        unit-variance DSS convention.

    Notes
    -----
    For $g(s)=s^3$, $g'(s)=3s^2$ and the unit-variance assumption gives
    $E[g'(s)] = 3$, so $\\beta=-3$.
    Legacy: `beta_pow3.m`
    """
    return -3.0


def beta_gauss(source: np.ndarray, a: float = 1.0) -> float:
    r"""Compute beta for Gaussian denoiser.

    Parameters
    ----------
    source : ndarray
        Source samples used to estimate the expectation.
    a : float, default=1.0
        Gaussian nonlinearity scale.

    Returns
    -------
    beta : float
        Estimated coefficient
        ``-mean((1 - a * source**2) * exp(-a * source**2 / 2))``.

    Notes
    -----
    Formula: $\\beta = -E[(1 - a s^2) \\exp(-a s^2 / 2)]$.
    Legacy: `beta_gauss.m`
    """
    s2 = source**2
    return -np.mean((1 - a * s2) * np.exp(-a * s2 / 2))
