"""Nonlinearities for iterative DSS."""

from __future__ import annotations

import numpy as np

from .base import NonlinearDenoiser


class TanhMaskDenoiser(NonlinearDenoiser):
    """Scaled hyperbolic-tangent nonlinearity.

    The transform is ``tanh(alpha * source)``. When ``normalize=True``, the
    source is divided by its standard deviation before the transform and the
    result is rescaled by the same value.

    Parameters
    ----------
    alpha : float, default=1.0
        Tanh scale.
    normalize : bool, default=True
        Whether to standardize and rescale the source around the transform.
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
    """Residual hyperbolic-tangent nonlinearity.

    The transform is ``source - tanh(alpha * source)``.

    Parameters
    ----------
    alpha : float, default=1.0
        Tanh scale.
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
    """Gaussian nonlinearity for iterative DSS.

    The transform is ``source * exp(-a * source**2 / 2)``.

    Parameters
    ----------
    a : float, default=1.0
        Gaussian scale.
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
    """Squared-source nonlinearity for iterative DSS.

    The transform is ``source**2``.
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
    """Configurable ICA contrast nonlinearity.

    Parameters
    ----------
    nonlinearity : {"tanh", "cube", "gauss"}, default="tanh"
        Transform to apply: ``tanh(alpha * source)``, ``source**3``, or
        ``source * exp(-0.5 * (alpha * source)**2)``.
    alpha : float, default=1.0
        Scale used by the ``"tanh"`` and ``"gauss"`` transforms.
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
    """Uniformly smoothed hyperbolic-tangent nonlinearity.

    Parameters
    ----------
    alpha : float, default=1.0
        Tanh scale.
    window : int, default=10
        Uniform-filter size; values below 3 are set to 3.

    Notes
    -----
    The implementation applies :func:`scipy.ndimage.uniform_filter1d` with its
    default last-axis behavior before applying ``tanh``.
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
    """Return the tanh fixed-point coefficient.

    The coefficient is ``-mean(1 - tanh(source)**2)``.

    Parameters
    ----------
    source : ndarray
        Source samples.
    """
    return -np.mean(1 - np.tanh(source) ** 2)


def beta_pow3(source: np.ndarray) -> float:
    """Return the cubic fixed-point coefficient ``-3.0``.

    Parameters
    ----------
    source : ndarray
        Source samples; used only to match the fixed-point coefficient API.
    """
    return -3.0


def beta_gauss(source: np.ndarray, a: float = 1.0) -> float:
    """Return the Gaussian fixed-point coefficient.

    The coefficient is ``-mean((1 - a * source**2) * exp(-a * source**2 / 2))``.

    Parameters
    ----------
    source : ndarray
        Source samples.
    a : float, default=1.0
        Gaussian scale.
    """
    s2 = source**2
    return -np.mean((1 - a * s2) * np.exp(-a * s2 / 2))
