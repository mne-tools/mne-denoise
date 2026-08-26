"""Reference-aware bias operator for linear DSS."""

from __future__ import annotations

import numpy as np

from .base import LinearDenoiser


class ReferenceBias(LinearDenoiser):
    """Return the part of primary data predictable from external references.

    The operator builds a time-domain least-squares prediction

    ``X_hat = X R.T (R R.T + ridge I)^-1 R``

    where ``X`` is the primary sensor data and ``R`` contains temporally aligned
    reference channels. DSS applied with this bias ranks spatial directions by
    reference-predictable variance. The reference is explicit constructor state,
    so benchmark registries can label the method as reference-aware.

    Parameters
    ----------
    reference : ndarray, shape (n_reference, n_times)
        Temporally aligned external reference data.
    ridge : float, default=1e-8
        Relative diagonal loading applied to the reference covariance.
    """

    def __init__(self, reference: np.ndarray, *, ridge: float = 1e-8) -> None:
        self.reference = np.asarray(reference, dtype=np.float64)
        self.ridge = float(ridge)
        if self.reference.ndim != 2 or self.reference.shape[0] == 0:
            raise ValueError("reference must have shape (n_reference, n_times)")
        if not np.all(np.isfinite(self.reference)):
            raise ValueError("reference must contain only finite values")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Project primary data onto the row space of the reference."""
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 3:
            channels, times, epochs = data.shape
            flat = data.transpose(0, 2, 1).reshape(channels, epochs * times)
            reference = self.reference
            if reference.shape[1] == times:
                reference = np.tile(reference, epochs)
            predicted = self._predict(flat, reference)
            return predicted.reshape(channels, epochs, times).transpose(0, 2, 1)
        if data.ndim != 2:
            raise ValueError("data must be 2D or 3D")
        return self._predict(data, self.reference)

    def _predict(self, data: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if data.shape[1] != reference.shape[1]:
            raise ValueError(
                "data and reference must have the same number of time samples"
            )
        centered_data = data - data.mean(axis=1, keepdims=True)
        centered_ref = reference - reference.mean(axis=1, keepdims=True)
        gram = centered_ref @ centered_ref.T
        loading = self.ridge * max(float(np.trace(gram)) / gram.shape[0], 1.0)
        coefficients = np.linalg.solve(
            gram + loading * np.eye(gram.shape[0]),
            centered_ref @ centered_data.T,
        ).T
        return coefficients @ centered_ref


__all__ = ["ReferenceBias"]
