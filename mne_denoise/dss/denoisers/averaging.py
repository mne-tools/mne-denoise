"""Averaging bias functions for DSS."""

from __future__ import annotations

import numpy as np

from .base import LinearDenoiser


class AverageBias(LinearDenoiser):
    """Averaging bias for repeatable DSS structure.

    Parameters
    ----------
    axis : {"epochs", "datasets"}, default="epochs"
        For ``"epochs"``, input has shape ``(n_channels, n_times, n_epochs)``.
        For ``"datasets"``, input has shape ``(n_datasets, n_channels, n_times)``.
    weights : array-like or None, default=None
        Epoch weights, optionally a ``(n_times, n_epochs)`` observation-weight
        matrix, or one weight per dataset depending on ``axis``.

    Notes
    -----
    ``axis="datasets"`` is a low-level dataset-first bias operation; it is not a
    second NumPy input layout for the channel-first :class:`~mne_denoise.dss.DSS`
    estimator.
    """

    def __init__(self, axis: str = "epochs", weights: np.ndarray | None = None) -> None:
        if axis not in ("epochs", "datasets"):
            raise ValueError(f"axis must be 'epochs' or 'datasets', got {axis!r}")
        self.axis = axis
        self.weights = weights

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the averaging bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times, n_epochs) or (n_datasets, n_channels, n_times)
            Shape is determined by ``axis``.

        Returns
        -------
        ndarray
            Weighted-average data with the input shape.
        """
        if self.axis == "epochs":
            return self._apply_epochs(data)
        else:  # datasets
            return self._apply_datasets(data)

    def _apply_epochs(self, data: np.ndarray) -> np.ndarray:
        """Average across epochs (last axis)."""
        if data.ndim != 3:
            raise ValueError(
                f"AverageBias(axis='epochs') requires 3D data "
                f"(n_channels, n_times, n_epochs), got shape {data.shape}"
            )

        n_channels, n_times, n_epochs = data.shape

        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("weights must be finite and non-negative")
            if weights.shape == (n_epochs,):
                if weights.sum() <= 0:
                    raise ValueError("weights must have a positive sum")
                avg = np.tensordot(
                    data, weights / weights.sum(), axes=(2, 0)
                )  # (n_ch, n_times)
            elif weights.shape == (n_times, n_epochs):
                weight_per_time = weights.sum(axis=1)
                if not np.any(weight_per_time > 0):
                    raise ValueError("weights must contain a positive observation")
                weighted_sum = np.einsum("cte,te->ct", data, weights, optimize=True)
                avg = np.divide(
                    weighted_sum,
                    weight_per_time[np.newaxis, :],
                    out=np.zeros_like(weighted_sum, dtype=float),
                    where=weight_per_time[np.newaxis, :] > 0,
                )
            else:
                if weights.ndim == 1:
                    raise ValueError(
                        f"weights length ({weights.size}) must match "
                        f"n_epochs ({n_epochs})"
                    )
                raise ValueError(
                    "weights must have shape "
                    f"({n_epochs},) or ({n_times}, {n_epochs}); got {weights.shape}"
                )
        else:
            avg = data.mean(axis=2)

        # Broadcast average to all epochs
        biased = np.broadcast_to(avg[:, :, np.newaxis], data.shape).copy()
        return biased

    def _apply_datasets(self, data: np.ndarray) -> np.ndarray:
        """Average across datasets."""
        if data.ndim != 3:
            raise ValueError("AverageBias(axis='datasets') requires 3D data.")

        # Typically, for group DSS (JDSS), the input data shape might be
        # (n_datasets, n_channels, n_times). We assume axis=0 corresponds to datasets.

        n_datasets, n_channels, n_times = data.shape

        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != (n_datasets,):
                if weights.ndim == 1:
                    raise ValueError(
                        f"weights length ({weights.size}) must match "
                        f"n_datasets ({n_datasets})"
                    )
                raise ValueError(
                    f"weights must have shape ({n_datasets},); got {weights.shape}"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("weights must be finite and non-negative")
            if weights.sum() <= 0:
                raise ValueError("weights must have a positive sum")
            weights = weights / weights.sum()
            avg = np.tensordot(weights, data, axes=(0, 0))  # (n_ch, n_times)
        else:
            avg = data.mean(axis=0)

        # Broadcast average to all datasets
        biased = np.broadcast_to(avg[np.newaxis, :, :], data.shape).copy()
        return biased
