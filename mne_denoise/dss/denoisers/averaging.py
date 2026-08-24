"""Averaging bias functions for DSS.

Implements trial/epoch and group/dataset averaging to enhance reproducible patterns.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
.. [3] de Cheveigné & Parra (2014). Joint denoising source separation. NeuroImage, 98, 489-496.
"""

from __future__ import annotations

import numpy as np

from .base import LinearDenoiser


class AverageBias(LinearDenoiser):
    """Bias function for finding repeatable components via averaging.

    Maximizes the reproducibility of patterns across trials (epochs) or
    datasets (subjects). This LinearDenoiser covers:
    - Trial averaging (axis='epochs'): for evoked response enhancement
    - Dataset averaging (axis='datasets'): for group-level repeatability (JDSS)

    Parameters
    ----------
    axis : str
        Dimension to average over:
        - 'epochs' (default): Average across trials. Input shape: (n_channels, n_times, n_epochs)
        - 'datasets': Average across datasets/subjects. Input shape: (n_datasets, n_channels, n_times)
    weights : array-like, optional
        Weights for averaging. For ``axis='epochs'``, either one weight per
        epoch or a ``(n_times, n_epochs)`` observation-weight matrix. For
        ``axis='datasets'``, one weight per dataset. If None, use uniform
        weighting.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import AverageBias
    >>> # For evoked response enhancement (like old TrialAverageBias)
    >>> epochs_data = np.random.randn(64, 100, 50)  # channels x times x trials
    >>> bias = AverageBias(axis="epochs")
    >>> biased = bias.apply(epochs_data)

    >>> # For group-level repeatability (like old JDSS)
    >>> group_data = np.random.randn(10, 64, 100)  # subjects x channels x times
    >>> bias = AverageBias(axis="datasets")
    >>> biased = bias.apply(group_data)

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.4 "DENOISING OF QUASIPERIODIC SIGNALS"
    de Cheveigné & Parra (2014). Joint denoising source separation.
    """

    def __init__(self, axis: str = "epochs", weights: np.ndarray | None = None) -> None:
        if axis not in ("epochs", "datasets"):
            raise ValueError(f"axis must be 'epochs' or 'datasets', got {axis!r}")
        self.axis = axis
        self.weights = weights

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply averaging bias.

        Parameters
        ----------
        data : ndarray
            Input data.
            - For axis='epochs': shape (n_channels, n_times, n_epochs)
            - For axis='datasets': shape (n_datasets, n_channels, n_times)

        Returns
        -------
        biased : ndarray, same shape as input
            Data where each slice is replaced by the weighted average.
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
