"""Temporal DSS variants.

Time-shift DSS augments every sensor with delayed copies, then composes the
package's :class:`DSS` estimator with :class:`AverageBias` across trials. The
initial public contract implements the repeated-trial contrast evaluated by
de Cheveigne (2010); arbitrary bias operators are intentionally outside that
claim. :func:`smooth_dss` remains the lightweight ordinary-DSS smoothing
configuration.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from mne.epochs import BaseEpochs

from ..._cca import canonical_correlation
from ..._data import extract_data_from_mne, reconstruct_mne_object
from ..._logging import logger, verbose
from ..._spatial import fit_mixing_matrix
from ..._validation import check_channel_layout, check_positive_integer, resolve_sfreq
from ..denoisers import AverageBias, SmoothingBias
from ..linear import DSS

_ACTIONS = frozenset({"extract", "retain", "subtract"})
_DISTORTION_CONTROLS = frozenset({None, "cca"})


def _resolve_lags(
    *,
    lag_samples: Sequence[int] | None,
    lag_times: Sequence[float] | None,
    sfreq: float | None,
) -> tuple[tuple[int, ...], tuple[float, ...] | None, float | None]:
    """Resolve exactly one physical or sample lag declaration."""

    def _as_samples(values: Sequence[int]) -> tuple[int, ...]:
        if isinstance(values, str | bytes):
            raise TypeError(
                "lag_samples must be a one-dimensional sequence of integers"
            )
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("lag_samples must be a non-empty one-dimensional sequence")
        samples = []
        for value in array.tolist():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("lag_samples must contain only integers")
            samples.append(int(value))
        resolved = tuple(sorted(set(samples)))
        if len(resolved) < 2 or 0 not in resolved:
            raise ValueError(
                "lag_samples must contain zero and at least one nonzero lag"
            )
        return resolved

    if (lag_samples is None) == (lag_times is None):
        raise ValueError("Provide exactly one of lag_samples or lag_times")
    if lag_samples is not None:
        samples = _as_samples(lag_samples)
        times = (
            tuple(sample / sfreq for sample in samples) if sfreq is not None else None
        )
        return samples, times, sfreq

    sfreq = resolve_sfreq(sfreq, None, context="lag_times")
    if isinstance(lag_times, str | bytes):
        raise TypeError("lag_times must be a one-dimensional sequence")
    raw_values = np.asarray(lag_times, dtype=object)
    if any(isinstance(value, bool) for value in raw_values.reshape(-1).tolist()):
        raise TypeError("lag_times must not contain booleans")
    values = np.asarray(lag_times, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "lag_times must be a non-empty finite one-dimensional sequence"
        )
    sample_values = values * sfreq
    rounded = np.rint(sample_values)
    if not np.allclose(sample_values, rounded, rtol=0.0, atol=1e-10):
        raise ValueError("Every lag_time must fall exactly on the sampling grid")
    samples = _as_samples([int(value) for value in rounded])
    return samples, tuple(sample / sfreq for sample in samples), sfreq


def _validate_epoched_array(data: Any) -> np.ndarray:
    """Return finite float data in channel-by-time-by-epoch orientation."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(
            "TimeShiftDSS requires epoched data shaped (n_channels, n_times, n_epochs)"
        )
    if min(data.shape) < 1 or data.shape[1] < 2:
        raise ValueError("TimeShiftDSS input dimensions must be non-empty")
    if data.shape[2] < 2:
        raise ValueError("TimeShiftDSS requires at least two repeated epochs")
    if not np.all(np.isfinite(data)):
        raise ValueError("TimeShiftDSS input must contain only finite values")
    return data


def _lag_augment(
    data: np.ndarray, lags: tuple[int, ...]
) -> tuple[np.ndarray, int, int]:
    """Stack lag-major sensor blocks without wrapping or joining epochs."""
    start = max(lags)
    stop = data.shape[1] + min(lags)
    if stop - start < 2:
        raise ValueError(
            "The lag span leaves fewer than two common time samples per epoch"
        )
    blocks = [data[:, start - lag : stop - lag, :] for lag in lags]
    return np.concatenate(blocks, axis=0), start, stop


def _observation_weights(
    weights: np.ndarray | None,
    *,
    n_times: int,
    n_epochs: int,
    lags: tuple[int, ...],
    start: int,
    stop: int,
) -> np.ndarray:
    """Validate weights and dilate zeros through every lagged observation."""
    if weights is None:
        return np.ones((stop - start, n_epochs), dtype=np.float64)

    base = np.asarray(weights, dtype=np.float64)
    if base.shape == (n_times,):
        base = np.broadcast_to(base[:, np.newaxis], (n_times, n_epochs)).copy()
    elif base.shape != (n_times, n_epochs):
        raise ValueError(
            "sample_weight must have shape "
            f"({n_times},) or ({n_times}, {n_epochs}); got {base.shape}"
        )
    if not np.all(np.isfinite(base)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(base < 0):
        raise ValueError("sample_weight must be non-negative")

    shifted = [base[start - lag : stop - lag, :] for lag in lags]
    valid = np.minimum.reduce(shifted)
    if not np.any(valid > 0):
        raise ValueError("sample_weight leaves no positive-weight lag observations")
    return valid


class TimeShiftDSS(BaseEstimator, TransformerMixin):
    """Trial-average DSS in a lag-augmented sensor space.

    Parameters
    ----------
    lag_samples : sequence of int | None
        Explicit lag grid in samples. It must contain zero and at least one
        nonzero lag. Positive lags contribute ``X(t - lag)``.
    lag_times : sequence of float | None
        Explicit lag grid in seconds. Every value must lie on the sampling
        grid. Exactly one lag representation must be provided.
    sfreq : float | None
        Sampling frequency for array data when ``lag_times`` is used. MNE
        metadata is authoritative and must agree with a supplied value.
    n_components : int
        Number of lag-space DSS components to fit. Selection is explicit;
        there is no in-sample automatic selector.
    rank : int
        Explicit whitening rank in the augmented feature space.
    n_select : int | None
        Size of the leading component subspace used by :meth:`score` and by
        sensor-space ``retain`` or ``subtract``. It is required for those
        operations; extraction itself can leave it unset.
    component_action : {'extract', 'retain', 'subtract'}
        Component extraction or sensor-space operation. Sensor operations
        preserve input shape and leave samples outside the common lag support
        unchanged.
    center : bool
        If ``False`` (default), use source-aligned uncentered second moments.
        If ``True``, fit one weighted augmented-feature mean and reuse it for
        every transform. Epoch-wise and transform-batch centering are never
        performed.
    distortion_control : {None, 'cca'}
        Optional paper step 7. ``'cca'`` rotates the fitted reproducible
        subspace to the single variate most correlated with undelayed training
        data. It returns one component and requires ``n_select=1`` for sensor
        operations.
    reg : float
        Relative numerical rank tolerance used by DSS and optional CCA.
    verbose : bool | str | int | None
        Logging verbosity.

    Notes
    -----
    Array input is ``(n_channels, n_times, n_epochs)``. MNE Epochs input is
    accepted natively. Continuous and Evoked inputs are intentionally rejected
    by this initial repeated-trial implementation.

    Lag augmentation is the TSDSS-specific layer. The fitted decomposition is
    available as ``dss_`` and is an ordinary :class:`DSS` configured with
    ``AverageBias(axis="epochs")``.

    Component interpretation and parameter choice require held-out and
    surrogate validation.
    """

    def __init__(
        self,
        *,
        lag_samples: Sequence[int] | None = None,
        lag_times: Sequence[float] | None = None,
        sfreq: float | None = None,
        n_components: int,
        rank: int,
        n_select: int | None = None,
        component_action: str = "extract",
        center: bool = False,
        distortion_control: str | None = None,
        reg: float = 1e-9,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.lag_samples = lag_samples
        self.lag_times = lag_times
        self.sfreq = sfreq
        self.n_components = n_components
        self.rank = rank
        self.n_select = n_select
        self.component_action = component_action
        self.center = center
        self.distortion_control = distortion_control
        self.reg = reg
        self.verbose = verbose

    def _validate_parameters(self) -> None:
        """Validate constructor state without mutating it."""
        check_positive_integer(self.n_components, name="n_components")
        check_positive_integer(self.rank, name="rank")
        if self.n_select is not None:
            check_positive_integer(self.n_select, name="n_select")
        if self.component_action not in _ACTIONS:
            raise ValueError(f"component_action must be one of {sorted(_ACTIONS)}")
        if self.component_action != "extract" and self.n_select is None:
            raise ValueError("n_select is required for retain and subtract actions")
        if not isinstance(self.center, bool):
            raise TypeError("center must be a bool")
        if self.distortion_control not in _DISTORTION_CONTROLS:
            raise ValueError("distortion_control must be None or 'cca'")
        if self.distortion_control == "cca" and self.n_select not in (None, 1):
            raise ValueError("CCA distortion control supports only n_select=1")
        if isinstance(self.reg, bool) or not isinstance(self.reg, Real):
            raise TypeError("reg must be a positive finite real number")
        if not np.isfinite(self.reg) or self.reg <= 0:
            raise ValueError("reg must be a positive finite real number")

    def _prepare_epochs(
        self,
        X: BaseEpochs | np.ndarray,
        *,
        fitting: bool,
    ) -> tuple[np.ndarray, float | None, str, Any, np.ndarray | None]:
        """Use shared extraction while enforcing the fitted epoch contract."""
        data, data_sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(
            X,
            ch_names=None if fitting else self._mne_ch_names_,
            channel_first_epochs=True,
            exclude_bads=fitting,
        )
        is_mne = mne_type == "epochs"
        if not is_mne and not isinstance(X, np.ndarray):
            raise TypeError("TimeShiftDSS supports MNE Epochs or NumPy arrays")
        if not fitting and is_mne != self._fit_was_mne_:
            raise TypeError("Transform input must use the container family used in fit")
        data = _validate_epoched_array(data)
        if fitting:
            self._fit_was_mne_ = is_mne
            self._mne_ch_names_ = ch_names
            if orig is not None:
                fitted = orig.copy()
                if picks is not None:
                    fitted.pick(picks)
                self.info_ = fitted.info
            else:
                self.info_ = None
        else:
            if self.sfreq_ is not None and data_sfreq is not None:
                resolve_sfreq(self.sfreq_, data_sfreq)
            check_channel_layout(
                "TimeShiftDSS",
                n_channels=data.shape[0],
                fitted_n_channels=self.n_features_in_,
                ch_names=ch_names,
                fitted_ch_names=self._mne_ch_names_,
            )
        return data, data_sfreq, mne_type, orig, picks

    @verbose
    def fit(
        self,
        X: BaseEpochs | np.ndarray,
        y: None = None,
        *,
        sample_weight: np.ndarray | None = None,
        verbose: bool | str | int | None = None,
    ) -> TimeShiftDSS:
        """Fit lag-augmented repeated-trial DSS filters."""
        del y
        self._validate_parameters()
        data, data_sfreq, _, _, _ = self._prepare_epochs(X, fitting=True)
        effective_sfreq = resolve_sfreq(
            self.sfreq,
            data_sfreq,
            context="lag_times",
            required=self.lag_times is not None,
        )
        lags, lag_times, effective_sfreq = _resolve_lags(
            lag_samples=self.lag_samples,
            lag_times=self.lag_times,
            sfreq=effective_sfreq,
        )
        augmented, start, stop = _lag_augment(data, lags)
        weights = _observation_weights(
            sample_weight,
            n_times=data.shape[1],
            n_epochs=data.shape[2],
            lags=lags,
            start=start,
            stop=stop,
        )
        n_features = augmented.shape[0]
        rank = check_positive_integer(self.rank, name="rank")
        n_components = check_positive_integer(self.n_components, name="n_components")
        if rank > n_features:
            raise ValueError(
                f"rank={rank} exceeds {n_features} augmented sensor-lag features"
            )
        if n_components > rank:
            raise ValueError("n_components cannot exceed rank")
        if self.n_select is not None and self.n_select > n_components:
            raise ValueError("n_select cannot exceed n_components")

        weight_flat = weights.reshape(-1)
        self.dss_ = DSS(
            bias=AverageBias(axis="epochs", weights=weights),
            n_components=n_components,
            rank=rank,
            reg=float(self.reg),
            normalize_input=False,
            center=self.center,
            cov_method="empirical",
            component_action="extract",
            # TimeShiftDSS owns the user-facing report.  The nested ordinary
            # DSS fit is a numerical implementation detail.
            verbose="WARNING",
        )
        # TimeShiftDSS owns the high-level report; the ordinary DSS fit is
        # only the lag-space numerical implementation.
        self.dss_.fit(augmented, weights=weights, verbose="WARNING")
        filters = self.dss_.filters_
        eigenvalues = self.dss_.eigenvalues_
        if filters.shape[0] < n_components:
            raise ValueError(
                f"n_components={n_components} exceeds the fitted numerical "
                f"whitening rank ({filters.shape[0]})"
            )
        feature_mean = self.dss_.mean_
        sources = self.dss_.transform(augmented)
        zero_index = lags.index(0)
        sensor_mean = feature_mean[
            zero_index * data.shape[0] : (zero_index + 1) * data.shape[0]
        ]

        self.cca_correlations_ = None
        self.cca_rotation_ = None
        self.cca_source_mean_ = None
        if self.distortion_control == "cca":
            source_2d = sources.reshape(sources.shape[0], -1)
            sensor_2d = data[:, start:stop, :].reshape(data.shape[0], -1)
            cca_source_mean = (source_2d @ weight_flat / weight_flat.sum())[
                :, np.newaxis
            ]
            cca_sensor_mean = (sensor_2d @ weight_flat / weight_flat.sum())[
                :, np.newaxis
            ]
            source_coefficients, _, correlations, _, _ = canonical_correlation(
                source_2d.T,
                sensor_2d.T,
                sample_weight=weight_flat,
                rtol=float(self.reg),
            )
            if correlations.size == 0:
                raise ValueError("CCA input has no variance above the rank threshold")
            rotation = source_coefficients[:, :1].T
            canonical = rotation @ (source_2d - cca_source_mean)
            canonical = canonical.reshape((1, *sources.shape[1:]))
            patterns = fit_mixing_matrix(
                data[:, start:stop, :] - cca_sensor_mean.reshape(data.shape[0], 1, 1),
                canonical,
                sample_weight=weights,
            )
            self.cca_rotation_ = rotation
            self.cca_source_mean_ = cca_source_mean
            self.cca_correlations_ = correlations
            sensor_mean = cca_sensor_mean
        else:
            sensors = data[:, start:stop, :] - sensor_mean.reshape(data.shape[0], 1, 1)
            patterns = fit_mixing_matrix(sensors, sources, sample_weight=weights)

        effective_observations = weight_flat.sum() ** 2 / np.dot(
            weight_flat, weight_flat
        )
        if n_features / effective_observations >= 0.5:
            warnings.warn(
                "The augmented feature count approaches the Kish effective "
                "observation count; TimeShiftDSS is at high risk of overfitting. "
                "Use held-out and surrogate validation.",
                UserWarning,
                stacklevel=2,
            )

        self.filters_ = (
            filters if self.cca_rotation_ is None else self.cca_rotation_ @ filters
        )
        self.patterns_ = patterns
        self.eigenvalues_ = eigenvalues
        self.feature_mean_ = feature_mean
        self.sensor_mean_ = sensor_mean
        self.lag_samples_ = lags
        self.lag_times_ = lag_times
        self.sfreq_ = effective_sfreq
        self.n_features_in_ = data.shape[0]
        self.n_augmented_features_ = n_features
        self.positive_weight_observations_ = int(np.count_nonzero(weight_flat > 0))
        self.effective_observations_ = float(effective_observations)
        self.valid_slice_ = slice(start, stop)
        logger.info(
            "TimeShiftDSS: lags=%s sample(s), rank=%d, components=%d, "
            "action=%s, distortion_control=%s, effective observations=%.1f.",
            self.lag_samples_,
            rank,
            self.filters_.shape[0],
            self.component_action,
            self.distortion_control or "none",
            self.effective_observations_,
        )
        return self

    def _sources(self, data: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Apply the frozen lag-space transform."""
        augmented, start, stop = _lag_augment(data, self.lag_samples_)
        sources = self.dss_.transform(augmented)
        sources_2d = sources.reshape(sources.shape[0], -1)
        if self.cca_rotation_ is not None:
            sources_2d = self.cca_rotation_ @ (sources_2d - self.cca_source_mean_)
        sources = sources_2d.reshape((sources_2d.shape[0], stop - start, data.shape[2]))
        return sources, start, stop

    def score(
        self,
        X: BaseEpochs | np.ndarray,
        y: None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Score the leading fitted subspace on held-out repeated trials.

        The score is the trial-average power divided by total power, summed
        over the first ``n_select`` components. A fixed ``n_select`` therefore
        defines one scalar model score suitable for whole-epoch validation.
        """
        del y
        check_is_fitted(self, "dss_")
        if self.n_select is None:
            raise ValueError("score requires an explicit n_select")
        data, _, _, _, _ = self._prepare_epochs(X, fitting=False)
        sources, start, stop = self._sources(data)
        selected = sources[: int(self.n_select)]
        weights = _observation_weights(
            sample_weight,
            n_times=data.shape[1],
            n_epochs=data.shape[2],
            lags=self.lag_samples_,
            start=start,
            stop=stop,
        )
        weight_per_time = weights.sum(axis=1)
        valid_times = weight_per_time > 0
        average = np.einsum("cte,te->ct", selected, weights, optimize=True)
        average[:, valid_times] /= weight_per_time[valid_times]
        evoked_power = float(
            np.sum(average[:, valid_times] ** 2 * weight_per_time[valid_times])
            / weight_per_time[valid_times].sum()
        )
        total_power = float(
            np.sum(selected**2 * weights[np.newaxis, :, :]) / weights.sum()
        )
        return evoked_power / total_power if total_power > 0 else 0.0

    @verbose
    def transform(
        self,
        X: BaseEpochs | np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ) -> BaseEpochs | np.ndarray:
        """Extract components or apply the fitted sensor-space operation."""
        check_is_fitted(self, "dss_")
        self._validate_parameters()
        data, _, mne_type, orig, picks = self._prepare_epochs(X, fitting=False)
        sources, start, stop = self._sources(data)
        if self.component_action == "extract":
            if orig is not None:
                return np.transpose(sources, (2, 0, 1))
            return sources

        count = int(self.n_select)
        selected = self.patterns_[:, :count] @ sources[:count].reshape(count, -1)
        selected = selected.reshape(data.shape[0], stop - start, data.shape[2])
        if self.component_action == "retain":
            valid_output = selected + self.sensor_mean_.reshape(data.shape[0], 1, 1)
        else:
            valid_output = data[:, start:stop, :] - selected
        output = data.copy()
        output[:, start:stop, :] = valid_output
        if orig is None:
            return output
        return reconstruct_mne_object(
            np.transpose(output, (2, 0, 1)),
            orig,
            mne_type,
            picks=picks,
        )


def smooth_dss(
    window: int = 10,
    *,
    n_components: int | None = None,
    **dss_kws,
) -> DSS:
    """Create an ordinary DSS configured for temporally smooth sources.

    Parameters
    ----------
    window : int
        Smoothing window in samples.
    n_components : int | None
        Number of DSS components to fit. ``None`` keeps the available rank.
    **dss_kws
        Additional keyword arguments passed to :class:`DSS`.

    Returns
    -------
    dss : DSS
        DSS configured with :class:`~mne_denoise.dss.SmoothingBias`.
    """
    bias = SmoothingBias(window=window)
    return DSS(bias=bias, n_components=n_components, **dss_kws)


__all__ = ["TimeShiftDSS", "smooth_dss"]
