"""DSS-guided soft Artifact Subspace Reconstruction (experimental).

``GuidedASR`` deliberately reuses the public :class:`~mne_denoise.asr.ASR`
fit and reconstruction path.  It never replaces ASR's transient-abnormality
detector.  Instead, an optional bank of deterministic DSS bias operators scores
the signal that hard ASR would remove in each processing window.  A preserve
bias can rescue target-like residual activity, while an artifact bias keeps the
hard-ASR suppression.  This makes the extra information used by the method
explicit and inspectable.

The estimator is experimental.  Soft reconstruction therefore requires
``experimental=True``.  ``reconstruction='hard'`` is an exact pass-through to
the selected ASR backend and is useful as a matched control.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import extract_data_from_mne, reconstruct_mne_object
from .core import ASR

_EPS = float(np.finfo(np.float64).eps)


def _normalize_cov(cov: np.ndarray) -> np.ndarray:
    """Return a symmetric covariance with mean eigenvalue equal to one."""
    cov = np.asarray(cov, dtype=np.float64)
    cov = (cov + cov.T) / 2.0
    trace = float(np.trace(cov))
    if trace > _EPS:
        cov = cov * (cov.shape[0] / trace)
    return cov


def _soft_component_weights(
    variances: np.ndarray,
    directions: np.ndarray,
    thresholds_squared: np.ndarray,
    *,
    forced_keep: np.ndarray,
    artifact_cov: np.ndarray | None,
    preserve_cov: np.ndarray | None,
    soft_weight: str,
    scale: float,
) -> np.ndarray:
    """Compute bounded keep weights for documented component-level analyses.

    This helper is exposed for validation of the weighting rule.  The production
    estimator applies the same artifact-versus-preserve logic to ASR's removed
    signal by processing window because the stable ASR API intentionally does
    not expose mutable internal eigensystems.
    """
    variances = np.asarray(variances, dtype=float)
    thresholds_squared = np.asarray(thresholds_squared, dtype=float)
    directions = np.asarray(directions, dtype=float)
    forced_keep = np.asarray(forced_keep, dtype=bool)
    if soft_weight not in {"wiener", "logistic"}:
        raise ValueError("soft_weight must be 'wiener' or 'logistic'")
    if scale <= 0:
        raise ValueError("scale must be positive")

    ratio = thresholds_squared / np.maximum(variances, _EPS)
    if soft_weight == "wiener":
        base = np.clip(ratio, 0.0, 1.0)
    else:
        base = 1.0 / (1.0 + np.exp(-scale * (ratio - 1.0)))

    def _direction_scores(cov: np.ndarray | None) -> np.ndarray:
        if cov is None:
            return np.zeros_like(base)
        scores = np.diag(directions.T @ _normalize_cov(cov) @ directions)
        return np.maximum(scores, 0.0)

    artifact = _direction_scores(artifact_cov)
    preserve = _direction_scores(preserve_cov)
    if artifact_cov is not None or preserve_cov is not None:
        clipped = np.clip(base, 1e-9, 1.0 - 1e-9)
        logits = np.log(clipped / (1.0 - clipped))
        # The multiplier makes a unit trace-normalized bias decisive while the
        # ``scale`` parameter retains a clear monotonic aggressiveness control.
        logits += 8.0 * scale * (preserve - artifact)
        base = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    base[forced_keep] = 1.0
    return np.clip(base, 0.0, 1.0)


def _bias_energy_fraction(data: np.ndarray, biases: tuple[Any, ...]) -> float:
    """Score how much residual energy is selected by a bias-operator bank."""
    if not biases:
        return 0.0
    centered = np.asarray(data, dtype=np.float64)
    centered = centered - centered.mean(axis=1, keepdims=True)
    denominator = float(np.sum(centered * centered)) + _EPS
    scores: list[float] = []
    for bias in biases:
        try:
            selected = np.asarray(bias.apply(centered), dtype=np.float64)
        except (ValueError, RuntimeError):
            # Very short edge windows can be shorter than a zero-phase filter's
            # padding requirement.  Such windows carry no positive bias evidence.
            continue
        scores.append(float(np.sum(selected * selected)) / denominator)
    return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0


class GuidedASR(ASR):
    """ASR with explicit artifact and preservation bias operators.

    Parameters are identical to :class:`ASR`, with the additions below.

    Parameters
    ----------
    reconstruction : {'hard', 'soft'}, default='soft'
        ``'hard'`` returns the underlying ASR result exactly. ``'soft'`` blends
        ASR's removed signal using window-local bias evidence.
    artifact_biases : sequence of LinearDenoiser | None
        Bias operators identifying residual activity that should remain removed.
    preserve_biases : sequence of LinearDenoiser | None
        Bias operators identifying residual activity that should be rescued.
    soft_weight : {'wiener', 'logistic'}, default='wiener'
        Bounded mapping used to combine artifact and preservation evidence.
    soft_weight_scale : float, default=1.0
        Positive contrast applied to the evidence ratio.

    Notes
    -----
    Bias scores are evaluated on the transform data.  GuidedASR is consequently
    an adaptive/local estimator, not a fixed linear transform.  Benchmark
    registries must label this information access explicitly.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        cutoff: float = 20.0,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        method: str = "riemannian_windowed",
        experimental: bool = False,
        calibration: str = "auto",
        picks: str | list[str] | list[int] | None = "eeg",
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.075,
        ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
        blocksize: int = 10,
        max_dims: float | int = 0.66,
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        cov_estimator: str = "geometric_median",
        regularization: float = 1e-8,
        filter_kind: str = "none",
        window_criterion: float | int | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
        *,
        reconstruction: str = "soft",
        artifact_biases: list[Any] | tuple[Any, ...] | None = None,
        preserve_biases: list[Any] | tuple[Any, ...] | None = None,
        soft_weight: str = "wiener",
        soft_weight_scale: float = 1.0,
    ) -> None:
        super().__init__(
            sfreq=sfreq,
            cutoff=cutoff,
            window_length=window_length,
            window_overlap=window_overlap,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction,
            method=method,
            experimental=experimental,
            calibration=calibration,
            picks=picks,
            calibration_window_length=calibration_window_length,
            calibration_window_overlap=calibration_window_overlap,
            ref_max_bad_channels=ref_max_bad_channels,
            ref_tolerances=ref_tolerances,
            blocksize=blocksize,
            max_dims=max_dims,
            reject_by_annotation=reject_by_annotation,
            skip_by_annotation=skip_by_annotation,
            cov_estimator=cov_estimator,
            regularization=regularization,
            filter_kind=filter_kind,
            window_criterion=window_criterion,
            window_criterion_tolerances=window_criterion_tolerances,
            lookahead=lookahead,
            stepsize=stepsize,
            max_mem_mb=max_mem_mb,
            copy=copy,
            store_reconstruction_matrices=store_reconstruction_matrices,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.reconstruction = reconstruction
        self.artifact_biases = artifact_biases
        self.preserve_biases = preserve_biases
        self.soft_weight = soft_weight
        self.soft_weight_scale = soft_weight_scale

    def fit(self, X, y=None, *, calibration=None, calibration_mask=None):
        """Fit the ASR state and record the information supplied to guidance."""
        self._validate_guidance()
        super().fit(
            X,
            y=y,
            calibration=calibration,
            calibration_mask=calibration_mask,
        )
        self.artifact_biases_ = tuple(self.artifact_biases or ())
        self.preserve_biases_ = tuple(self.preserve_biases or ())
        self.guidance_state_ = {
            "reconstruction": self.reconstruction,
            "artifact_biases": [type(b).__name__ for b in self.artifact_biases_],
            "preserve_biases": [type(b).__name__ for b in self.preserve_biases_],
            "information_access": "transform-local deterministic bias scoring",
        }
        return self

    def transform(
        self,
        X,
        y=None,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Apply hard ASR and, in soft mode, rescue preserve-biased residuals."""
        del y
        hard, diagnostics = super().transform(
            X, copy=copy, return_diagnostics=True
        )
        if self.reconstruction == "hard":
            diagnostics = dict(diagnostics)
            diagnostics.update(
                {
                    "covariance_geometry": "guided-hard-control",
                    "reconstruction": "hard",
                    "soft_weights": np.empty((0, self.n_channels_)),
                    "mean_soft_weight": 0.0,
                }
            )
            self._store_transform_diagnostics(diagnostics)
            return (hard, diagnostics) if return_diagnostics else hard

        original_data, _, mne_type, original_inst, picks, _ = extract_data_from_mne(
            X, auto_pick=True
        )
        hard_data, _, _, _, _, _ = extract_data_from_mne(hard, auto_pick=True)
        if mne_type == "epochs":
            guided_epochs = []
            epoch_weights = []
            for original_epoch, hard_epoch in zip(original_data, hard_data):
                guided, weights = self._soft_blend_2d(original_epoch, hard_epoch)
                guided_epochs.append(guided)
                epoch_weights.append(weights)
            guided_data = np.asarray(guided_epochs)
            weights = (
                np.concatenate(epoch_weights, axis=0)
                if epoch_weights
                else np.empty((0, self.n_channels_))
            )
        else:
            guided_data, weights = self._soft_blend_2d(original_data, hard_data)

        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "covariance_geometry": "guided",
                "reconstruction": "soft",
                "soft_weights": weights,
                "mean_soft_weight": float(weights.mean()) if weights.size else 0.0,
                "guidance_state": dict(self.guidance_state_),
            }
        )
        self._store_transform_diagnostics(diagnostics)
        cleaned = reconstruct_mne_object(
            guided_data, original_inst, mne_type, picks=picks, verbose=False
        )
        return (cleaned, diagnostics) if return_diagnostics else cleaned

    def _soft_blend_2d(
        self, original: np.ndarray, hard: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        original = np.asarray(original, dtype=np.float64)
        hard = np.asarray(hard, dtype=np.float64)
        removed = original - hard
        n_times = original.shape[1]
        window = max(2, int(round(self.window_length * self.sfreq_)))
        step = max(1, int(round(window * (1.0 - self.window_overlap))))
        starts = list(range(0, max(1, n_times - window + 1), step))
        final_start = max(0, n_times - window)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)

        removed_acc = np.zeros_like(removed)
        overlap_acc = np.zeros(n_times, dtype=np.float64)
        rows: list[np.ndarray] = []
        taper = np.hanning(window) if window > 2 else np.ones(window)
        taper = np.maximum(taper, 1e-6)
        for start in starts:
            stop = min(start + window, n_times)
            residual = removed[:, start:stop]
            artifact = _bias_energy_fraction(residual, self.artifact_biases_)
            preserve = _bias_energy_fraction(residual, self.preserve_biases_)
            keep = self._keep_fraction(artifact, preserve)
            suppression = 1.0 - keep
            local_taper = taper[: stop - start]
            removed_acc[:, start:stop] += suppression * residual * local_taper
            overlap_acc[start:stop] += local_taper
            rows.append(np.full(original.shape[0], keep, dtype=np.float64))
        overlap_acc = np.maximum(overlap_acc, _EPS)
        guided = original - removed_acc / overlap_acc
        return guided, np.vstack(rows)

    def _keep_fraction(self, artifact: float, preserve: float) -> float:
        if not self.artifact_biases_ and not self.preserve_biases_:
            return 0.0
        if not self.artifact_biases_:
            return float(np.clip(preserve * self.soft_weight_scale, 0.0, 1.0))
        if not self.preserve_biases_:
            return 0.0
        if artifact + preserve <= _EPS:
            return 0.0
        if self.soft_weight == "wiener":
            a = artifact**self.soft_weight_scale
            p = preserve**self.soft_weight_scale
            return float(p / max(a + p, _EPS))
        log_ratio = self.soft_weight_scale * np.log(
            (preserve + _EPS) / (artifact + _EPS)
        )
        return float(1.0 / (1.0 + np.exp(-np.clip(log_ratio, -60.0, 60.0))))

    def _validate_guidance(self) -> None:
        if self.reconstruction not in {"hard", "soft"}:
            raise ValueError("reconstruction must be 'hard' or 'soft'")
        if self.soft_weight not in {"wiener", "logistic"}:
            raise ValueError("soft_weight must be 'wiener' or 'logistic'")
        if self.soft_weight_scale <= 0:
            raise ValueError("soft_weight_scale must be positive")
        if self.reconstruction == "soft" and not self.experimental:
            raise ValueError(
                "GuidedASR soft reconstruction is experimental; pass "
                "experimental=True to opt in"
            )


__all__ = ["GuidedASR"]
