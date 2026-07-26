"""DSS operator adapters for the operator-completeness campaign.

The v1 benchmark reaches DSS through exactly two registry ids (``dss`` /
``dss_average_bias`` and ``eog_dss``). Every other DSS variant that the paper
reports -- the cycle-average cardiac operator, the comb-bias SSVEP operator, the
reference-aware operator, and the whole ``IterativeDSS`` family -- is written
inline inside a runner. Inline methods cannot be dispatched by
``comparators.get`` and therefore cannot be swept by
:func:`mne_denoise.benchmarks.sweep.method_runs`, which is why no DSS operating
point curve exists anywhere in the evidence base.

This module registers those operators without touching :mod:`.adapters`, so the
importable tree stays a strict superset of the tree every completed v1 attempt
executed.

Two things are deliberate and load-bearing:

**Polarity is part of the method identity, not a tunable.** A target-aware bias
(comb, bandpass, average) *keeps* the subspace it concentrates; an artifact bias
(cycle-average, reference, line) *subtracts* it. The same estimator with the
wrong polarity silently inverts every metric without raising, so polarity is a
constructor argument, is recorded in ``ComparatorResult.parameters``, and each
registered id names it.

**Denoising need not be exact.** Sarela & Valpola (2005) sec. 4 note that it is
enough for the denoising function to remove more noise than signal, because the
re-estimation step constrains the source to the subspace spanned by the data.
A mismatched contrast therefore degrades gracefully, which is what makes the
eleven-way contrast screen interpretable rather than a set of failures.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import adapters as _adapters  # noqa: F401  - ensure base ids register first
from .comparators import Comparator, ComparatorMeta, ComparatorResult, register

# Canonical contrast id -> denoiser class name. Mirrors the mapping already used
# by scripts/run_ground_truth_arm.py so the screening ids match the completed
# generic-BSS factorial exactly.
CONTRASTS: dict[str, str] = {
    "tanh": "TanhMaskDenoiser",
    "robust_tanh": "RobustTanhDenoiser",
    "gauss": "GaussDenoiser",
    "skew": "SkewDenoiser",
    "smooth_tanh": "SmoothTanhDenoiser",
    "kurtosis": "KurtosisDenoiser",
    "wiener_mask": "WienerMaskDenoiser",
    "variance_mask": "VarianceMaskDenoiser",
    "dct": "DCTDenoiser",
    "quasi_periodic": "QuasiPeriodicDenoiser",
    "spectrogram": "SpectrogramDenoiser",
}

VALID_POLARITIES = ("keep", "subtract")


def _is_epochs(obj: Any) -> bool:
    return hasattr(obj, "get_data") and obj.__class__.__name__.startswith("Epochs")


def _data_of(obj: Any) -> np.ndarray:
    return np.asarray(obj.get_data()) if hasattr(obj, "get_data") else np.asarray(obj)


def _with_data(obj: Any, values: np.ndarray) -> Any:
    """Return a copy of ``obj`` carrying ``values`` (MNE object or bare array)."""
    if not hasattr(obj, "get_data"):
        return values
    out = obj.copy()
    out._data = values
    return out


def _apply_polarity(evaluation: Any, estimate: np.ndarray, polarity: str) -> Any:
    """``keep`` returns the recovered subspace; ``subtract`` removes it."""
    original = _data_of(evaluation)
    estimate = np.asarray(estimate, dtype=float)
    if estimate.shape != original.shape:
        estimate = estimate.reshape(original.shape)
    values = estimate if polarity == "keep" else original - estimate
    return _with_data(evaluation, values)


def _degenerate_reason(cleaned: Any, original: np.ndarray) -> str | None:
    """Detect an output that is not a cleaning result at all.

    A degenerate operator that annihilates the recording scores a perfect
    attenuation and a zero preservation, and nothing in the contract rejects it:
    the v1 cardiac operator returned an identically zero signal for every
    participant and was recorded as ``success``. Only the paired preservation
    endpoint revealed it. Failing closed here makes that a terminal status
    instead of a headline number.
    """
    values = _data_of(cleaned)
    if not np.all(np.isfinite(values)):
        return "output contains non-finite samples"
    if np.allclose(values, 0.0):
        return "output is identically zero; the operator annihilated the signal"
    scale = float(np.std(original))
    if scale > 0 and float(np.std(values)) < 1e-12 * scale:
        return "output has no variance relative to the input"
    return None


class _LinearDSSOperator(Comparator):
    """Linear DSS under an explicit bias operator and polarity.

    The bias is built from ``ctx`` at fit time because several biases need
    information the config cannot carry: ``cycle_average`` needs event sample
    indices, ``reference`` needs the reference channel data itself. Those must be
    expressed **relative to the training object**, since the bias is applied to
    the training data to form the biased covariance. Nothing is needed at
    transform time -- the fitted spatial filters are applied directly, which is
    what keeps the leakage barrier intact.
    """

    #: bias kind -> (needs_ctx_key, reference_aware)
    _NEEDS = {
        "cycle_average": ("event_samples", False),
        "reference": ("ref_channels", True),
        "comb": (None, False),
        "line": (None, False),
        "bandpass": (None, False),
        "average": (None, False),
    }

    def __init__(
        self,
        comparator_id: str,
        bias: str,
        polarity: str,
        n_components: int = 3,
        **params: Any,
    ) -> None:
        if bias not in self._NEEDS:
            raise ValueError(f"unknown bias {bias!r}; expected one of {sorted(self._NEEDS)}")
        if polarity not in VALID_POLARITIES:
            raise ValueError(f"polarity {polarity!r} must be one of {VALID_POLARITIES}")
        needs_ctx, reference_aware = self._NEEDS[bias]
        super().__init__(
            ComparatorMeta(
                comparator_id,
                fit_scope="train_only",
                rank_reducing=(polarity == "keep"),
                reference_aware=reference_aware,
            ),
            bias=bias, polarity=polarity, n_components=n_components, **params,
        )
        self.bias = bias
        self.polarity = polarity
        self.n_components = int(n_components)
        self.needs_ctx = needs_ctx

    def _make_bias(self, train, ctx):
        from mne_denoise.dss import denoisers as den

        if self.bias == "cycle_average":
            # CycleAverageBias multiplies `window` by `sfreq` whenever sfreq is
            # given, so the window MUST be expressed in seconds here. Passing a
            # window already converted to samples together with sfreq converts it
            # twice; that is what silently produced a degenerate cardiac operator
            # in the v1 arms. The unit is named in the ctx key to make it hard to
            # get wrong, and the resulting sample window is validated below.
            window_s = tuple(ctx.get("event_window_s", (-0.1, 0.2)))
            bias = den.CycleAverageBias(
                event_samples=np.asarray(ctx["event_samples"], dtype=int),
                window=window_s,
                sfreq=float(ctx["sfreq"]),
            )
            pre, post = bias.window
            span = post - pre
            if not 0 < span <= int(2.0 * float(ctx["sfreq"])):
                raise ValueError(
                    f"cycle-average window {window_s} s resolves to {bias.window} samples "
                    f"({span / float(ctx['sfreq']):.3g} s); expected a sub-2 s cycle window. "
                    "Give the window in SECONDS -- CycleAverageBias scales it by sfreq."
                )
            return bias
        if self.bias == "reference":
            reference = ctx.get("reference_data")
            if reference is None:
                reference = train.copy().pick(list(ctx["ref_channels"])).get_data()
            reference = np.asarray(reference, dtype=float)
            if reference.ndim == 3:  # epoched reference -> concatenate trials
                reference = reference.transpose(1, 0, 2).reshape(reference.shape[1], -1)
            return den.ReferenceBias(reference=reference, ridge=float(self.params.get("ridge", 1e-8)))
        if self.bias == "comb":
            return den.CombFilterBias(
                fundamental_freq=float(ctx["stim_freq"]),
                sfreq=float(ctx["sfreq"]),
                n_harmonics=int(self.params.get("n_harmonics", 3)),
            )
        if self.bias == "line":
            return den.LineNoiseBias(
                freq=float(ctx.get("line_freq", 50.0)),
                sfreq=float(ctx["sfreq"]),
                n_harmonics=self.params.get("n_harmonics"),
            )
        if self.bias == "bandpass":
            return den.BandpassBias(
                freq_band=tuple(ctx.get("band", (8.0, 12.0))),
                sfreq=float(ctx["sfreq"]),
            )
        return den.AverageBias()

    def _fit(self, train, ctx):
        from mne_denoise.dss import DSS

        if self.needs_ctx and self.needs_ctx not in ctx and "reference_data" not in ctx:
            return None  # reported as skipped_missing_channels in _transform
        rtype = "epochs" if _is_epochs(train) else ("raw" if hasattr(train, "get_data") else "array")
        try:
            bias = self._make_bias(train, ctx)
        except (ValueError, KeyError) as exc:
            # Only transform() is wrapped by the Comparator contract, so a bad
            # bias specification would otherwise kill the whole shard instead of
            # producing one recorded failed attempt.
            return {"error": f"{type(exc).__name__}: {exc}"}
        dss = DSS(bias=bias, n_components=self.n_components, return_type=rtype)
        dss.fit(train)
        return dss

    def _transform(self, evaluation, payload, ctx):
        if payload is None:
            return ComparatorResult(
                status="skipped_missing_channels",
                error=f"{self.meta.comparator_id} needs ctx[{self.needs_ctx!r}]",
            )
        if isinstance(payload, dict) and "error" in payload:
            return ComparatorResult(
                status="failed_numerical", error=payload["error"],
                parameters={"bias": self.bias, "polarity": self.polarity,
                            "n_components": self.n_components},
            )
        estimate = _data_of(payload.transform(evaluation))
        cleaned = _apply_polarity(evaluation, estimate, self.polarity)
        original = _data_of(evaluation)
        reason = _degenerate_reason(cleaned, original)
        if reason is not None:
            return ComparatorResult(
                status="failed_numerical", error=reason,
                parameters={"bias": self.bias, "polarity": self.polarity,
                            "n_components": self.n_components},
            )
        n_channels = original.shape[-2]
        eigenvalues = getattr(payload, "eigenvalues_", None)
        return ComparatorResult(
            cleaned=cleaned,
            status="success",
            rank_after=(self.n_components if self.polarity == "keep"
                        else int(n_channels - self.n_components)),
            diagnostics={
                "eigenvalues": (eigenvalues[: self.n_components].tolist()
                                if eigenvalues is not None else None),
            },
            parameters={"bias": self.bias, "polarity": self.polarity,
                        "n_components": self.n_components},
        )


class _IterativeDSSOperator(Comparator):
    """Nonlinear DSS with the contrast selected by name.

    Exposing the contrast as a keyword argument is what lets the existing sweep
    grammar screen all eleven in one arm::

        sweep:
          iterative_dss: {param: contrast, grid: [tanh, kurtosis, ...]}

    ``IterativeDSS`` is stochastic, so ``deterministic`` is False, the seed is
    recorded on every attempt, and a component that never converges yields
    ``failed_convergence`` rather than a silently retained filter.
    """

    def __init__(
        self,
        contrast: str = "tanh",
        n_components: int = 3,
        polarity: str = "subtract",
        method: str = "symmetric",
        max_iter: int = 100,
        random_state: int = 0,
        min_converged_fraction: float = 0.5,
        **params: Any,
    ) -> None:
        if contrast not in CONTRASTS:
            raise ValueError(f"unknown contrast {contrast!r}; expected one of {sorted(CONTRASTS)}")
        if polarity not in VALID_POLARITIES:
            raise ValueError(f"polarity {polarity!r} must be one of {VALID_POLARITIES}")
        super().__init__(
            ComparatorMeta(
                "iterative_dss",
                fit_scope="train_only",
                rank_reducing=(polarity == "keep"),
                deterministic=False,
            ),
            contrast=contrast, n_components=n_components, polarity=polarity,
            method=method, max_iter=max_iter, random_state=random_state, **params,
        )
        self.contrast = contrast
        self.n_components = int(n_components)
        self.polarity = polarity
        self.method = method
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.min_converged_fraction = float(min_converged_fraction)

    def _denoiser(self):
        from mne_denoise.dss import denoisers as den

        cls = getattr(den, CONTRASTS[self.contrast])
        kwargs = {k: v for k, v in self.params.items()
                  if k not in ("contrast", "n_components", "polarity", "method",
                               "max_iter", "random_state", "min_converged_fraction")}
        return cls(**kwargs)

    def _fit(self, train, ctx):
        from mne_denoise.dss import iterative_dss
        from mne_denoise.dss.denoisers import beta_tanh

        data = _data_of(train)
        if data.ndim == 3:  # (n_epochs, n_channels, n_times) -> concatenate trials
            data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
        train_mean = data.mean(axis=1, keepdims=True)
        unmixing, _, _, convergence = iterative_dss(
            data,
            self._denoiser(),
            self.n_components,
            method=self.method,
            beta=beta_tanh if self.contrast == "tanh" else None,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        convergence = np.asarray(convergence)
        converged = (float(np.mean(convergence[:, 1])) if convergence.ndim == 2 else None)
        return {"unmixing": np.asarray(unmixing), "train_mean": train_mean,
                "converged_fraction": converged,
                "iterations_mean": (float(np.mean(convergence[:, 0]))
                                    if convergence.ndim == 2 else None)}

    def _transform(self, evaluation, payload, ctx):
        converged = payload["converged_fraction"]
        if converged is not None and converged < self.min_converged_fraction:
            return ComparatorResult(
                status="failed_convergence",
                error=(f"only {converged:.2f} of {self.n_components} components converged "
                       f"within max_iter={self.max_iter}"),
                random_seed=self.random_state,
                parameters={"contrast": self.contrast, "polarity": self.polarity,
                            "n_components": self.n_components,
                            "converged_fraction": converged},
            )
        unmixing = payload["unmixing"]
        original = _data_of(evaluation)
        flat = (original.transpose(1, 0, 2).reshape(original.shape[1], -1)
                if original.ndim == 3 else original)
        sources = unmixing @ (flat - payload["train_mean"])
        estimate = np.linalg.pinv(unmixing) @ sources          # back-project to sensors
        estimate = estimate.reshape(original.transpose(1, 0, 2).shape).transpose(1, 0, 2) \
            if original.ndim == 3 else estimate
        cleaned = _apply_polarity(evaluation, estimate, self.polarity)
        reason = _degenerate_reason(cleaned, original)
        if reason is not None:
            return ComparatorResult(
                status="failed_numerical", error=reason, random_seed=self.random_state,
                parameters={"contrast": self.contrast, "polarity": self.polarity,
                            "n_components": self.n_components},
            )
        return ComparatorResult(
            cleaned=cleaned,
            status="success",
            rank_after=(self.n_components if self.polarity == "keep"
                        else int(original.shape[-2] - self.n_components)),
            random_seed=self.random_state,
            diagnostics={"converged_fraction": converged,
                         "iterations_mean": payload["iterations_mean"]},
            parameters={"contrast": self.contrast, "polarity": self.polarity,
                        "n_components": self.n_components, "method": self.method},
        )


def _linear(comparator_id: str, bias: str, polarity: str):
    return lambda **p: _LinearDSSOperator(comparator_id, bias=bias, polarity=polarity, **p)


# Artifact biases subtract the subspace they concentrate; target-aware biases keep it.
register("dss_cycle_average_subtract", _linear("dss_cycle_average_subtract", "cycle_average", "subtract"))
register("dss_reference_bias_subtract", _linear("dss_reference_bias_subtract", "reference", "subtract"))
register("dss_line_bias_subtract", _linear("dss_line_bias_subtract", "line", "subtract"))
register("dss_comb_bias_keep", _linear("dss_comb_bias_keep", "comb", "keep"))
register("dss_bandpass_bias_keep", _linear("dss_bandpass_bias_keep", "bandpass", "keep"))
register("iterative_dss", lambda **p: _IterativeDSSOperator(**p))
