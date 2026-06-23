"""Method + comparator adapters wrapping native ``mne-denoise`` denoisers and
baseline filters behind the :class:`~mne_denoise.benchmarks.comparators.Comparator`
fit/transform contract.

Each adapter registers under a stable ``comparator_id``.  A runner does::

    import mne_denoise.benchmarks.adapters  # noqa: F401  (registers everything)
    from mne_denoise.benchmarks import comparators
    cmp = comparators.get("zapline", n_remove="auto")
    state = cmp.fit(train, ctx)
    result = cmp.transform(evaluation, state, ctx)   # -> ComparatorResult

``ctx`` is a dict the runner supplies, carrying at least ``sfreq`` and (for the
line arms) ``line_freq``; reference-aware arms also pass ``ref_picks``.

This module is imported for its side effects (registration).  Import errors of
optional native methods are deferred to call time so a partial install still
loads the registry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .comparators import (
    Comparator,
    ComparatorMeta,
    ComparatorResult,
    register,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sfreq(evaluation: Any, ctx: dict) -> float:
    sf = ctx.get("sfreq")
    if sf is None and hasattr(evaluation, "info"):
        sf = float(evaluation.info["sfreq"])
    if sf is None:
        raise ValueError("sfreq not available (pass via ctx or an MNE object)")
    return float(sf)


def _line_harmonics(line_freq: float, sfreq: float, n: int | None = None) -> list[float]:
    """Line fundamental + harmonics strictly below Nyquist."""
    nyq = sfreq / 2.0
    out, k = [], 1
    while line_freq * k < nyq - 1.0:
        out.append(line_freq * k)
        if n is not None and len(out) >= n:
            break
        k += 1
    return out


# ---------------------------------------------------------------------------
# ZapLine / ZapLine-plus  (native, per-recording unsupervised)
# ---------------------------------------------------------------------------
class _ZapLine(Comparator):
    """``zapline`` (standard) and ``zapline_plus`` (adaptive) line-noise removal."""

    def __init__(self, adaptive: bool = False, n_remove: Any = "auto", **params: Any) -> None:
        cid = "zapline_plus" if adaptive else "zapline"
        super().__init__(
            ComparatorMeta(
                cid,
                fit_scope="per_recording_unsupervised",
                requires_fit=False,
                deterministic=True,
            ),
            adaptive=adaptive,
            n_remove=n_remove,
            **params,
        )
        self.adaptive = bool(adaptive)
        self.n_remove = n_remove

    def _fit(self, train, ctx):  # per-recording: nothing learned from train
        return None

    def _transform(self, evaluation, payload, ctx):
        from mne_denoise.zapline import ZapLine

        sfreq = _sfreq(evaluation, ctx)
        line_freq = ctx.get("line_freq")
        zl = ZapLine(
            sfreq=sfreq,
            line_freq=(None if self.adaptive else line_freq),
            n_remove=self.n_remove,
            adaptive=self.adaptive,
        )
        cleaned = zl.fit_transform(evaluation)
        n_removed = getattr(zl, "n_removed_", None)
        if n_removed is None and getattr(zl, "adaptive_results_", None):
            n_removed = zl.adaptive_results_.get("n_removed")
        return ComparatorResult(
            cleaned=cleaned,
            status="success",
            diagnostics={"n_removed": (int(n_removed) if n_removed is not None else None)},
            parameters={"n_remove": self.n_remove, "adaptive": self.adaptive},
        )


# ---------------------------------------------------------------------------
# notch (FIR) and non-spatial line removal (spectrum-fit / CleanLine-like)
# ---------------------------------------------------------------------------
class _Notch(Comparator):
    """``notch`` = zero-phase FIR notch; ``non_spatial_line`` = MNE ``spectrum_fit``
    multi-taper sinusoid removal (an approved CleanLine substitute)."""

    def __init__(self, method: str = "fir", **params: Any) -> None:
        cid = "non_spatial_line" if method == "spectrum_fit" else "notch"
        super().__init__(
            ComparatorMeta(cid, fit_scope="per_recording_unsupervised", requires_fit=False),
            method=method,
            **params,
        )
        self.method = method

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        if not hasattr(evaluation, "notch_filter"):
            return ComparatorResult(
                status="skipped_missing_channels",
                error="notch/non_spatial_line require an MNE Raw object",
            )
        sfreq = _sfreq(evaluation, ctx)
        line_freq = ctx.get("line_freq")
        freqs = _line_harmonics(line_freq, sfreq)
        raw = evaluation.copy()
        if self.method == "spectrum_fit":
            raw.notch_filter(
                freqs=freqs, picks="data", method="spectrum_fit",
                filter_length="10s", verbose=False,
            )
        else:
            raw.notch_filter(freqs=freqs, picks="data", verbose=False)
        return ComparatorResult(
            cleaned=raw,
            status="success",
            diagnostics={"freqs": freqs},
            parameters={"method": self.method},
        )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
register("zapline", lambda **p: _ZapLine(adaptive=False, **p))
register("zapline_plus", lambda **p: _ZapLine(adaptive=True, **p))
register("notch", lambda **p: _Notch(method="fir", **p))
register("non_spatial_line", lambda **p: _Notch(method="spectrum_fit", **p))
