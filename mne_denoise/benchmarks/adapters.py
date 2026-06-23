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
# Linear DSS  (native; train_only — evoked enhancement / oscillatory extraction)
# ---------------------------------------------------------------------------
class _DSS(Comparator):
    """``dss`` — linear DSS denoising. Filters learned on TRAIN (leakage barrier),
    applied to EVAL. Bias defaults to ``AverageBias`` (trial-reproducibility →
    evoked enhancement); ``bias='bandpass'`` targets an oscillatory band via ctx.
    Keeps the top ``n_components`` and reconstructs."""

    def __init__(self, n_components: int = 5, bias: str = "average", **params: Any) -> None:
        super().__init__(
            ComparatorMeta("dss", fit_scope="train_only", rank_reducing=True),
            n_components=n_components, bias=bias, **params,
        )
        self.n_components = int(n_components)
        self.bias = bias

    def _make_bias(self, ctx):
        from mne_denoise.dss import denoisers as _den

        if self.bias == "bandpass":
            band = ctx.get("band", (1.0, 40.0))
            return _den.BandpassBias(freq_band=tuple(band), sfreq=ctx["sfreq"])
        return _den.AverageBias()

    def _fit(self, train, ctx):
        from mne_denoise.dss import DSS

        rtype = "epochs" if hasattr(train, "get_data") and train.__class__.__name__.startswith("Epochs") else "raw"
        dss = DSS(bias=self._make_bias(ctx), n_components=self.n_components, return_type=rtype)
        dss.fit(train)
        return dss

    def _transform(self, evaluation, payload, ctx):
        dss = payload
        cleaned = dss.transform(evaluation)
        evs = getattr(dss, "eigenvalues_", None)
        return ComparatorResult(
            cleaned=cleaned, status="success", rank_after=self.n_components,
            diagnostics={"eigenvalues": (evs[: self.n_components].tolist() if evs is not None else None)},
            parameters={"n_components": self.n_components, "bias": self.bias},
        )


# ---------------------------------------------------------------------------
# ICA controls (rank-matched dimensionality control; ICLabel ocular rejection)
# ---------------------------------------------------------------------------
class _ICARankMatched(Comparator):
    """``ica_rank_matched`` — rank-k ICA fit on TRAIN, reconstructed on EVAL with
    no component removed (a dimensionality-matched control for DSS/PCA)."""

    def __init__(self, n_components: int = 5, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("ica_rank_matched", fit_scope="train_only", rank_reducing=True),
            n_components=n_components, **params,
        )
        self.k = int(n_components)

    def _fit(self, train, ctx):
        import mne

        ica = mne.preprocessing.ICA(n_components=self.k, max_iter="auto", random_state=97, verbose=False)
        ica.fit(train, verbose=False)
        ica.exclude = []
        return ica

    def _transform(self, evaluation, payload, ctx):
        ica = payload
        cleaned = ica.apply(evaluation.copy(), verbose=False)
        return ComparatorResult(cleaned=cleaned, status="success", rank_after=self.k,
                                parameters={"n_components": self.k})


class _ICAICLabel(Comparator):
    """``ica_iclabel_rejection`` — ICA fit on TRAIN, ocular ICs labelled by ICLabel
    (falls back to EOG-correlation), removed on EVAL. Reference-blind ocular comparator."""

    def __init__(self, n_components: int | None = None, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("ica_iclabel_rejection", fit_scope="train_only",
                           optional_dependency="mne-icalabel"),
            n_components=n_components, **params,
        )
        self.n_components = n_components

    def _fit(self, train, ctx):
        import mne

        ica = mne.preprocessing.ICA(n_components=self.n_components, max_iter="auto",
                                    random_state=97, verbose=False)
        ica.fit(train, verbose=False)
        excl: list[int] = []
        try:
            from mne_icalabel import label_components

            labels = label_components(train, ica, method="iclabel")["labels"]
            excl = [i for i, lab in enumerate(labels) if lab in ("eye blink", "eye")]
        except Exception:  # noqa: BLE001 - fall back to EOG correlation if available
            try:
                excl, _ = ica.find_bads_eog(train, verbose=False)
            except Exception:  # noqa: BLE001
                excl = []
        ica.exclude = list(excl)
        return ica

    def _transform(self, evaluation, payload, ctx):
        ica = payload
        cleaned = ica.apply(evaluation.copy(), verbose=False)
        return ComparatorResult(cleaned=cleaned, status="success",
                                diagnostics={"n_excluded": len(ica.exclude)})


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
from .comparators import _REGISTRY  # noqa: E402  (for the pca alias)

register("dss", lambda **p: _DSS(**p))
register("ica_rank_matched", lambda **p: _ICARankMatched(**p))
register("ica_iclabel_rejection", lambda **p: _ICAICLabel(**p))
if "rank_matched_pca" not in _REGISTRY:
    register("rank_matched_pca", _REGISTRY["pca_reconstruct"])  # alias the validated PCA control
register("zapline", lambda **p: _ZapLine(adaptive=False, **p))
register("zapline_plus", lambda **p: _ZapLine(adaptive=True, **p))
register("notch", lambda **p: _Notch(method="fir", **p))
register("non_spatial_line", lambda **p: _Notch(method="spectrum_fit", **p))
