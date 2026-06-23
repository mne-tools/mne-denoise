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
# Spatial rank-matched PCA  (handles Epochs per-trial AND continuous arrays)
# ---------------------------------------------------------------------------
class _SpatialPCA(Comparator):
    """``rank_matched_pca`` — spatial PCA across channels, fitted on TRAIN, applied
    to EVAL. For Epochs it reconstructs each trial from the top-k spatial
    components (the dimensionality-matched control for DSS); for a Raw/array it
    reconstructs the 2-D signal."""

    def __init__(self, n_components: int = 5, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("rank_matched_pca", fit_scope="train_only", rank_reducing=True),
            n_components=n_components, **params,
        )
        self.k = int(n_components)

    @staticmethod
    def _chan_time(x):
        return x.transpose(1, 0, 2).reshape(x.shape[1], -1) if x.ndim == 3 else x

    def _fit(self, train, ctx):
        from sklearn.decomposition import PCA

        x = train.get_data(copy=False) if hasattr(train, "get_data") else np.asarray(train)
        X = self._chan_time(x)  # (n_channels, n_samples)
        pca = PCA(n_components=min(self.k, X.shape[0]), svd_solver="full")
        pca.fit(X.T)
        return pca

    def _transform(self, evaluation, payload, ctx):
        pca = payload
        if hasattr(evaluation, "get_data"):
            x = evaluation.get_data(copy=False)
            out = np.stack([pca.inverse_transform(pca.transform(x[i].T)).T for i in range(x.shape[0])])
            cleaned = evaluation.copy()
            cleaned._data = out
            return ComparatorResult(cleaned=cleaned, status="success", rank_after=int(pca.n_components_))
        x = np.asarray(evaluation)
        recon = pca.inverse_transform(pca.transform(x.T)).T
        return ComparatorResult(cleaned=recon, status="success", rank_after=int(pca.n_components_))


# ---------------------------------------------------------------------------
# ASR / rASR  (native; calibration_then_transform)
# ---------------------------------------------------------------------------
class _ASR(Comparator):
    """``asr`` (Euclidean) / ``rasr`` (``method='riemannian_windowed'``). Calibrated
    on clean TRAIN data, then transforms EVAL. ``cutoff`` is the swept aggressiveness."""

    def __init__(self, cutoff: float = 20.0, method: str = "standard", **params: Any) -> None:
        cid = "rasr" if method == "riemannian_windowed" else "asr"
        super().__init__(
            ComparatorMeta(cid, fit_scope="calibration_then_transform"),
            cutoff=cutoff, method=method, **params,
        )
        self.cutoff = float(cutoff)
        self.method = method

    def _fit(self, train, ctx):
        from mne_denoise.asr import ASR

        asr = ASR(sfreq=ctx.get("sfreq"), cutoff=self.cutoff, method=self.method)
        asr.fit(train)
        return asr

    def _transform(self, evaluation, payload, ctx):
        cleaned = payload.transform(evaluation)
        return ComparatorResult(cleaned=cleaned, status="success",
                                diagnostics={"cutoff": self.cutoff, "method": self.method},
                                parameters={"cutoff": self.cutoff})


# ---------------------------------------------------------------------------
# iCanClean / reference regression  (reference-aware; need ctx['ref_channels'])
# ---------------------------------------------------------------------------
class _ICanClean(Comparator):
    """``icanclean`` — reference-coupled CCA; removes the EEG subspace shared with
    the reference channels (``ctx['ref_channels']``)."""

    def __init__(self, threshold: float = 0.7, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("icanclean", fit_scope="window_local", reference_aware=True),
            threshold=threshold, **params,
        )
        self.threshold = threshold

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        ref = ctx.get("ref_channels")
        if not ref:
            return ComparatorResult(status="skipped_missing_channels",
                                    error="icanclean needs ctx['ref_channels']")
        from mne_denoise.icanclean import ICanClean

        icc = ICanClean(sfreq=float(evaluation.info["sfreq"]), ref_channels=list(ref),
                        threshold=self.threshold)
        cleaned = icc.fit_transform(evaluation.copy())
        return ComparatorResult(cleaned=cleaned, status="success",
                                diagnostics={"threshold": self.threshold})


class _RefRegression(Comparator):
    """``regression`` — least-squares regress the reference channels out of EEG
    (per-recording; reference-aware baseline)."""

    def __init__(self, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("regression", fit_scope="window_local", reference_aware=True), **params,
        )

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        ref = ctx.get("ref_channels")
        if not ref:
            return ComparatorResult(status="skipped_missing_channels",
                                    error="regression needs ctx['ref_channels']")
        from mne_denoise.qa.coupling import regress_out

        raw = evaluation.copy()
        eeg = raw.copy().pick("eeg")
        ed = eeg.get_data()
        rd = raw.copy().pick(list(ref)).get_data()
        cleaned = eeg.copy()
        if ed.ndim == 3:  # epoched: regress across the concatenated time axis
            ntr, nch, nt = ed.shape
            E = ed.transpose(1, 0, 2).reshape(nch, -1)
            R = rd.transpose(1, 0, 2).reshape(rd.shape[1], -1)
            cleaned._data = regress_out(E, R).reshape(nch, ntr, nt).transpose(1, 0, 2)
        else:
            cleaned._data = regress_out(ed, rd)
        return ComparatorResult(cleaned=cleaned, status="success",
                                diagnostics={"n_ref": len(ref)})


# ---------------------------------------------------------------------------
# EOG-DSS  (ocular: remove the EEG subspace coupled to the EOG channels)
# ---------------------------------------------------------------------------
def _flat(x):
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1) if x.ndim == 3 else x


class _EOGDSS(Comparator):
    """``eog_dss`` — learn (on TRAIN) the rank-k spatial subspace of EEG most coupled
    to the EOG channels (``ctx['eog_channels']``) and project it out of EVAL. A spatial
    ocular filter; the frozen cycle-average (blink-locked) variant is a refinement."""

    def __init__(self, n_components: int = 2, **params: Any) -> None:
        super().__init__(ComparatorMeta("eog_dss", fit_scope="train_only", rank_reducing=True),
                         n_components=n_components, **params)
        self.k = int(n_components)

    def _fit(self, train, ctx):
        eog = ctx.get("eog_channels") or ctx.get("ref_channels")
        if not eog:
            return None
        Xe = _flat(train.copy().pick("eeg").get_data())
        Xo = _flat(train.copy().pick(list(eog)).get_data())
        B = Xe @ Xo.T @ np.linalg.pinv(Xo @ Xo.T)         # ocular spatial patterns (n_ch, n_eog)
        U, _, _ = np.linalg.svd(B, full_matrices=False)
        return U[:, : self.k]                              # ocular subspace basis (n_ch, k)

    def _transform(self, evaluation, payload, ctx):
        if payload is None:
            return ComparatorResult(status="skipped_missing_channels",
                                    error="eog_dss needs ctx['eog_channels']")
        P = payload
        eeg = evaluation.copy().pick("eeg")
        proj = np.eye(P.shape[0]) - P @ P.T               # remove the ocular subspace
        x = eeg.get_data()
        eeg._data = np.einsum("ij,ejt->eit", proj, x) if x.ndim == 3 else proj @ x
        return ComparatorResult(cleaned=eeg, status="success", rank_after=int(P.shape[0] - self.k),
                                diagnostics={"n_components": self.k})


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
register("eog_dss", lambda **p: _EOGDSS(**p))
register("asr", lambda **p: _ASR(method="standard", **p))
register("rasr", lambda **p: _ASR(method="riemannian_windowed", **p))
register("icanclean", lambda **p: _ICanClean(**p))
register("regression", lambda **p: _RefRegression(**p))
register("dss", lambda **p: _DSS(**p))
register("dss_average_bias", lambda **p: _DSS(bias="average", **p))
register("ica_rank_matched", lambda **p: _ICARankMatched(**p))
register("ica_iclabel_rejection", lambda **p: _ICAICLabel(**p))
register("rank_matched_pca", lambda **p: _SpatialPCA(**p))
register("zapline", lambda **p: _ZapLine(adaptive=False, **p))
register("zapline_plus", lambda **p: _ZapLine(adaptive=True, **p))
register("notch", lambda **p: _Notch(method="fir", **p))
register("non_spatial_line", lambda **p: _Notch(method="spectrum_fit", **p))
