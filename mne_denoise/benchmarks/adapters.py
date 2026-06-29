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


def _line_harmonics(line_freq, sfreq: float, n: int | None = None) -> list[float]:
    """Line fundamental(s) + harmonics strictly below Nyquist. Accepts a single
    frequency or a list (e.g. dual 50+60 Hz injection)."""
    nyq = sfreq / 2.0
    bases = line_freq if isinstance(line_freq, (list, tuple)) else [line_freq]
    out: list[float] = []
    for base in bases:
        k = 1
        while base * k < nyq - 1.0:
            out.append(float(base * k))
            if n is not None and len(out) >= n:
                break
            k += 1
    return sorted(set(out))


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

        x = train.get_data() if hasattr(train, "get_data") else np.asarray(train)
        X = self._chan_time(x)  # (n_channels, n_samples)
        pca = PCA(n_components=min(self.k, X.shape[0]), svd_solver="full")
        pca.fit(X.T)
        return pca

    def _transform(self, evaluation, payload, ctx):
        pca = payload
        if hasattr(evaluation, "get_data"):
            x = evaluation.get_data()
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
# TSPCA  (time-shift PCA/regression) + SSP-EOG  (fill the two known gaps)
# ---------------------------------------------------------------------------
class _TSPCA(Comparator):
    """``tspca`` — time-shift PCA / regression (de Cheveigné & Simon 2007). Subtract
    from EEG the part predictable from time-LAGGED copies of the reference channels
    (distinct from zero-lag ``regression``). Reference-aware (``ctx['ref_channels']``)."""

    def __init__(self, n_shifts: int = 3, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("tspca", fit_scope="window_local", reference_aware=True),
            n_shifts=n_shifts, **params,
        )
        self.n_shifts = int(n_shifts)

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        ref = ctx.get("ref_channels")
        if not ref:
            return ComparatorResult(status="skipped_missing_channels", error="tspca needs ctx['ref_channels']")
        from mne_denoise.qa.coupling import regress_out

        raw = evaluation.copy()
        eeg = raw.copy().pick("eeg")
        ed = eeg.get_data()
        rd = raw.copy().pick(list(ref)).get_data()
        shifts = range(-self.n_shifts, self.n_shifts + 1)
        cleaned = eeg.copy()
        if ed.ndim == 3:
            ntr, nch, nt = ed.shape
            E = ed.transpose(1, 0, 2).reshape(nch, -1)
            R = rd.transpose(1, 0, 2).reshape(rd.shape[1], -1)
            Rl = np.vstack([np.roll(R, s, axis=1) for s in shifts])
            cleaned._data = regress_out(E, Rl).reshape(nch, ntr, nt).transpose(1, 0, 2)
        else:
            Rl = np.vstack([np.roll(rd, s, axis=1) for s in shifts])
            cleaned._data = regress_out(ed, Rl)
        return ComparatorResult(cleaned=cleaned, status="success",
                                diagnostics={"n_shifts": self.n_shifts})


class _SSPEOG(Comparator):
    """``ssp_eog`` — SSP-style projectors from the blink-weighted EEG covariance learned
    on TRAIN (top-k PCs of the artifact subspace, the standard SSP convention), projected
    out of EVAL. Distinct from EOG-DSS (which uses regression patterns)."""

    def __init__(self, n_components: int = 1, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("ssp_eog", fit_scope="train_only", rank_reducing=True),
            n_components=n_components, **params,
        )
        self.k = int(n_components)

    def _fit(self, train, ctx):
        eog = ctx.get("eog_channels") or ctx.get("ref_channels")
        if not eog:
            return None
        Xe = _flat(train.copy().pick("eeg").get_data())
        Xo = _flat(train.copy().pick(list(eog)).get_data())
        w = np.abs(Xo).sum(0)
        w = w / (w.sum() + 1e-12)                          # emphasise blink samples
        Cw = (Xe * w) @ Xe.T                               # blink-weighted covariance
        U, _, _ = np.linalg.svd(Cw, full_matrices=False)
        return U[:, : self.k]

    def _transform(self, evaluation, payload, ctx):
        if payload is None:
            return ComparatorResult(status="skipped_missing_channels", error="ssp_eog needs ctx['eog_channels']")
        P = payload
        eeg = evaluation.copy().pick("eeg")
        proj = np.eye(P.shape[0]) - P @ P.T
        x = eeg.get_data()
        eeg._data = np.einsum("ij,ejt->eit", proj, x) if x.ndim == 3 else proj @ x
        return ComparatorResult(cleaned=eeg, status="success", rank_after=int(P.shape[0] - self.k),
                                diagnostics={"n_components": self.k})


# ---------------------------------------------------------------------------
# Literature-standard comparators (wearable-review-driven: BSS-CCA, SSA, wavelet, EMD)
# ---------------------------------------------------------------------------
class _AutoCCA(Comparator):
    """``autocca`` -- reference-free BSS-CCA (De Clercq 2006). CCA of the signal vs its
    one-sample lag orders components by autocorrelation; the low-autocorrelation
    (broadband EMG/muscle) components are dropped and the signal is reconstructed.
    Fitted on TRAIN, applied to EVAL -- the reference-free, canonical muscle counterpart
    to iCanClean."""

    def __init__(self, rho_threshold: float = 0.9, n_keep: Any = None, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("autocca", fit_scope="train_only", rank_reducing=True),
            rho_threshold=rho_threshold, n_keep=n_keep, **params,
        )
        self.rho_threshold = float(rho_threshold)
        self.n_keep = n_keep

    def _fit(self, train, ctx):
        from mne_denoise.cca import AutoCCA

        x = train.get_data() if hasattr(train, "get_data") else np.asarray(train)
        return AutoCCA(rho_threshold=self.rho_threshold, n_keep=self.n_keep).fit(_flat(x))

    def _transform(self, evaluation, payload, ctx):
        if hasattr(evaluation, "get_data"):
            x = evaluation.get_data()
            out = (np.stack([payload.transform(x[i]) for i in range(x.shape[0])])
                   if x.ndim == 3 else payload.transform(x))
            cleaned = evaluation.copy()
            cleaned._data = out
            return ComparatorResult(cleaned=cleaned, status="success", rank_after=int(payload.n_kept_),
                                    diagnostics={"n_kept": int(payload.n_kept_),
                                                 "rho_threshold": self.rho_threshold})
        out = payload.transform(np.asarray(evaluation))
        return ComparatorResult(cleaned=out, status="success", rank_after=int(payload.n_kept_),
                                diagnostics={"n_kept": int(payload.n_kept_)})


class _SSA(Comparator):
    """``ssa`` -- Singular Spectrum Analysis: per-channel eigentriple decomposition that
    drops slow / quasi-periodic (ocular drift, cardiac) components by their dominant
    frequency and reconstructs by diagonal averaging. Per-recording / unsupervised."""

    def __init__(self, drop_freq_max: float = 3.0, drop_band: Any = None,
                 window_length: Any = None, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("ssa", fit_scope="window_local"),
            drop_freq_max=drop_freq_max, drop_band=drop_band, window_length=window_length, **params,
        )
        self.drop_freq_max = float(drop_freq_max)
        self.drop_band = drop_band
        self.window_length = window_length

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        from mne_denoise.ssa import SSA

        ssa = SSA(sfreq=_sfreq(evaluation, ctx), window_length=self.window_length,
                  drop_freq_max=self.drop_freq_max, drop_band=self.drop_band)
        if hasattr(evaluation, "get_data"):
            x = evaluation.get_data()
            out = (np.stack([ssa.transform(x[i]) for i in range(x.shape[0])])
                   if x.ndim == 3 else ssa.transform(x))
            cleaned = evaluation.copy()
            cleaned._data = out
            return ComparatorResult(cleaned=cleaned, status="success",
                                    diagnostics={"drop_freq_max": self.drop_freq_max, "drop_band": self.drop_band})
        out = ssa.transform(np.asarray(evaluation))
        return ComparatorResult(cleaned=out, status="success",
                                diagnostics={"drop_freq_max": self.drop_freq_max})


class _Wavelet(Comparator):
    """``wavelet_threshold`` -- per-channel universal soft-thresholding (remove HF
    noise/muscle); ``wica`` -- wavelet-enhanced ICA (Castellanos & Makarov 2006): extract
    and subtract each ICA source's sparse high-amplitude transients. The most-used
    wearable artifact methods for ocular and muscular contamination."""

    def __init__(self, method: str = "threshold", wavelet: Any = None,
                 n_components: Any = None, **params: Any) -> None:
        cid = "wica" if method == "wica" else "wavelet_threshold"
        scope = "train_only" if method == "wica" else "window_local"
        super().__init__(
            ComparatorMeta(cid, fit_scope=scope, optional_dependency="PyWavelets"),
            method=method, wavelet=wavelet, n_components=n_components, **params,
        )
        self.method = method
        self.wavelet = wavelet
        self.n_components = n_components

    def _fit(self, train, ctx):
        if self.method != "wica":
            return None
        from mne_denoise.wavelet import WaveletICA

        x = train.get_data() if hasattr(train, "get_data") else np.asarray(train)
        return WaveletICA(n_components=self.n_components, wavelet=self.wavelet or "coif5").fit(_flat(x))

    def _transform(self, evaluation, payload, ctx):
        if self.method == "wica":
            if hasattr(evaluation, "get_data"):
                x = evaluation.get_data()
                out = (np.stack([payload.transform(x[i]) for i in range(x.shape[0])])
                       if x.ndim == 3 else payload.transform(x))
                cleaned = evaluation.copy()
                cleaned._data = out
                return ComparatorResult(cleaned=cleaned, status="success",
                                        diagnostics={"method": "wica", "wavelet": self.wavelet or "coif5"})
            out = payload.transform(np.asarray(evaluation))
            return ComparatorResult(cleaned=out, status="success", diagnostics={"method": "wica"})
        from mne_denoise.wavelet import wavelet_denoise_multichannel

        wav = self.wavelet or "sym5"
        if hasattr(evaluation, "get_data"):
            x = evaluation.get_data()
            out = (np.stack([wavelet_denoise_multichannel(x[i], wav) for i in range(x.shape[0])])
                   if x.ndim == 3 else wavelet_denoise_multichannel(x, wav))
            cleaned = evaluation.copy()
            cleaned._data = out
            return ComparatorResult(cleaned=cleaned, status="success",
                                    diagnostics={"method": "wavelet_threshold", "wavelet": wav})
        out = wavelet_denoise_multichannel(np.asarray(evaluation), wav)
        return ComparatorResult(cleaned=out, status="success", diagnostics={"method": "wavelet_threshold"})


class _EMD(Comparator):
    """``emd``/``eemd`` -- Empirical Mode Decomposition: decompose each channel into IMFs,
    drop the high-frequency (muscle/EMG) IMFs, reconstruct. Per-recording / unsupervised.
    ``eemd`` is the noise-assisted ensemble variant (slower; sensitivity tier)."""

    def __init__(self, method: str = "emd", freq_cutoff: float = 30.0, max_imf: int = 8,
                 trials: int = 20, **params: Any) -> None:
        super().__init__(
            ComparatorMeta(method, fit_scope="window_local", optional_dependency="EMD-signal"),
            method=method, freq_cutoff=freq_cutoff, max_imf=max_imf, trials=trials, **params,
        )
        self.method = method
        self.freq_cutoff = float(freq_cutoff)
        self.max_imf = int(max_imf)
        self.trials = int(trials)

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        from mne_denoise.emd import EMDDenoiser

        d = EMDDenoiser(sfreq=_sfreq(evaluation, ctx), method=self.method,
                        freq_cutoff=self.freq_cutoff, max_imf=self.max_imf, trials=self.trials)
        if hasattr(evaluation, "get_data"):
            x = evaluation.get_data()
            out = (np.stack([d.transform(x[i]) for i in range(x.shape[0])])
                   if x.ndim == 3 else d.transform(x))
            cleaned = evaluation.copy()
            cleaned._data = out
            return ComparatorResult(cleaned=cleaned, status="success",
                                    diagnostics={"method": self.method, "freq_cutoff": self.freq_cutoff})
        out = d.transform(np.asarray(evaluation))
        return ComparatorResult(cleaned=out, status="success", diagnostics={"method": self.method})


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
register("tspca", lambda **p: _TSPCA(**p))
register("ssp_eog", lambda **p: _SSPEOG(**p))
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
# wearable-review-driven literature-standard comparators
register("autocca", lambda **p: _AutoCCA(**p))
register("ssa", lambda **p: _SSA(**p))
register("wavelet_threshold", lambda **p: _Wavelet(method="threshold", **p))
register("wica", lambda **p: _Wavelet(method="wica", **p))
register("emd", lambda **p: _EMD(method="emd", **p))
register("eemd", lambda **p: _EMD(method="eemd", **p))
