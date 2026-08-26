"""Apply the paper-grounded *baseline* preprocessing of an arm config.

The baseline is the minimal common stage at which the method under test and every
comparator receive identical data (see ``docs/benchmarks/PREPROCESSING.md``).  It
adopts each dataset's acquisition/reference/filter conventions but NOT its
artifact-removal steps (those are under test).  This module applies the
*raw-level* steps of a config's ``baseline_preprocessing.steps`` block:

    highpass_hz, cleanline_hz, reference/offline_reference, bad_channel_interp,
    resample_hz (skipped when keep_native_sfreq is set — the line-arm guard).

Dataset-loading choices (raw FIF vs SSS, BTi reader), epoching, and mag/grad
separation are the runner's responsibility, not this module's.
"""

from __future__ import annotations

from typing import Any


def _apply_reference(raw, ref, verbose) -> list[str]:
    if not ref or str(ref).lower() in ("as_recorded", "none", "cms_drl", "nose", "fcz", "cpz"):
        # online references / no offline re-reference requested
        return [] if not ref else [f"reference_kept({ref})"]
    r = str(ref).lower()
    if "average" in r:  # average_all_33, common_average_per_layer_full_rank, ...
        raw.set_eeg_reference("average", projection=False, verbose=verbose)
        return ["reference=average"]
    if "mastoid" in r:
        picks = [c for c in ("TP9", "TP10", "M1", "M2") if c in raw.ch_names]
        if picks:
            raw.set_eeg_reference(picks, verbose=verbose)
            return [f"reference={picks}"]
        return ["reference_mastoid_unavailable"]
    return [f"reference_unhandled({ref})"]


def apply_baseline(raw, baseline_block: dict | None, *, verbose: bool = False):
    """Return ``(raw_copy, applied)`` after applying the raw-level baseline steps.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded recording at the dataset's native state.
    baseline_block : dict
        The config's ``baseline_preprocessing`` mapping (with a ``steps`` dict).
    """
    raw = raw.copy()
    steps = (baseline_block or {}).get("steps", {}) or {}
    applied: list[str] = []

    # 1. high-pass (slow-drift removal only)
    hp = steps.get("highpass_hz")
    if hp not in (None, False):
        raw.filter(l_freq=float(hp), h_freq=None, picks="data", verbose=verbose)
        applied.append(f"highpass={hp}Hz")

    # 2. baseline line removal — ONLY when the arm declares cleanline (e.g. the
    #    muscle arm at 60 Hz, where line noise is not the artifact under test).
    #    Line arms deliberately omit this so the line content reaches the methods.
    cl = steps.get("cleanline_hz")
    if cl not in (None, False):
        from .adapters import _line_harmonics

        raw.notch_filter(
            _line_harmonics(float(cl), raw.info["sfreq"]),
            picks="data", method="spectrum_fit", verbose=verbose,
        )
        applied.append(f"cleanline={cl}Hz")

    # 3. re-reference (offline)
    applied += _apply_reference(raw, steps.get("offline_reference") or steps.get("reference"), verbose)

    # 4. bad-channel interpolation (only if bads are marked)
    if steps.get("bad_channel_interp") and raw.info["bads"]:
        raw.interpolate_bads(verbose=verbose)
        applied.append("interpolate_bads")

    # 5. resample — only if explicitly requested AND not pinned to native rate.
    #    keep_native_sfreq is the line-arm guard (no resample before line eval).
    rs = steps.get("resample_hz")
    if rs not in (None, False) and not steps.get("keep_native_sfreq"):
        raw.resample(float(rs), verbose=verbose)
        applied.append(f"resample={rs}Hz")

    return raw, applied


def branch_context(cfg: dict) -> dict:
    """Build the ``ctx`` dict passed to comparator fit/transform from a config."""
    ds = cfg.get("dataset", {}) or {}
    bp = cfg.get("baseline_preprocessing", {}) or {}
    ctx: dict[str, Any] = {}
    lf = ds.get("line_freq_hz")
    if lf is None:
        lf = bp.get("mains_hz")
    if lf not in (None, False):
        ctx["line_freq"] = [float(x) for x in lf] if isinstance(lf, list) else float(lf)
    ref = (ds.get("reference_channels") or {})
    if ref:
        ctx["reference_channels"] = ref
    return ctx
