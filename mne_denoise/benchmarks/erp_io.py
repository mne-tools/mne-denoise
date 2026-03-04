"""Disk I/O helpers for the ERP DSS benchmark deferred-group workflow.

This module provides utilities to **save** per-subject intermediate
results (evoked arrays, difference-wave arrays, scalar metrics) during
single-subject processing, and later **load + aggregate** them for
group-level visualization â€” *without* re-running preprocessing.

Typical workflow
----------------

**Step 1 â€” Per-subject (can be parallelised / run on a cluster):**

>>> from mne_denoise.viz.erp_io import save_subject_erp_results
>>> save_subject_erp_results(
...     deriv_root / sub / "erp_dss",
...     subject=sub,
...     pipe_evokeds={"C0": ev0, "C1": ev1, "C2": ev2},
...     diff_waves=diff_waves,  # {(cond, pipe): 1-D array}
...     effect_sizes=effect_sizes,  # {(cond, pipe): float}
...     times_ms=times_ms,
...     metrics_df=df_metrics,  # or dict -> converted
... )

**Step 2 â€” Group aggregation (after all subjects are done):**

>>> from mne_denoise.viz.erp_io import aggregate_erp_results
>>> agg = aggregate_erp_results(deriv_root, subjects=None)
>>> # agg is an ERPGroupData namedtuple with fields:
>>> #   .df, .all_evokeds, .all_diff_waves, .all_effect_sizes, .times_ms

**Step 3 â€” Group plots (no reprocessing needed):**

>>> from mne_denoise.viz import (
...     plot_grand_average_evokeds,
...     plot_erp_grand_condition_interaction,
...     plot_forest,
...     plot_null_distribution,
... )
>>> plot_grand_average_evokeds(agg.all_evokeds, fname=fig_dir / "grand_avg.png")
"""

from __future__ import annotations

import json
import logging
from collections import namedtuple
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# â”€â”€ Public data container â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ERPGroupData = namedtuple(
    "ERPGroupData",
    ["df", "all_evokeds", "all_diff_waves", "all_effect_sizes", "times_ms"],
)
ERPGroupData.__doc__ = """\
Aggregated group-level ERP data.

Fields
------
df : DataFrame
    Long-form scalar metrics (all subjects Ã— pipelines).
all_evokeds : dict
    ``{pipe_tag: list[Evoked]}`` â€” one Evoked per subject.
all_diff_waves : dict
    ``{(condition, pipe_tag): ndarray (n_subjects, n_times)}``
all_effect_sizes : dict
    ``{(condition, pipe_tag): ndarray (n_subjects,)}``
times_ms : ndarray | None
    Common time vector in milliseconds (from the first subject).
"""

# â”€â”€ File-name conventions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_METRICS_TSV = "metrics.tsv"
_EVOKEDS_NPZ = "evokeds.npz"
_DIFF_WAVES_NPZ = "diff_waves.npz"
_META_JSON = "erp_io_meta.json"


# =====================================================================
# save_subject_erp_results
# =====================================================================


def save_subject_erp_results(
    out_dir,
    *,
    subject,
    pipe_evokeds=None,
    diff_waves=None,
    effect_sizes=None,
    times_ms=None,
    metrics_df=None,
    metrics_dict=None,
    pipe_order=None,
):
    """Save per-subject ERP results to disk for later group aggregation.

    Parameters
    ----------
    out_dir : str | Path
        Per-subject output directory.  Created if it does not exist.
        Typical: ``{deriv_root}/{sub}/erp_dss``.
    subject : str
        Subject identifier (e.g. ``"sub-01"``).
    pipe_evokeds : dict | None
        ``{pipe_tag: Evoked}`` â€” MNE Evoked objects per pipeline.
        Stored as ``{pipe_tag}_data`` (n_ch, n_times) + ``{pipe_tag}_ch_names``
        + ``times`` in :file:`evokeds.npz`.
    diff_waves : dict | None
        ``{(condition, pipe_tag): 1-D array}`` â€” difference waves (ÂµV).
        Stored in :file:`diff_waves.npz`.
    effect_sizes : dict | None
        ``{(condition, pipe_tag): float}`` â€” Hedges' *g* per cell.
        Stored in :file:`erp_io_meta.json`.
    times_ms : ndarray | None
        Time vector in milliseconds.  Saved in :file:`evokeds.npz`.
    metrics_df : DataFrame | None
        If given, saved as :file:`metrics.tsv` (appending ``subject``
        and ``pipeline`` columns if not present).
    metrics_dict : dict | None
        ``{pipe_tag: dict_of_scalars}`` â€” alternative to *metrics_df*.
        Converted to a DataFrame with ``subject`` and ``pipeline`` cols.
    pipe_order : list of str | None
        Pipeline ordering hint saved to metadata.

    Notes
    -----
    Calling this function a second time for the same subject overwrites
    existing files.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # â”€â”€ Metrics TSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if metrics_df is not None:
        metrics_df.to_csv(out_dir / _METRICS_TSV, sep="\t", index=False)
    elif metrics_dict is not None:
        rows = []
        for ptag, m in metrics_dict.items():
            row = {"subject": subject, "pipeline": ptag, **m}
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / _METRICS_TSV, sep="\t", index=False)

    # â”€â”€ Evokeds NPZ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pipe_evokeds is not None:
        npz_kw = {}
        for ptag, ev in pipe_evokeds.items():
            npz_kw[f"{ptag}_data"] = np.asarray(ev.data if hasattr(ev, "data") else ev)
            if hasattr(ev, "ch_names"):
                npz_kw[f"{ptag}_ch_names"] = np.array(ev.ch_names)
        if times_ms is not None:
            npz_kw["times_ms"] = np.asarray(times_ms)
        elif pipe_evokeds:
            # Derive times from first Evoked
            first = next(iter(pipe_evokeds.values()))
            if hasattr(first, "times"):
                npz_kw["times_ms"] = first.times * 1000
        np.savez_compressed(out_dir / _EVOKEDS_NPZ, **npz_kw)

    # â”€â”€ Diff-waves NPZ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if diff_waves is not None:
        npz_kw = {}
        for (cond, ptag), arr in diff_waves.items():
            key = f"{cond}__{ptag}"  # e.g. "lab__C2"
            npz_kw[key] = np.asarray(arr)
        if times_ms is not None:
            npz_kw["times_ms"] = np.asarray(times_ms)
        np.savez_compressed(out_dir / _DIFF_WAVES_NPZ, **npz_kw)

    # â”€â”€ Metadata JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    meta = {"subject": subject}
    if pipe_order is not None:
        meta["pipe_order"] = list(pipe_order)
    if pipe_evokeds is not None:
        meta["pipe_tags"] = sorted(pipe_evokeds.keys())
    if effect_sizes is not None:
        # Serialise (cond, pipe) -> float
        meta["effect_sizes"] = {
            f"{cond}__{ptag}": float(val) for (cond, ptag), val in effect_sizes.items()
        }
    if diff_waves is not None:
        meta["diff_wave_keys"] = [f"{cond}__{ptag}" for cond, ptag in diff_waves]
    with open(out_dir / _META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved ERP results for %s â†’ %s", subject, out_dir)


# =====================================================================
# load_subject_erp_results
# =====================================================================


def load_subject_erp_results(sub_dir):
    """Load a single subject's saved ERP results from disk.

    Parameters
    ----------
    sub_dir : str | Path
        The per-subject directory (e.g. ``{deriv_root}/sub-01/erp_dss``).

    Returns
    -------
    result : dict
        Keys (present only if the corresponding file exists):

        * ``"subject"`` â€” subject label
        * ``"df"`` â€” scalar metrics DataFrame
        * ``"pipe_evokeds"`` â€” ``{pipe_tag: ndarray (n_ch, n_times)}``
        * ``"ch_names"`` â€” ``{pipe_tag: list[str]}``
        * ``"times_ms"`` â€” 1-D array
        * ``"diff_waves"`` â€” ``{(cond, pipe): 1-D array}``
        * ``"effect_sizes"`` â€” ``{(cond, pipe): float}``
    """
    import pandas as pd

    sub_dir = Path(sub_dir)
    result = {}

    # â”€â”€ Metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    meta_path = sub_dir / _META_JSON
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    result["subject"] = meta.get("subject", sub_dir.parent.name)

    # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tsv_path = sub_dir / _METRICS_TSV
    if tsv_path.exists():
        result["df"] = pd.read_csv(tsv_path, sep="\t")

    # â”€â”€ Evokeds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    npz_path = sub_dir / _EVOKEDS_NPZ
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        pipe_evokeds = {}
        ch_names = {}
        for key in data.files:
            if key == "times_ms":
                result["times_ms"] = data[key]
            elif key.endswith("_data"):
                ptag = key[: -len("_data")]
                pipe_evokeds[ptag] = data[key]
            elif key.endswith("_ch_names"):
                ptag = key[: -len("_ch_names")]
                ch_names[ptag] = list(data[key])
        result["pipe_evokeds"] = pipe_evokeds
        result["ch_names"] = ch_names

    # â”€â”€ Diff waves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dw_path = sub_dir / _DIFF_WAVES_NPZ
    if dw_path.exists():
        data = np.load(dw_path, allow_pickle=True)
        diff_waves = {}
        for key in data.files:
            if key == "times_ms":
                if "times_ms" not in result:
                    result["times_ms"] = data[key]
            elif "__" in key:
                cond, ptag = key.split("__", 1)
                diff_waves[(cond, ptag)] = data[key]
        result["diff_waves"] = diff_waves

    # â”€â”€ Effect sizes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "effect_sizes" in meta:
        es = {}
        for key, val in meta["effect_sizes"].items():
            cond, ptag = key.split("__", 1)
            es[(cond, ptag)] = val
        result["effect_sizes"] = es

    return result


# =====================================================================
# aggregate_erp_results â€” the "deferred group" entry point
# =====================================================================


def aggregate_erp_results(deriv_root, *, subjects=None):
    """Aggregate per-subject ERP results from disk into group containers.

    This is the key function enabling the *deferred-group* workflow:
    run subjects independently (even on different machines), then call
    this once to collect everything needed for group-level plots.

    Parameters
    ----------
    deriv_root : str | Path
        Derivatives root (e.g. ``data/runabout/derivatives/mne-denoise``).
    subjects : list of str | None
        Subjects to include.  If *None*, auto-discovers all ``sub-*``
        directories that contain an ``erp_dss`` subfolder.

    Returns
    -------
    data : ERPGroupData
        Named tuple with fields:

        * ``df`` â€” DataFrame of scalar metrics (all subjects Ã— pipes).
        * ``all_evokeds`` â€” ``{pipe_tag: [array, ...]}`` per subject.
        * ``all_diff_waves`` â€” ``{(cond, pipe): (n_sub, n_times)}``.
        * ``all_effect_sizes`` â€” ``{(cond, pipe): (n_sub,)}``.
        * ``times_ms`` â€” common time vector or *None*.
    """
    import pandas as pd

    deriv_root = Path(deriv_root)

    if subjects is None:
        subjects = sorted(
            d.name
            for d in deriv_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-") and (d / "erp_dss").exists()
        )
    if not subjects:
        log.warning("No subjects found in %s", deriv_root)
        return ERPGroupData(pd.DataFrame(), {}, {}, {}, None)

    # â”€â”€ Collect per-subject data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df_frames = []
    evokeds_by_pipe = {}  # {pipe: [array_sub1, array_sub2, ...]}
    dw_by_key = {}  # {(cond, pipe): [array_sub1, ...]}
    es_by_key = {}  # {(cond, pipe): [float_sub1, ...]}
    times_ms = None

    for sub in subjects:
        sub_dir = deriv_root / sub / "erp_dss"
        if not sub_dir.exists():
            log.warning("Skipping %s â€” no erp_dss dir", sub)
            continue

        r = load_subject_erp_results(sub_dir)

        if "df" in r:
            df_frames.append(r["df"])

        if "times_ms" in r and times_ms is None:
            times_ms = r["times_ms"]

        if "pipe_evokeds" in r:
            for ptag, arr in r["pipe_evokeds"].items():
                evokeds_by_pipe.setdefault(ptag, []).append(arr)

        if "diff_waves" in r:
            for key, arr in r["diff_waves"].items():
                dw_by_key.setdefault(key, []).append(arr)

        if "effect_sizes" in r:
            for key, val in r["effect_sizes"].items():
                es_by_key.setdefault(key, []).append(val)

    # â”€â”€ Assemble outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = pd.concat(df_frames, ignore_index=True) if df_frames else pd.DataFrame()

    # Stack diff waves: (n_sub, n_times)
    all_diff_waves = {key: np.array(arrs) for key, arrs in dw_by_key.items()}

    # Stack effect sizes: (n_sub,)
    all_effect_sizes = {key: np.array(vals) for key, vals in es_by_key.items()}

    log.info(
        "Aggregated %d subjects: %d metric rows, %d pipe tags, %d diff-wave keys",
        len(subjects),
        len(df),
        len(evokeds_by_pipe),
        len(all_diff_waves),
    )

    return ERPGroupData(
        df=df,
        all_evokeds=evokeds_by_pipe,
        all_diff_waves=all_diff_waves,
        all_effect_sizes=all_effect_sizes,
        times_ms=times_ms,
    )
