"""Disk I/O helpers for the line-noise benchmark deferred-group workflow.

This module provides utilities to **save** per-subject benchmark
results (scalar metrics, per-condition metrics, model metadata) during
single-subject processing, and later **load + aggregate** them for
group-level visualization — *without* re-running the benchmark.

Typical workflow
----------------

**Step 1 — Per-subject (can be parallelised / run on a cluster):**

>>> from mne_denoise.viz.benchmark_io import save_subject_benchmark_results
>>> save_subject_benchmark_results(
...     deriv_root / sub / "line_noise" / method,
...     subject=sub,
...     method=method,
...     metrics={"peak_attenuation_db": 12.3, "R_f0": 0.85},
...     condition_metrics=[
...         {"condition": "lab", "peak_attenuation_db": 11.0},
...         {"condition": "campus", "peak_attenuation_db": 13.5},
...     ],
...     model_info={"n_components": 4, "line_freq": 50.0},
... )

**Step 2 — Group aggregation (after all subjects are done):**

>>> from mne_denoise.viz.benchmark_io import aggregate_benchmark_results
>>> grp = aggregate_benchmark_results(deriv_root, methods=["zapline", "dss"])
>>> # grp is a LineNoiseGroupData namedtuple with fields:
>>> #   .df_metrics, .df_all, .df_cond

**Step 3 — Group plots (no reprocessing needed):**

>>> from mne_denoise.viz import (
...     plot_metric_bars,
...     plot_tradeoff_scatter,
...     plot_psd_gallery,
... )
>>> plot_metric_bars(
...     grp.df_all,
...     group_col="method",
...     metric_cols=[
...         "R_f0",
...         "peak_attenuation_db",
...         "below_noise_pct",
...         "overclean_proportion",
...         "underclean_proportion",
...     ],
...     metric_labels=[
...         "R(f0) - 1",
...         "Peak Attenuation (dB)",
...         "Sub-Peak dPower (%) - 0",
...         "Overclean Fraction",
...         "Underclean Fraction",
...     ],
...     lower_better=[True, False, True, True, True],
...     fname=fig_dir / "metric_bars.png",
... )
"""

from __future__ import annotations

import json
import logging
from collections import namedtuple
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# — Public data container ——————————————————————————————————————

LineNoiseGroupData = namedtuple(
    "LineNoiseGroupData",
    ["df_metrics", "df_all", "df_cond"],
)
LineNoiseGroupData.__doc__ = """\
Aggregated group-level line-noise benchmark data.

Fields
------
df_metrics : DataFrame
    Master long-form DataFrame (all subjects × methods × conditions).
    Contains a ``condition`` column: ``"all"`` for whole-recording
    rows, and the actual condition label otherwise.
df_all : DataFrame
    Subset of *df_metrics* where ``condition == "all"``
    (whole-recording metrics).
df_cond : DataFrame
    Subset of *df_metrics* where ``condition != "all"``
    (per-condition metrics).
"""

# — File-name conventions ——————————————————————————————————————

_METRICS_TSV = "metrics.tsv"
_CONDITION_METRICS_TSV = "condition_metrics.tsv"
_MODEL_JSON = "model.json"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# =====================================================================
# save_subject_benchmark_results
# =====================================================================


def save_subject_benchmark_results(
    out_dir,
    *,
    subject,
    method,
    metrics=None,
    condition_metrics=None,
    model_info=None,
):
    """Save per-subject line-noise benchmark results to disk.

    Parameters
    ----------
    out_dir : str | Path
        Per-subject / per-method output directory.  Created if it
        does not exist.
        Typical: ``{deriv_root}/{sub}/line_noise/{method}``.
    subject : str
        Subject identifier (e.g. ``"sub-01"``).
    method : str
        Method name (e.g. ``"zapline"``, ``"dss"``).
    metrics : dict | None
        Scalar whole-recording metrics — ``{metric_name: value}``.
        Saved as a single-row :file:`metrics.tsv` with ``subject``
        and ``method`` columns prepended.
    condition_metrics : list[dict] | DataFrame | None
        Per-condition metrics.  Each dict must contain a
        ``"condition"`` key—plus the same metric columns.
        Saved as :file:`condition_metrics.tsv`.
    model_info : dict | None
        Arbitrary model metadata (hyper-parameters, timing, etc.).
        Saved as :file:`model.json` using a NumPy-aware encoder.

    Notes
    -----
    Calling this function a second time for the same subject / method
    overwrites existing files.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # — Whole-recording metrics TSV ————————————————————————————
    if metrics is not None:
        row = {"subject": subject, "method": method, **metrics}
        pd.DataFrame([row]).to_csv(out_dir / _METRICS_TSV, sep="\t", index=False)

    # — Per-condition metrics TSV ——————————————————————————————
    if condition_metrics is not None:
        if isinstance(condition_metrics, pd.DataFrame):
            df_cond = condition_metrics.copy()
        else:
            df_cond = pd.DataFrame(condition_metrics)
        if "subject" not in df_cond.columns:
            df_cond.insert(0, "subject", subject)
        if "method" not in df_cond.columns:
            df_cond.insert(1, "method", method)
        df_cond.to_csv(out_dir / _CONDITION_METRICS_TSV, sep="\t", index=False)

    # — Model metadata JSON ————————————————————————————————————
    if model_info is not None:
        info = {"subject": subject, "method": method, **model_info}
        with open(out_dir / _MODEL_JSON, "w") as f:
            json.dump(info, f, indent=2, cls=_NumpyEncoder)

    log.info("Saved benchmark results for %s/%s → %s", subject, method, out_dir)


# =====================================================================
# load_subject_benchmark_results
# =====================================================================


def load_subject_benchmark_results(sub_dir, *, methods=None):
    """Load a single subject's saved benchmark results from disk.

    Parameters
    ----------
    sub_dir : str | Path
        The per-subject benchmark directory (e.g.
        ``{deriv_root}/sub-01/line_noise``).  Expected to contain one
        sub-folder per method, each with ``metrics.tsv`` and
        optionally ``condition_metrics.tsv`` and ``model.json``.
    methods : list of str | None
        Restrict to these methods.  If *None*, auto-discovers all
        sub-directories that contain a :file:`metrics.tsv`.

    Returns
    -------
    result : dict
        Keys:

        * ``"subject"`` — subject label.
        * ``"df"`` — whole-recording metrics DataFrame.
        * ``"df_cond"`` — per-condition metrics DataFrame (may be empty).
        * ``"model_info"`` — ``{method: dict}`` of model JSON metadata.
    """
    import pandas as pd

    sub_dir = Path(sub_dir)
    subject = sub_dir.parent.name  # e.g. "sub-01"

    if methods is None:
        methods = sorted(
            d.name
            for d in sub_dir.iterdir()
            if d.is_dir() and (d / _METRICS_TSV).exists()
        )

    dfs = []
    dfs_cond = []
    model_infos = {}

    for method in methods:
        method_dir = sub_dir / method

        tsv = method_dir / _METRICS_TSV
        if tsv.exists():
            dfs.append(pd.read_csv(tsv, sep="\t"))

        cond_tsv = method_dir / _CONDITION_METRICS_TSV
        if cond_tsv.exists():
            dfs_cond.append(pd.read_csv(cond_tsv, sep="\t"))

        model_json = method_dir / _MODEL_JSON
        if model_json.exists():
            with open(model_json) as f:
                model_infos[method] = json.load(f)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df_cond = pd.concat(dfs_cond, ignore_index=True) if dfs_cond else pd.DataFrame()

    return {
        "subject": subject,
        "df": df,
        "df_cond": df_cond,
        "model_info": model_infos,
    }


# =====================================================================
# aggregate_benchmark_results — the "deferred group" entry point
# =====================================================================


def aggregate_benchmark_results(deriv_root, *, subjects=None, methods=None):
    """Aggregate per-subject benchmark results into group containers.

    This is the key function enabling the *deferred-group* workflow:
    run subjects independently (even on different machines), then call
    this once to collect everything needed for group-level plots.

    Parameters
    ----------
    deriv_root : str | Path
        Derivatives root (e.g. ``data/runabout/derivatives/mne-denoise``).
    subjects : list of str | None
        Subjects to include.  If *None*, auto-discovers all ``sub-*``
        directories that contain a ``line_noise`` subfolder.
    methods : list of str | None
        Restrict to these methods.  If *None*, discovers all
        available methods per subject.

    Returns
    -------
    data : LineNoiseGroupData
        Named tuple with fields:

        * ``df_metrics`` — master DataFrame (all rows).
        * ``df_all`` — whole-recording subset (``condition == "all"``).
        * ``df_cond`` — per-condition subset (``condition != "all"``).
    """
    import pandas as pd

    deriv_root = Path(deriv_root)

    if subjects is None:
        subjects = sorted(
            d.name
            for d in deriv_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-") and (d / "line_noise").exists()
        )
    if not subjects:
        log.warning("No subjects found in %s", deriv_root)
        return LineNoiseGroupData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # — Collect per-subject data ———————————————————————————————
    df_all_frames = []
    df_cond_frames = []

    for sub in subjects:
        sub_dir = deriv_root / sub / "line_noise"
        if not sub_dir.exists():
            log.warning("Skipping %s — no line_noise dir", sub)
            continue

        r = load_subject_benchmark_results(sub_dir, methods=methods)

        if not r["df"].empty:
            df_whole = r["df"].copy()
            df_whole["condition"] = "all"
            df_all_frames.append(df_whole)

        if not r["df_cond"].empty:
            df_cond_frames.append(r["df_cond"])

    # — Assemble outputs ———————————————————————————————————————
    df_all = (
        pd.concat(df_all_frames, ignore_index=True) if df_all_frames else pd.DataFrame()
    )
    df_cond = (
        pd.concat(df_cond_frames, ignore_index=True)
        if df_cond_frames
        else pd.DataFrame()
    )
    df_metrics = pd.concat([df_all, df_cond], ignore_index=True)

    log.info(
        "Aggregated %d subjects: %d whole-recording rows, %d condition rows",
        len(subjects),
        len(df_all),
        len(df_cond),
    )

    return LineNoiseGroupData(
        df_metrics=df_metrics,
        df_all=df_all,
        df_cond=df_cond,
    )
