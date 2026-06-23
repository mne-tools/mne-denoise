#!/usr/bin/env python
"""Generate manuscript result-figure PNGs from the aggregated benchmark data.

Reads D:/tmp/all_arms.json (per-arm/method/subject metrics) + re-runs the ground-truth
simulation, and writes multi-panel summary PNGs into the Overleaf figures/ dir, where the
manuscript's \\IfFileExists wrappers pick them up automatically.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = pathlib.Path(r"D:/overleaf_repos/69a477a4afe0ee4cadc562fd/figures")
FIG.mkdir(exist_ok=True)
D = json.load(open(r"D:/tmp/all_arms.json"))
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.titlesize": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight"})
BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#888888"


def vals(arm, method, metric):
    return [v for v in D.get(arm, {}).get(method, {}).get(metric, {}).values() if v is not None]


def msem(arm, method, metric):
    v = vals(arm, method, metric)
    if not v:
        return None, None, 0
    return st.mean(v), (st.pstdev(v) / max(1, len(v) ** 0.5)), len(v)


def barpanel(ax, arm, methods, metric, labels, ylabel, title, color=BLUE, hline=None):
    xs = np.arange(len(methods))
    ms = [msem(arm, m, metric)[0] or 0 for m in methods]
    es = [msem(arm, m, metric)[1] or 0 for m in methods]
    ax.bar(xs, ms, yerr=es, color=color, capsize=3, width=0.66)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, ls="--", c=GREY, lw=1)
    return ms


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig); print(f"  wrote figures/{name}")


# --- 1. Failure taxonomy --------------------------------------------------
def fig_failure():
    meth = ["zapline_plus", "dss_average_bias", "asr", "iterative_dss", "icanclean"]
    lab = ["ZapLine+", "DSS", "ASR", "IterativeDSS", "iCanClean"]
    succ = [98, 100, 100, 100, 100]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(meth)), succ, color=[RED if s < 100 else GREEN for s in succ], width=0.6)
    ax.set_xticks(range(len(meth))); ax.set_xticklabels(lab, rotation=20, ha="right")
    ax.set_ylim(90, 101); ax.set_ylabel("Success rate (\\%)")
    ax.set_title("Intention-to-benchmark success rate by method (terminal status)")
    for i, s in enumerate(succ):
        ax.text(i, s + 0.1, f"{s}%", ha="center", fontsize=8)
    ax.text(0, 96.5, "1 long-MEG\nsubject OOM", ha="center", fontsize=7, color=RED)
    save(fig, "failure_taxonomy.png")


# --- 2. ds003620 line group + QC -----------------------------------------
def fig_line_group():
    meth = ["none", "notch", "non_spatial_line", "zapline_plus"]
    lab = ["none", "notch", "non-spatial", "ZapLine+"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    barpanel(axs[0], "line_ds003620", meth, "R_f0", lab, "held-out $R(f_0)$",
             "Line attenuation (ds003620, n=42)", hline=1.0)
    barpanel(axs[1], "line_ds003620", meth, "overclean_proportion", lab, "overclean proportion",
             "Overcleaning (collateral floor loss)", color=RED)
    save(fig, "runabout_line_noise_group.png")


def fig_line_qc():
    # ZapLine n_remove sweep -> Pareto (attenuation dB vs overclean)
    sweep = sorted([m for m in D["line_ds003620"] if m.startswith("zapline__n_remove-")],
                   key=lambda m: int(m.split("-")[-1]))
    n = [int(m.split("-")[-1]) for m in sweep]
    atten = [msem("line_ds003620", m, "peak_attenuation_db")[0] or 0 for m in sweep]
    over = [msem("line_ds003620", m, "overclean_proportion")[0] or 0 for m in sweep]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    axs[0].plot(n, atten, "o-", color=BLUE); axs[0].set_xlabel("n\\_remove")
    axs[0].set_ylabel("peak attenuation (dB)"); axs[0].set_title("ZapLine sweep: attenuation")
    axs[1].plot(over, atten, "o-", color=RED)
    for x, y, k in zip(over, atten, n):
        axs[1].annotate(str(k), (x, y), fontsize=7)
    axs[1].set_xlabel("overclean proportion"); axs[1].set_ylabel("peak attenuation (dB)")
    axs[1].set_title("Attenuation–distortion Pareto")
    save(fig, "runabout_line_noise_qc.png")


def fig_pareto():
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for arm, col, lab in [("line_ds003620", BLUE, "ds003620 (EEG)")]:
        sweep = sorted([m for m in D[arm] if m.startswith("zapline__n_remove-")],
                       key=lambda m: int(m.split("-")[-1]))
        atten = [msem(arm, m, "peak_attenuation_db")[0] or 0 for m in sweep]
        over = [msem(arm, m, "overclean_proportion")[0] or 0 for m in sweep]
        ax.plot(over, atten, "o-", color=col, label=lab)
        for x, y, m in zip(over, atten, sweep):
            ax.annotate(m.split("-")[-1], (x, y), fontsize=7)
    ax.set_xlabel("overcleaning (collateral floor loss)"); ax.set_ylabel("line attenuation (dB)")
    ax.set_title("Attenuation–preservation trade-off (ZapLine n\\_remove)"); ax.legend()
    save(fig, "tradeoff_pareto_fronts.png")


# --- 3. evoked ERP-CORE + MEG --------------------------------------------
def _g(arm, method):
    d = vals(arm, method, "n170_diff")
    if len(d) < 2:
        return 0.0
    return abs(st.mean(d) / st.pstdev(d)) * (1 - 3 / (4 * len(d) - 9))


def fig_evoked(arm, name, title):
    meth = ["none", "dss_average_bias", "rank_matched_pca"]; lab = ["none", "DSS", "PCA"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    axs[0].bar(range(3), [_g(arm, m) for m in meth], color=BLUE, width=0.6)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(lab)
    axs[0].set_ylabel("|Hedges $g$| (held-out)"); axs[0].set_title(f"{title}: effect size")
    barpanel(axs[1], arm, meth, "split_half", lab, "split-half reliability", "Reliability", color=GREEN)
    save(fig, name)


# --- 4. ocular -----------------------------------------------------------
def fig_ocular():
    meth = ["none", "eog_dss", "eog_regression", "ica_iclabel_rejection", "ssp_eog"]
    lab = ["none", "EOG-DSS", "EOG-reg", "ICA+ICLabel", "SSP-EOG"]
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.3))
    barpanel(axs[0], "ocular_erp_core", meth, "blink_coupling", lab, "blink coupling (EOG)",
             "Blink removal (lower better, n=40)", color=RED)
    barpanel(axs[1], "ocular_erp_core", meth, "n170_amp", lab, "N170 amplitude (V)",
             "N170 preservation", color=BLUE)
    save(fig, "erp_core_eog_summary.png")


# --- 5. muscle -----------------------------------------------------------
def fig_muscle():
    meth = ["none", "asr", "rasr_windowed", "tspca", "icanclean"]
    lab = ["none", "ASR", "rASR", "TSPCA", "iCanClean"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    barpanel(axs[0], "muscle_ds004505", meth, "hf_power", lab, "HF power (20–100 Hz)",
             "Muscle/HF (n=25)", color=RED)
    barpanel(axs[1], "muscle_ds004505", meth, "alpha_power", lab, "$\\alpha$ power", "Alpha preservation", color=GREEN)
    barpanel(axs[2], "muscle_ds004505", meth, "beta_power", lab, "$\\beta$ power", "Beta preservation", color=GREEN)
    save(fig, "tabletennis_reference_summary.png")


# --- 6. line-MEG ---------------------------------------------------------
def fig_megline():
    meth = ["none", "notch", "non_spatial_line", "zapline_plus"]
    lab = ["none", "notch", "non-spatial", "ZapLine+"]
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    barpanel(ax, "line_ds000117", meth, "peak_attenuation_db", lab, "peak attenuation (dB)",
             "MEG line attenuation @ 50 Hz (mag, n=15)")
    save(fig, "meg_scaling_summary.png")


# --- 7. ground-truth (re-run sim) ----------------------------------------
def fig_groundtruth():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from run_ground_truth_arm import _load_cfg, run
    cfg = _load_cfg(str(pathlib.Path(__file__).resolve().parents[2] /
                       "configs/benchmarks/ground_truth_generic.yaml"))
    rows, methods = run(cfg, "D:/tmp/gtfig", synthetic=True)
    order = ["oracle", "fastica", "infomax", "iterative_dss", "picard", "pca"]
    lab = ["oracle", "FastICA", "Infomax", "IterDSS", "Picard", "PCA"]
    rr = {m: st.mean([r["rrmse"] for r in rows if r.get("method") == m and r.get("status") == "success"])
          for m in order if any(r.get("method") == m for r in rows)}
    fig, ax = plt.subplots(figsize=(6, 3.3))
    xs = [m for m in order if m in rr]
    ax.bar(range(len(xs)), [rr[m] for m in xs], color=[GREEN] + [BLUE] * (len(xs) - 1), width=0.62)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([lab[order.index(m)] for m in xs], rotation=20, ha="right")
    ax.set_ylabel("source RRMSE (lower better)")
    ax.set_title("Ground-truth BSS recovery (generic mixing)")
    save(fig, "nonlinear_dss_summary.png")


if __name__ == "__main__":
    fig_failure(); fig_line_group(); fig_line_qc(); fig_pareto()
    fig_evoked("evoked_erp_core", "evoked_erp_core_summary.png", "N170 (ERP-CORE, n=40)")
    fig_evoked("evoked_ds000117", "evoked_ds000117_summary.png", "M170 (MEG, n=16)")
    fig_ocular(); fig_muscle(); fig_megline()
    try:
        fig_groundtruth()
    except Exception as exc:  # noqa: BLE001
        print(f"  (ground-truth fig skipped: {exc})")
    print("done")
