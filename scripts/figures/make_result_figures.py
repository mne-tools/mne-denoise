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
try:
    D = json.load(open(r"D:/tmp/all_arms.json"))
except FileNotFoundError:
    D = {}   # all_arms.json absent: data-driven figs no-op; self-contained figs (ssvep/cardiac/mobile) still run
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.titlesize": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight"})
BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#888888"


def vals(arm, method, metric):
    return [v for v in D.get(arm, {}).get(method, {}).get(metric, {}).values() if v is not None]


def msem(arm, method, metric):
    # median + IQR/2 (robust; matches the paper's median aggregation, avoids outlier-driven means)
    v = vals(arm, method, metric)
    if not v:
        return None, None, 0
    med = st.median(v)
    if len(v) >= 4:
        q = st.quantiles(v, n=4)
        spread = (q[2] - q[0]) / 2
    else:
        spread = st.pstdev(v) / max(1, len(v) ** 0.5)
    return med, spread, len(v)


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
    # env-segmented (lab/oval/campus recovered from the participants.tsv counterbalancing)
    envs = ["lab", "oval", "campus"]; elab = ["Lab", "Field", "Campus"]
    meth = ["none", "notch", "non_spatial_line", "zapline_plus"]; mlab = ["none", "notch", "non-sp", "ZapLine+"]
    mcol = [GREY, RED, "#DD8452", BLUE]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
    x = np.arange(len(envs)); w = 0.2
    for j, (m, c) in enumerate(zip(meth, mcol)):
        vals = [msem("line_ds003620_env", f"{e}__{m}", "R_f0")[0] or 0 for e in envs]
        axs[0].bar(x + (j - 1.5) * w, vals, w, label=mlab[j], color=c)
    axs[0].axhline(1.0, ls="--", c=GREY, lw=1)
    axs[0].set_xticks(x); axs[0].set_xticklabels(elab); axs[0].set_ylabel("held-out $R(f_0)$")
    axs[0].set_title("Line attenuation by mobility (ds003620, n=41)"); axs[0].legend(fontsize=6.5, ncol=2)
    nrm = [msem("line_ds003620_env", f"{e}__zapline_plus", "n_removed")[0] or 0 for e in envs]
    axs[1].bar(x, nrm, color=BLUE, width=0.6)
    for i, v in enumerate(nrm):
        axs[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axs[1].set_xticks(x); axs[1].set_xticklabels(elab); axs[1].set_ylabel("ZapLine+ components removed")
    axs[1].set_title("Adaptive removal $\\propto$ contamination")
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
    meth = ["none", "eog_dss", "eog_regression", "ica_iclabel_rejection", "ssp_eog", "ssa", "wica", "wavelet_threshold"]
    lab = ["none", "EOG-DSS", "EOG-reg", "ICA+ICLabel", "SSP-EOG", "SSA", "wICA", "wavelet"]
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.3))
    barpanel(axs[0], "ocular_erp_core", meth, "blink_coupling", lab, "blink coupling (EOG)",
             "Blink removal (lower better, n=40)", color=RED)
    barpanel(axs[1], "ocular_erp_core", meth, "n170_amp", lab, "N170 amplitude (V)",
             "N170 amplitude (raw; contrast $g$ in text)", color=BLUE)
    save(fig, "erp_core_eog_summary.png")


# --- 5. muscle -----------------------------------------------------------
def fig_muscle():
    meth = ["none", "asr__cutoff-20", "rasr_windowed", "tspca", "icanclean", "autocca__rho_threshold-0.8", "wavelet_threshold"]
    lab = ["none", "ASR", "rASR", "TSPCA", "iCanClean", "autoCCA", "wavelet"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    barpanel(axs[0], "muscle_ds004505", meth, "emg_coupling", lab, "EMG--scalp coupling",
             "EMG coupling (lower better, n=25)", color=RED)
    barpanel(axs[1], "muscle_ds004505", meth, "alpha_power", lab, "$\\alpha$ power", "Alpha preservation", color=GREEN)
    barpanel(axs[2], "muscle_ds004505", meth, "beta_power", lab, "$\\beta$ power", "Beta preservation", color=GREEN)
    save(fig, "tabletennis_reference_summary.png")


# --- 6. line-MEG ---------------------------------------------------------
def fig_megline():
    meth = ["none", "notch", "non_spatial_line", "zapline_plus"]
    lab = ["none", "notch", "non-spatial", "ZapLine+"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    barpanel(axs[0], "line_ds000117", meth, "peak_attenuation_db", lab, "peak attenuation (dB)",
             "Elekta ds000117 (mag, n=15)")
    barpanel(axs[1], "line_meg_masc", meth, "peak_attenuation_db", lab, "",
             "KIT meg-MASC (mag, n=11)")
    fig.suptitle("MEG line attenuation @ 50 Hz across systems (contamination-proportional)", fontsize=9.5)
    save(fig, "meg_scaling_summary.png")


# --- 7. ground-truth (re-run sim) ----------------------------------------
def fig_groundtruth():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from run_ground_truth_arm import _load_cfg, run
    cfg = _load_cfg(str(pathlib.Path(__file__).resolve().parents[2] /
                       "configs/benchmarks/ground_truth_generic.yaml"))
    rows, methods = run(cfg, "D:/tmp/gtfig", synthetic=True)
    order = ["oracle", "jade", "fastica", "infomax", "sobi", "iterative_dss", "picard", "pca"]
    lab = ["oracle", "JADE", "FastICA", "Infomax", "SOBI", "IterDSS", "Picard", "PCA"]
    rr = {m: st.mean([r["rrmse"] for r in rows if r.get("method") == m and r.get("status") == "success"])
          for m in order if any(r.get("method") == m for r in rows)}
    fig, ax = plt.subplots(figsize=(6, 3.3))
    xs = [m for m in order if m in rr]
    ax.bar(range(len(xs)), [rr[m] for m in xs], color=[GREEN] + [BLUE] * (len(xs) - 1), width=0.62)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([lab[order.index(m)] for m in xs], rotation=20, ha="right")
    ax.set_ylabel("source RRMSE (lower better)")
    ax.set_title("Ground-truth BSS recovery (generic mixing)")
    save(fig, "nonlinear_dss_summary.png")


# --- 8. phantom (reference-aware, ds004784) ------------------------------
def fig_phantom():
    meth = ["none", "icanclean", "tspca"]; lab = ["none", "iCanClean", "TSPCA"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3))
    barpanel(axs[0], "phantom_ds004784", meth, "emg_coupling", lab,
             "residual coupling $\\rho$ to EXG", "Artifact removal (phantom, n=1)", color=RED)
    barpanel(axs[1], "phantom_ds004784", meth, "hf_power", lab,
             "HF power (20--100 Hz)", "HF artifact level", color=BLUE)
    save(fig, "phantom_ds004784_summary.png")


# --- 9. mobile auditory ERP (ds003620, reframed) -------------------------
def fig_mobile_erp():
    p = pathlib.Path(__file__).resolve().parents[2] / "docs/benchmarks/filled_values/mobile_erp_ds003620.json"
    m = json.load(open(p))
    g = m.get("mobile_erp_ds003620.g.c0", 0); lat = m.get("mobile_erp_ds003620.latency_ms.c0", 0)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2))
    axs[0].bar([0], [g], color=BLUE, width=0.5)
    axs[0].axhline(0.5, ls="--", c=GREY, lw=1); axs[0].axhline(0.8, ls=":", c=GREY, lw=1)
    axs[0].set_xticks([0]); axs[0].set_xticklabels(["C0 (none)"]); axs[0].set_ylim(0, 1.0)
    axs[0].set_ylabel("vertex N1--P2 effect size (Hedges $g$)")
    axs[0].set_title("Mobile auditory ERP (ds003620, n=41)")
    axs[0].text(0, g + 0.02, f"{g:.2f}", ha="center", fontsize=10)
    axs[1].bar([0], [lat], color=GREEN, width=0.5); axs[1].axhspan(150, 250, color=GREEN, alpha=0.12)
    axs[1].set_xticks([0]); axs[1].set_xticklabels(["P2 peak"]); axs[1].set_ylabel("latency (ms)")
    axs[1].set_title("P2 peak latency"); axs[1].text(0, lat + 4, f"{lat:.0f} ms", ha="center", fontsize=10)
    save(fig, "runabout_erp_summary.png")


# --- 10. SSVEP comb-DSS (Tsinghua) ---------------------------------------
def fig_ssvep():
    p = pathlib.Path(__file__).resolve().parents[2] / "docs/benchmarks/filled_values/ssvep_tsinghua.json"
    m = json.load(open(p))
    sr, sd, n = m["ssvep_tsinghua.snr_raw"], m["ssvep_tsinghua.snr_dss"], m["ssvep_tsinghua.n"]
    ac, ad = m["ssvep_tsinghua.acc_cca_pct"], m["ssvep_tsinghua.acc_dss_cca_pct"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axs[0].bar([0, 1], [sr, sd], color=[GREY, BLUE], width=0.55)
    axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["raw", "comb-DSS"])
    axs[0].set_ylabel("stimulus-harmonic SNR (dB)")
    axs[0].set_title(f"SSVEP target enhancement (Tsinghua, n={n})")
    axs[0].annotate(f"+{m['ssvep_tsinghua.gain_db']:.1f} dB", xy=(1, sd), xytext=(0.5, sd + 0.5),
                    ha="center", fontsize=9.5, color=BLUE)
    axs[1].bar([0, 1], [ac, ad], color=[GREY, BLUE], width=0.55)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["CCA", "DSS+CCA"]); axs[1].set_ylim(80, 100)
    axs[1].set_ylabel("40-class accuracy (\\%)")
    axs[1].set_title("Decoding (already saturated; DSS-redundant)")
    for i, v in enumerate([ac, ad]):
        axs[1].text(i, v + 0.4, f"{v:.1f}\\%", ha="center", fontsize=8)
    save(fig, "ssvep_dss_summary.png")


# --- 11. cardiac (CAP, reference-coupled) --------------------------------
def fig_cardiac():
    p = pathlib.Path(__file__).resolve().parents[2] / "docs/benchmarks/filled_values/cardiac_cap.json"
    m = json.load(open(p))
    meth = ["none", "ecg_regression", "ica_ecg", "ssp_ecg", "cardiac_dss", "wavelet_threshold"]
    lab = ["none", "ECG-reg\n(lagged)", "ICA-ECG", "SSP-ECG", "CycleAvg\nDSS", "wavelet"]
    qn, nn = st.median(m["none"]["qrs"]), st.median(m["none"]["neu"])
    removed = [100 * (1 - st.median(m[x]["qrs"]) / qn) for x in meth]
    retained = [100 * st.median(m[x]["neu"]) / nn for x in meth]
    col = [GREY, GREEN, GREEN, RED, RED, BLUE]   # green=clean removers, red=over-cleaners, blue=null
    nx = range(len(meth))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    axs[0].bar(nx, removed, color=col, width=0.62)
    axs[0].set_xticks(nx); axs[0].set_xticklabels(lab, fontsize=8)
    axs[0].set_ylabel("\\% QRS-locked cardiac removed"); axs[0].set_title("Cardiac removal (CAP, n=8)")
    for i, v in enumerate(removed):
        axs[0].text(i, v + 1.5, f"{v:.0f}\\%", ha="center", fontsize=8)
    axs[1].bar(nx, retained, color=col, width=0.62); axs[1].axhline(100, ls="--", c=GREY, lw=1)
    axs[1].set_xticks(nx); axs[1].set_xticklabels(lab, fontsize=8)
    axs[1].set_ylabel("\\% neural band retained"); axs[1].set_title("Preservation (4--30 Hz)")
    for i, v in enumerate(retained):
        axs[1].text(i, v + 1.5, f"{v:.0f}\\%", ha="center", fontsize=8)
    save(fig, "ecg_artifact_summary.png")


def fig_lowdensity():
    import json as _json
    OR = "#DD8452"
    specs = [
        ("D:/tmp/ld_muscle.json", "Muscle (ds004505, n=6)", "emg_coupling", "alpha_power",
         [("autocca", "autoCCA", RED), ("asr", "ASR", BLUE), ("rasr_windowed", "rASR", GREEN),
          ("ica_fixed", "ICA", OR), ("wavelet_threshold", "wavelet", GREY)]),
        ("D:/tmp/ld_ocular.json", "Ocular (ERP-CORE, n=6)", "blink_coupling", "n170_amp",
         [("eog_regression", "EOG-reg", BLUE), ("eog_dss", "EOG-DSS", RED), ("ssa", "SSA", GREEN),
          ("ica_iclabel_rejection", "ICA", OR), ("wavelet_threshold", "wavelet", GREY)]),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.6))
    for col, (path, title, rem, pres, methods) in enumerate(specs):
        try:
            DD = _json.load(open(path))
        except FileNotFoundError:
            continue
        for row, (metric, ylab) in enumerate([(rem, "removal (ratio vs none)"),
                                              (pres, "preservation (ratio vs none)")]):
            ax = axs[row, col]
            ns_last = []
            for mid, lab, c in methods:
                d = DD.get(mid, {}).get(metric, {})
                if not d:
                    continue
                ns = sorted(int(n) for n in d)
                ax.plot(ns, [d[str(n)]["ratio_vs_none"] for n in ns], "o-", color=c, label=lab, ms=4)
                ns_last = ns
            ax.axhline(1.0, ls="--", c=GREY, lw=0.8)
            if ns_last:
                ax.set_xscale("log", base=2); ax.set_xticks(ns_last); ax.set_xticklabels(ns_last)
            ax.set_xlabel("channels"); ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(title)
            if row == 0:
                ax.legend(fontsize=7, ncol=2)
    save(fig, "lowdensity_summary.png")


if __name__ == "__main__":
    fig_failure(); fig_line_group(); fig_line_qc(); fig_pareto()
    fig_evoked("evoked_erp_core", "evoked_erp_core_summary.png", "N170 (ERP-CORE, n=40)")
    fig_evoked("evoked_ds000117", "evoked_ds000117_summary.png", "M170 (MEG, n=16)")
    fig_ocular(); fig_muscle(); fig_megline(); fig_phantom(); fig_mobile_erp()
    fig_ssvep(); fig_cardiac(); fig_lowdensity()
    try:
        fig_groundtruth()
    except Exception as exc:  # noqa: BLE001
        print(f"  (ground-truth fig skipped: {exc})")
    print("done")
