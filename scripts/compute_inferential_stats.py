#!/usr/bin/env python
"""Phase-1 inferential statistics for the mne-denoise benchmark arms.

Reads the per-subject metric values (D:/tmp/all_arms_persubject.json, the nested
{arm: {method: {metric: {subject: value}}}} extracted from the Fir metrics.tsv) and
computes, per arm, the inferential endpoints the manuscript's section 4.5 promises but
only displayed as point estimates: one-sample / paired Hedges g, BCa bootstrap 95% CIs,
Wilcoxon signed-rank p-values, and equivalence (TOST 90% CI vs the preservation margin).

No denoisers are re-run; this is pure post-hoc statistics on the saved per-subject metrics.
Emits D:/tmp/inferential_stats.json + a readable report.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

RNG = np.random.default_rng(42)
SRC = json.load(open("D:/tmp/all_arms_persubject.json"))
OUT = {}


def hedges_g(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * (1 - 3 / (4 * n - 9)))


def vals(arm, method, metric):
    d = SRC.get(arm, {}).get(method, {}).get(metric, {})
    return d  # {subject: value}


def aligned(*dicts):
    keys = sorted(set.intersection(*[set(d) for d in dicts])) if dicts else []
    return [np.array([d[k] for k in keys], float) for d in dicts]


def median_ci(x, B=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x[0]) if len(x) else 0.0, None, None)
    boot = np.median(x[RNG.integers(0, len(x), (B, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bca(boot, obs, jack):
    boot = np.asarray(boot)
    z0 = stats.norm.ppf(np.clip((boot < obs).mean(), 1e-6, 1 - 1e-6))
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    def alpha(p):
        z = stats.norm.ppf(p)
        return float(stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    return (float(np.percentile(boot, 100 * alpha(0.025))),
            float(np.percentile(boot, 100 * alpha(0.975))))


def delta_g(none_arr, dss_arr, B=10000):
    """Paired Δg = g(dss) - g(none), BCa 95% CI by resampling subjects."""
    n = len(none_arr)
    th = lambda idx: hedges_g(dss_arr[idx]) - hedges_g(none_arr[idx])
    obs = th(np.arange(n))
    boot = np.array([th(RNG.integers(0, n, n)) for _ in range(B)])
    jack = np.array([th(np.delete(np.arange(n), i)) for i in range(n)])
    lo, hi = _bca(boot, obs, jack)
    return obs, lo, hi


def wilcoxon(a, b):
    a, b = aligned(a, b)
    if len(a) < 3 or np.allclose(a, b):
        return None
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return None


def equivalence(none_d, dss_d, margin=0.30):
    """Paired dz of (dss - none) on a preservation metric + 90% CI; equivalent if CI subset of +/-margin."""
    a, b = aligned(none_d, dss_d)
    if len(a) < 3:
        return None
    diff = b - a
    sd = diff.std(ddof=1)
    if sd == 0:
        return {"dz": 0.0, "ci90": [0.0, 0.0], "equivalent": True, "margin": margin}
    dz = diff.mean() / sd
    n = len(diff)
    boot = np.array([(lambda i: (b[i] - a[i]).mean() / (b[i] - a[i]).std(ddof=1)
                      if (b[i] - a[i]).std(ddof=1) else 0.0)(RNG.integers(0, n, n)) for _ in range(10000)])
    lo, hi = np.percentile(boot, [5, 95])
    return {"dz": round(float(dz), 3), "ci90": [round(float(lo), 3), round(float(hi), 3)],
            "equivalent": bool(lo > -margin and hi < margin), "margin": margin}


def report(title, lines):
    print(f"\n### {title}")
    for ln in lines:
        print("  " + ln)


# ---------- EVOKED arms (Claim 3): N170, M170, P300 ----------------------------
for arm, label, mneg in [("evoked_erp_core", "N170 (ERP-CORE)", "rank_matched_pca"),
                         ("evoked_ds000117", "M170 (ds000117)", "rank_matched_pca"),
                         ("p300_brain_invaders", "P300 (BrainInvaders)", "rank_matched_pca")]:
    none_d = vals(arm, "none", "n170_diff")
    dss_d = vals(arm, "dss_average_bias", "n170_diff")
    pca_d = vals(arm, mneg, "n170_diff")
    if not none_d:
        continue
    gn, gd, gp = (abs(hedges_g(list(d.values()))) for d in (none_d, dss_d, pca_d))
    na, da = aligned(none_d, dss_d)
    dg, lo, hi = delta_g(na, da)
    w = wilcoxon(none_d, dss_d)
    shn = median_ci(list(vals(arm, "none", "split_half").values()))
    shd = median_ci(list(vals(arm, "dss_average_bias", "split_half").values()))
    eq = equivalence(vals(arm, "none", "n170_amp_A"), vals(arm, "dss_average_bias", "n170_amp_A"))
    OUT[arm] = {"n": len(na), "g_none": round(gn, 2), "g_dss": round(gd, 2), "g_pca": round(gp, 2),
                "delta_g": round(abs(dg), 2), "delta_g_ci": [round(abs(hi), 2), round(abs(lo), 2)],
                "wilcoxon_p": w, "split_half_none": round(shn[0], 2), "split_half_dss": round(shd[0], 2),
                "equiv": eq}
    report(f"{label}  n={len(na)}", [
        f"g: none={gn:.2f}, DSS={gd:.2f}, PCA={gp:.2f}",
        f"paired Delta g (DSS-none) = {dg:+.2f}, 95% BCa CI [{lo:+.2f}, {hi:+.2f}], Wilcoxon p={w}",
        f"split-half reliability: none={shn[0]:.2f} [{shn[1]:.2f},{shn[2]:.2f}], DSS={shd[0]:.2f} [{shd[1]:.2f},{shd[2]:.2f}]",
        f"N170/amp equivalence (DSS vs none): {eq}"])

# ---------- LINE arms: env (per environment), meg_masc, ds000117 --------------
envk = {"lab": "lab", "oval": "field", "campus": "campus"}
for de, ke in envk.items():
    rf = {m.split("__")[1]: vals("line_ds003620_env", m, "R_f0")
          for m in SRC.get("line_ds003620_env", {}) if m.startswith(de + "__")}
    if not rf:
        continue
    none_d, zlp_d = rf.get("none", {}), rf.get("zapline_plus", {})
    w = wilcoxon(none_d, zlp_d)
    mc = {k: median_ci(list(v.values())) for k, v in rf.items()}
    OUT.setdefault("line_ds003620_env", {})[ke] = {
        "wilcoxon_none_vs_zlp_p": w, "n": len(none_d),
        **{k: {"med": round(m[0], 2), "ci": [round(m[1], 2), round(m[2], 2)]} for k, m in mc.items()}}
    report(f"line env={ke}  n={len(none_d)}", [
        f"R(f0) medians: " + ", ".join(f"{k}={m[0]:.2f}[{m[1]:.2f},{m[2]:.2f}]" for k, m in mc.items()),
        f"Wilcoxon none vs ZapLine+ p={w}"])

for arm, lab in [("line_meg_masc", "KIT MEG"), ("line_ds000117", "Elekta MEG")]:
    pa = {m: median_ci(list(vals(arm, m, "peak_attenuation_db").values()))
          for m in ("none", "notch", "non_spatial_line", "zapline_plus") if vals(arm, m, "peak_attenuation_db")}
    if not pa:
        continue
    OUT[arm] = {m: {"db": round(v[0], 1), "ci": [round(v[1], 1), round(v[2], 1)]} for m, v in pa.items()}
    report(f"{lab} line ({arm})", [", ".join(f"{m}={v[0]:.1f}dB[{v[1]:.1f},{v[2]:.1f}]" for m, v in pa.items())])

# ---------- OCULAR ------------------------------------------------------------
oc = {m: median_ci(list(vals("ocular_erp_core", m, "blink_coupling").values()))
      for m in SRC.get("ocular_erp_core", {}) if vals("ocular_erp_core", m, "blink_coupling")}
if oc:
    w = wilcoxon(vals("ocular_erp_core", "none", "blink_coupling"), vals("ocular_erp_core", "eog_dss", "blink_coupling"))
    eq = equivalence(vals("ocular_erp_core", "none", "n170_amp"), vals("ocular_erp_core", "eog_dss", "n170_amp"))
    OUT["ocular_erp_core"] = {"wilcoxon_none_vs_eogdss_p": w, "equiv_n170": eq,
                              **{m: {"med": round(v[0], 3), "ci": [round(v[1], 3), round(v[2], 3)]} for m, v in oc.items()}}
    report("OCULAR (blink coupling, lower=better)", [
        ", ".join(f"{m}={v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}]" for m, v in oc.items()),
        f"Wilcoxon none vs EOG-DSS p={w}; N170 equivalence {eq}"])

# ---------- MUSCLE -----------------------------------------------------------
mu = {m: median_ci(list(vals("muscle_ds004505", m, "emg_coupling").values()))
      for m in SRC.get("muscle_ds004505", {}) if vals("muscle_ds004505", m, "emg_coupling")}
if mu:
    OUT["muscle_ds004505"] = {m: {"med": round(v[0], 3), "ci": [round(v[1], 3), round(v[2], 3)]} for m, v in mu.items()}
    report("MUSCLE (neck-EMG coupling, lower=better)",
           [", ".join(f"{m}={v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}]" for m, v in mu.items())])

json.dump(OUT, open("D:/tmp/inferential_stats.json", "w"), indent=1, default=str)
print("\n[saved] D:/tmp/inferential_stats.json")
