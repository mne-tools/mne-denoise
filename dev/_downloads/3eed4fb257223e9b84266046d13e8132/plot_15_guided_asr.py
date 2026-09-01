r"""
Guided ASR: preserving neural activity that ASR would over-clean.
=================================================================

Standard ASR decides purely on variance, so a strong, spatially structured
**neural** burst (here a transient 10 Hz oscillation) looks just as "abnormal"
as an artifact and gets reconstructed away. ``GuidedASR`` keeps ASR's
abnormality detection but adds DSS-style **bias operators** that recognise the
neural direction and a continuous reconstruction weight that rescues it, while
still removing a genuine artifact.

This example builds a substrate with a known 10 Hz neural target and a known
broadband artifact, cleans it with standard ASR and with GuidedASR, and shows
that GuidedASR preserves the neural projection that standard ASR removes.

.. warning::

   GuidedASR is an unpublished, unvalidated experimental research prototype.
   This synthetic demonstration is not evidence of validity on real EEG. Check
   neural-signal preservation, artifact attenuation, and downstream endpoints
   independently before scientific use.
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, GuidedASR
from mne_denoise.dss.denoisers import BandpassBias, PeakFilterBias
from mne_denoise.viz import plot_guided_asr_weights

# %%
# Synthetic substrate
# -------------------
# A quiet 1/f-ish background (the calibration baseline), plus a transient 10 Hz
# neural target (fixed spatial pattern, second half) we want to PRESERVE, plus
# a high-amplitude broadband artifact (different pattern) we want to REMOVE.
sfreq = 250.0
rng = np.random.default_rng(7)
n_ch, n = 16, 15000
t = np.arange(n) / sfreq

background = 0.15 * rng.standard_normal((n_ch, n))
for f in (3.0, 6.0, 11.0, 23.0):
    for c in range(n_ch):
        background[c] += (1.0 / f) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))

p_neural = rng.standard_normal(n_ch)
p_neural /= np.linalg.norm(p_neural)
neural_ts = np.zeros(n)
neural_ts[n // 2 :] = 4.0 * np.sin(2 * np.pi * 10.0 * t[n // 2 :])
clean = background + np.outer(p_neural, neural_ts)  # ground truth

p_art = rng.standard_normal(n_ch)
p_art /= np.linalg.norm(p_art)
artifact = np.zeros((n_ch, n))
for start in (int(0.15 * n), int(0.30 * n)):
    seg = slice(start, start + int(0.4 * sfreq))
    artifact[:, seg] += 7.0 * np.outer(p_art, rng.standard_normal(seg.stop - seg.start))

contaminated = clean + artifact
calib = background[:, : int(8.0 * sfreq)]  # quiet baseline (no events, no artifact)

# %%
# Clean with standard ASR and with Guided ASR
# -------------------------------------------
# Both calibrate on the quiet baseline. GuidedASR is told the brain subspace
# (a 10 Hz peak filter) and the artifact subspace (a broadband band).
common = {
    "sfreq": sfreq,
    "cutoff": 8.0,
    "calibration": "manual",
    "picks": None,
    "verbose": False,
}

asr = ASR(method="riemannian_windowed", **common).fit(calib)
cleaned_asr = np.asarray(asr.transform(contaminated))

guided = GuidedASR(
    reconstruction="soft",
    experimental=True,
    guidance_strength=1.0,
    preserve_biases=[PeakFilterBias(10.0, sfreq)],
    artifact_biases=[BandpassBias((30.0, 80.0), sfreq)],
    **common,
).fit(contaminated, calibration=calib)
cleaned_guided = np.asarray(guided.transform(contaminated))

# %%
# Soft weights
# ------------
# Green = kept (w -> 1), red = suppressed (w -> 0). Standard ASR would show
# only 0/1; the graded values are where Guided ASR preserves structure.
plot_guided_asr_weights(guided, show=False)

# %%
# Neural projection: preserved vs over-cleaned
# --------------------------------------------
# Project each signal onto the neural spatial pattern over the event region.
ev = slice(n // 2, n)
proj_clean = (p_neural @ clean)[ev]
proj_asr = (p_neural @ cleaned_asr)[ev]
proj_guided = (p_neural @ cleaned_guided)[ev]
tt = t[ev]

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(tt, proj_clean, color="0.6", lw=1.4, label="ground truth (neural)")
ax.plot(tt, proj_asr, color="C3", lw=0.9, alpha=0.8, label="standard ASR (removed)")
ax.plot(tt, proj_guided, color="C0", lw=1.0, label="GuidedASR (preserved)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Neural-pattern projection")
ax.set_title("GuidedASR preserves the 10 Hz neural target ASR over-cleans")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


print(f"neural preservation  standard ASR: {_corr(proj_asr, proj_clean):+.3f}")
print(f"neural preservation  GuidedASR   : {_corr(proj_guided, proj_clean):+.3f}")

plt.show()
