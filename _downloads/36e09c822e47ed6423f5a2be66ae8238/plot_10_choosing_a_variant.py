r"""
Choosing an ASR variant.
========================

A runnable version of the "Choosing a variant" section of the ASR user guide.
``mne-denoise`` ships several ASR backends; this example runs the three you will
most often choose between on one substrate and prints a short recommendation:

- ``ASR(method="standard")`` -- the right default for most EEG;
- ``ASR(method="riemannian_windowed")`` -- Riemannian-robust calibration with a
  working ``cutoff``;
- ``JugglerASR(strategy="gev")`` -- sample-wise calibration that survives heavy
  contamination where the standard window selector starves.

See ``plot_06`` (cutoff), ``plot_07`` (Riemannian), ``plot_08`` (adaptive) and
``plot_09`` (Juggler) for the per-variant deep dives.
"""

# %%
# Synthetic substrate + helper
# ----------------------------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, JugglerASR
from mne_denoise.qa import variance_removed

rng = np.random.default_rng(3)
sfreq = 250.0
n_channels, n_times = 16, 10000  # 40 s
t = np.arange(n_times) / sfreq

brain = np.zeros((n_channels, n_times))
for ch in range(n_channels):
    phase = rng.uniform(0, 2 * np.pi)
    brain[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.05 * rng.standard_normal(
        n_times
    )
contaminated = brain.copy()
for start in np.linspace(600, n_times - 600, 8).astype(int):
    spatial = rng.standard_normal(n_channels)
    spatial /= np.linalg.norm(spatial)
    contaminated[:, start : start + 200] += 10.0 * np.outer(
        spatial, rng.standard_normal(200)
    )

estimators = {
    "standard": ASR(sfreq=sfreq, cutoff=20.0, picks=None, verbose=False),
    "riemannian_windowed": ASR(
        sfreq=sfreq,
        cutoff=20.0,
        method="riemannian_windowed",
        picks=None,
        verbose=False,
    ),
    "juggler-gev": JugglerASR(
        sfreq=sfreq, cutoff=20.0, strategy="gev", picks=None, verbose=False
    ),
}

# %%
# Run each variant and score it
# -----------------------------
rows = {}
for name, est in estimators.items():
    cleaned = np.asarray(est.fit_transform(contaminated))
    pct = variance_removed(contaminated, cleaned)
    corr = float(np.corrcoef(cleaned.ravel(), brain.ravel())[0, 1])
    rows[name] = (pct, corr)
    print(
        f"  {name:22s} variance removed={rows[name][0]:5.1f}%  corr-to-clean={corr:.3f}"
    )

# %%
# Recommendation
# --------------
print("\nQuick guide:")
print("  - reference-compatible start . ASR(method='standard'), then validate cutoff")
print("  - robust calibration + cutoff  ASR(method='riemannian_windowed')")
print("  - online / streaming ......... AdaptiveASR(variant='psw' or 'psp')")
print("  - extreme MoBI / dense bursts  JugglerASR(strategy='gev' or 'dbscan')")

# %%
# Side-by-side scores
# -------------------
names = list(rows)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(names, [rows[n][0] for n in names], color="C0")
ax1.set_ylabel("% variance removed")
ax1.set_title("Artifact suppression")
ax1.tick_params(axis="x", labelrotation=20)
ax2.bar(names, [rows[n][1] for n in names], color="C2")
ax2.set_ylim(0, 1)
ax2.set_ylabel("correlation to clean reference")
ax2.set_title("Signal fidelity")
ax2.tick_params(axis="x", labelrotation=20)
fig.suptitle("Choosing an ASR variant (same data, three backends)")
fig.tight_layout()

plt.show()
