r"""
ASR diagnostics and quality control.
====================================

Every fitted ASR estimator records what it did, so you can audit the cleaning
instead of trusting it blindly. This example tours the QC surface:

- ``get_diagnostics()`` -- the per-window reconstruction record;
- ``variance_removed()`` -- scalar before/after summaries;
- ``to_annotations(kind=...)`` -- repaired / rejected / calibration spans as
  ``mne.Annotations`` (mark, do not delete);
- the three ASR-specific diagnostic plots.
"""

# %%
# Fit ASR on a synthetic Raw
# --------------------------
import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_denoise.asr import ASR
from mne_denoise.qa import variance_removed
from mne_denoise.viz import (
    plot_asr_calibration_fraction,
    plot_asr_component_reconstruction,
    plot_asr_repair_timeline,
)

rng = np.random.default_rng(8)
sfreq = 250.0
n_channels, n_times = 12, 12000
t = np.arange(n_times) / sfreq
brain = np.vstack(
    [
        0.6 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))
        + 0.05 * rng.standard_normal(n_times)
        for _ in range(n_channels)
    ]
)
contaminated = brain.copy()
for start in np.linspace(500, n_times - 500, 9).astype(int):
    spatial = rng.standard_normal(n_channels)
    spatial /= np.linalg.norm(spatial)
    contaminated[:, start : start + 220] += 11.0 * np.outer(
        spatial, rng.standard_normal(220)
    )
info = mne.create_info([f"EEG{i:02d}" for i in range(n_channels)], sfreq, "eeg")
raw = mne.io.RawArray(contaminated, info, verbose="ERROR")

asr = ASR(cutoff=10.0, picks="eeg", window_criterion=0.3, verbose=False)
raw_clean = asr.fit_transform(raw)

# %%
# Diagnostics record + QA metrics
# -------------------------------
diag = asr.get_diagnostics()
print("get_diagnostics() keys:", sorted(diag))
print(
    f"  windows={diag['n_windows']}  "
    f"reconstructed windows={diag['fraction_reconstructed_windows']:.1%}  "
    f"samples modified={diag['fraction_reconstructed_samples']:.1%}"
)

from mne_denoise.qa import channel_variance_ratio, rms_change

print(
    f"QA: variance removed={variance_removed(raw.get_data(picks='eeg'), raw_clean.get_data(picks='eeg')):.1f}%  "
    f"rms change={rms_change(raw.get_data(picks='eeg'), raw_clean.get_data(picks='eeg')):.3g}  "
    f"median channel variance ratio={np.median(channel_variance_ratio(raw.get_data(picks='eeg'), raw_clean.get_data(picks='eeg'))):.2f}"
)

# %%
# Annotation export (mark, don't delete)
# --------------------------------------
for kind in ("repair", "rejection", "calibration"):
    try:
        ann = asr.to_annotations(kind=kind)
        total = float(np.sum(ann.duration))
        print(f"  to_annotations(kind={kind!r}): {len(ann)} spans, {total:.1f} s")
    except (ValueError, RuntimeError) as exc:
        print(f"  to_annotations(kind={kind!r}): not available ({exc})")

# %%
# The three ASR-specific diagnostic plots
# ---------------------------------------
plot_asr_repair_timeline(asr, show=False)
plot_asr_component_reconstruction(asr, show=False)
plot_asr_calibration_fraction(asr, show=False)

plt.show()
