r"""
A realistic pipeline: filter, ASR, then ICA.
============================================

ASR sits between high-pass filtering and ICA in a standard EEG pipeline: it
repairs high-amplitude transients *before* ICA, so the decomposition is not
dominated by a handful of huge bursts. This capstone runs the full
``filter -> ASR -> ICA`` workflow on real EEG (the MNE *sample* dataset) and
shows that ICA fit on ASR-cleaned data has less extreme component activations.

References
----------
* Mullen, T. R., et al. (2013). Real-time modeling and 3D visualization of
  source dynamics... EMBC 2013. doi:10.1109/EMBC.2013.6609968
* Chang, C.-Y., et al. (2020). Evaluation of Artifact Subspace
  Reconstruction... IEEE TBME, 67(4). doi:10.1109/TBME.2019.2930186
"""

# %%
# Load + filter real EEG
# ----------------------
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import kurtosis

from mne_denoise.asr import ASR
from mne_denoise.viz import plot_signal_overlay

sample = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample / "MEG" / "sample" / "sample_audvis_raw.fif", preload=True, verbose="ERROR"
)
raw.pick("eeg").crop(0, 60).resample(150, verbose="ERROR")
raw.set_eeg_reference("average", verbose="ERROR")
raw.filter(1.0, None, verbose="ERROR")  # 1) high-pass

# %%
# Apply ASR
# ---------
asr = ASR(cutoff=20.0, picks="eeg", verbose=False)
raw_asr = asr.fit_transform(raw.copy())  # 2) ASR burst repair

noisiest = int(np.argmax(np.var(raw.get_data(), axis=1)))
plot_signal_overlay(
    raw,
    raw_asr,
    raw.times,
    pick=raw.ch_names[noisiest],
    scale_after=False,
    before_label="filtered",
    after_label="filtered + ASR",
    x_label="Time (s)",
    y_label="Amplitude (V)",
    title=f"ASR burst repair on {raw.ch_names[noisiest]}",
    show=False,
)

# %%
# Fit ICA on raw vs ASR-cleaned
# -----------------------------
# Same settings on both; compare how extreme the recovered sources are.
ica_kw = {"n_components": 15, "max_iter": 300, "random_state": 97}
ica_raw = mne.preprocessing.ICA(**ica_kw).fit(raw, verbose="ERROR")  # 3) ICA
ica_asr = mne.preprocessing.ICA(**ica_kw).fit(raw_asr, verbose="ERROR")

src_raw = ica_raw.get_sources(raw).get_data()
src_asr = ica_asr.get_sources(raw_asr).get_data()
max_raw = float(np.max(np.abs(src_raw)))
max_asr = float(np.max(np.abs(src_asr)))
kurt_raw = float(np.mean(kurtosis(src_raw, axis=1)))
kurt_asr = float(np.mean(kurtosis(src_asr, axis=1)))
print(f"max |IC activation|:  raw={max_raw:.1f}  ASR={max_asr:.1f}")
print(f"mean IC excess kurtosis:  raw={kurt_raw:.1f}  ASR={kurt_asr:.1f}")

# %%
# ICA component "peakiness" before vs after ASR
# ---------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.bar(["raw", "ASR-cleaned"], [max_raw, max_asr], color=["0.6", "C0"])
ax1.set_ylabel("max |IC activation|")
ax1.set_title("Largest transient captured by ICA")
ax2.bar(["raw", "ASR-cleaned"], [kurt_raw, kurt_asr], color=["0.6", "C2"])
ax2.set_ylabel("mean IC excess kurtosis")
ax2.set_title("How burst-dominated the ICs are")
fig.suptitle("ICA after ASR is less dominated by transients")
fig.tight_layout()

plt.show()
