"""
SOUND: Suppressing Channel-Specific Noise with a Forward Model
================================================================

SOUND asks a question no purely statistical cleaner can: *is what this sensor
reports physically consistent with what its neighbours report, given the shape
of the head?* A channel whose signal cannot be explained by any cortical source
configuration is carrying noise, and SOUND estimates how much -- per channel,
iteratively -- then applies a Wiener filter that trusts each sensor in
proportion to how clean it is.

This example builds a synthetic recording in which three sensors are far
noisier than the rest, and shows that SOUND finds them without being told.

Authors: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_denoise._leadfield import make_spherical_leadfield
from mne_denoise.sound import SOUND

mne.set_log_level("ERROR")
rng = np.random.default_rng(0)

# %%
# Build a montage
# ---------------
# SOUND needs sensor positions: the lead field is what lets it distinguish
# "signal a brain could have produced" from "noise this sensor invented". With
# no ``forward`` supplied, a three-layer spherical model is built from the
# montage automatically.

montage = mne.channels.make_standard_montage("standard_1020")
ch_names = [
    ch
    for ch in montage.ch_names
    if ch not in ("T3", "T4", "T5", "T6", "A1", "A2")  # keep a clean 10-20 subset
][:32]
sfreq = 250.0
info = mne.create_info(ch_names, sfreq, "eeg")
info.set_montage(montage)
n_channels = len(ch_names)

# %%
# Simulate brain signal plus channel-specific noise
# -------------------------------------------------
# The brain part must be something a brain could actually produce, so we drive
# a handful of dipoles through the *same* spherical lead field SOUND will use.
# This matters: SOUND's whole criterion is field consistency, and signal
# generated with an arbitrary random mixing matrix is -- correctly -- treated
# as noise, because no source configuration explains it.
#
# The sensor noise is independent per channel, and three sensors get a lot
# more of it.

leadfield = make_spherical_leadfield(info, n_dipoles=5000)
n_times = int(30 * sfreq)
times = np.arange(n_times) / sfreq

n_sources = 4
source_ts = np.stack(
    [
        np.sin(2 * np.pi * f * times) * (1 + 0.3 * rng.standard_normal(n_times))
        for f in (6.0, 10.0, 14.0, 22.0)
    ]
)
active = rng.choice(leadfield.shape[1], size=n_sources, replace=False)
brain = leadfield[:, active] @ source_ts
brain /= np.abs(brain).max()

noise = 0.05 * rng.standard_normal((n_channels, n_times))
bad_channels = [3, 11, 24]
noise[bad_channels] *= 12.0  # three sensors go bad

data = brain + noise
data -= data.mean(axis=0, keepdims=True)  # average reference
raw = mne.io.RawArray(data * 1e-6, info)

print(f"Noisy sensors planted at: {[ch_names[i] for i in bad_channels]}")

# %%
# Fit SOUND
# ---------
# ``reference='best'`` (the default) follows ``tesa_sound``: it re-references
# to the least noisy channel while estimating, then hands back an
# average-referenced montage. That detour exists because SOUND's whitener
# assumes independent per-channel noise, and an average reference correlates
# every channel with every other.

sound = SOUND(n_iter=5, random_state=0, verbose=False)
cleaned = sound.fit_transform(raw)

# %%
# Did it find the bad channels?
# -----------------------------
# ``sigmas_`` is the estimated noise amplitude per channel. Nothing told SOUND
# which sensors were contaminated -- the estimate comes from how poorly each
# channel is predicted by the others through the forward model.
#
# Note that with ``reference='best'`` one channel is dropped during
# estimation, so ``sigmas_`` is one shorter than the montage.

sigmas = sound.sigmas_
kept = [i for i in range(n_channels) if i != sound.best_channel_]
ranked = np.argsort(sigmas)[::-1][:5]
print("\nNoisiest channels by estimated sigma:")
for rank, idx in enumerate(ranked, start=1):
    ch = ch_names[kept[idx]]
    flag = "  <-- planted" if kept[idx] in bad_channels else ""
    print(f"  {rank}. {ch:5s}  sigma = {sigmas[idx]:.3e}{flag}")

fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
colors = [
    "tab:red" if kept[i] in bad_channels else "tab:blue" for i in range(len(kept))
]
ax.bar(range(len(kept)), sigmas, color=colors)
ax.set_xticks(range(len(kept)))
ax.set_xticklabels([ch_names[i] for i in kept], rotation=90, fontsize=7)
ax.set_ylabel("estimated noise amplitude")
ax.set_title("SOUND noise estimates (red = sensors we deliberately corrupted)")
plt.show()

# %%
# Before and after
# ----------------
# SOUND does not delete the bad channels; it reconstructs them from what the
# rest of the montage and the head model imply they should have recorded.

clean_data = cleaned.get_data() * 1e6
noisy_data = raw.get_data() * 1e6
window = slice(0, int(3 * sfreq))

fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, layout="constrained")
for ax, idx in zip(axes, bad_channels, strict=True):
    ax.plot(times[window], noisy_data[idx, window], lw=0.8, alpha=0.7, label="before")
    ax.plot(times[window], clean_data[idx, window], lw=1.1, label="after")
    ax.set_ylabel(f"{ch_names[idx]}\n(µV)")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time (s)")
axes[0].set_title("Corrupted sensors, before and after SOUND")
plt.show()

# %%
# How much did the clean channels change?
# ---------------------------------------
# A denoiser that improves bad channels by mangling good ones is not much use.
# Correlating each channel against the noise-free ground truth shows where the
# gain actually came from.


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


good_channels = [i for i in range(n_channels) if i not in bad_channels]
truth = brain  # noise-free ground truth, same scale as the plotted data

before_bad = np.mean([_corr(noisy_data[i], truth[i]) for i in bad_channels])
after_bad = np.mean([_corr(clean_data[i], truth[i]) for i in bad_channels])
before_good = np.mean([_corr(noisy_data[i], truth[i]) for i in good_channels])
after_good = np.mean([_corr(clean_data[i], truth[i]) for i in good_channels])

print("\nCorrelation with the noise-free signal:")
print(f"  corrupted channels: {before_bad:.3f} -> {after_bad:.3f}")
print(f"  intact channels:    {before_good:.3f} -> {after_good:.3f}")

# %%
# Choosing ``lambda_``
# --------------------
# ``lambda_`` sets how aggressively SOUND suppresses. Mutanen et al. (2022)
# suggest starting near ``1 / SNR``.
#
# In this simulation more suppression is monotonically better, because the
# ground truth was generated *through the lead field* and so is perfectly
# forward-consistent: there is nothing for an aggressive filter to damage.
# Real recordings are not that kind. Genuine brain activity that the spherical
# model describes poorly -- deep sources, or anywhere the real head departs
# from a sphere -- gets suppressed along with the noise, and the sweep turns
# into a trade-off with an interior optimum. Sweep it on your own data rather
# than reading a best value off this example.

for lam in (0.01, 0.1, 0.5):
    est = SOUND(lambda_=lam, n_iter=5, random_state=0, verbose=False)
    out = est.fit_transform(raw).get_data() * 1e6
    bad_c = np.mean([_corr(out[i], truth[i]) for i in bad_channels])
    good_c = np.mean([_corr(out[i], truth[i]) for i in good_channels])
    print(f"  lambda_={lam:<5} corrupted {bad_c:.3f}   intact {good_c:.3f}")

# %%
# References
# ----------
# Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J. (2018).
# Automatic and robust noise suppression in EEG and MEG: The SOUND algorithm.
# *NeuroImage*, 166, 135-151.
