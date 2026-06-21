r"""
JugglerASR for dense short bursts
=================================

This example highlights Juggler-style calibration on a synthetic signal with
frequent short bursts that leave little fully clean time window support.
"""

import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, JugglerASR

rng = np.random.default_rng(21)
sfreq = 250.0
duration = 14.0
n_times = int(sfreq * duration)
n_channels = 8
times = np.arange(n_times) / sfreq

brain = np.zeros((n_channels, n_times), dtype=np.float64)
for ch_idx in range(n_channels):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    brain[ch_idx] = (
        0.45 * np.sin(2.0 * np.pi * 10.0 * times + phase)
        + 0.18 * np.sin(2.0 * np.pi * 6.0 * times + 0.4 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

data = brain.copy()
burst_mask = np.zeros(n_times, dtype=bool)
spatial = rng.standard_normal((n_channels, 2))
spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
for onset in np.arange(3.0, 11.5, 0.28):
    start = int(round(onset * sfreq))
    stop = min(n_times, start + int(round(0.10 * sfreq)))
    burst_mask[start:stop] = True
    data[:, start:stop] += spatial @ (7.0 * rng.standard_normal((2, stop - start)))

standard = ASR(
    sfreq=sfreq,
    cutoff=5.0,
    calibration="auto",
    filter_kind="asr",
    max_dims=0.5,
    verbose=False,
)
juggler = JugglerASR(
    sfreq=sfreq,
    cutoff=5.0,
    strategy="dbscan",
    max_dims=0.5,
    verbose=False,
)

clean_standard = standard.fit_transform(data)
clean_juggler = juggler.fit_transform(data)
standard_mask = np.zeros(n_times, dtype=bool)
standard_mask[standard.calibration_info_["clean_sample_mask"]] = True
juggler_mask = juggler.get_calibration_mask()

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")

axes[0].plot(times, data[0], color="0.7", lw=1.0, label="Noisy")
axes[0].plot(times, brain[0], color="k", lw=1.0, alpha=0.8, label="Underlying")
axes[0].set_title("Dense short bursts challenge window-based calibration")
axes[0].legend(loc="upper right")

axes[1].plot(times, clean_standard[0], color="tab:orange", lw=1.0, label="ASR")
axes[1].plot(times, clean_juggler[0], color="tab:blue", lw=1.0, label="JugglerASR")
axes[1].plot(times, brain[0], color="k", lw=1.0, alpha=0.6, label="Underlying")
axes[1].set_title("Burst repair after standard vs Juggler calibration")
axes[1].legend(loc="upper right")

axes[2].fill_between(
    times,
    0.0,
    1.0,
    where=standard_mask,
    color="tab:orange",
    alpha=0.45,
    label="ASR reference",
)
axes[2].fill_between(
    times,
    0.0,
    1.0,
    where=juggler_mask,
    color="tab:blue",
    alpha=0.45,
    label="Juggler reference",
)
axes[2].fill_between(
    times,
    0.0,
    1.0,
    where=burst_mask,
    color="tab:red",
    alpha=0.18,
    label="Burst artifact",
)
axes[2].set(
    xlabel="Time (s)",
    ylabel="Mask",
    yticks=[],
    title="Reference samples used for calibration",
)
axes[2].legend(loc="upper right")

print(f"ASR reference fraction: {standard_mask.mean():.2%}")
print(f"Juggler reference fraction: {juggler_mask.mean():.2%}")

plt.show()
