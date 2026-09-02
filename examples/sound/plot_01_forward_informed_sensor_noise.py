r"""
Forward-informed sensor-noise suppression with SOUND
====================================================

Can SOUND use an anatomical forward model to identify and reduce deliberately
added channel-specific noise while limiting distortion of the underlying
recording? This example uses EEG from the MNE Sample dataset together with the
public Sample forward solution, then plants independent broadband noise in
three sensors.

SOUND uses forward-model consistency to estimate channel-specific noise; a
signal that the forward model explains poorly is not thereby proven to be
noise. Forward-model mismatch is a reason to evaluate preservation of the
signal of interest separately. The unmodified filtered recording is the
reference substrate for this controlled comparison, not noise-free neural
ground truth. Quantitative and visual comparisons use a common average
reference because ``reference="best"`` returns that representation.

The use case is motivated by
:footcite:p:`mutanen2018_sound,mutanen2022_source_artifact`.

References
----------
.. footbibliography::
"""

# %%
# Load EEG and an explicit public forward solution
# -------------------------------------------------
import mne
import numpy as np
from matplotlib.patches import Patch

from mne_denoise.qa import rms_change
from mne_denoise.sound import SOUND
from mne_denoise.viz import (
    COLORS,
    FONTS,
    plot_signal_overlay,
    style_axes,
    themed_figure,
    themed_legend,
)

sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=False,
    verbose="ERROR",
)
forward = mne.read_forward_solution(
    sample_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif",
    verbose="ERROR",
)

raw.pick("eeg", exclude="bads").crop(0.0, 20.0).load_data()
raw.resample(200.0, verbose="ERROR")
raw.filter(1.0, 45.0, verbose="ERROR")

# Keep the filtered, good-channel recording in its original sensor coordinates
# for planting noise. The unmodified recording is the reference substrate, not
# a claim of noise-free neural ground truth.
reference = raw.copy()
reference_data = reference.get_data()
n_channels = reference_data.shape[0]

# %%
# Plant known channel-specific noise
# ----------------------------------
rng = np.random.default_rng(2018)
channel_scale = np.median(np.std(reference_data, axis=1))
corrupted_indices = np.array([5, n_channels // 2, n_channels - 5])
noise_multiplier = 3.0

corrupted_data = reference_data.copy()

for index in corrupted_indices:
    corrupted_data[index] += (
        noise_multiplier * channel_scale * rng.standard_normal(raw.n_times)
    )

corrupted = mne.io.RawArray(
    corrupted_data,
    reference.info.copy(),
    first_samp=reference.first_samp,
    verbose=False,
)
corrupted.set_annotations(reference.annotations.copy())
corrupted_channel_names = [reference.ch_names[index] for index in corrupted_indices]

# %%
# Fit SOUND once, then apply the fitted operator to both recordings
# -----------------------------------------------------------------
sound = SOUND(
    forward=forward,
    reference="best",
    n_iter=5,
    random_state=0,
    verbose=False,
)
cleaned_corrupted = sound.fit_transform(corrupted)
cleaned_reference = sound.transform(reference)

# SOUND with reference="best" returns an average-referenced representation.
# Put both the clean substrate and corrupted input in that same coordinate
# system before computing recovery and preservation metrics.
reference_avg = reference.copy().set_eeg_reference(
    "average", projection=False, verbose="ERROR"
)
corrupted_avg = corrupted.copy().set_eeg_reference(
    "average", projection=False, verbose="ERROR"
)
reference_avg_data = reference_avg.get_data()
corrupted_avg_data = corrupted_avg.get_data()
cleaned_corrupted_data = cleaned_corrupted.get_data()
cleaned_reference_data = cleaned_reference.get_data()

artifact_before = rms_change(
    corrupted_avg_data,
    reference_avg_data,
)
artifact_after = rms_change(
    cleaned_corrupted_data,
    cleaned_reference_data,
)
artifact_residual_ratio = artifact_after / artifact_before

reference_scale = np.sqrt(np.mean(reference_avg_data**2))
preservation_change = rms_change(
    cleaned_reference_data,
    reference_avg_data,
)
preservation_relative_change = preservation_change / reference_scale

sigma_channel_indices = np.flatnonzero(np.arange(n_channels) != sound.best_channel_)
sigma_channel_names = [reference.ch_names[index] for index in sigma_channel_indices]
rank_order = np.argsort(sound.sigmas_)[::-1]
top_indices = rank_order[:5]
print(f"Planted corrupted channels: {corrupted_channel_names}")
print(f"Selected best-reference channel: {reference.ch_names[sound.best_channel_]}")
print(f"Artifact residual ratio:          {artifact_residual_ratio:.3f}")
print(f"Clean-substrate relative change:  {preservation_relative_change:.3f}")
print("Top channels by estimated sigma:")
for position, index in enumerate(top_indices, start=1):
    print(f"  {position}. {sigma_channel_names[index]}: {sound.sigmas_[index]:.3e}")
for index in corrupted_indices:
    if index == sound.best_channel_:
        print(
            f"Rank of planted channel {reference.ch_names[index]}: selected reference"
        )
    else:
        sigma_position = int(np.flatnonzero(sigma_channel_indices == index)[0])
        rank = int(np.flatnonzero(rank_order == sigma_position)[0] + 1)
        print(f"Rank of planted channel {reference.ch_names[index]}: {rank}")
print(f"Final convergence value: {sound.convergence_[-1]:.3e}")

# %%
# Plot the fitted channel-noise diagnostic
# ------------------------------------------
sigma_colors = [COLORS["primary"] for _ in range(len(sigma_channel_indices))]
for index in corrupted_indices:
    if index != sound.best_channel_:
        sigma_position = int(np.flatnonzero(sigma_channel_indices == index)[0])
        sigma_colors[sigma_position] = COLORS["accent"]

fig, ax = themed_figure(figsize=(10.0, 3.8))
sigma_positions = np.arange(len(sigma_channel_indices))
ax.bar(sigma_positions, sound.sigmas_, color=sigma_colors)
tick_step = max(1, len(sigma_channel_indices) // 12)
tick_positions = np.arange(0, len(sigma_channel_indices), tick_step)
ax.set_xticks(tick_positions)
ax.set_xticklabels(
    [sigma_channel_names[index] for index in tick_positions], rotation=90
)
ax.set_xlabel("EEG channel")
ax.set_ylabel("Estimated noise amplitude")
ax.set_title(
    "SOUND channel-noise estimates "
    f"(best reference: {reference.ch_names[sound.best_channel_]})"
)
for index in corrupted_indices:
    if index != sound.best_channel_:
        sigma_position = int(np.flatnonzero(sigma_channel_indices == index)[0])
        ax.annotate(
            "planted",
            xy=(sigma_position, sound.sigmas_[sigma_position]),
            xytext=(
                sigma_position,
                float(np.max(sound.sigmas_)) * 1.08,
            ),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=FONTS["annotation"],
            color=COLORS["accent"],
            arrowprops={"arrowstyle": "-", "color": COLORS["accent"]},
        )
themed_legend(
    ax,
    handles=[
        Patch(color=COLORS["primary"], label="other channel"),
        Patch(color=COLORS["accent"], label="deliberately corrupted"),
    ],
    loc="upper right",
)
style_axes(ax, grid=True)
fig.tight_layout()

# %%
# Inspect one reconstructed sensor
# ---------------------------------
plot_signal_overlay(
    corrupted_avg,
    cleaned_corrupted,
    reference.times,
    pick=corrupted_channel_names[0],
    start=2.0,
    stop=4.0,
    scale_after=False,
    before_label="corrupted (average reference)",
    after_label="SOUND",
    reference=reference_avg_data[corrupted_indices[0]],
    reference_label="unmodified (average reference)",
    x_label="Time (s)",
    y_label="Amplitude (V)",
    title=f"Forward-informed reconstruction at {corrupted_channel_names[0]}",
    show=False,
)
