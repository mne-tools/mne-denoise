"""
SNS assumptions and diagnostics
===============================

Deterministic simulations illustrate when SNS works: spatially redundant
signals can be regenerated, independent sensor noise is attenuated, robust
fitting limits the influence of brief glitches, and successive passes make
progressively smaller changes.

The simulation uses 157 channels to exercise a dense sensor array [1]_. It is an
algorithm demonstration, not a reproduction of a particular experimental
dataset or figure.

References
----------
.. [1] de Cheveigné, A., & Simon, J. Z. (2008). Sensor noise suppression.
   Journal of Neuroscience Methods, 168(1), 195–202.
   https://doi.org/10.1016/j.jneumeth.2007.09.012
"""

import numpy as np

from mne_denoise.sns import compute_sns
from mne_denoise.viz import (
    plot_signal_overlay,
    style_axes,
    themed_figure,
    themed_legend,
)

rng = np.random.default_rng(21)
n_channels, n_sources, n_times = 157, 10, 2400
sources = rng.standard_normal((n_sources, n_times))
mixing = rng.standard_normal((n_channels, n_sources))
shared = mixing @ sources
sensor_noise = 0.35 * rng.standard_normal(shared.shape)
observed = shared + sensor_noise

# A redundant clean signal can be regenerated nearly exactly when enough
# neighbours span its source subspace.
clean_regenerated, _ = compute_sns(shared, n_neighbors=24)
shared_centered = shared - shared.mean(axis=1, keepdims=True)
preservation_error = np.linalg.norm(clean_regenerated - shared_centered)
preservation_error /= np.linalg.norm(shared_centered)

cleaned, _ = compute_sns(observed, n_neighbors=24)
before_error = np.linalg.norm(observed - shared_centered) / np.linalg.norm(
    shared_centered
)
after_error = np.linalg.norm(cleaned - shared_centered) / np.linalg.norm(
    shared_centered
)

# Brief, high-amplitude sensor glitches distort an ordinary covariance. Robust
# masking learns the spatial model from typical samples and transforms the full
# record, including the masked samples.
glitched = observed.copy()
glitched[3, 500:515] += 80.0
ordinary, _ = compute_sns(glitched, n_neighbors=24)
robust, robust_info = compute_sns(glitched, n_neighbors=24, outlier_threshold=8.0)
highlight = np.zeros(n_times, dtype=bool)
highlight[500:515] = True
plot_signal_overlay(
    glitched,
    robust,
    np.arange(n_times),
    pick=3,
    start=475,
    stop=540,
    scale_after=False,
    before_label="Glitched",
    after_label="Robust SNS",
    reference=shared_centered[3],
    reference_label="Shared signal",
    highlight_mask=highlight,
    highlight_label="Excluded while fitting",
    x_label="Sample",
    title="Robust fitting around a sensor glitch",
    show=False,
)


def covariance_spectrum(data):
    """Return the normalized empirical covariance spectrum."""
    centered = data - data.mean(axis=1, keepdims=True)
    values = np.linalg.eigvalsh(centered @ centered.T / centered.shape[1])[::-1]
    return values / values.sum()


spectrum_before = covariance_spectrum(observed)
spectrum_after = covariance_spectrum(cleaned)
pass_outputs = [observed - observed.mean(axis=1, keepdims=True)]
for n_iter in range(1, 5):
    output, _ = compute_sns(observed, n_neighbors=24, n_iter=n_iter)
    pass_outputs.append(output)
changes = [
    np.linalg.norm(pass_outputs[index] - pass_outputs[index - 1])
    / np.linalg.norm(pass_outputs[index - 1])
    for index in range(1, len(pass_outputs))
]

print(f"Clean-signal relative error: {preservation_error:.2e}")
print(f"Sensor-noise error before/after: {before_error:.3f} / {after_error:.3f}")
print(f"Robustly rejected samples: {robust_info['rejected_sample_count']}")

fig, axes = themed_figure(1, 3, figsize=(12, 3.6), constrained_layout=True)
axes[0].bar(["Observed", "SNS"], [before_error, after_error])
axes[0].set(title="Independent sensor noise", ylabel="Relative error")
axes[1].semilogy(spectrum_before[:50], label="Observed")
axes[1].semilogy(spectrum_after[:50], label="SNS")
axes[1].set(title="Covariance spectrum", xlabel="Component", ylabel="Variance fraction")
themed_legend(axes[1])
axes[2].plot(np.arange(1, 5), changes, "o-")
axes[2].set(
    title="Iterative convergence", xlabel="SNS passes", ylabel="Relative change"
)
for ax in axes:
    style_axes(ax, grid=True)
