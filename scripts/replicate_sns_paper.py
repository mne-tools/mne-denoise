"""Reproduce synthetic experiments for the SNS algorithm [1]_.

The script evaluates the paper's central algorithmic predictions on a fixed
157-sensor simulation: preservation of spatially redundant activity,
attenuation of sensor-specific noise, and diminishing changes across repeated
SNS passes. It uses only :mod:`mne_denoise.sns` and writes no files.

Run from the repository root with::

    python scripts/replicate_sns_paper.py

References
----------
.. [1] de Cheveigné, A., & Simon, J. Z. (2008). Sensor noise suppression.
   Journal of Neuroscience Methods, 168(1), 195–202.
   https://doi.org/10.1016/j.jneumeth.2007.09.012
"""

from __future__ import annotations

import numpy as np

from mne_denoise.sns import compute_sns


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compute relative root-sum-square error."""
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


def main() -> None:
    """Run and report the deterministic SNS experiments."""
    rng = np.random.default_rng(2008)
    n_channels, n_sources, n_times = 157, 10, 5000
    sources = rng.standard_normal((n_sources, n_times))
    shared = rng.standard_normal((n_channels, n_sources)) @ sources
    shared -= shared.mean(axis=1, keepdims=True)
    observed = shared + 0.35 * rng.standard_normal(shared.shape)

    regenerated, _ = compute_sns(shared, n_neighbors=24)
    cleaned, info = compute_sns(observed, n_neighbors=24)
    outputs = [observed - observed.mean(axis=1, keepdims=True)]
    for n_iter in range(1, 5):
        outputs.append(compute_sns(observed, n_neighbors=24, n_iter=n_iter)[0])
    changes = [
        np.linalg.norm(current - previous) / np.linalg.norm(previous)
        for previous, current in zip(outputs[:-1], outputs[1:], strict=True)
    ]

    print("SNS synthetic paper experiments")
    print(f"channels: {n_channels}; latent sources: {n_sources}")
    print(f"clean-signal regeneration error: {relative_error(regenerated, shared):.3e}")
    print(f"sensor-noise error before SNS: {relative_error(observed, shared):.4f}")
    print(f"sensor-noise error after SNS:  {relative_error(cleaned, shared):.4f}")
    print(f"effective neighbours: {info['n_neighbors']}")
    print("relative changes by pass: " + ", ".join(f"{value:.4f}" for value in changes))


if __name__ == "__main__":
    main()
