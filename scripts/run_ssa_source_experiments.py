"""Run source-oriented SSA experiments without overstating replication.

The default experiment reproduces the documented Teixeira et al. sinusoid
parameter grid: N=500, period=26, SNR in {20, 5} dB, embedding dimensions M in
{11, 36}, and q in {1, 3, 5}. The SNR=5 results are compared with their Table 1
and repeated over deterministic seeds to expose sensitivity to the omitted
noise realization and k-means initialization. The paper does not provide the
exact phase, noise realization, or clustering settings, so equality with the
published MSE values is neither expected nor claimed.

With ``--wine-data``, the script also reproduces the Basic SSA fortified-wine
example distributed by the GistaT authors: N=174, L=84, trend eigentriple 1,
and seasonal pairs 2-3, 4-5, 6-7, and 8-9. The third-party data are intentionally
not bundled with this package.

The undefined Teixeira "funny curve," unavailable clinical EEG recordings, and
multichannel figures cannot be reproduced from the published source. A separate
broadband-EEG surrogate reports both artifact attenuation and desired-signal
preservation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mne_denoise.ssa import (
    compute_local_ssa,
    ssa_decompose,
    ssa_w_correlation,
)

_TEIXEIRA_TABLE_1_SINUSOID = {
    (11, 1): 0.041,
    (11, 3): 0.031,
    (11, 5): 0.317,
    (36, 1): 0.012,
    (36, 3): 0.006,
    (36, 5): 0.009,
}


def _snr_noise(signal: np.ndarray, snr_db: float, rng) -> np.ndarray:
    scale = np.sqrt(np.mean(signal**2) / (10 ** (snr_db / 10)))
    return scale * rng.standard_normal(signal.size)


def _band_power(signal: np.ndarray, sfreq: float, low: float, high: float) -> float:
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / sfreq)
    power = np.abs(np.fft.rfft(signal)) ** 2
    return float(power[(frequencies >= low) & (frequencies < high)].sum())


def _local_sinusoid_mse(
    periodic: np.ndarray,
    observed: np.ndarray,
    window_length: int,
    n_clusters: int,
    random_state: int,
) -> tuple[float, float, list[int]]:
    residual, info = compute_local_ssa(
        observed[np.newaxis],
        window_length=window_length,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    extracted = observed - residual[0]
    mse = float(np.mean((extracted - periodic) ** 2))
    residual_fraction = float(np.linalg.norm(residual[0]) / np.linalg.norm(observed))
    dimensions = info["subspace_dimensions"][0].tolist()
    return mse, residual_fraction, dimensions


def run_periodic_grid(n_sensitivity_seeds: int) -> None:
    """Evaluate the Teixeira sinusoid grid and seed sensitivity."""
    samples = np.arange(500)
    periodic = np.sin(2 * np.pi * samples / 26)
    print("Teixeira sinusoid parameter grid (documented settings, new draw)")
    for snr_db in (20.0, 5.0):
        observed = periodic + _snr_noise(
            periodic, snr_db, np.random.default_rng(2006 + int(snr_db))
        )
        for window_length in (11, 36):
            for n_clusters in (1, 3, 5):
                mse, residual_fraction, dimensions = _local_sinusoid_mse(
                    periodic,
                    observed,
                    window_length,
                    n_clusters,
                    random_state=0,
                )
                target = _TEIXEIRA_TABLE_1_SINUSOID.get((window_length, n_clusters))
                comparison = "" if snr_db != 5 else f", published={target:.3f}"
                print(
                    f"SNR={snr_db:>4.0f} dB M={window_length:>2} "
                    f"q={n_clusters}: MSE={mse:.5f}{comparison}, "
                    f"residual_fraction={residual_fraction:.4f}, k={dimensions}"
                )

    if n_sensitivity_seeds < 1:
        return
    results = {configuration: [] for configuration in _TEIXEIRA_TABLE_1_SINUSOID}
    for seed in range(n_sensitivity_seeds):
        observed = periodic + _snr_noise(
            periodic, 5.0, np.random.default_rng(10_000 + seed)
        )
        for window_length, n_clusters in results:
            mse, _, _ = _local_sinusoid_mse(
                periodic,
                observed,
                window_length,
                n_clusters,
                random_state=seed,
            )
            results[(window_length, n_clusters)].append(mse)
    print(f"\nSNR=5 sensitivity over {n_sensitivity_seeds} prespecified seeds")
    for configuration, values in results.items():
        values = np.asarray(values)
        target = _TEIXEIRA_TABLE_1_SINUSOID[configuration]
        within_range = values.min() <= target <= values.max()
        print(
            f"M={configuration[0]:>2} q={configuration[1]}: "
            f"median={np.median(values):.5f}, "
            f"range=[{values.min():.5f}, {values.max():.5f}], "
            f"published={target:.3f}, within_range={within_range}"
        )


def _load_fortified_wine(path: Path) -> np.ndarray:
    """Load the Fort column from the GistaT ``Wine.dat`` text file."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("%", "#", "Fort")):
            continue
        rows.append(float(stripped.split()[0]))
    values = np.asarray(rows)
    if values.size != 174 or not np.isfinite(values).all():
        raise ValueError("wine data must contain the 174 finite Fort observations")
    return values


def run_fortified_wine(path: Path) -> None:
    """Reproduce the official Basic SSA fortified-wine grouping."""
    series = _load_fortified_wine(path)
    components, info = ssa_decompose(series, window_length=84)
    groups = ((1, 2), (3, 4), (5, 6), (7, 8))
    periods = []
    for group in groups:
        reconstructed = components[list(group)].sum(axis=0)
        spectrum = np.abs(np.fft.rfft(reconstructed - reconstructed.mean()))
        peak = int(np.argmax(spectrum[1:]) + 1)
        periods.append(series.size / peak)
    pair_ratios = [
        info["singular_values"][left] / info["singular_values"][right]
        for left, right in groups
    ]
    correlation = ssa_w_correlation(components[:9], window_length=84)
    reconstruction_error = np.max(np.abs(components.sum(axis=0) - series))
    print("\nGistaT fortified-wine Basic SSA example")
    print(f"N={series.size}, L={info['window_length']}")
    print(f"complete-reconstruction max error: {reconstruction_error:.3e}")
    print(
        "dominant periods for pairs 2-3, 4-5, 6-7, 8-9: "
        + ", ".join(f"{period:.3g}" for period in periods)
        + " months"
    )
    print(
        "paired singular-value ratios: "
        + ", ".join(f"{ratio:.5f}" for ratio in pair_ratios)
    )
    print(
        "max |w-correlation| between trend ET 1 and ETs 2-9: "
        f"{np.max(np.abs(correlation[0, 1:9])):.5f}"
    )


def run_preservation_check() -> None:
    """Report artifact attenuation and broadband desired-signal preservation."""
    sfreq = 128.0
    rng = np.random.default_rng(83)
    times = np.arange(1280) / sfreq
    neural = 0.5 * rng.standard_normal(times.size)
    artifact = 6.0 * np.sin(2 * np.pi * 0.5 * times)
    observed = neural + artifact + 0.05 * rng.standard_normal(times.size)
    cleaned, _ = compute_local_ssa(
        observed[np.newaxis],
        window_length=41,
        n_clusters=6,
        random_state=0,
    )
    cleaned = cleaned[0]
    attenuation = 10 * np.log10(
        _band_power(observed, sfreq, 0, 3) / _band_power(cleaned, sfreq, 0, 3)
    )
    gain = np.dot(cleaned, neural) / np.dot(neural, neural)
    correlation = np.corrcoef(cleaned, neural)[0, 1]
    print("\nBroadband preservation surrogate")
    print(f"0-3 Hz attenuation: {attenuation:.2f} dB")
    print(f"Desired-signal gain: {gain:.3f}")
    print(f"Desired-signal correlation: {correlation:.3f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=10,
        help="number of prespecified SNR=5 seeds (default: 10; 0 disables)",
    )
    parser.add_argument(
        "--wine-data",
        type=Path,
        help="path to the GistaT Wine.dat file for the Basic SSA reproduction",
    )
    args = parser.parse_args()
    if args.sensitivity_seeds < 0:
        parser.error("--sensitivity-seeds must be nonnegative")
    return args


if __name__ == "__main__":
    arguments = _parse_args()
    run_periodic_grid(arguments.sensitivity_seeds)
    if arguments.wine_data is not None:
        run_fortified_wine(arguments.wine_data)
    run_preservation_check()
