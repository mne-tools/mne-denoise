#!/usr/bin/env python
"""
SOUND & SSP-SIR — validation against Mutanen et al. (2016, 2018).

Regenerates every number and figure in ``validation/sound_sspsir/README.md``.

Simulation 2 reproduces the 2016 paper's reported figures. Simulation 1 does
not fully reproduce the 2018 paper's -- see the "Replication status" section of
that README. Correctness of the implementation rests primarily on the parity
suite (``tests/parity/test_sound_sspsir_parity.py``).

Simulation 1 — SOUND (Mutanen et al. 2018, Figs. 3-4)
    Follows the paper's "Simulation analysis" section: 60 sensors, T = 146,
    neural data ``Ybar = L J`` from 8 dipoles (2 per lobe of the left
    hemisphere, one superficial and one ~1 cm deeper) with smooth staggered
    waveforms; Appendix-B noise ``N = (B^(m) o L) eps`` with ``m`` = NCI ones
    per column; ``SNR = trace(Ybar Ybar^T)/trace(N N^T)`` (Eq. 16); rNR per
    Eq. (14). Reports the correlation between true and estimated per-channel
    noise levels, and rNR for SOUND vs DDWiener/SNS.

Simulation 2 — SSP-SIR (Mutanen et al. 2016, "Simulation analysis")
    Computes the EEG topography of every cortical dipole, scales it so the
    maximum voltage in the array is 10 uV, passes it through SSP-SIR, and
    compares the result with the original via correlation coefficient (CC) and
    relative error (RE) -- i.e. it measures how much SSP-SIR *distorts* clean
    neuronal topographies. The artifact subspace is swept over spatial
    frequencies via the skull/skin conductivity contrast, following Mutanen
    et al. (2024), so that its signal-space angle to the neuronal topographies
    varies; that angle is what dominates SSP-SIR's performance.

References
----------
Mutanen, T. P., et al. (2016). Recovering TMS-evoked EEG responses masked by
muscle artifacts. NeuroImage, 139, 157-166.
Mutanen, T. P., et al. (2018). Automatic and robust noise suppression in EEG
and MEG: The SOUND algorithm. NeuroImage, 166, 135-151.
Mutanen, T. P., Ilmoniemi, I., Atti, I., Metsomaa, J., & Ilmoniemi, R. J.
(2024). A simulation study: comparing independent component analysis and
signal-space projection - source-informed reconstruction for rejecting muscle
artifacts evoked by transcranial magnetic stimulation. Frontiers in Human
Neuroscience, 18, 1324958.

Usage
-----
  # Full run reproducing the report (slow: ~100 runs per cell):
  python scripts/run_sound_sspsir_validation.py

  # Quick smoke run:
  python scripts/run_sound_sspsir_validation.py --n-runs 5 --n-dipoles 50

  # Custom output directory:
  python scripts/run_sound_sspsir_validation.py --out-dir /tmp/validation

Output contract
---------------
  {out_dir}/
      fig1_sound_validation.png     <-- noise-estimate correlation + rNR
      fig2_sspsir_validation.png    <-- CC/RE vs M, CC histogram
      results.json                  <-- every scalar quoted in the README
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from mne_denoise._leadfield import (  # noqa: E402
    fibonacci_sphere,
    make_spherical_leadfield,
)
from mne_denoise.sound.core import compute_sound  # noqa: E402
from mne_denoise.sspsir.core import compute_sspsir  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "validation" / "sound_sspsir"


def build_leadfield(n_channels: int, n_dipoles: int) -> np.ndarray:
    """Average-referenced spherical lead field for a standard_1020 subset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_spherical_leadfield(_montage_info(n_channels), n_dipoles=n_dipoles)


def _montage_info(n_channels: int):
    names = mne.channels.make_standard_montage("standard_1020").ch_names[:n_channels]
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage("standard_1020")
    return info


def build_source_leadfield(n_channels: int) -> np.ndarray:
    """Lead field of the 8 dipoles used to generate the neural data.

    Mutanen et al. (2018), 'Simulation analysis': two dipoles in each of the
    frontal, parietal, temporal and occipital lobes of the left hemisphere, one
    situated superficially and the other about 1 cm deeper.
    """
    info = _montage_info(n_channels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sphere = mne.make_sphere_model(
            r0="auto",
            head_radius="auto",
            info=info,
            relative_radii=(81 / 88, 85 / 88, 1.0),
            sigmas=(0.33, 0.33 / 50, 0.33),
            verbose=False,
        )
        head_radius = float(sphere["layers"][-1]["rad"])
        lobes = np.array(
            [
                [0.4, 0.8, 0.45],  # frontal
                [0.0, 0.15, 0.99],  # parietal
                [0.85, 0.0, 0.0],  # temporal
                [0.2, -0.85, 0.45],  # occipital
            ]
        )
        lobes[:, 0] *= -1.0  # left hemisphere
        lobes /= np.linalg.norm(lobes, axis=1, keepdims=True)
        superficial = 76 / 88 * head_radius
        positions, normals = [], []
        for direction in lobes:
            for radius in (superficial, superficial - 0.010):
                positions.append(direction * radius + sphere["r0"])
                normals.append(direction)
        src = mne.setup_volume_source_space(
            pos={"rr": np.array(positions), "nn": np.array(normals)},
            sphere_units="m",
            verbose=False,
        )
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
        )
        fwd = mne.convert_forward_solution(
            fwd, force_fixed=True, use_cps=False, verbose=False
        )
    gain = np.asarray(fwd["sol"]["data"], dtype=float)
    return gain - gain.mean(axis=0, keepdims=True)


def source_waveforms(n_times: int) -> np.ndarray:
    """Eight smooth, staggered source waveforms (Mutanen et al. 2018, Fig. 2)."""
    t = np.arange(n_times)
    peaks = np.linspace(0.17, 0.82, 8) * n_times
    widths = np.linspace(10, 14, 8)
    amps = [1.0, 0.7, 0.9, 0.6, 0.85, 0.6, 0.8, 0.55]
    return np.array(
        [
            a * np.exp(-((t - p) ** 2) / (2.0 * w**2))
            for p, w, a in zip(peaks, widths, amps, strict=True)
        ]
    )


# --------------------------------------------------------------------------
# Simulation 1: SOUND
# --------------------------------------------------------------------------
def noise_mixing_factor(leadfield: np.ndarray) -> np.ndarray:
    """Square (S x S) factor whose Gram equals ``leadfield @ leadfield.T``.

    Appendix B writes the noise mixing matrix as ``B^(m) o L`` with ``L`` an
    S x S object satisfying ``L L^T = Sigma``. Any square root satisfies that,
    and the paper does not say which; the entry-wise product with ``B^(m)``
    makes the choice matter. We use the symmetric root ``U S U^T``, which keeps
    per-channel noise amplitudes in a plausible range -- the rectangular root
    ``U S`` inherits the lead field's singular-value spread and produces noise
    levels differing by many orders of magnitude across channels, which is
    inconsistent with the paper's Fig. 3B (levels spanning roughly 8x).
    """
    u, s, _ = np.linalg.svd(leadfield, full_matrices=False)
    n_channels = leadfield.shape[0]
    u, s = u[:, :n_channels], s[:n_channels]
    return (u * s) @ u.T


def appendix_b_noise(
    factor: np.ndarray, nci: int, n_times: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated sensor noise per Mutanen et al. (2018), Appendix B.

    ``N = A eps = (B^(m) o L) eps``, where ``B^(m)`` is binary with ``m = NCI``
    ones per column, so each of the S independent noise sources is recorded by
    NCI channels. NCI=1 gives a diagonal noise covariance; NCI=S gives
    ``Sigma = L L^T``.

    The binary design is balanced (each channel also receives NCI sources).
    Drawing each column's rows independently instead would, by the coupon-
    collector effect, leave roughly S/e channels with no noise at all at NCI=1,
    which cannot be what the paper intends.

    Returns the noise and the true per-channel noise amplitudes.
    """
    n_channels = factor.shape[0]
    mask = np.zeros((n_channels, n_channels))
    for _ in range(min(nci, n_channels)):
        for col, row in enumerate(rng.permutation(n_channels)):
            mask[row, col] = 1.0
    mixing = mask * factor
    noise = mixing @ rng.standard_normal((n_channels, n_times))
    return noise, np.sqrt((mixing**2).sum(axis=1))


def relative_noise_reduction(
    clean: np.ndarray, noisy: np.ndarray, cleaned: np.ndarray
) -> float:
    """rNR as defined in Mutanen et al. (2018), Eq. (14).

    ``100% * sum_s (sigma_s - sigmahat_s)/nu_s / sum_s (sigma_s/nu_s)``, with
    ``sigma_s`` / ``sigmahat_s`` the noise standard deviation in channel s
    before / after cleaning and ``nu_s`` that of the noiseless signal. 0% if no
    noise is removed, 100% if the noise is removed perfectly.
    """
    nu = clean.std(axis=1)
    sigma = (noisy - clean).std(axis=1)
    sigma_hat = (cleaned - clean).std(axis=1)
    return float(100.0 * np.sum((sigma - sigma_hat) / nu) / np.sum(sigma / nu))


def run_sound_simulation(
    leadfield: np.ndarray,
    source_leadfield: np.ndarray,
    snr: float,
    nci: int,
    n_runs: int,
    n_times: int,
    seed: int,
) -> dict:
    """Noise-estimate correlation and relative noise reduction for one cell."""
    n_channels = leadfield.shape[0]
    factor = noise_mixing_factor(leadfield)
    waveforms = source_waveforms(n_times)
    correlations, rnr_sound, rnr_ddwiener = [], [], []

    # Neural data Ybar = L J from the eight dipoles; fixed across runs, as in
    # the paper (only the noise is re-randomised).
    clean = source_leadfield @ waveforms
    clean -= clean.mean(axis=0, keepdims=True)

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        noise, true_sigmas = appendix_b_noise(factor, nci, n_times, rng)
        # SNR = trace(Ybar Ybar^T) / trace(N N^T)  (Eq. 16; a power ratio).
        scale = np.sqrt(np.trace(clean @ clean.T) / np.trace(noise @ noise.T) / snr)
        noise, true_sigmas = noise * scale, true_sigmas * scale
        data = clean + noise
        data -= data.mean(axis=0, keepdims=True)

        operator, sigmas, _ = compute_sound(
            data, leadfield, lambda_=0.1, n_iter=5, random_state=run
        )
        correlations.append(np.corrcoef(sigmas, true_sigmas)[0, 1])
        rnr_sound.append(relative_noise_reduction(clean, data, operator @ data))

        # DDWiener/SNS baseline: reconstruct each channel from all the others.
        cov = data @ data.T
        gamma = np.mean(np.diag(cov))
        pred = np.empty_like(data)
        for i in range(n_channels):
            others = np.array([j for j in range(n_channels) if j != i])
            pred[i] = cov[i, others] @ np.linalg.solve(
                cov[np.ix_(others, others)] + gamma * np.eye(n_channels - 1),
                data[others],
            )
        rnr_ddwiener.append(relative_noise_reduction(clean, data, pred))

    return {
        "snr": snr,
        "nci": nci,
        "noise_correlation": float(np.mean(correlations)),
        "noise_correlation_sd": float(np.std(correlations)),
        "rnr_sound": float(np.mean(rnr_sound)),
        "rnr_ddwiener": float(np.mean(rnr_ddwiener)),
    }


# --------------------------------------------------------------------------
# Simulation 2: SSP-SIR
# --------------------------------------------------------------------------
def _sphere(info, skull_divisor: float):
    return mne.make_sphere_model(
        r0="auto",
        head_radius="auto",
        info=info,
        relative_radii=(81 / 88, 85 / 88, 1.0),
        sigmas=(0.33, 0.33 / skull_divisor, 0.33),
        verbose=False,
    )


def _leadfield_at(info, sphere, positions, normals) -> np.ndarray:
    src = mne.setup_volume_source_space(
        pos={"rr": positions, "nn": normals}, sphere_units="m", verbose=False
    )
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False
    )
    fwd = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=False, verbose=False
    )
    gain = np.asarray(fwd["sol"]["data"], dtype=float)
    return gain - gain.mean(axis=0, keepdims=True)


def artifact_subspace(
    info, sphere, head_radius: float, skull_divisor: float, k: int
) -> np.ndarray:
    """Muscle-artifact topographies of a given spatial frequency.

    Following Mutanen et al. (2024): lateral current sources in the right
    hemisphere produce the bipolar lateral potential patterns characteristic of
    real TMS-evoked muscle artifacts, and the skull/skin conductivity contrast
    controls their spatial frequency -- a larger divisor smears the topography,
    lowering its spatial frequency and making it more similar to neuronal
    patterns.
    """
    directions = fibonacci_sphere(5000)
    lateral = (directions[:, 0] > 0.6) & (np.abs(directions[:, 2]) < 0.4)
    directions = directions[lateral]
    tangential = np.cross(directions, np.array([0.0, 0.0, 1.0]))
    tangential /= np.linalg.norm(tangential, axis=1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gain = _leadfield_at(
            info,
            _sphere(info, skull_divisor),
            directions * (76 / 88 * head_radius) + sphere["r0"],
            tangential,
        )
    return np.linalg.svd(gain, full_matrices=False)[0][:, :k]


def run_sspsir_simulation(
    n_channels: int,
    n_dipoles: int,
    subspace_dims: list[int],
    skull_divisors: list[float],
    M: int,
    seed: int,
) -> dict:
    """Distortion of neuronal topographies by SSP-SIR (Mutanen et al. 2016).

    The paper's 'Simulation analysis': compute the EEG topography of every
    cortical dipole, scale it so the maximum voltage in the array is 10 uV,
    pass it through SSP-SIR, and compare the result with the original via
    correlation coefficient (CC) and relative error (RE). The artifact subspace
    is swept over spatial frequencies so that its signal-space angle to the
    neuronal topographies varies, which is the factor Mutanen et al. (2024)
    identify as dominating SSP-SIR's performance.
    """
    info = _montage_info(n_channels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sphere = _sphere(info, 50.0)
        head_radius = float(sphere["layers"][-1]["rad"])

        # SIR lead field: radial dipoles on the 76 mm shell (the paper's
        # spherical model, as used for the SIR step).
        shell = fibonacci_sphere(5000)
        leadfield = _leadfield_at(
            info, sphere, shell * (76 / 88 * head_radius) + sphere["r0"], shell
        )

        # Neuronal topographies: dipoles spread through the brain volume at
        # varying depth and orientation, standing in for the folded cortical
        # surface the paper samples with its BEM model.
        rng = np.random.default_rng(seed)
        directions = fibonacci_sphere(n_dipoles)
        depths = rng.uniform(0.55, 1.0, n_dipoles)
        orientations = rng.standard_normal((n_dipoles, 3))
        orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
        topographies = _leadfield_at(
            info,
            sphere,
            directions * (depths * (76 / 88 * head_radius))[:, None] + sphere["r0"],
            orientations,
        )
    # Scaled so the maximum voltage in the EEG array is 10 uV.
    topographies = (
        topographies / np.abs(topographies).max(axis=0, keepdims=True) * 10e-6
    )
    neural_space = np.linalg.svd(topographies, full_matrices=False)[0][:, :30]
    norms = np.linalg.norm(topographies, axis=0)

    def score(estimate):
        re = np.linalg.norm(estimate - topographies, axis=0) / norms
        cc = np.array(
            [
                np.corrcoef(estimate[:, i], topographies[:, i])[0, 1]
                for i in range(n_dipoles)
            ]
        )
        return {
            "cc": float(np.nanmean(cc)),
            "frac_above_0p9": float(np.nanmean(cc > 0.9)),
            "re": float(np.mean(re)),
        }

    cells = []
    for divisor in skull_divisors:
        for k in subspace_dims:
            topos = artifact_subspace(info, sphere, head_radius, divisor, k)
            # Smallest principal angle between the artifact and neuronal spaces.
            angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.linalg.svd(neural_space.T @ topos, compute_uv=False)[-1],
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                operator = compute_sspsir(leadfield, topos, M)
            cell = {"skull_divisor": divisor, "k": k, "angle_deg": angle}
            cell.update(score(operator @ topographies))
            cells.append(cell)

    # Uncleaned baseline: the artifact topography superimposed at its peak.
    topos = artifact_subspace(info, sphere, head_radius, 1.0, 1)
    artifact = topos[:, :1] / np.abs(topos[:, :1]).max() * 100e-6
    uncleaned = score(topographies + artifact)

    return {"cells": cells, "uncleaned": uncleaned, "M": M}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_sound(cells: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ncis = sorted({c["nci"] for c in cells})
    snrs = sorted({c["snr"] for c in cells})

    for snr in snrs:
        row = [c for c in cells if c["snr"] == snr]
        row.sort(key=lambda c: c["nci"])
        axes[0].plot(
            [c["nci"] for c in row],
            [c["noise_correlation"] for c in row],
            "o-",
            label=f"SNR = {snr:g}",
        )
    axes[0].set_xlabel("Noise-correlation index (NCI)")
    axes[0].set_ylabel("corr(true σ, estimated σ)")
    axes[0].set_title("SOUND noise-level estimation")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.98, ls="--", c="grey", lw=1, label="paper ≈ 0.98")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    mid = snrs[len(snrs) // 2]
    row = sorted([c for c in cells if c["snr"] == mid], key=lambda c: c["nci"])
    axes[1].plot(
        [c["nci"] for c in row], [c["rnr_sound"] for c in row], "o-", label="SOUND"
    )
    axes[1].plot(
        [c["nci"] for c in row],
        [c["rnr_ddwiener"] for c in row],
        "s--",
        label="DDWiener",
    )
    axes[1].set_xlabel("Noise-correlation index (NCI)")
    axes[1].set_ylabel("Relative noise reduction (%)  [Eq. 14]")
    axes[1].set_title(f"Noise reduction (SNR = {mid:g})")
    axes[1].set_xticks(ncis)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("SOUND validation — Mutanen et al. (2018)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sspsir(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    dims = sorted({c["k"] for c in results["cells"]})
    markers = {dims[0]: "o-", dims[-1]: "s--"}

    for k in dims:
        row = sorted(
            (c for c in results["cells"] if c["k"] == k), key=lambda c: c["angle_deg"]
        )
        angles = [c["angle_deg"] for c in row]
        axes[0].plot(
            angles, [c["cc"] for c in row], markers.get(k, "^:"), label=f"k = {k}"
        )
        axes[1].plot(
            angles,
            [c["re"] * 100 for c in row],
            markers.get(k, "^:"),
            label=f"k = {k}",
        )

    axes[0].axhspan(0.94, 0.97, color="tab:green", alpha=0.15, label="paper CC range")
    axes[0].axhline(results["uncleaned"]["cc"], ls=":", c="tab:red", label="uncleaned")
    axes[0].set_ylabel("Topography correlation (CC)")
    axes[0].set_title("SSP-SIR topography preservation")
    axes[0].set_ylim(0, 1.05)

    axes[1].axhspan(21, 36, color="tab:green", alpha=0.15, label="paper RE range")
    axes[1].set_ylabel("Relative error (%)")
    axes[1].set_title("Distortion introduced by SSP-SIR")

    for ax in axes:
        ax.set_xlabel("Artifact / neuronal signal-space angle [deg]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"SSP-SIR validation — Mutanen et al. (2016), M = {results['M']}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-dipoles", type=int, default=500)
    # Paper defaults: 60 sensors, T = 146 samples.
    parser.add_argument("--n-channels", type=int, default=60)
    parser.add_argument("--n-times", type=int, default=146)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    print(f"Building lead field ({args.n_channels} channels)...")
    leadfield = build_leadfield(args.n_channels, n_dipoles=2000)
    source_leadfield = build_source_leadfield(args.n_channels)

    print("Simulation 1: SOUND")
    cells = []
    for snr in (0.33, 1.0, 3.0, 9.0):  # paper's SNR grid
        for nci in (1, 3, 9):  # paper's NCI grid
            cell = run_sound_simulation(
                leadfield,
                source_leadfield,
                snr,
                nci,
                args.n_runs,
                args.n_times,
                args.seed,
            )
            cells.append(cell)
            print(
                f"  SNR={snr:<5g} NCI={nci:<3d} "
                f"corr={cell['noise_correlation']:.3f}  "
                f"rNR SOUND={cell['rnr_sound']:5.1f}% "
                f"DDWiener={cell['rnr_ddwiener']:6.1f}%"
            )

    print("Simulation 2: SSP-SIR")
    # The paper measured 9-, 9- and 6-dimensional artifact subspaces in its
    # three subjects, and truncated the SIR inverse at M = 30.
    subspace_dims = [6, 9]
    skull_divisors = [1.0, 20.0, 60.0, 200.0]
    sspsir = run_sspsir_simulation(
        args.n_channels,
        args.n_dipoles,
        subspace_dims,
        skull_divisors,
        M=30,
        seed=args.seed,
    )
    for cell in sspsir["cells"]:
        print(
            f"  skull 1/{cell['skull_divisor']:<5g} k={cell['k']}  "
            f"angle={cell['angle_deg']:4.1f}deg  CC={cell['cc']:.3f}  "
            f">0.9: {cell['frac_above_0p9'] * 100:3.0f}%  "
            f"RE={cell['re'] * 100:3.0f}%"
        )
    print(
        f"  uncleaned CC={sspsir['uncleaned']['cc']:.2f} "
        f"RE={sspsir['uncleaned']['re'] * 100:.0f}%"
    )
    print("  paper (M=30): CC 0.94-0.97, 84-94% > 0.9, RE 21-36%")

    plot_sound(cells, args.out_dir / "fig1_sound_validation.png")
    plot_sspsir(sspsir, args.out_dir / "fig2_sspsir_validation.png")

    payload = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "sound": cells,
        "sspsir": sspsir,
    }
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote figures and results.json to {args.out_dir}")


if __name__ == "__main__":
    main()
