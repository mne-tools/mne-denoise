"""Reproduce the synthetic BSS-CCA experiments of De Clercq et al. [1]_.

What this script reproduces
---------------------------
The paper's *simulation* protocol, on surrogate data:

- the mixing model of Eq. (8), ``X(lambda) = B + lambda * M``, superimposing a
  muscle-artifact signal on brain activity at a controlled SNR;
- the SNR definition of Eq. (11) and the RRMSE error measure of Eq. (12);
- Fig. 4, RRMSE as a function of the number of lowest-autocorrelated CCA
  components excluded from the reconstruction;
- Fig. 5, RRMSE as a function of SNR;
- the paper's observation that muscle activity occupies the lowest
  autocorrelated components, and that brain activity does not.

The paper's acquisition parameters are matched where they are stated: 21
channels, 250 Hz, 10 s epochs, average reference, and three brain conditions
dominated by delta, alpha, and beta activity (B1, B2, B3).

What this script does NOT reproduce
-----------------------------------
- The paper's actual data. The 21-channel clinical EEG recordings, the
  SOBI-extracted muscle components of Fig. 1, and the ictal recording of
  Figs. 6-7 are not public. Brain and muscle sources here are **surrogates**.
- Therefore the paper's absolute RRMSE values (0.11 for BSS-CCA at SNR 0.65)
  are not expected to be matched, and are not asserted.
- The comparators. No low-pass Butterworth filter or ICA(JADE) is run, so the
  ranking of BSS-CCA against them is not re-derived.
- The ictal EEG results (Figs. 6-7) and the semi-automatic scrolling appendix,
  both of which are qualitative and rely on expert visual assessment.

Every reported endpoint pairs artifact attenuation with a preservation
measure, so improvement cannot be claimed by simply removing more signal.

Run from the repository root with::

    python scripts/replicate_declercq_paper.py

References
----------
.. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
   Van Huffel, S. (2006). Canonical correlation analysis applied to remove
   muscle artifacts from the electroencephalogram. IEEE Transactions on
   Biomedical Engineering, 53(12), 2583-2587.
   https://doi.org/10.1109/TBME.2006.879459
"""

from __future__ import annotations

import numpy as np

from mne_denoise.bss_cca import compute_bss_cca

SFREQ = 250.0
N_CHANNELS = 21
N_TIMES = int(SFREQ * 10)
N_MUSCLE_SOURCES = 3

# Dominant band of the paper's three brain conditions B1, B2, B3.
CONDITIONS = {
    "B1 (delta)": (1.0, 4.0),
    "B2 (alpha)": (8.0, 13.0),
    "B3 (beta)": (13.0, 30.0),
}


def rms(X: np.ndarray) -> float:
    """Root mean square over channels and samples, Eq. (9)."""
    return float(np.sqrt(np.mean(np.square(X))))


def rrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Relative root mean squared error, Eq. (12)."""
    return rms(estimate - reference) / rms(reference)


def band_limited_sources(
    rng: np.random.Generator, n_sources: int, fmin: float, fmax: float
) -> np.ndarray:
    """Random sources whose power is confined to ``[fmin, fmax]``."""
    freqs = np.fft.rfftfreq(N_TIMES, 1.0 / SFREQ)
    out = np.zeros((n_sources, N_TIMES))
    for index in range(n_sources):
        spectrum = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
            freqs.size
        )
        spectrum[(freqs < fmin) | (freqs > fmax)] = 0.0
        out[index] = np.fft.irfft(spectrum, n=N_TIMES)
    return out


def average_reference(X: np.ndarray) -> np.ndarray:
    """Apply the paper's average-referenced montage."""
    return X - X.mean(axis=0, keepdims=True)


def build_condition(
    rng: np.random.Generator, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return average-referenced surrogate brain and muscle signals.

    The brain signal is fifteen sources carrying a dominant rhythm plus a
    substantial broadband 1-45 Hz background. The background is what makes the
    problem non-trivial: it gives the neural subspace genuine low-autocorrelation
    content that partially overlaps the muscle subspace, so perfect separation
    is not available and the error measures mean something.

    Note that independent per-channel sensor noise is deliberately *not* added.
    Such noise is itself broadband and weakly autocorrelated, so BSS-CCA
    assigns it to the same low-ranking components as EMG; with enough of it the
    correct operating point is "remove nearly everything" rather than "remove
    the muscle sources", which is a property of the noise model, not of the
    algorithm. The paper's B matrices are real EEG recordings and carry no such
    added term.
    """
    n_sources = 15
    dominant = band_limited_sources(rng, n_sources, fmin, fmax)
    background = band_limited_sources(rng, n_sources, 1.0, 45.0)
    sources = dominant / np.linalg.norm(dominant) + 0.6 * background / np.linalg.norm(
        background
    )
    brain = average_reference(rng.standard_normal((N_CHANNELS, n_sources)) @ sources)

    # EMG: broadband, dominated by high frequencies (Goncharova et al. 2003).
    # Capped below Nyquist to reflect the band-limited acquisition of the paper.
    muscle_sources = band_limited_sources(rng, N_MUSCLE_SOURCES, 20.0, 0.4 * SFREQ)
    muscle = average_reference(
        rng.standard_normal((N_CHANNELS, N_MUSCLE_SOURCES)) @ muscle_sources
    )
    return brain, muscle


def mix_at_snr(brain: np.ndarray, muscle: np.ndarray, snr: float) -> np.ndarray:
    """Superimpose muscle on brain at a target SNR, Eqs. (8) and (11)."""
    lam = rms(brain) / (snr * rms(muscle))
    return brain + lam * muscle


def band_power(X: np.ndarray, fmin: float, fmax: float) -> float:
    """Total power in a frequency band."""
    spectrum = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], 1.0 / SFREQ)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(spectrum[:, band].sum())


def main() -> None:
    """Run and report the synthetic experiments."""
    rng = np.random.default_rng(2006)

    print("BSS-CCA synthetic experiments (De Clercq et al., 2006)")
    print(
        f"{N_CHANNELS} channels, {SFREQ:.0f} Hz, {N_TIMES / SFREQ:.0f} s, "
        f"average reference, {N_MUSCLE_SOURCES} muscle sources"
    )
    print("Surrogate data: absolute values are NOT comparable to the paper.\n")

    # -- Fig. 4: RRMSE versus number of components removed -------------------
    print("Fig. 4 - RRMSE versus components excluded (SNR = 0.65)")
    counts = list(range(0, 9))
    for label, (fmin, fmax) in CONDITIONS.items():
        brain, muscle = build_condition(rng, fmin, fmax)
        observed = mix_at_snr(brain, muscle, 0.65)
        brain_c = brain - brain.mean(axis=1, keepdims=True)
        errors = [
            rrmse(
                compute_bss_cca(observed, n_remove=k, preserve_mean=False)[0], brain_c
            )
            for k in counts
        ]
        best = int(np.argmin(errors))
        row = "  ".join(f"{k}:{e:.3f}" for k, e in zip(counts, errors, strict=True))
        print(f"  {label:12s} {row}")
        print(
            f"  {'':12s} minimum at {best} removed "
            f"(true number of muscle sources: {N_MUSCLE_SOURCES})"
        )
    print()

    # -- Fig. 5: RRMSE versus SNR --------------------------------------------
    print("Fig. 5 - RRMSE versus SNR (removing 3 components)")
    snrs = [0.2, 0.4, 0.65, 1.0, 2.0, 4.0]
    for label, (fmin, fmax) in CONDITIONS.items():
        brain, muscle = build_condition(rng, fmin, fmax)
        brain_c = brain - brain.mean(axis=1, keepdims=True)
        cells = []
        for snr in snrs:
            observed = mix_at_snr(brain, muscle, snr)
            cleaned, _ = compute_bss_cca(
                observed, n_remove=N_MUSCLE_SOURCES, preserve_mean=False
            )
            centered = observed - observed.mean(axis=1, keepdims=True)
            cells.append(
                f"{snr}:{rrmse(cleaned, brain_c):.3f}/{rrmse(centered, brain_c):.3f}"
            )
        print(f"  {label:12s} " + "  ".join(cells))
    print("  (format SNR:after/before; lower is better)\n")

    # -- Component ordering ---------------------------------------------------
    print("Ordering - muscle occupies the lowest autocorrelated components")
    brain, muscle = build_condition(rng, 8.0, 13.0)
    observed = mix_at_snr(brain, muscle, 0.65)
    _, info = compute_bss_cca(observed, n_remove=N_MUSCLE_SOURCES)
    correlations = info["correlations"]
    print(f"  canonical correlations: {np.round(correlations, 3)}")
    print(
        f"  lowest {N_MUSCLE_SOURCES}: "
        f"{np.round(correlations[-N_MUSCLE_SOURCES:], 3)}  "
        f"rest: {correlations[:-N_MUSCLE_SOURCES].min():.3f} and above"
    )
    print(
        f"  separation margin: "
        f"{correlations[-N_MUSCLE_SOURCES - 1] - correlations[-N_MUSCLE_SOURCES]:.3f}"
    )
    print(
        f"  any anti-correlated component: "
        f"{bool((info['autocorrelations'] < 0).any())}\n"
    )

    # -- Attenuation paired with preservation --------------------------------
    print("Endpoints - attenuation must be paired with preservation")
    for label, (fmin, fmax) in CONDITIONS.items():
        brain, muscle = build_condition(rng, fmin, fmax)
        observed = mix_at_snr(brain, muscle, 0.65)
        cleaned, _ = compute_bss_cca(
            observed, n_remove=N_MUSCLE_SOURCES, preserve_mean=False
        )
        brain_c = brain - brain.mean(axis=1, keepdims=True)
        emg_before = band_power(observed, 60.0, SFREQ / 2)
        emg_after = band_power(cleaned, 60.0, SFREQ / 2)
        neural_reference = band_power(brain_c, fmin, fmax)
        neural_after = band_power(cleaned, fmin, fmax)
        print(
            f"  {label:12s} EMG band retained {emg_after / emg_before:6.1%}   "
            f"neural band retained {neural_after / neural_reference:6.1%}   "
            f"RRMSE {rrmse(cleaned, brain_c):.3f}"
        )


if __name__ == "__main__":
    main()
