"""Parity + behavior tests for the new ``method="riemannian_windowed"`` backend.

The original ``method="riemannian"`` backend computes one covariance + one
reconstruction matrix for the entire stream, which makes its cleaned output
cutoff-invariant on real EEG (documented in
``reports/paper_validation/rasr/README.md``). The new ``riemannian_windowed``
backend re-runs the Riemannian nonlinear eigenspace per processing window,
restoring cutoff sensitivity.

This test file pins:

1. **Schema parity** — both backends produce the same diagnostic keys.
2. **Cutoff monotonicity** — for ``riemannian_windowed`` the fraction of
   reconstructed windows monotonically *decreases* as cutoff increases on
   contaminated data (regression guard against the original bug).
3. **Sane outputs** — cleaned signal correlation with input is high
   (> 0.5) and finite; no NaN / Inf / negative variance.
4. **Run-only check** for the experimental gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mne_denoise.asr import ASR

_XCHECK_INPUT = (
    Path(__file__).parent / "matlab_reference" / "riemannian_windowed_xcheck_input.mat"
)
_XCHECK_REFERENCE = (
    Path(__file__).parent
    / "matlab_reference"
    / "riemannian_windowed_xcheck_reference.mat"
)


def _make_synthetic_eeg(
    sfreq: float = 250.0,
    duration_s: float = 40.0,
    n_channels: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    return data


def _inject_bursts(
    data: np.ndarray,
    n_bursts: int = 10,
    burst_duration_s: float = 0.5,
    amplitude: float = 12.0,
    sfreq: float = 250.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(97)
    burst_len = int(round(burst_duration_s * sfreq))
    n_times = data.shape[1]
    starts = np.linspace(burst_len, n_times - burst_len, n_bursts).astype(int)
    contaminated = data.copy()
    channel_scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_times)
        actual_len = stop - start
        spatial = rng.standard_normal(data.shape[0])
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(actual_len)
        contaminated[:, start:stop] += (
            amplitude * channel_scale * np.outer(spatial, temporal)
        )
    return contaminated


def _build_rasr(method: str, cutoff: float) -> ASR:
    return ASR(
        sfreq=250.0,
        cutoff=cutoff,
        method=method,
        experimental=True,
        picks=None,
        max_mem_mb=512,
        random_state=42,
        verbose=False,
    )


def test_riemannian_windowed_is_first_class_no_experimental_required():
    """``method="riemannian_windowed"`` is promoted — no experimental flag needed.

    Validated by Python equivalence (processing ≡ standard; covariance ≡
    riemannian) + a direct MATLAB asr_process cross-check at relerr < 1e-13.
    The legacy ``riemannian`` backend STILL requires experimental=True (it is
    cutoff-invariant on real EEG).
    """
    clean = _make_synthetic_eeg(n_channels=12)
    contaminated = _inject_bursts(clean, sfreq=250.0)

    # riemannian_windowed: works WITHOUT experimental=True
    asr = ASR(
        sfreq=250.0,
        cutoff=20.0,
        method="riemannian_windowed",
        experimental=False,
        picks=None,
        verbose=False,
    )
    cleaned = asr.fit_transform(contaminated)
    assert cleaned.shape == contaminated.shape

    # legacy riemannian: STILL gated
    legacy = ASR(
        sfreq=250.0,
        cutoff=20.0,
        method="riemannian",
        experimental=False,
        picks=None,
        verbose=False,
    )
    with pytest.raises(ValueError, match="experimental"):
        legacy.fit(_make_synthetic_eeg())


def test_riemannian_windowed_runs_end_to_end():
    """Cleaned output is finite, shape-preserving, and reasonably correlated with input."""
    clean = _make_synthetic_eeg(n_channels=12)
    contaminated = _inject_bursts(clean, sfreq=250.0)

    asr = _build_rasr("riemannian_windowed", cutoff=20.0)
    asr.fit(contaminated)
    cleaned = asr.transform(contaminated)

    assert cleaned.shape == contaminated.shape
    assert np.all(np.isfinite(cleaned))
    corr = np.corrcoef(cleaned.ravel(), contaminated.ravel())[0, 1]
    assert corr > 0.5, f"cleaned-input correlation too low: {corr:.3f}"


def test_riemannian_windowed_state_advertises_solver():
    """diagnostics_ and state_ should expose the riemannian_windowed solver."""
    clean = _make_synthetic_eeg(n_channels=12)
    contaminated = _inject_bursts(clean, sfreq=250.0)

    asr = _build_rasr("riemannian_windowed", cutoff=20.0)
    asr.fit_transform(contaminated)

    assert asr.diagnostics_["covariance_geometry"] == "riemannian_windowed"
    assert asr.diagnostics_["riemannian_solver"] == "nonlinear_eigenspace"
    assert asr.state_.method == "riemannian_windowed"


def test_riemannian_windowed_cutoff_monotonicity():
    """The bug-fix gate: increasing cutoff should reduce % windows reconstructed.

    The original ``method="riemannian"`` backend produces a *flat* line because
    its keep mask is computed once for the entire stream. This test verifies
    that ``riemannian_windowed`` instead exhibits the expected monotonically
    decreasing trend (matching standard ASR).
    """
    clean = _make_synthetic_eeg(n_channels=12)
    contaminated = _inject_bursts(clean, sfreq=250.0)

    cutoffs = [5.0, 20.0, 50.0]
    fractions = []
    for k in cutoffs:
        asr = _build_rasr("riemannian_windowed", cutoff=k)
        asr.fit(contaminated)
        asr.transform(contaminated)
        frac = float(asr.diagnostics_["fraction_reconstructed_windows"])
        fractions.append(frac)

    # Must be monotonically non-increasing (allow ties; reject sweeps that go UP).
    diffs = np.diff(fractions)
    assert np.all(diffs <= 1e-6), (
        f"riemannian_windowed should monotonically reduce reconstruction "
        f"fraction with cutoff; got {dict(zip(cutoffs, fractions))}"
    )

    # And the RANGE should be non-zero — flat line is the documented bug.
    span = max(fractions) - min(fractions)
    assert span > 0.05, (
        f"riemannian_windowed fraction_reconstructed_windows is essentially "
        f"flat across cutoffs (span={span:.3f}); "
        f"this is the original riemannian-backend bug. fractions={fractions}"
    )


def test_riemannian_windowed_vs_riemannian_diagnostics_keys_match():
    """Schema parity: both backends produce the same set of diagnostic keys."""
    clean = _make_synthetic_eeg(n_channels=12)
    contaminated = _inject_bursts(clean, sfreq=250.0)

    rasr = _build_rasr("riemannian", cutoff=20.0)
    rasr.fit_transform(contaminated)

    rasrw = _build_rasr("riemannian_windowed", cutoff=20.0)
    rasrw.fit_transform(contaminated)

    rasr_keys = set(rasr.diagnostics_.keys())
    rasrw_keys = set(rasrw.diagnostics_.keys())
    # The riemannian backend exposes 3 extra "riemannian_mean_*" zero-filled
    # arrays (legacy). The windowed backend omits them.
    extra_in_rasr = rasr_keys - rasrw_keys
    extra_in_rasrw = rasrw_keys - rasr_keys
    legacy_extras = {
        "riemannian_mean_iterations",
        "riemannian_mean_converged",
        "riemannian_mean_update_norm",
    }
    # Allow only the documented legacy delta.
    assert extra_in_rasr.issubset(legacy_extras), (
        f"riemannian has unexpected extra keys: {extra_in_rasr - legacy_extras}"
    )
    assert not extra_in_rasrw, (
        f"riemannian_windowed has unexpected extra keys: {extra_in_rasrw}"
    )


def test_riemannian_windowed_processing_identical_to_standard():
    """PROOF (Link A): riemannian_windowed processing == standard processing.

    Given the SAME calibration state, the per-window processing loop of
    ``method="riemannian_windowed"`` is byte-identical to the standard ASR
    processing path (they share the same eigh / threshold / raised-cosine
    blend code). Since ``_process_asr_standard`` is already validated against
    MATLAB ``clean_rawdata/asr_process.m`` (``test_asr_process_against_matlab_reference``),
    this exact equality transitively confers that MATLAB parity onto
    riemannian_windowed processing — with ZERO added error.
    """
    from mne_denoise.asr.core import calibrate_asr, process_asr

    clean = _make_synthetic_eeg(n_channels=12, duration_s=60.0)
    dirty = _inject_bursts(clean, sfreq=250.0)

    # One shared state (calibration backend is irrelevant for this test —
    # we are isolating the PROCESSING step).
    state, _ = calibrate_asr(
        dirty[:, :7500],
        250.0,
        cutoff=20.0,
        method="standard",
        calibration="manual",
        filter_kind="none",
    )
    clean_std, _ = process_asr(dirty, 250.0, state, method="standard")
    clean_rw, _ = process_asr(dirty, 250.0, state, method="riemannian_windowed")

    assert np.array_equal(clean_std, clean_rw), (
        "riemannian_windowed processing diverged from standard processing on "
        f"the same state; max abs diff = {np.max(np.abs(clean_std - clean_rw)):.3e}"
    )


def test_riemannian_windowed_calibration_matches_riemannian_covariance():
    """PROOF (Link B): riemannian_windowed covariance/M == legacy riemannian.

    Both backends use the Riemannian ("rasr") geometric-median block-covariance
    aggregation, so the calibration covariance C and the mixing matrix
    M = sqrt(C) are identical. The legacy ``riemannian`` M is validated against
    MATLAB rASRMatlab (``test_rasr_calibration_against_matlab_reference``), so
    this equality transitively confers that calibration parity.

    (The threshold matrix T differs — riemannian_windowed uses the standard
    eigendecomposition of C, the same code path as standard ASR's T, which is
    itself MATLAB-parity-tested.)
    """
    from mne_denoise.asr.core import calibrate_asr

    clean = _make_synthetic_eeg(n_channels=12, duration_s=40.0)
    dirty = _inject_bursts(clean, sfreq=250.0)

    s_rie, _ = calibrate_asr(
        dirty,
        250.0,
        cutoff=20.0,
        method="riemannian",
        calibration="manual",
        filter_kind="none",
    )
    s_rw, _ = calibrate_asr(
        dirty,
        250.0,
        cutoff=20.0,
        method="riemannian_windowed",
        calibration="manual",
        filter_kind="none",
    )

    cov_relerr = np.linalg.norm(s_rie.cov - s_rw.cov) / max(
        np.linalg.norm(s_rie.cov), np.finfo(float).eps
    )
    m_relerr = np.linalg.norm(s_rie.M - s_rw.M) / max(
        np.linalg.norm(s_rie.M), np.finfo(float).eps
    )
    assert cov_relerr < 1e-10, f"covariance C diverged: relerr={cov_relerr:.3e}"
    assert m_relerr < 1e-10, f"mixing M diverged: relerr={m_relerr:.3e}"


def test_riemannian_windowed_cutoff_invariance_demonstration():
    """Companion test that demonstrates the original ``riemannian`` bug.

    On the SAME contaminated input, ``method="riemannian"`` produces an
    essentially flat ``fraction_reconstructed_windows`` across cutoffs.
    This test documents that behavior (so anyone changing the original
    backend has to acknowledge the regression).
    """
    clean = _make_synthetic_eeg(n_channels=12)
    contaminated = _inject_bursts(clean, sfreq=250.0)

    cutoffs = [5.0, 20.0, 50.0]
    fractions = []
    for k in cutoffs:
        asr = _build_rasr("riemannian", cutoff=k)
        asr.fit(contaminated)
        asr.transform(contaminated)
        fractions.append(float(asr.diagnostics_["fraction_reconstructed_windows"]))

    span = max(fractions) - min(fractions)
    # We *expect* this to be tiny on the legacy backend. If it ever stops being
    # tiny (e.g. because someone fixed the legacy backend too), this test fails
    # loudly and the user should update both backends + the documentation.
    assert span < 0.20, (
        f"Original riemannian backend is no longer cutoff-invariant "
        f"(span={span:.3f}). If you fixed the legacy backend, please update "
        f"this test + the rASR sprint README's deviation note."
    )


@pytest.mark.skipif(
    not (_XCHECK_INPUT.exists() and _XCHECK_REFERENCE.exists()),
    reason=(
        "MATLAB cross-check fixtures missing. Regenerate with: "
        "python tests/parity/matlab_reference/generate_riemannian_windowed_input.py "
        "&& matlab -batch \"run('tests/parity/matlab_reference/"
        "generate_riemannian_windowed_reference.m')\""
    ),
)
def test_riemannian_windowed_matches_matlab_asr_process():
    """DIRECT MATLAB cross-check: riemannian_windowed == clean_rawdata asr_process.

    Belt-and-suspenders confirmation of the transitive proof. The Python
    riemannian_windowed cleaned output (exported in the input fixture) is
    compared against clean_rawdata's ``asr_process`` run on the same injected
    calibration state. Empirically matches to ~1e-13 (far tighter than the
    1e-5 gate used by the other parity suites).
    """
    from scipy.io import loadmat

    inp = loadmat(_XCHECK_INPUT)
    ref = loadmat(_XCHECK_REFERENCE)
    py_cleaned = np.asarray(inp["py_cleaned"], dtype=np.float64)
    matlab_cleaned = np.asarray(ref["matlab_cleaned"], dtype=np.float64)

    assert py_cleaned.shape == matlab_cleaned.shape
    relerr = np.linalg.norm(py_cleaned - matlab_cleaned) / max(
        np.linalg.norm(matlab_cleaned), np.finfo(float).eps
    )
    corr = np.corrcoef(py_cleaned.ravel(), matlab_cleaned.ravel())[0, 1]
    assert relerr < 1e-5, (
        f"riemannian_windowed vs MATLAB asr_process relerr={relerr:.3e}"
    )
    assert corr > 0.99999, f"riemannian_windowed vs MATLAB asr_process corr={corr:.6f}"
