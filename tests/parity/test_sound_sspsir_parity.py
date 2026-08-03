"""Parity tests for SOUND and SSP-SIR against the reference MATLAB.

Two layers:

``TestTransliterationParity``
    Runs everywhere, including CI. Compares against
    ``matlab_reference/sound_sspsir_reference.py``, a statement-by-statement
    port of ``SOUND.m`` / ``DDWiener.m`` / ``tesa_sound.m`` / ``tesa_sspsir.m``
    that keeps the reference's source-space formulation. The package computes
    the same quantities through folded channel-space products, so agreement is
    evidence about the algebra rather than about shared code.

``TestMatlabEngineParity``
    Runs only when the MATLAB Engine is installed *and* the reference toolboxes
    are on the MATLAB path (Mutanen's Sound-Demo-Package and EEGLAB's TESA;
    neither can be vendored here for licensing reasons). This is the ground
    truth the transliteration itself is answerable to.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from mne_denoise.sound import compute_sound, compute_sound_ref_best
from mne_denoise.sound.core import _ddwiener
from mne_denoise.sspsir import SSPSIR, compute_sir, compute_sspsir

from .matlab_reference import sound_sspsir_reference as ref

HAS_MATLAB = bool(
    importlib.util.find_spec("matlab") and importlib.util.find_spec("matlab.engine")
)

LAMBDA, N_ITER, SEED = 0.1, 6, 42
N_CHANNELS, N_TIMES, N_SOURCES = 20, 500, 300
SFREQ = 1000.0


@pytest.fixture(scope="module")
def fixtures():
    """Shared data, lead field, and the channel order SOUND will visit."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_CHANNELS, N_TIMES))
    data -= data.mean(axis=0, keepdims=True)
    leadfield = rng.standard_normal((N_CHANNELS, N_SOURCES))
    leadfield -= leadfield.mean(axis=0, keepdims=True)
    # compute_sound draws its permutations from default_rng(random_state); the
    # transliteration is handed the same sequence so the stochastic update
    # order matches.
    order_rng = np.random.default_rng(SEED)
    orders = [order_rng.permutation(N_CHANNELS) for _ in range(N_ITER)]
    # ref_best drops one channel, so its inner solver permutes n - 1 indices.
    drop_rng = np.random.default_rng(SEED)
    orders_drop = [drop_rng.permutation(N_CHANNELS - 1) for _ in range(N_ITER)]
    return {
        "data": data,
        "leadfield": leadfield,
        "orders": orders,
        "orders_drop": orders_drop,
    }


def rel_diff(actual, expected):
    return np.max(np.abs(actual - expected)) / np.max(np.abs(expected))


class TestTransliterationParity:
    """Package vs a literal port of the reference MATLAB."""

    def test_ddwiener(self, fixtures):
        _, expected = ref.ddwiener(fixtures["data"])
        assert rel_diff(_ddwiener(fixtures["data"]), expected) < 1e-12

    def test_sound_sigmas_and_corrected_data(self, fixtures):
        data, leadfield = fixtures["data"], fixtures["leadfield"]
        expected_data, expected_sigmas, expected_conv = ref.sound(
            data, leadfield, N_ITER, LAMBDA, fixtures["orders"]
        )
        operator, sigmas, convergence = compute_sound(
            data, leadfield, lambda_=LAMBDA, n_iter=N_ITER, random_state=SEED
        )
        assert rel_diff(sigmas, expected_sigmas) < 1e-12
        assert rel_diff(convergence, expected_conv) < 1e-10
        assert rel_diff(operator @ data, expected_data) < 1e-10

    def test_sound_ref_best_matches_tesa_sound(self, fixtures):
        """The folded ref_best operator equals tesa_sound's explicit pipeline."""
        data, leadfield = fixtures["data"], fixtures["leadfield"]
        expected_data, expected_sigmas, expected_best = ref.tesa_sound(
            data, leadfield, N_ITER, LAMBDA, fixtures["orders_drop"]
        )
        operator, sigmas, _, best = compute_sound_ref_best(
            data, leadfield, lambda_=LAMBDA, n_iter=N_ITER, random_state=SEED
        )
        assert best == expected_best
        assert rel_diff(sigmas, expected_sigmas) < 1e-12
        assert rel_diff(operator @ data, expected_data) < 1e-10

    def test_sspsir_suppressed_branch(self, fixtures):
        data, leadfield = fixtures["data"], fixtures["leadfield"]
        topos = np.linalg.svd(data)[0][:, :4]
        expected = ref.ssp_sir(data, leadfield, topos, M=12, filt_ker=None)
        operator = compute_sspsir(leadfield, topos, M=12)
        assert rel_diff(operator @ data, expected) < 1e-10

    def test_sspsir_unprojected_branch(self, fixtures):
        """orig_data_SIR: the branch that was missing before the crossfade."""
        data, leadfield = fixtures["data"], fixtures["leadfield"]
        llt = leadfield @ leadfield.T
        expected = llt @ ref._truncated_pinv(llt, 12) @ data
        assert rel_diff(compute_sir(leadfield, 12) @ data, expected) < 1e-10

    def test_sspsir_full_crossfade(self, fixtures):
        """The blended output matches tesa_sspsir's data_correct."""
        data, leadfield = fixtures["data"], fixtures["leadfield"]
        topos = np.linalg.svd(data)[0][:, :4]
        times = np.arange(N_TIMES) / SFREQ
        kernel = ref.manual_filt_ker(times, 0.005, 0.050, 0.010)
        expected = ref.ssp_sir(data, leadfield, topos, M=12, filt_ker=kernel)
        ours = kernel * (compute_sspsir(leadfield, topos, M=12) @ data) + (
            1.0 - kernel
        ) * (compute_sir(leadfield, 12) @ data)
        assert rel_diff(ours, expected) < 1e-10

    def test_automatic_kernel_matches_reference(self):
        """The sliding-RMS crossfade kernel matches TESA's conv-based one.

        The package uses ``uniform_filter1d(..., mode='nearest')`` where TESA
        uses ``conv(..., 'same')`` (zero padded), so the two differ within half
        a window of each edge. Compare the interior.
        """
        rng = np.random.default_rng(3)
        evoked = rng.standard_normal((N_CHANNELS, N_TIMES))
        evoked -= evoked.mean(axis=0, keepdims=True)
        b, a = butter(2, 100.0 / (SFREQ / 2.0), btype="high")
        data_high = filtfilt(b, a, evoked, axis=1)

        estimator = SSPSIR(n_components=2, sfreq=SFREQ)
        _, ours = estimator._svd_input(evoked, SFREQ, None)
        expected = ref.automatic_filt_ker(data_high, SFREQ)

        edge = int(round(SFREQ / 1000.0 * 50.0))
        interior = slice(edge, -edge)
        assert rel_diff(ours[interior], expected[interior]) < 1e-10

    def test_manual_kernel_matches_reference(self):
        times = (np.arange(N_TIMES) - 100) / SFREQ
        estimator = SSPSIR(
            n_components=2, sfreq=SFREQ, art_window=(0.005, 0.050), smooth_length=0.010
        )
        rng = np.random.default_rng(4)
        evoked = rng.standard_normal((N_CHANNELS, N_TIMES))
        _, ours = estimator._svd_input(evoked, SFREQ, times)
        expected = ref.manual_filt_ker(times, 0.005, 0.050, 0.010)
        assert rel_diff(ours, expected) < 1e-12


@pytest.mark.skipif(not HAS_MATLAB, reason="MATLAB Engine not available")
class TestMatlabEngineParity:
    """Package vs the actual reference ``.m`` files, via the MATLAB Engine.

    Requires the Sound-Demo-Package and TESA on the MATLAB path. Neither is
    vendored here: the Sound-Demo-Package is licensed for academic use only and
    TESA is GPL, so both must be installed by whoever runs these.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        import matlab.engine

        eng = matlab.engine.start_matlab()
        try:
            extra = Path(__file__).parent / "matlab_reference"
            eng.addpath(str(extra), nargout=0)
            if not eng.exist("SOUND", nargout=1) or not eng.exist(
                "DDWiener", nargout=1
            ):
                pytest.skip(
                    "SOUND.m / DDWiener.m not on the MATLAB path; install "
                    "https://github.com/tuomasmutanen/Sound-Demo-Package"
                )
            yield eng
        finally:
            eng.quit()

    def test_ddwiener_matches_matlab(self, engine, fixtures):
        import matlab

        data = fixtures["data"]
        _, sigmas_mat = engine.DDWiener(matlab.double(data.tolist()), nargout=2)
        expected = np.asarray(sigmas_mat).ravel()
        assert rel_diff(_ddwiener(data), expected) < 1e-10

    def test_transliteration_matches_matlab(self, engine, fixtures):
        """Pin the CI-side transliteration to the real MATLAB it stands in for."""
        import matlab

        data = fixtures["data"]
        _, sigmas_mat = engine.DDWiener(matlab.double(data.tolist()), nargout=2)
        _, sigmas_ref = ref.ddwiener(data)
        assert rel_diff(sigmas_ref, np.asarray(sigmas_mat).ravel()) < 1e-10
