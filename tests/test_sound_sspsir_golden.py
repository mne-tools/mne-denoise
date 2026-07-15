"""Golden-value regression tests for SOUND and SSP-SIR.

The ``compute_sound`` and ``compute_sspsir_operator`` outputs were verified to
match the original Mutanen MATLAB reference (``SOUND.m`` / ``DDWiener.m`` and the
SSP-SIR core of ``tesa_sspsir.m``) to machine precision (~1e-16) by running the
reference code under GNU Octave on identical inputs. Octave is not available in
CI, so these pinned golden values lock that validated behaviour in and guard
against numerical regressions.
"""

import numpy as np

from mne_denoise.sound.core import compute_sound
from mne_denoise.sspsir.core import compute_sspsir_operator

# Reference values produced by the parity-validated implementation.
SOUND_SIGMAS = [
    0.8074480171,
    0.7525807024,
    0.7844724296,
    0.6217834583,
    0.6124545364,
    0.6382410109,
    0.6087057907,
    0.6213894433,
    0.5076292126,
    0.7560042278,
    0.5397326800,
    0.7680682941,
]
SOUND_OP_FRO = 2.9322038764
SSPSIR_OP_FRO = 2.4834163967
SSPSIR_ROW0 = [
    0.4471950881,
    -0.0297988324,
    -0.0048990576,
    -0.1766279907,
    -0.1575304654,
    -0.1815472674,
    -0.1585807087,
    0.1222171822,
    0.1566610179,
    -0.1229042232,
    -0.0554524910,
    0.2358425729,
]


def _fixed_inputs():
    """Deterministic (data, lead field, rng) matching the golden-value run."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_src = 12, 60, 40
    leadfield = rng.standard_normal((n_ch, n_src))
    leadfield -= leadfield.mean(0, keepdims=True)
    data = rng.standard_normal((n_ch, n_times))
    data -= data.mean(0, keepdims=True)
    return data, leadfield, rng


def test_sound_golden_values():
    """SOUND noise estimate + operator match the validated reference."""
    data, leadfield, _ = _fixed_inputs()
    operator, sigmas, _ = compute_sound(
        data, leadfield, lambda_=0.1, n_iter=12, random_state=0
    )
    np.testing.assert_allclose(sigmas, SOUND_SIGMAS, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(operator), SOUND_OP_FRO, rtol=1e-5)


def test_sspsir_golden_values():
    """SSP-SIR operator matches the validated reference."""
    _, leadfield, rng = _fixed_inputs()
    artifact = np.linalg.svd(rng.standard_normal((12, 12)))[0][:, :3]
    operator = compute_sspsir_operator(leadfield, artifact, M=6)
    np.testing.assert_allclose(np.linalg.norm(operator), SSPSIR_OP_FRO, rtol=1e-5)
    np.testing.assert_allclose(operator[0], SSPSIR_ROW0, rtol=1e-5, atol=1e-7)

