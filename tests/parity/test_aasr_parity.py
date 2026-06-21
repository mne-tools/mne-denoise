"""Adaptive ASR parity tests against the local MATLAB AASR reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from mne_denoise.asr import AdaptiveASR

MATLAB_REF_DIR = Path(__file__).parent / "matlab_reference"


def _reference_params():
    paths = sorted(MATLAB_REF_DIR.glob("aasr_case_reference_*.mat"))
    if paths:
        return [
            pytest.param(path, id=path.stem.replace("aasr_case_reference_", ""))
            for path in paths
        ]
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason=(
                    "Missing adaptive-ASR MATLAB reference artifacts. Generate them with "
                    "tests/parity/matlab_reference/generate_aasr_input.py and "
                    "generate_aasr_reference.m."
                )
            ),
            id="missing",
        )
    ]


AASR_CASES = _reference_params()


def _load_reference(path: Path | None) -> dict:
    if path is None:
        pytest.skip("Missing MATLAB adaptive-ASR reference artifact")
    return loadmat(path, squeeze_me=True)


def _array(ref: dict, key: str) -> np.ndarray:
    return np.asarray(ref[key], dtype=np.float64)


def _scalar(ref: dict, key: str) -> float:
    return float(np.asarray(ref[key]).squeeze())


def _str(ref: dict, key: str) -> str:
    value = ref[key]
    if isinstance(value, str):
        return value
    return str(np.asarray(value).squeeze())


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).eps))


def _threshold_error(asr: AdaptiveASR, ref: dict) -> tuple[float, float, float]:
    mat_t = _array(ref, "T")
    signs = np.sign(np.sum(asr.T_ * mat_t, axis=1))
    signs[signs == 0] = 1.0
    t_error = _relative_error(asr.T_ * signs[:, np.newaxis], mat_t)
    threshold_error = _relative_error(
        asr.thresholds_.ravel(),
        _array(ref, "thresholds").ravel(),
    )
    threshold_corr = np.corrcoef(
        asr.thresholds_.ravel(),
        _array(ref, "thresholds").ravel(),
    )[0, 1]
    return t_error, threshold_error, float(threshold_corr)


@pytest.mark.parametrize("ref_path", AASR_CASES)
def test_aasr_reference_schema(ref_path: Path | None):
    """Each adaptive-ASR reference artifact contains the required fields."""
    ref = _load_reference(ref_path)
    required = {
        "case_name",
        "variant",
        "n_updates",
        "data",
        "sfreq",
        "cutoff",
        "update_chunk_1",
        "M",
        "T",
        "thresholds",
        "cleaned",
    }
    missing = sorted(required - set(ref))
    assert missing == []
    assert ref["data"].ndim == 2
    assert ref["M"].shape[0] == ref["M"].shape[1]
    assert ref["T"].shape == ref["M"].shape
    assert ref["cleaned"].shape == ref["data"].shape


@pytest.mark.parametrize("ref_path", AASR_CASES)
def test_aasr_against_matlab_reference(ref_path: Path | None):
    """Adaptive ASR PSP/PSW match the MATLAB AASR reference outputs."""
    ref = _load_reference(ref_path)
    variant = _str(ref, "variant")
    n_updates = int(round(_scalar(ref, "n_updates")))
    asr = AdaptiveASR(
        sfreq=_scalar(ref, "sfreq"),
        cutoff=_scalar(ref, "cutoff"),
        variant=variant,
        verbose=False,
    )
    asr.fit(_array(ref, "update_chunk_1"))
    for update_idx in range(2, n_updates + 1):
        asr.partial_fit(_array(ref, f"update_chunk_{update_idx}"))

    asr.reset_process_state()
    py_cleaned = asr.transform(_array(ref, "data"))

    m_error = _relative_error(asr.M_, _array(ref, "M"))
    t_error, threshold_error, threshold_corr = _threshold_error(asr, ref)
    relerr = _relative_error(py_cleaned, _array(ref, "cleaned"))
    corr = np.corrcoef(py_cleaned.ravel(), _array(ref, "cleaned").ravel())[0, 1]

    print(f"AASR parity [{ref_path.stem}]: M relerr={m_error:.4g}")
    print(f"AASR parity [{ref_path.stem}]: T relerr={t_error:.4g}")
    print(
        f"AASR parity [{ref_path.stem}]: "
        f"threshold relerr={threshold_error:.4g} corr={threshold_corr:.4g}"
    )
    print(f"AASR parity [{ref_path.stem}]: cleaned relerr={relerr:.4g}")
    print(f"AASR parity [{ref_path.stem}]: cleaned corr={corr:.4g}")

    assert asr.calibration_info_["adaptive_variant"] == variant
    assert asr.diagnostics_["adaptive_variant"] == variant
    assert m_error < 1e-5
    assert t_error < 1e-4
    assert threshold_error < 1e-4
    assert threshold_corr > 0.99999
    assert relerr < 1e-5
    assert corr > 0.99999
