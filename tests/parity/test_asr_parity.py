"""ASR parity tests against MATLAB clean_rawdata and rASRMatlab artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from mne_denoise.asr import ASR, calibrate_asr

MATLAB_REF_DIR = Path(__file__).parent / "matlab_reference"
LEGACY_STD_REF = MATLAB_REF_DIR / "asr_reference_results.mat"
LEGACY_RASR_REF = MATLAB_REF_DIR / "rasr_reference_results.mat"


def _reference_params(pattern: str, legacy_path: Path, prefix: str):
    paths = sorted(MATLAB_REF_DIR.glob(pattern))
    if paths:
        return [pytest.param(path, id=path.stem.replace(prefix, "")) for path in paths]
    if legacy_path.exists():
        return [pytest.param(legacy_path, id="legacy")]
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason=(
                    f"Missing MATLAB reference artifacts matching {pattern}. "
                    "Generate them with tests/parity/matlab_reference/"
                    "generate_asr_input.py plus the corresponding MATLAB scripts."
                )
            ),
            id="missing",
        )
    ]


STANDARD_CASES = _reference_params(
    "asr_case_reference_*.mat",
    LEGACY_STD_REF,
    "asr_case_reference_",
)
RASR_CASES = _reference_params(
    "rasr_case_reference_*.mat",
    LEGACY_RASR_REF,
    "rasr_case_reference_",
)


def _load_reference(path: Path | None) -> dict:
    if path is None:
        pytest.skip("Missing MATLAB reference artifact")
    return loadmat(path, squeeze_me=True)


def _scalar(ref: dict, key: str, default: float | None = None) -> float:
    if key not in ref:
        if default is None:
            raise KeyError(key)
        return float(default)
    return float(np.asarray(ref[key]).squeeze())


def _bool_scalar(ref: dict, key: str, default: bool = False) -> bool:
    if key not in ref:
        return bool(default)
    return bool(_scalar(ref, key))


def _array(ref: dict, key: str) -> np.ndarray:
    return np.asarray(ref[key], dtype=np.float64)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).eps))


def _threshold_error(state, ref: dict) -> tuple[float, float, float]:
    mat_t = _array(ref, "T")
    signs = np.sign(np.sum(state.T * mat_t, axis=1))
    signs[signs == 0] = 1.0
    t_error = _relative_error(state.T * signs[:, np.newaxis], mat_t)
    threshold_error = _relative_error(
        state.thresholds.ravel(),
        _array(ref, "thresholds").ravel(),
    )
    threshold_corr = np.corrcoef(
        state.thresholds.ravel(),
        _array(ref, "thresholds").ravel(),
    )[0, 1]
    return t_error, threshold_error, float(threshold_corr)


def _calibrate_against_reference(ref: dict, *, method: str):
    use_auto = _bool_scalar(ref, "use_auto_calibration", default=False)
    fit_input = _array(ref, "data") if use_auto else _array(ref, "calibration")
    kwargs = {
        "cutoff": _scalar(ref, "cutoff"),
        "window_length": _scalar(ref, "window_length"),
        "window_overlap": _scalar(ref, "window_overlap"),
        "calibration": "auto" if use_auto else "manual",
        "calibration_window_length": _scalar(ref, "ref_window_length", 1.0),
        "calibration_window_overlap": _scalar(ref, "window_overlap"),
        "ref_max_bad_channels": _scalar(ref, "ref_max_bad_channels", 0.075),
        "ref_tolerances": tuple(_array(ref, "ref_tolerances").ravel().tolist())
        if "ref_tolerances" in ref
        else (-np.inf, 5.5),
        "blocksize": int(round(_scalar(ref, "blocksize"))),
        "max_dropout_fraction": _scalar(ref, "max_dropout_fraction"),
        "min_clean_fraction": _scalar(ref, "min_clean_fraction"),
        "filter_kind": "none",
        "method": method,
    }
    return calibrate_asr(fit_input, _scalar(ref, "sfreq"), **kwargs)


@pytest.mark.parametrize("ref_path", STANDARD_CASES)
def test_asr_matlab_reference_schema(ref_path: Path | None):
    """Each standard ASR reference artifact contains the required fields."""
    ref = _load_reference(ref_path)
    required = {
        "data",
        "calibration",
        "sfreq",
        "cutoff",
        "blocksize",
        "window_length",
        "window_overlap",
        "max_dropout_fraction",
        "min_clean_fraction",
        "max_dims",
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


@pytest.mark.parametrize("ref_path", STANDARD_CASES)
def test_asr_calibration_against_matlab_reference(ref_path: Path | None):
    """Standard ASR calibration tracks MATLAB clean_rawdata across cases."""
    ref = _load_reference(ref_path)
    state, diagnostics = _calibrate_against_reference(ref, method="standard")

    m_error = _relative_error(state.M, _array(ref, "M"))
    t_error, threshold_error, threshold_corr = _threshold_error(state, ref)

    print(f"ASR calibration parity [{ref_path.stem}]: M relerr={m_error:.4g}")
    print(f"ASR calibration parity [{ref_path.stem}]: T relerr={t_error:.4g}")
    print(
        f"ASR calibration parity [{ref_path.stem}]: "
        f"threshold relerr={threshold_error:.4g} corr={threshold_corr:.4g}"
    )

    if _bool_scalar(ref, "use_auto_calibration", default=False):
        np.testing.assert_array_equal(
            np.asarray(diagnostics["clean_sample_mask"], dtype=bool),
            np.asarray(ref["reference_sample_mask"], dtype=bool).ravel(),
        )
        assert (
            diagnostics["calibration_samples"]
            == _array(ref, "calibration_used").shape[1]
        )

    assert m_error < 0.01
    assert t_error < 0.15
    assert threshold_error < 0.15
    assert threshold_corr > 0.99


@pytest.mark.parametrize("ref_path", STANDARD_CASES)
def test_asr_process_against_matlab_reference(ref_path: Path | None):
    """Standard ASR cleaned output matches MATLAB clean_rawdata across cases."""
    ref = _load_reference(ref_path)
    use_auto = _bool_scalar(ref, "use_auto_calibration", default=False)
    data = _array(ref, "data")
    sfreq = _scalar(ref, "sfreq")
    asr = ASR(
        sfreq=sfreq,
        cutoff=_scalar(ref, "cutoff"),
        window_length=_scalar(ref, "window_length"),
        window_overlap=_scalar(ref, "window_overlap"),
        calibration="auto" if use_auto else "manual",
        calibration_window_length=_scalar(ref, "ref_window_length", 1.0),
        calibration_window_overlap=_scalar(ref, "window_overlap"),
        ref_max_bad_channels=_scalar(ref, "ref_max_bad_channels", 0.075),
        ref_tolerances=tuple(_array(ref, "ref_tolerances").ravel().tolist())
        if "ref_tolerances" in ref
        else (-np.inf, 5.5),
        blocksize=int(round(_scalar(ref, "blocksize"))),
        max_dropout_fraction=_scalar(ref, "max_dropout_fraction"),
        min_clean_fraction=_scalar(ref, "min_clean_fraction"),
        max_dims=_scalar(ref, "max_dims"),
        filter_kind="none",
    )
    fit_input = data if use_auto else _array(ref, "calibration")
    asr.fit(fit_input)
    py_cleaned = asr.transform(data)

    mat_cleaned = _array(ref, "cleaned")
    relerr = _relative_error(py_cleaned, mat_cleaned)
    corr = np.corrcoef(py_cleaned.ravel(), mat_cleaned.ravel())[0, 1]

    print(f"ASR process parity [{ref_path.stem}]: cleaned relerr={relerr:.4g}")
    print(f"ASR process parity [{ref_path.stem}]: cleaned corr={corr:.4g}")

    assert relerr < 1e-5
    assert corr > 0.99999


@pytest.mark.parametrize("ref_path", RASR_CASES)
def test_rasr_calibration_against_matlab_reference(ref_path: Path | None):
    """Experimental rASR calibration tracks MATLAB rASRMatlab across cases."""
    ref = _load_reference(ref_path)
    state, diagnostics = _calibrate_against_reference(ref, method="riemannian")

    m_error = _relative_error(state.M, _array(ref, "M"))
    t_error, threshold_error, threshold_corr = _threshold_error(state, ref)

    print(f"rASR calibration parity [{ref_path.stem}]: M relerr={m_error:.4g}")
    print(f"rASR calibration parity [{ref_path.stem}]: T relerr={t_error:.4g}")
    print(
        f"rASR calibration parity [{ref_path.stem}]: "
        f"threshold relerr={threshold_error:.4g} corr={threshold_corr:.4g}"
    )

    if _bool_scalar(ref, "use_auto_calibration", default=False):
        np.testing.assert_array_equal(
            np.asarray(diagnostics["clean_sample_mask"], dtype=bool),
            np.asarray(ref["reference_sample_mask"], dtype=bool).ravel(),
        )
        assert (
            diagnostics["calibration_samples"]
            == _array(ref, "calibration_used").shape[1]
        )

    assert diagnostics["covariance_geometry"] == "riemannian"
    assert state.riemannian_solver == "nonlinear_eigenspace"
    # The nonlinear eigenspace basis is less numerically stable than standard ASR
    # across equivalent calibration solutions, so process parity is the stronger
    # oracle for the experimental backend.
    assert m_error < 1e-5
    assert t_error < 0.07
    assert threshold_error < 0.07
    assert threshold_corr > 0.98


@pytest.mark.parametrize("ref_path", RASR_CASES)
def test_rasr_process_against_matlab_reference(ref_path: Path | None):
    """Experimental rASR cleaned output matches MATLAB rASRMatlab across cases."""
    ref = _load_reference(ref_path)
    use_auto = _bool_scalar(ref, "use_auto_calibration", default=False)
    data = _array(ref, "data")
    sfreq = _scalar(ref, "sfreq")
    asr = ASR(
        sfreq=sfreq,
        cutoff=_scalar(ref, "cutoff"),
        window_length=_scalar(ref, "window_length"),
        window_overlap=_scalar(ref, "window_overlap"),
        calibration="auto" if use_auto else "manual",
        calibration_window_length=_scalar(ref, "ref_window_length", 1.0),
        calibration_window_overlap=_scalar(ref, "window_overlap"),
        ref_max_bad_channels=_scalar(ref, "ref_max_bad_channels", 0.075),
        ref_tolerances=tuple(_array(ref, "ref_tolerances").ravel().tolist())
        if "ref_tolerances" in ref
        else (-np.inf, 5.5),
        blocksize=int(round(_scalar(ref, "blocksize"))),
        max_dropout_fraction=_scalar(ref, "max_dropout_fraction"),
        min_clean_fraction=_scalar(ref, "min_clean_fraction"),
        max_dims=_scalar(ref, "max_dims"),
        filter_kind="none",
        method="riemannian",
        experimental=True,
    )
    fit_input = data if use_auto else _array(ref, "calibration")
    asr.fit(fit_input)
    py_cleaned = asr.transform(data)

    mat_cleaned = _array(ref, "cleaned")
    relerr = _relative_error(py_cleaned, mat_cleaned)
    corr = np.corrcoef(py_cleaned.ravel(), mat_cleaned.ravel())[0, 1]

    print(f"rASR process parity [{ref_path.stem}]: cleaned relerr={relerr:.4g}")
    print(f"rASR process parity [{ref_path.stem}]: cleaned corr={corr:.4g}")

    assert asr.diagnostics_["covariance_geometry"] == "riemannian"
    assert asr.diagnostics_["riemannian_solver"] == "nonlinear_eigenspace"
    assert relerr < 1e-5
    assert corr > 0.99999
