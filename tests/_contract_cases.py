"""Small registry of representative public estimator capabilities.

The registry is intentionally capability-based.  A case is included in a
contract group only when that contract is part of the estimator's public
semantics; it is not a promise that every estimator has the same interface or
output shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mne_denoise.asr import ASR, AdaptiveASR, GuidedASR, JugglerASR
from mne_denoise.bss_cca import BSSCCA
from mne_denoise.dss import DSS
from mne_denoise.dss.variants import TimeShiftDSS
from mne_denoise.icanclean import ICanClean
from mne_denoise.sns import SNS
from mne_denoise.sound import SOUND
from mne_denoise.spectrum_interpolation import SpectrumInterpolation
from mne_denoise.ssa import LocalSingularSpectrumAnalysis, SingularSpectrumAnalysis
from mne_denoise.sspsir import SSPSIR
from mne_denoise.zapline import ZapLine

CLONEABLE = "cloneable"
FIT_RETURNS_SELF = "fit_returns_self"
FIT_TRANSFORM_COMPOSES = "fit_transform_composes"
NOT_FITTED = "not_fitted"
NUMPY_NO_MUTATION = "numpy_no_mutation"
NUMPY_LAYOUT = "numpy_layout"
MNE_RAW = "mne_raw"
MNE_EPOCHS = "mne_epochs"
MNE_EVOKED = "mne_evoked"
FITTED_CHANNEL_COUNT = "fitted_channel_count"
FITTED_CHANNEL_ORDER = "fitted_channel_order"
FITTED_CHANNEL_NAMES = "fitted_channel_names"
SFREQ_AWARE = "sfreq_aware"
CALLBACK_TRANSPARENT = "callback_transparent"


EstimatorFactory = Callable[[], Any]
MNEEstimatorFactory = Callable[[tuple[str, ...]], Any]


@dataclass(frozen=True)
class EstimatorCase:
    """One public estimator and the contracts it is eligible to exercise."""

    name: str
    make_estimator: EstimatorFactory
    make_array: Callable[[], np.ndarray] | None
    capabilities: frozenset[str]
    mne_factory: MNEEstimatorFactory | None = None
    callback_factory: EstimatorFactory | None = None


def _identity_bias(data: np.ndarray) -> np.ndarray:
    """Use a deterministic, public-callable DSS bias in contract tests."""
    return data


def _make_array() -> np.ndarray:
    """Return deterministic multichannel data with full numerical support."""
    rng = np.random.default_rng(20260828)
    n_channels, n_times = 6, 400
    time = np.arange(n_times) / 200.0
    sources = np.vstack(
        (
            np.sin(2 * np.pi * 8.0 * time),
            np.sin(2 * np.pi * 17.0 * time + 0.3),
            rng.standard_normal(n_times),
        )
    )
    mixing = rng.standard_normal((n_channels, sources.shape[0]))
    return mixing @ sources + 0.05 * rng.standard_normal((n_channels, n_times))


def _make_bsscca() -> BSSCCA:
    return BSSCCA(n_remove=1, lag_samples=1, verbose=False)


def _make_bsscca_callback() -> BSSCCA:
    return BSSCCA(
        n_remove=1,
        lag_samples=1,
        sfreq=200.0,
        segment_len=1.0,
        verbose=False,
    )


def _make_sns() -> SNS:
    return SNS(n_neighbors=2, verbose=False)


def _make_spectrum_interpolation() -> SpectrumInterpolation:
    return SpectrumInterpolation(
        sfreq=200.0,
        line_freq=50.0,
        n_harmonics=1,
        verbose=False,
    )


def _make_dss() -> DSS:
    return DSS(
        bias=_identity_bias,
        n_components=3,
        n_select=1,
        component_action="subtract",
        normalize_input=False,
        verbose=False,
    )


def _make_icanclean() -> ICanClean:
    return ICanClean(
        sfreq=200.0,
        primary_channels=[0, 1, 2, 3],
        ref_channels=[4, 5],
        segment_len=1.0,
        verbose=False,
    )


def _make_icanclean_mne(names: tuple[str, ...]) -> ICanClean:
    return ICanClean(
        sfreq=200.0,
        primary_channels=list(names[:4]),
        ref_channels=[names[4]],
        segment_len=1.0,
        verbose=False,
    )


def _make_ssa() -> SingularSpectrumAnalysis:
    return SingularSpectrumAnalysis(sfreq=200.0, window_length=20, verbose=False)


def _make_local_ssa() -> LocalSingularSpectrumAnalysis:
    return LocalSingularSpectrumAnalysis(
        sfreq=200.0,
        window_length=20,
        n_clusters=2,
        random_state=0,
        verbose=False,
    )


def _make_zapline() -> ZapLine:
    return ZapLine(
        sfreq=200.0,
        line_freq=50.0,
        n_select=1,
        n_harmonics=1,
        nfft=200,
        verbose=False,
    )


def _make_asr() -> ASR:
    return ASR(
        sfreq=200.0,
        calibration="manual",
        filter_kind="none",
        calibration_window_length=1.0,
        window_length=0.5,
        picks=None,
        verbose=False,
    )


def _make_guided_asr() -> GuidedASR:
    return GuidedASR(
        sfreq=200.0,
        calibration="manual",
        filter_kind="none",
        calibration_window_length=1.0,
        window_length=0.5,
        picks=None,
        reconstruction="hard",
        verbose=False,
    )


def _make_adaptive_asr() -> AdaptiveASR:
    return AdaptiveASR(
        sfreq=200.0,
        variant="psw",
        calibration_window_length=1.0,
        window_length=0.5,
        update_window_length=0.1,
        picks=None,
        verbose=False,
    )


def _make_juggler_asr() -> JugglerASR:
    return JugglerASR(
        sfreq=200.0,
        calibration_window_length=1.0,
        window_length=0.5,
        picks=None,
        verbose=False,
    )


def _make_sound() -> SOUND:
    """SOUND is registered for cloneability without inventing a forward model."""
    return SOUND(n_iter=1, random_state=0, verbose=False)


def _make_sspsir() -> SSPSIR:
    """SSP-SIR is registered for cloneability without a forward model."""
    return SSPSIR(n_components=1, blend="constant", verbose=False)


def _make_time_shift_dss() -> TimeShiftDSS:
    """Time-shift DSS has a deliberate, variant-specific fit layout."""
    return TimeShiftDSS(
        lag_samples=[1],
        n_components=2,
        rank=4,
        n_select=1,
        component_action="subtract",
        verbose=False,
    )


# These estimators expose a two-dimensional NumPy path; their MNE Epochs paths
# carry the repeated-epoch layout instead of accepting 3-D arrays directly.
_NUMPY_NO_MUTATION_ONLY = frozenset({NUMPY_NO_MUTATION})
_SAME_SHAPE = frozenset({NUMPY_NO_MUTATION, NUMPY_LAYOUT})
_ALL_MNE = frozenset({MNE_RAW, MNE_EPOCHS, MNE_EVOKED})
_RAW_EPOCHS = frozenset({MNE_RAW, MNE_EPOCHS})


# DSS and ZapLine retain their public RuntimeError pre-fit behavior rather
# than being forced into the sklearn NotFittedError group.
ESTIMATOR_CASES = (
    EstimatorCase(
        "bss_cca",
        _make_bsscca,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                NOT_FITTED,
                *_SAME_SHAPE,
                *_ALL_MNE,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
                CALLBACK_TRANSPARENT,
            }
        ),
        callback_factory=_make_bsscca_callback,
    ),
    EstimatorCase(
        "sns",
        _make_sns,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                NOT_FITTED,
                *_SAME_SHAPE,
                *_ALL_MNE,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                CALLBACK_TRANSPARENT,
            }
        ),
        callback_factory=_make_sns,
    ),
    EstimatorCase(
        "spectrum_interpolation",
        _make_spectrum_interpolation,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                NOT_FITTED,
                *_SAME_SHAPE,
                *_ALL_MNE,
                SFREQ_AWARE,
            }
        ),
    ),
    EstimatorCase(
        "dss_subtract",
        _make_dss,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                *_SAME_SHAPE,
                *_ALL_MNE,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_NAMES,
            }
        ),
    ),
    EstimatorCase(
        "icanclean",
        _make_icanclean,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                *_NUMPY_NO_MUTATION_ONLY,
                *_ALL_MNE,
                FITTED_CHANNEL_NAMES,
            }
        ),
        mne_factory=_make_icanclean_mne,
    ),
    EstimatorCase(
        "basic_ssa",
        _make_ssa,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                NOT_FITTED,
                *_SAME_SHAPE,
                *_ALL_MNE,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
            }
        ),
    ),
    EstimatorCase(
        "local_ssa",
        _make_local_ssa,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                NOT_FITTED,
                *_SAME_SHAPE,
                *_ALL_MNE,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
            }
        ),
    ),
    EstimatorCase(
        "zapline",
        _make_zapline,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                *_SAME_SHAPE,
                *_ALL_MNE,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_NAMES,
            }
        ),
    ),
    EstimatorCase(
        "asr",
        _make_asr,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                *_NUMPY_NO_MUTATION_ONLY,
                *_RAW_EPOCHS,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
            }
        ),
    ),
    EstimatorCase(
        "guided_asr",
        _make_guided_asr,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                *_NUMPY_NO_MUTATION_ONLY,
                *_RAW_EPOCHS,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
            }
        ),
    ),
    # AdaptiveASR has fit_transform modes that deliberately calibrate and
    # transform per window, so it is not in the ordinary composition group.
    EstimatorCase(
        "adaptive_asr_psw",
        _make_adaptive_asr,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                *_NUMPY_NO_MUTATION_ONLY,
                *_RAW_EPOCHS,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
            }
        ),
    ),
    EstimatorCase(
        "juggler_asr",
        _make_juggler_asr,
        _make_array,
        frozenset(
            {
                CLONEABLE,
                FIT_RETURNS_SELF,
                FIT_TRANSFORM_COMPOSES,
                *_NUMPY_NO_MUTATION_ONLY,
                *_RAW_EPOCHS,
                FITTED_CHANNEL_COUNT,
                FITTED_CHANNEL_ORDER,
                FITTED_CHANNEL_NAMES,
                SFREQ_AWARE,
            }
        ),
    ),
    EstimatorCase("sound", _make_sound, None, frozenset({CLONEABLE})),
    EstimatorCase("sspsir", _make_sspsir, None, frozenset({CLONEABLE})),
    EstimatorCase("time_shift_dss", _make_time_shift_dss, None, frozenset({CLONEABLE})),
)


CALLBACK_CASES = tuple(
    case for case in ESTIMATOR_CASES if CALLBACK_TRANSPARENT in case.capabilities
)
