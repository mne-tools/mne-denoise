"""Cross-method tests for the structured progress callback surface."""

from __future__ import annotations

import inspect

import pytest

from mne_denoise.asr import (
    ASR,
    AdaptiveASR,
    GuidedASR,
    JugglerASR,
    calibrate_asr,
    process_asr,
    process_guided_asr,
    select_juggler_reference_samples,
)
from mne_denoise.bss_cca import BSSCCA, compute_bss_cca
from mne_denoise.dss import DSS, IterativeDSS, iterative_dss, iterative_dss_one
from mne_denoise.dss.variants import narrowband_scan
from mne_denoise.icanclean import ICanClean, compute_icanclean, null_r2_threshold
from mne_denoise.sns import SNS, compute_sns, compute_sns_weights
from mne_denoise.sound import SOUND, compute_sound, compute_sound_ref_best
from mne_denoise.ssa import (
    LocalSingularSpectrumAnalysis,
    SingularSpectrumAnalysis,
    compute_basic_ssa,
    compute_local_ssa,
    local_ssa_clean_channel,
    ssa_clean_channel,
    ssa_decompose,
    ssa_w_correlation,
)
from mne_denoise.zapline import ZapLine

CALLBACK_APIS = (
    ("compute_sound", compute_sound),
    ("compute_sound_ref_best", compute_sound_ref_best),
    ("iterative_dss_one", iterative_dss_one),
    ("iterative_dss", iterative_dss),
    ("calibrate_asr", calibrate_asr),
    ("process_asr", process_asr),
    ("process_guided_asr", process_guided_asr),
    ("narrowband_scan", narrowband_scan),
    ("compute_bss_cca", compute_bss_cca),
    ("compute_icanclean", compute_icanclean),
    ("compute_basic_ssa", compute_basic_ssa),
    ("compute_local_ssa", compute_local_ssa),
    ("compute_sns", compute_sns),
    ("compute_sns_weights", compute_sns_weights),
    ("SOUND.fit", SOUND.fit),
    ("SOUND.fit_transform", SOUND.fit_transform),
    ("IterativeDSS.fit", IterativeDSS.fit),
    ("IterativeDSS.fit_transform", IterativeDSS.fit_transform),
    ("ASR.fit", ASR.fit),
    ("ASR.transform", ASR.transform),
    ("ASR.fit_transform", ASR.fit_transform),
    ("GuidedASR.fit", GuidedASR.fit),
    ("GuidedASR.transform", GuidedASR.transform),
    ("GuidedASR.fit_transform", GuidedASR.fit_transform),
    ("JugglerASR.fit", JugglerASR.fit),
    ("JugglerASR.transform", JugglerASR.transform),
    ("JugglerASR.fit_transform", JugglerASR.fit_transform),
    ("AdaptiveASR.fit", AdaptiveASR.fit),
    ("AdaptiveASR.transform", AdaptiveASR.transform),
    ("AdaptiveASR.fit_transform", AdaptiveASR.fit_transform),
    ("DSS.fit_transform", DSS.fit_transform),
    ("ZapLine.fit_transform", ZapLine.fit_transform),
    ("BSSCCA.fit", BSSCCA.fit),
    ("BSSCCA.fit_transform", BSSCCA.fit_transform),
    ("ICanClean.transform", ICanClean.transform),
    ("ICanClean.fit_transform", ICanClean.fit_transform),
    ("SingularSpectrumAnalysis.transform", SingularSpectrumAnalysis.transform),
    (
        "SingularSpectrumAnalysis.fit_transform",
        SingularSpectrumAnalysis.fit_transform,
    ),
    (
        "LocalSingularSpectrumAnalysis.transform",
        LocalSingularSpectrumAnalysis.transform,
    ),
    (
        "LocalSingularSpectrumAnalysis.fit_transform",
        LocalSingularSpectrumAnalysis.fit_transform,
    ),
    ("SNS.fit", SNS.fit),
    ("SNS.fit_transform", SNS.fit_transform),
)


CALLBACK_FREE_APIS = (
    ("SOUND.transform", SOUND.transform),
    ("IterativeDSS.transform", IterativeDSS.transform),
    ("DSS.fit", DSS.fit),
    ("DSS.transform", DSS.transform),
    ("ZapLine.fit", ZapLine.fit),
    ("ZapLine.transform", ZapLine.transform),
    ("BSSCCA.transform", BSSCCA.transform),
    ("ICanClean.fit", ICanClean.fit),
    ("AdaptiveASR.partial_fit", AdaptiveASR.partial_fit),
    ("SNS.transform", SNS.transform),
    ("SingularSpectrumAnalysis.fit", SingularSpectrumAnalysis.fit),
    ("LocalSingularSpectrumAnalysis.fit", LocalSingularSpectrumAnalysis.fit),
    ("ssa_clean_channel", ssa_clean_channel),
    ("local_ssa_clean_channel", local_ssa_clean_channel),
    ("ssa_decompose", ssa_decompose),
    ("ssa_w_correlation", ssa_w_correlation),
    ("select_juggler_reference_samples", select_juggler_reference_samples),
    ("null_r2_threshold", null_r2_threshold),
)


CALLBACK_ESTIMATORS = (
    SOUND,
    IterativeDSS,
    ASR,
    GuidedASR,
    JugglerASR,
    AdaptiveASR,
    DSS,
    ZapLine,
    BSSCCA,
    ICanClean,
    SingularSpectrumAnalysis,
    LocalSingularSpectrumAnalysis,
    SNS,
)


@pytest.mark.parametrize("name, operation", CALLBACK_APIS)
def test_public_operations_expose_callback(name, operation):
    """Every audited callback-aware public operation names its callback."""
    assert operation is not None, name
    assert "callback" in inspect.signature(operation).parameters


@pytest.mark.parametrize("name, operation", CALLBACK_FREE_APIS)
def test_deliberate_callback_free_operations_omit_callback(name, operation):
    """Deliberate silent operations do not grow a meaningless callback API."""
    assert "callback" not in inspect.signature(operation).parameters, name


@pytest.mark.parametrize("estimator", CALLBACK_ESTIMATORS)
def test_callback_is_absent_from_every_estimator_constructor(estimator):
    """Callbacks are runtime observer state, not estimator parameters."""
    assert "callback" not in inspect.signature(estimator.__init__).parameters


_LEGACY_CALLBACK_APIS = {"calibrate_asr", "compute_icanclean"}


@pytest.mark.parametrize(
    "name, operation",
    [item for item in CALLBACK_APIS if item[0] not in _LEGACY_CALLBACK_APIS],
)
def test_modern_callback_parameters_are_keyword_only(name, operation):
    """Modern callback APIs expose callback after their keyword-only boundary."""
    parameter = inspect.signature(operation).parameters["callback"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
