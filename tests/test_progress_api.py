"""Cross-method tests for the structured progress callback surface."""

from __future__ import annotations

import inspect

import numpy as np
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
from mne_denoise.progress import ProgressEvent
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
from tests._contract_cases import CALLBACK_CASES

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


_LEGACY_CALLBACK_APIS = {"calibrate_asr", "compute_icanclean"}


def test_callback_aware_public_signatures_are_consistent():
    """Callback-aware operations expose keyword-only runtime observer state."""
    for name, operation in CALLBACK_APIS:
        parameters = inspect.signature(operation).parameters
        assert "callback" in parameters, f"{name}: callback is missing"
        if name not in _LEGACY_CALLBACK_APIS:
            assert parameters["callback"].kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name}: callback must be keyword-only"
            )

    for estimator in CALLBACK_ESTIMATORS:
        assert "callback" not in inspect.signature(estimator.__init__).parameters, (
            f"{estimator.__name__}: callback must not be an estimator parameter"
        )


def test_callback_free_public_signatures_are_intentional():
    """Deliberate silent operations do not grow a meaningless callback API."""
    for name, operation in CALLBACK_FREE_APIS:
        assert "callback" not in inspect.signature(operation).parameters, (
            f"{name}: callback is not allowed on this operation"
        )


@pytest.mark.parametrize("case", CALLBACK_CASES, ids=lambda case: case.name)
def test_representative_public_callback_contracts(case):
    """Representative estimators cover callback transparency and failures."""
    factory = case.callback_factory
    assert factory is not None, f"{case.name}: callback factory is missing"
    data = case.make_array()

    expected = factory().fit_transform(data)
    events = []
    actual = factory().fit_transform(data, callback=events.append)
    np.testing.assert_allclose(
        actual,
        expected,
        err_msg=f"{case.name}: callback changed the numerical result",
    )
    assert events, f"{case.name}: callback received no progress events"
    assert all(isinstance(event, ProgressEvent) for event in events), (
        f"{case.name}: callback received a non-ProgressEvent value"
    )

    error = RuntimeError(f"{case.name} callback failed")

    def callback(_event):
        raise error

    with pytest.raises(RuntimeError) as caught:
        factory().fit_transform(data, callback=callback)
    assert caught.value is error, f"{case.name}: callback exception was replaced"

    with pytest.raises(TypeError, match="callback must be callable or None"):
        factory().fit_transform(data, callback=1)
