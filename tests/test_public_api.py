"""Tests for the intentional public API facades."""

import importlib

import mne_denoise

PUBLIC_FACADES = (
    "mne_denoise",
    "mne_denoise.asr",
    "mne_denoise.dss",
    "mne_denoise.dss.denoisers",
    "mne_denoise.dss.variants",
    "mne_denoise.dss.segmentation",
    "mne_denoise.dss.selection",
    "mne_denoise.icanclean",
    "mne_denoise.overcorrection",
    "mne_denoise.qa",
    "mne_denoise.sns",
    "mne_denoise.sound",
    "mne_denoise.spectrum_interpolation",
    "mne_denoise.ssa",
    "mne_denoise.sspsir",
    "mne_denoise.viz",
    "mne_denoise.zapline",
    "mne_denoise.bss_cca",
)


def test_public_facades_have_unique_existing_all():
    """Every declared public facade has a unique, existing ``__all__``."""
    for module_name in PUBLIC_FACADES:
        module = importlib.import_module(module_name)
        public_names = getattr(module, "__all__", None)
        if public_names is None:
            continue

        assert len(public_names) == len(set(public_names)), module_name
        for name in public_names:
            assert hasattr(module, name), f"{module_name}.{name} is missing"


def test_flattened_method_modules_expose_canonical_api():
    """The single-module denoisers expose their canonical public paths."""
    from mne_denoise.bss_cca import BSSCCA, compute_bss_cca
    from mne_denoise.icanclean import ICanClean, compute_icanclean
    from mne_denoise.sns import SNS, compute_sns, compute_sns_weights
    from mne_denoise.sound import SOUND, compute_sound, compute_sound_ref_best
    from mne_denoise.spectrum_interpolation import (
        SpectrumInterpolation,
        interpolate_spectrum,
    )
    from mne_denoise.sspsir import SSPSIR, compute_sir, compute_sspsir

    assert all(
        callable(item)
        for item in (
            BSSCCA,
            compute_bss_cca,
            ICanClean,
            compute_icanclean,
            SNS,
            compute_sns,
            compute_sns_weights,
            SOUND,
            compute_sound,
            compute_sound_ref_best,
            SpectrumInterpolation,
            interpolate_spectrum,
            SSPSIR,
            compute_sir,
            compute_sspsir,
        )
    )


def test_public_root_module_namespaces_and_facades():
    """Root imports expose the documented modules and utility facades."""
    from mne_denoise import compute_covariance, quantify_overcorrection
    from mne_denoise.qa import (
        compute_all_qa_metrics,
        peak_attenuation_db,
        suppression_ratio,
    )
    from mne_denoise.viz import plot_psd_comparison, set_theme

    assert mne_denoise.qa is not None
    assert mne_denoise.viz is not None
    assert all(
        callable(item)
        for item in (
            compute_covariance,
            quantify_overcorrection,
            peak_attenuation_db,
            suppression_ratio,
            compute_all_qa_metrics,
            plot_psd_comparison,
            set_theme,
        )
    )


def test_canonical_imports_cover_public_method_families():
    """Representative imports cover each documented method namespace."""
    from mne_denoise.asr import ASR, AdaptiveASR
    from mne_denoise.bss_cca import BSSCCA
    from mne_denoise.dss import (
        DSS,
        CovarianceSegmenter,
        IterativeDSS,
        TimeShiftDSS,
        VarianceMaskDenoiser,
        compute_dss,
        iterative_dss,
        iterative_dss_one,
    )
    from mne_denoise.dss.selection import detect_eigenvalue_knee
    from mne_denoise.icanclean import ICanClean
    from mne_denoise.qa import peak_attenuation_db
    from mne_denoise.sns import SNS
    from mne_denoise.sound import SOUND
    from mne_denoise.spectrum_interpolation import SpectrumInterpolation
    from mne_denoise.ssa import (
        LocalSingularSpectrumAnalysis,
        SingularSpectrumAnalysis,
    )
    from mne_denoise.sspsir import SSPSIR
    from mne_denoise.viz import plot_psd_comparison, set_theme
    from mne_denoise.zapline import ZapLine

    assert all(
        callable(item)
        for item in (
            ASR,
            AdaptiveASR,
            BSSCCA,
            CovarianceSegmenter,
            DSS,
            IterativeDSS,
            TimeShiftDSS,
            VarianceMaskDenoiser,
            compute_dss,
            iterative_dss,
            iterative_dss_one,
            detect_eigenvalue_knee,
            ICanClean,
            peak_attenuation_db,
            SNS,
            SOUND,
            SpectrumInterpolation,
            LocalSingularSpectrumAnalysis,
            SingularSpectrumAnalysis,
            SSPSIR,
            plot_psd_comparison,
            set_theme,
            ZapLine,
        )
    )
