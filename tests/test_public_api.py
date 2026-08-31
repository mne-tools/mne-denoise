"""Tests for the intentional public API facades and import paths."""

import importlib

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
    "mne_denoise.progress",
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


CANONICAL_PUBLIC_PATHS = (
    ("mne_denoise", "compute_covariance"),
    ("mne_denoise", "quantify_overcorrection"),
    ("mne_denoise.asr", "ASR"),
    ("mne_denoise.asr", "AdaptiveASR"),
    ("mne_denoise.asr", "GuidedASR"),
    ("mne_denoise.asr", "JugglerASR"),
    ("mne_denoise.asr", "calibrate_asr"),
    ("mne_denoise.asr", "process_asr"),
    ("mne_denoise.bss_cca", "BSSCCA"),
    ("mne_denoise.bss_cca", "compute_bss_cca"),
    ("mne_denoise.dss", "DSS"),
    ("mne_denoise.dss", "IterativeDSS"),
    ("mne_denoise.dss", "TimeShiftDSS"),
    ("mne_denoise.dss", "CovarianceSegmenter"),
    ("mne_denoise.dss", "VarianceMaskDenoiser"),
    ("mne_denoise.dss", "compute_dss"),
    ("mne_denoise.dss", "iterative_dss"),
    ("mne_denoise.dss", "iterative_dss_one"),
    ("mne_denoise.dss.selection", "detect_eigenvalue_knee"),
    ("mne_denoise.icanclean", "ICanClean"),
    ("mne_denoise.icanclean", "compute_icanclean"),
    ("mne_denoise.icanclean", "null_r2_threshold"),
    ("mne_denoise.progress", "ProgressEvent"),
    ("mne_denoise.progress", "TqdmProgress"),
    ("mne_denoise.qa", "compute_all_qa_metrics"),
    ("mne_denoise.qa", "peak_attenuation_db"),
    ("mne_denoise.qa", "suppression_ratio"),
    ("mne_denoise.sns", "SNS"),
    ("mne_denoise.sns", "compute_sns"),
    ("mne_denoise.sns", "compute_sns_weights"),
    ("mne_denoise.sound", "SOUND"),
    ("mne_denoise.sound", "compute_sound"),
    ("mne_denoise.sound", "compute_sound_ref_best"),
    ("mne_denoise.spectrum_interpolation", "SpectrumInterpolation"),
    ("mne_denoise.spectrum_interpolation", "interpolate_spectrum"),
    ("mne_denoise.ssa", "LocalSingularSpectrumAnalysis"),
    ("mne_denoise.ssa", "SingularSpectrumAnalysis"),
    ("mne_denoise.sspsir", "SSPSIR"),
    ("mne_denoise.sspsir", "compute_sir"),
    ("mne_denoise.sspsir", "compute_sspsir"),
    ("mne_denoise.viz", "plot_psd_comparison"),
    ("mne_denoise.viz", "set_theme"),
    ("mne_denoise.zapline", "ZapLine"),
)


def test_public_facades_declare_unique_existing_all():
    """Every declared public facade has a unique, existing ``__all__``."""
    for module_name in PUBLIC_FACADES:
        module = importlib.import_module(module_name)
        public_names = getattr(module, "__all__", None)
        assert public_names is not None, f"{module_name} must declare __all__"

        assert len(public_names) == len(set(public_names)), module_name
        for name in public_names:
            assert hasattr(module, name), f"{module_name}.{name} is missing"


def test_canonical_public_paths_are_importable():
    """Documented public classes and functions remain importable canonically."""
    for module_name, name in CANONICAL_PUBLIC_PATHS:
        module = importlib.import_module(module_name)
        public_object = getattr(module, name, None)
        assert callable(public_object), f"{module_name}.{name} is not callable"
