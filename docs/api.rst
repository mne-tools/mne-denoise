API reference
=============

API stability
-------------

MNE-Denoise is under active development. Before version 1.0, public APIs may
change without a mandatory formal deprecation cycle or advance warning. Such
changes should still be deliberate, avoid gratuitous breakage, and be recorded
in release notes.

Names or modules beginning with ``_`` are private and may change at any time.
APIs explicitly marked experimental or research prototypes have weaker
stability guarantees. Starting with version 1.0, stable public API changes
should normally follow a documented deprecation process.

The supported public API is the set of user-facing names and namespaces
intentionally documented in this reference or as canonical imports in
user-facing documentation; an arbitrary importable non-private submodule is
not automatically a stability promise.

Package utilities
-----------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.compute_covariance

ASR
---
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.asr.ASR
   mne_denoise.asr.AdaptiveASR
   mne_denoise.asr.JugglerASR
   mne_denoise.asr.GuidedASR
   mne_denoise.asr.calibrate_asr
   mne_denoise.asr.compute_clean_window_mask
   mne_denoise.asr.fit_rms_distribution
   mne_denoise.asr.process_asr
   mne_denoise.asr.process_guided_asr
   mne_denoise.asr.select_juggler_reference_samples

.. warning::

   ``GuidedASR`` and ``process_guided_asr`` are unpublished, unvalidated
   experimental research prototypes. Their current evidence is limited to unit
   tests and synthetic benchmarks; independently validate signal preservation
   and artifact attenuation before scientific use.

DSS
---
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.compute_dss
   mne_denoise.dss.DSS
   mne_denoise.dss.TimeShiftDSS
   mne_denoise.dss.iterative_dss
   mne_denoise.dss.iterative_dss_one
   mne_denoise.dss.IterativeDSS

DSS segmentation
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.segmentation.CovarianceSegmenter
   mne_denoise.dss.segmentation.FixedWindowSegmenter

DSS component selection
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.selection.auto_select_components
   mne_denoise.dss.selection.auto_select_components_robust
   mne_denoise.dss.selection.detect_eigenvalue_knee
   mne_denoise.dss.selection.iterative_outlier_removal

ZapLine
-------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.zapline.ZapLine

Spectrum interpolation
----------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.spectrum_interpolation.SpectrumInterpolation
   mne_denoise.spectrum_interpolation.interpolate_spectrum


iCanClean
---------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.icanclean.ICanClean
   mne_denoise.icanclean.compute_icanclean

SOUND
-----
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.sound.SOUND
   mne_denoise.sound.compute_sound
   mne_denoise.sound.compute_sound_ref_best

SSP-SIR
-------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.sspsir.SSPSIR
   mne_denoise.sspsir.compute_sspsir
   mne_denoise.sspsir.compute_sir

Overcorrection metrics
----------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.quantify_overcorrection

BSS-CCA
-------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.bss_cca.BSSCCA
   mne_denoise.bss_cca.compute_bss_cca

SNS
---
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.sns.SNS
   mne_denoise.sns.compute_sns
   mne_denoise.sns.compute_sns_weights

SSA
---
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.ssa.SingularSpectrumAnalysis
   mne_denoise.ssa.LocalSingularSpectrumAnalysis
   mne_denoise.ssa.ssa_decompose
   mne_denoise.ssa.ssa_w_correlation
   mne_denoise.ssa.compute_basic_ssa
   mne_denoise.ssa.ssa_clean_channel
   mne_denoise.ssa.compute_local_ssa
   mne_denoise.ssa.local_ssa_clean_channel

Denoisers
---------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.denoisers.LinearDenoiser
   mne_denoise.dss.denoisers.AverageBias
   mne_denoise.dss.denoisers.CycleAverageBias
   mne_denoise.dss.denoisers.BandpassBias
   mne_denoise.dss.denoisers.LineNoiseBias
   mne_denoise.dss.denoisers.PeakFilterBias
   mne_denoise.dss.denoisers.CombFilterBias
   mne_denoise.dss.denoisers.LagAverageBias
   mne_denoise.dss.denoisers.SmoothingBias
   mne_denoise.dss.denoisers.SpectrogramBias
   mne_denoise.dss.denoisers.VarianceMaskDenoiser
   mne_denoise.dss.denoisers.NonlinearDenoiser
   mne_denoise.dss.denoisers.TanhMaskDenoiser
   mne_denoise.dss.denoisers.RobustTanhDenoiser
   mne_denoise.dss.denoisers.KurtosisDenoiser
   mne_denoise.dss.denoisers.SkewDenoiser
   mne_denoise.dss.denoisers.GaussDenoiser
   mne_denoise.dss.denoisers.WienerMaskDenoiser
   mne_denoise.dss.denoisers.SpectrogramDenoiser
   mne_denoise.dss.denoisers.DCTDenoiser
   mne_denoise.dss.denoisers.QuasiPeriodicDenoiser
   mne_denoise.dss.denoisers.SmoothTanhDenoiser
   mne_denoise.dss.denoisers.beta_tanh
   mne_denoise.dss.denoisers.beta_pow3
   mne_denoise.dss.denoisers.beta_gauss

Variants
--------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.variants.smooth_dss
   mne_denoise.dss.variants.narrowband_dss
   mne_denoise.dss.variants.narrowband_scan
   mne_denoise.dss.variants.ssvep_dss

Quality Assurance
-----------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.qa.peak_attenuation_db
   mne_denoise.qa.suppression_ratio
   mne_denoise.qa.noise_surround_ratio
   mne_denoise.qa.below_noise_distortion_db
   mne_denoise.qa.spectral_distortion
   mne_denoise.qa.overclean_proportion
   mne_denoise.qa.underclean_proportion
   mne_denoise.qa.geometric_mean_psd_ratio
   mne_denoise.qa.variance_removed
   mne_denoise.qa.compute_all_qa_metrics
   mne_denoise.qa.rms_change
   mne_denoise.qa.max_abs_change
   mne_denoise.qa.channel_variance_ratio

Visualization
-------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.viz.plot_asr_repair_timeline
   mne_denoise.viz.plot_asr_calibration_fraction
   mne_denoise.viz.plot_asr_component_reconstruction
   mne_denoise.viz.plot_guided_asr_weights
   mne_denoise.viz.plot_component_summary
   mne_denoise.viz.plot_component_selector
   mne_denoise.viz.plot_component_time_series
   mne_denoise.viz.plot_component_spectrogram
   mne_denoise.viz.plot_component_score_curve
   mne_denoise.viz.plot_window_score_traces
   mne_denoise.viz.plot_component_patterns
   mne_denoise.viz.plot_component_epochs_image
   mne_denoise.viz.plot_psd_comparison
   mne_denoise.viz.plot_psd_gallery
   mne_denoise.viz.plot_psd_overlay
   mne_denoise.viz.plot_psd_zoom_comparison
   mne_denoise.viz.plot_evoked_gfp_comparison
   mne_denoise.viz.plot_channel_time_course_comparison
   mne_denoise.viz.plot_power_ratio_map
   mne_denoise.viz.plot_spectrogram_comparison
   mne_denoise.viz.plot_signal_overlay
   mne_denoise.viz.plot_component_psd_comparison
   mne_denoise.viz.plot_grand_average_evokeds
   mne_denoise.viz.plot_narrowband_score_scan
   mne_denoise.viz.plot_time_frequency_mask
   mne_denoise.viz.plot_metric_bars
   mne_denoise.viz.plot_window_count_series
   mne_denoise.viz.plot_tradeoff_scatter
   mne_denoise.viz.plot_metric_comparison
   mne_denoise.viz.plot_metric_slopes
   mne_denoise.viz.plot_metric_violins
   mne_denoise.viz.plot_null_distribution
   mne_denoise.viz.plot_forest
   mne_denoise.viz.plot_harmonic_attenuation
   mne_denoise.viz.plot_metric_tradeoff_summary

   mne_denoise.viz.plot_denoising_summary
   mne_denoise.viz.plot_component_cleaning_summary
   mne_denoise.viz.plot_signal_diagnostics_summary
   mne_denoise.viz.plot_condition_interaction_summary
   mne_denoise.viz.plot_group_condition_interaction_summary
   mne_denoise.viz.plot_endpoint_metrics_summary

Visualization theme
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.viz.set_theme
   mne_denoise.viz.use_theme
   mne_denoise.viz.get_theme_rc
   mne_denoise.viz.get_color
   mne_denoise.viz.get_series_color
   mne_denoise.viz.style_axes
   mne_denoise.viz.themed_figure
   mne_denoise.viz.themed_legend

The public theme constants are ``mne_denoise.viz.COLORS``,
``mne_denoise.viz.FONTS``, ``mne_denoise.viz.METHOD_COLORS``,
``mne_denoise.viz.SERIES_COLORS``, ``mne_denoise.viz.SEQUENTIAL_CMAP``,
``mne_denoise.viz.DIVERGING_CMAP``, ``mne_denoise.viz.DEFAULT_FIGSIZE``, and
``mne_denoise.viz.DEFAULT_DPI``.

``plot_component_selector`` return object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: mne_denoise.viz.ComponentSelector
   :members: apply, excluded
   :exclude-members: __init__
