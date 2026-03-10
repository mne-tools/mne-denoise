API reference
=============

DSS
---
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.compute_dss
   mne_denoise.dss.DSS
   mne_denoise.dss.iterative_dss
   mne_denoise.dss.IterativeDSS

ZapLine
-------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.zapline.ZapLine

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
   mne_denoise.dss.denoisers.TimeShiftBias
   mne_denoise.dss.denoisers.SmoothingBias
   mne_denoise.dss.denoisers.SpectrogramBias
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

Variants
--------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.dss.variants.time_shift_dss
   mne_denoise.dss.variants.smooth_dss
   mne_denoise.dss.variants.narrowband_dss
   mne_denoise.dss.variants.narrowband_scan
   mne_denoise.dss.variants.ssvep_dss
   mne_denoise.dss.variants.tsr

Visualization
-------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   mne_denoise.viz.plot_component_summary
   mne_denoise.viz.plot_component_time_series
   mne_denoise.viz.plot_component_spectrogram
   mne_denoise.viz.plot_component_score_curve
   mne_denoise.viz.plot_component_patterns
   mne_denoise.viz.plot_component_epochs_image
   mne_denoise.viz.plot_psd_comparison
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
