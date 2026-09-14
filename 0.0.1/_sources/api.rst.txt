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
   mne_denoise.viz.plot_psd_comparison
   mne_denoise.viz.plot_time_course_comparison
   mne_denoise.viz.plot_spectrogram_comparison
   mne_denoise.viz.plot_overlay_comparison
   mne_denoise.viz.plot_narrowband_scan
   mne_denoise.viz.plot_tf_mask

   mne_denoise.viz.plot_zapline_analytics
   mne_denoise.viz.plot_cleaning_summary
   mne_denoise.viz.plot_component_scores
   mne_denoise.viz.plot_zapline_patterns
   mne_denoise.viz.plot_zapline_psd_comparison
