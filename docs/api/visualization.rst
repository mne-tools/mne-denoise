Visualization
=============

.. note::

   Visualization helpers require the optional ``viz`` dependency:
   ``pip install "mne-denoise[viz]"``.

All names below are available from the public :mod:`mne_denoise.viz` facade.

ASR diagnostics
---------------

.. currentmodule:: mne_denoise.viz

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_asr_repair_timeline
   plot_asr_calibration_fraction
   plot_asr_component_reconstruction
   plot_guided_asr_weights

Component diagnostics
---------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_component_summary
   plot_component_patterns
   plot_component_score_curve
   plot_component_epochs_image
   plot_component_time_series
   plot_component_spectrogram
   plot_window_score_traces

Signal comparisons
------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_channel_time_course_comparison
   plot_evoked_gfp_comparison
   plot_power_ratio_map
   plot_signal_overlay
   plot_grand_average_evokeds

Spectral and time-frequency plots
---------------------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_narrowband_score_scan
   plot_time_frequency_mask
   plot_psd_gallery
   plot_psd_comparison
   plot_psd_overlay
   plot_psd_zoom_comparison
   plot_spectrogram_comparison
   plot_component_psd_comparison

Statistical summaries
---------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_metric_bars
   plot_tradeoff_scatter
   plot_metric_comparison
   plot_harmonic_attenuation
   plot_metric_slopes
   plot_metric_violins
   plot_null_distribution
   plot_forest
   plot_window_count_series

Summary figures
---------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_denoising_summary
   plot_component_cleaning_summary
   plot_signal_diagnostics_summary
   plot_condition_interaction_summary
   plot_group_condition_interaction_summary
   plot_endpoint_metrics_summary
   plot_metric_tradeoff_summary

Interactive component selection
-------------------------------

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   ComponentSelector

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   plot_component_selector

Theme helpers
-------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   get_color
   get_series_color
   style_axes
   themed_figure
   themed_legend
   get_theme_rc
   use_theme
   set_theme
