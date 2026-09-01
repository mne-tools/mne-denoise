DSS
===

Core DSS
--------

.. currentmodule:: mne_denoise.dss

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   DSS
   compute_dss
   IterativeDSS
   iterative_dss
   iterative_dss_one

Biases
------

.. currentmodule:: mne_denoise.dss.denoisers

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   LinearDenoiser
   AverageBias
   CycleAverageBias
   BandpassBias
   LineNoiseBias
   PeakFilterBias
   CombFilterBias
   LagAverageBias
   SmoothingBias
   SpectrogramBias

Nonlinear denoisers
-------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   NonlinearDenoiser
   TanhMaskDenoiser
   RobustTanhDenoiser
   KurtosisDenoiser
   SkewDenoiser
   GaussDenoiser
   WienerMaskDenoiser
   VarianceMaskDenoiser
   SpectrogramDenoiser
   DCTDenoiser
   QuasiPeriodicDenoiser
   SmoothTanhDenoiser
   beta_tanh
   beta_pow3
   beta_gauss

Segmentation and component selection
------------------------------------

.. currentmodule:: mne_denoise.dss.segmentation

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   CovarianceSegmenter
   FixedWindowSegmenter

.. currentmodule:: mne_denoise.dss.selection

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   auto_select_components
   auto_select_components_robust
   detect_eigenvalue_knee
   iterative_outlier_removal

Variants
--------

.. currentmodule:: mne_denoise.dss

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   TimeShiftDSS

.. currentmodule:: mne_denoise.dss.variants

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   smooth_dss
   narrowband_dss
   narrowband_scan
   ssvep_dss
