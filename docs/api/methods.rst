Denoising methods
=================

Artifact Subspace Reconstruction
--------------------------------

.. currentmodule:: mne_denoise.asr

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   ASR
   AdaptiveASR
   JugglerASR
   GuidedASR
   calibrate_asr
   compute_clean_window_mask
   fit_rms_distribution
   process_asr
   process_guided_asr
   select_juggler_reference_samples

Sensor Noise Suppression
------------------------

.. currentmodule:: mne_denoise.sns

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   SNS
   compute_sns
   compute_sns_weights

SOUND
-----

.. currentmodule:: mne_denoise.sound

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   SOUND
   compute_sound
   compute_sound_ref_best

Spectrum interpolation
----------------------

.. currentmodule:: mne_denoise.spectrum_interpolation

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   SpectrumInterpolation
   interpolate_spectrum

ZapLine
-------

.. currentmodule:: mne_denoise.zapline

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   ZapLine

BSS-CCA
-------

.. currentmodule:: mne_denoise.bss_cca

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   BSSCCA
   compute_bss_cca

iCanClean
---------

.. currentmodule:: mne_denoise.icanclean

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   ICanClean
   compute_icanclean
   null_r2_threshold

Singular Spectrum Analysis
--------------------------

.. currentmodule:: mne_denoise.ssa

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   SingularSpectrumAnalysis
   LocalSingularSpectrumAnalysis
   compute_basic_ssa
   compute_local_ssa
   local_ssa_clean_channel
   ssa_clean_channel
   ssa_decompose
   ssa_w_correlation

SSP-SIR
-------

.. currentmodule:: mne_denoise.sspsir

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   SSPSIR
   compute_sir
   compute_sspsir
