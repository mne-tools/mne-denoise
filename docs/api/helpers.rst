Advanced method helpers
=======================

These secondary interfaces expose reusable building blocks for composing or
inspecting denoising workflows. Start with the primary estimators on the
:doc:`API reference <../api>` page when choosing a method.

ASR helpers
-----------

.. currentmodule:: mne_denoise.asr

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   compute_clean_window_mask
   fit_rms_distribution
   process_guided_asr
   select_juggler_reference_samples

DSS building blocks
-------------------

Additional iterative helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.dss

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   iterative_dss_one

Segmentation
~~~~~~~~~~~~

.. currentmodule:: mne_denoise.dss.segmentation

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   CovarianceSegmenter
   FixedWindowSegmenter

Component selection
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.dss.selection

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   auto_select_components
   auto_select_components_robust
   detect_eigenvalue_knee
   iterative_outlier_removal

DSS convenience variants
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.dss.variants

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   smooth_dss
   narrowband_dss
   narrowband_scan
   ssvep_dss

Nonlinear helper functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.dss.denoisers

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   beta_tanh
   beta_pow3
   beta_gauss

iCanClean helper
----------------

.. currentmodule:: mne_denoise.icanclean

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   null_r2_threshold

SNS helper
----------

.. currentmodule:: mne_denoise.sns

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   compute_sns_weights

SOUND helper
------------

.. currentmodule:: mne_denoise.sound

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   compute_sound_ref_best

SSA helpers
-----------

.. currentmodule:: mne_denoise.ssa

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   ssa_decompose
   ssa_w_correlation
   ssa_clean_channel
   local_ssa_clean_channel
