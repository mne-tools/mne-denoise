API reference
=============

.. note::

   **mne-denoise is under active development.** Until version 1.0, the
   public API may evolve between releases. For reproducible analyses,
   record the mne-denoise version used in your work.

Names documented in this reference are public unless explicitly marked
experimental. Underscore-prefixed implementation details are private.

Primary denoising API
---------------------

The main interfaces follow the artifact and method flow in the
:doc:`Methods guide <methods>` and are listed here first.

Artifact Subspace Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.asr

.. autosummary::
   :nosignatures:

   ASR
   AdaptiveASR
   JugglerASR
   GuidedASR
   calibrate_asr
   process_asr

Sensor Noise Suppression
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.sns

.. autosummary::
   :nosignatures:

   SNS
   compute_sns

SOUND
~~~~~

.. currentmodule:: mne_denoise.sound

.. autosummary::
   :nosignatures:

   SOUND
   compute_sound

Spectrum interpolation
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.spectrum_interpolation

.. autosummary::
   :nosignatures:

   SpectrumInterpolation
   interpolate_spectrum

ZapLine
~~~~~~~

.. currentmodule:: mne_denoise.zapline

.. autosummary::
   :nosignatures:

   ZapLine

DSS
~~~

.. currentmodule:: mne_denoise.dss

.. autosummary::
   :nosignatures:

   DSS
   compute_dss
   IterativeDSS
   iterative_dss
   TimeShiftDSS

Singular Spectrum Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.ssa

.. autosummary::
   :nosignatures:

   SingularSpectrumAnalysis
   LocalSingularSpectrumAnalysis
   compute_basic_ssa
   compute_local_ssa

BSS-CCA
~~~~~~~

.. currentmodule:: mne_denoise.bss_cca

.. autosummary::
   :nosignatures:

   BSSCCA
   compute_bss_cca

iCanClean
~~~~~~~~~

.. currentmodule:: mne_denoise.icanclean

.. autosummary::
   :nosignatures:

   ICanClean
   compute_icanclean

SSP-SIR
~~~~~~~

.. currentmodule:: mne_denoise.sspsir

.. autosummary::
   :nosignatures:

   SSPSIR
   compute_sspsir
   compute_sir

Additional API
--------------

.. toctree::
   :hidden:
   :maxdepth: 1

   Advanced method helpers <api/helpers>
   Evaluation and QA <api/evaluation>
   Visualization <api/visualization>
   Utilities <api/utilities>

The secondary reference pages collect reusable building blocks and supporting
interfaces:

* :doc:`Advanced method helpers <api/helpers>`
* :doc:`Evaluation and QA <api/evaluation>`
* :doc:`Visualization <api/visualization>`
* :doc:`Utilities <api/utilities>`
