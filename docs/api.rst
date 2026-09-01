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
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   ASR
   AdaptiveASR
   JugglerASR
   GuidedASR

.. autosummary::
   :toctree: generated/
   :nosignatures:

   calibrate_asr
   process_asr

Sensor Noise Suppression
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.sns

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   SNS

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_sns

SOUND
~~~~~

.. currentmodule:: mne_denoise.sound

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   SOUND

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_sound

Spectrum interpolation
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.spectrum_interpolation

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   SpectrumInterpolation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   interpolate_spectrum

ZapLine
~~~~~~~

.. currentmodule:: mne_denoise.zapline

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   ZapLine

DSS
~~~

Core
^^^^

.. currentmodule:: mne_denoise.dss

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   DSS
   IterativeDSS
   TimeShiftDSS

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_dss
   iterative_dss

Biases and linear denoisers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mne_denoise.dss

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
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
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
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

Singular Spectrum Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: mne_denoise.ssa

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   SingularSpectrumAnalysis
   LocalSingularSpectrumAnalysis

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_basic_ssa
   compute_local_ssa

BSS-CCA
~~~~~~~

.. currentmodule:: mne_denoise.bss_cca

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   BSSCCA

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_bss_cca

iCanClean
~~~~~~~~~

.. currentmodule:: mne_denoise.icanclean

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   ICanClean

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_icanclean

SSP-SIR
~~~~~~~

.. currentmodule:: mne_denoise.sspsir

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_no_members.rst
   :nosignatures:

   SSPSIR

.. autosummary::
   :toctree: generated/
   :nosignatures:

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
