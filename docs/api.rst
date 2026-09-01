API reference
=============

.. note::

   **mne-denoise is under active development.** Until version 1.0, the
   public API may evolve between releases. For reproducible analyses,
   record the mne-denoise version used in your work.

Names documented in this reference are public unless explicitly marked
experimental. Underscore-prefixed implementation details are private.

:doc:`Denoising methods <api/methods>`
--------------------------------------

Estimator and functional APIs for ASR, BSS-CCA, iCanClean, SNS, SOUND,
spectrum interpolation, SSA, SSP-SIR, and ZapLine.

:doc:`DSS <api/dss>`
--------------------

Core DSS estimators, biases, denoisers, segmentation, component selection,
and variants.

:doc:`Evaluation and QA <api/evaluation>`
-----------------------------------------

Metrics for artifact attenuation, signal change, and overcorrection.

:doc:`Visualization <api/visualization>`
----------------------------------------

Optional plotting, diagnostic, summary, and component-selection helpers.

:doc:`Utilities <api/utilities>`
-------------------------------------

Shared covariance and progress-callback utilities.

.. toctree::
   :hidden:
   :maxdepth: 1

   Denoising methods <api/methods>
   DSS <api/dss>
   Evaluation and QA <api/evaluation>
   Visualization <api/visualization>
   Utilities <api/utilities>
