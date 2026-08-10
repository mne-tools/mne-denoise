===========
mne-denoise
===========

`mne-denoise` provides artifact removal for M/EEG, matched to the structure of
the contamination. Line noise, transient movement bursts, reference-correlated
artifacts and target-response enhancement are different problems that need
different information, so the package implements each as a separate
scikit-learn-style estimator that operates directly on MNE objects and exposes
an inspectable fitted state.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Your problem
     - Method
     - Information it uses
   * - Power-line noise, possibly non-stationary
     - :class:`~mne_denoise.zapline.ZapLine`
     - narrowband spatial structure at the line frequency
   * - Line noise, conservative and phase-preserving
     - ``spectrum_interpolation``
     - the spectral neighbourhood of the peak
   * - Large transient / movement artifacts
     - :class:`~mne_denoise.asr.ASR` and variants
     - abnormal covariance relative to a clean baseline
   * - Recorded noise reference channels available
     - :class:`~mne_denoise.icanclean.ICanClean`
     - correlation between scalp and reference channels
   * - Channel-specific sensor noise
     - ``sns``
     - what neighbouring sensors agree on
   * - Enhance a response you can define
     - :class:`~mne_denoise.dss.DSS`
     - a bias you declare (trial average, band, period, ...)

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting-started
   dss
   asr
   sns
   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Project information

   development
