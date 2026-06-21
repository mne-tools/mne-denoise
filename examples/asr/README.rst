ASR Examples
============

Examples demonstrating Artifact Subspace Reconstruction (ASR) for burst-artifact
repair in EEG (and MEG) data --- from a basic clean to a full preprocessing
pipeline, with each method example grounded in its source paper.

Getting started
---------------

- ``plot_01_asr_basics.py``: Standard ASR on synthetic multichannel bursts.
- ``plot_02_mne_raw_qc.py``: MNE ``Raw`` usage with repair annotations and an
  optional clean_windows-style final rejection mask.
- ``plot_05_asr_visualization.py``: The ASR-specific ``mne_denoise.viz`` plots
  (repair timeline, component reconstruction, calibration fraction) alongside the
  generic before/after and PSD helpers.

Cutoff and variants
-------------------

- ``plot_06_cutoff_tuning.py``: How ``cutoff`` trades data modified against
  variance removed (Chang 2020).
- ``plot_07_riemannian_asr.py``: Riemannian (``method="riemannian_windowed"``)
  vs standard ASR on real blinks (Blum 2019).
- ``plot_03_adaptive_asr.py``: AASR-style streaming with ``fit`` /
  ``partial_fit`` / ``transform``.
- ``plot_08_adaptive_variants.py``: Adaptive ``psp`` vs ``psw`` vs ``mw`` on
  non-stationary data, with the moving-window adaptation trajectory (Tsai).
- ``plot_04_juggler_asr.py``: Juggler DBSCAN calibration on dense short bursts.
- ``plot_09_juggler_strategies.py``: Juggler ``dbscan`` vs ``gev`` reference
  selection under heavy contamination (Kim 2025).
- ``plot_10_choosing_a_variant.py``: Standard vs Riemannian vs Juggler on one
  substrate, with a short recommendation.

I/O, QC, and pipelines
----------------------

- ``plot_11_epochs_and_meg.py``: ASR on ``mne.Epochs`` and on MEG magnetometers.
- ``plot_12_diagnostics_qc.py``: ``get_diagnostics`` / ``variance_removed``
  / ``to_annotations`` and the three ASR diagnostic plots.
- ``plot_13_pipeline_filter_asr_ica.py``: A realistic ``filter -> ASR -> ICA``
  workflow on real EEG.

Notes
-----

Most examples use synthetic data so they run without downloads; ``plot_07`` and
``plot_13`` (and the MEG part of ``plot_11``) use the MNE *sample* dataset.
Apply ASR to real EEG only after bad-channel handling, referencing, and
high-pass filtering in the surrounding MNE workflow.
