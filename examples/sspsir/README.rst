SSP-SIR Examples
================

Overview
--------

Examples demonstrating SSP-SIR, which suppresses TMS-evoked muscle artifacts by
projecting out the artifact subspace and reconstructing the brain signal lost to
that projection through a forward model.

Files
-----

- ``plot_01_sspsir_basics.py``: SSP-SIR on a simulated TMS-evoked response buried
  under muscle bursts; reading ``n_components`` off the singular-value elbow,
  why the projection is crossfaded rather than applied to the whole epoch, and
  inspecting the removed topographies via ``projs_``.

Data Requirements
-----------------

- All sections run directly with no external data.
- The lead field is built from the montage with a three-layer spherical head
  model, so no MRI or forward solution is needed.

References
----------

- Mutanen, Kukkonen, Nieminen, Stenroos, Sarvas & Ilmoniemi (2016). Recovering
  TMS-evoked EEG responses masked by muscle artifacts. NeuroImage.
- Mutanen et al. (2024). A simulation study: comparing independent component
  analysis and signal-space projection - source-informed reconstruction for
  rejecting muscle artifacts evoked by transcranial magnetic stimulation.
  Frontiers in Human Neuroscience.
- Mutanen et al. (2022). Source-based artifact-rejection techniques for
  TMS-EEG. Journal of Neuroscience Methods.
- Hernandez-Pavon et al. (2022). Removing artifacts from TMS-evoked EEG: A
  methods review and a unifying theoretical framework. Journal of Neuroscience
  Methods.
