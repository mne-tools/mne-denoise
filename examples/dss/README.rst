DSS examples
============

These examples show four scientifically distinct ways to define structure of
interest with Denoising Source Separation.

The evoked example uses trial reproducibility on real held-out somatosensory
MEG data. The cardiac example combines real Sample EEG and real R-peak timing
with a controlled planted cardiac artifact, then uses an independent clean-input
preservation control. The narrowband example uses controlled spectral structure
to recover a known target source. The TimeShiftDSS example extends
reproducibility into lag-augmented spatiotemporal space and uses held-out and
surrogate validation.

A DSS bias defines what the decomposition emphasizes; it does not by itself
establish that a selected component is neural signal or artifact.
