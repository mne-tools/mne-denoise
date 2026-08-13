DSS Examples
============

Overview
--------

Examples demonstrating Denoising Source Separation (DSS) across evoked,
spectral, temporal, and blind-separation use cases.

Files
-----

- ``plot_01_dss_fundamentals.py``: Core DSS concepts with trial-average and bandpass biases.
- ``plot_02_artifact_correction.py``: Blink and heartbeat correction with DSS.
- ``plot_03_evoked_responses.py``: Evoked-response denoising and contrast-focused DSS.
- ``plot_04_spectral_dss.py``: Frequency-specific component extraction on synthetic and real data.
- ``plot_05_periodic_dss.py``: Periodic signal extraction for SSVEP and quasi-periodic structure.
- ``plot_06_temporal_dss.py``: Time-shift and smoothness biases for temporally structured signals.
- ``plot_07_spectrogram_dss.py``: Time-frequency masking with spectrogram-based DSS.
- ``plot_08_blind_source_separation.py``: Blind source separation and FastICA equivalence.
- ``plot_09_custom_bias.py``: Defining custom DSS biases.
- ``plot_10_benchmarking.py``: Efficiency benchmarking against PCA, ICA, and averaging.
- ``plot_11_wiener_masking.py``: Adaptive Wiener masking for bursty signals.
- ``plot_12_joint_dss.py``: Joint DSS for multi-dataset repeatability.
- ``plot_13_cardiac_composition.py``: Explicit, held-out cardiac DSS composition with attenuation and preservation checks.

Data Requirements
-----------------

- Synthetic sections run directly with no external data.
- Examples using MNE datasets download and cache them through MNE when needed.

References
----------

- Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res.
- de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
- de Cheveigné & Parra (2014). Joint decorrelation. NeuroImage.
