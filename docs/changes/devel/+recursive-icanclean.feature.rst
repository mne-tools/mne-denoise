Added the explicitly experimental
:class:`mne_denoise.icanclean.RecursiveICanClean` prototype. It combines
sample-indexed recursive joint-covariance updates with regularized CCA and the
iCanClean subtraction rule; supports contamination-gated and frozen adaptation,
sample- or second-valued memory/update contracts, transport-latency diagnostics,
MNE channel/type/unit and Raw-timeline alignment, leakage-free frozen transforms,
and lossless checksum-validated JSON state replay. Published iCanClean describes
recursive CCA only as future work, so the API and documentation do not transfer
ordinary iCanClean validation claims to this implementation.
