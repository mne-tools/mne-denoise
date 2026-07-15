# Development, reanalysis, and locked-test declaration

This registry prevents retrospective relabelling of data after algorithm development.
The original benchmark outputs and plots were inspected before the NeuroImage protocol
was frozen; those datasets therefore support pre-specified reanalysis or external
utility, not untouched confirmatory claims.

| Dataset or substrate | Used in development/debugging? | Used for tuning? | Final role | Locked unit |
|---|---:|---:|---|---|
| ds003620 | yes | yes | frozen reanalysis and environmental boundary test | none; all available recordings were previously inspected |
| ds004505 | yes | yes | frozen reanalysis of movement/reference regimes | none; all available subjects were previously inspected |
| ds000117 | yes | yes | frozen EEG/MEG reanalysis | none; existing line and evoked outputs were inspected |
| ERP CORE N170 | yes | yes | frozen ocular/evoked reanalysis | none; all 40-subject outputs were inspected |
| Tsinghua SSVEP | yes | yes | frozen external-utility reanalysis | none; group outputs were inspected |
| BETA SSVEP | no | no | locked lower-SNR external replication | complete subjects listed in its pre-run manifest |
| EEGdenoiseNet source pool | yes | yes | source library for simulations | new source-set assignments and seeds created only after protocol freeze |
| ds004784 repeat 1 | yes | yes | development and parameter selection | none |
| ds004784 repeat 2 | no | no | locked technical replication across Clean/Eyes/Jaw/Motion/Neck/All | complete repeat-2 recordings; hash before first numerical load |
| Klados clean/contaminated pairs | yes | yes | external paired known-truth validation | none; all loadable trials were previously inspected |
| New synthetic transient/reference/BSS/forward/streaming seeds | no | no | locked known-target evidence | seed lists and generator commit frozen before generation |

## Freeze record

- Candidate protocol authored: 2026-07-14.
- Candidate protocol ID: `neuroimage-benchmark-v1`.
- Freeze tag: created only after the integration worktree is clean and all selected
  configurations pass `python -m mne_denoise.benchmarks validate`.
- Preregistration: the archived protocol manifest and Git tag are the timestamped
  specification; an external registration may mirror the same files but cannot alter them.
- Any post-freeze change requires a new protocol ID, a machine-readable diff, and a
  sensitivity analysis retaining the original result.

## Holdout policy

- Hold out complete subjects, complete technical repeats, or new simulation seeds; never
  split adjacent windows from the same recording between development and test.
- Real datasets previously inspected are labelled reanalyses even when their rerun is
  fully pre-specified.
- BETA and ds004784 repeat 2 are not numerically loaded until their dataset manifests,
  target mappings, inclusion rules, and analysis configs are hashed.
- Subject exclusion uses raw-data QC evaluated before method comparison and is written to
  the terminal-status registry for every attempted unit.
- Defaults are not changed after locked outputs are opened. A method failure remains in
  the denominator.
