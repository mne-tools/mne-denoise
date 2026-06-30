# LICENSE NOTICE — this subdirectory only is GPL-3.0

**The `mne_denoise/mara/` package (this directory and everything under it) is
licensed under the GNU General Public License, version 3 (GPL-3.0).**

This is an exception to the rest of `mne-denoise`, which is BSD-3-Clause. Code
and data in this directory are a derivative of the MARA EEGLAB plug-in and are
therefore governed by MARA's GPL-3.0 license — copyleft propagates to derivative
works, so this module must remain GPL-3.0 and cannot be relicensed under BSD.

## Provenance

- **Upstream:** https://github.com/irenne/MARA (MARA, "Multiple Artifact
  Rejection Algorithm"), an EEGLAB plug-in.
- **Copyright (C) 2013 Irene Winkler and Eric Waldburger**, Berlin Institute of
  Technology, Germany. Licensed GPL-2.0-or-later by the original authors; this
  derivative is distributed under GPL-3.0.
- **Reference:** I. Winkler, S. Haufe, and M. Tangermann, "Automatic
  classification of artifactual ICA-components for artifact removal in EEG
  signals," *Behavioral and Brain Functions*, 7:30, 2011.
  DOI: 10.1186/1744-9081-7-30.

## What is derived from MARA

1. **`core.py`** — a Python reimplementation of MARA's six-feature extraction
   (Current Density Norm, Range Within Pattern, Mean Local Skewness, lambda,
   8–13 Hz band power, FitError), the parametric log-spectrum fit
   `P(f) = exp(x1) * f^(-exp(x2)) - x3`, and the sLORETA current-density operator
   (`get_M100_ADE` / `sloreta_invweights` in the original `MARA.m`). The linear
   discriminant weight vector + bias are the published MARA classifier, derived
   from the 1290 hand-labelled training components in `fv_training_MARA.mat`.

2. **`data/inv_matrix_icbm152.mat`** — vendored verbatim from the MARA repo
   (the ICBM152 forward leadfield `L` and channel labels `clab`), required to
   build the Current Density Norm operator. This file is GPL-3.0.

## Why isolated

`mne-denoise` proper is BSD-3-Clause. To keep that licensing clean, MARA lives in
this single self-contained directory, is **not** imported by the top-level
`mne_denoise` package, and is loaded only on demand by the optional benchmark
comparator `mara`. Importing or distributing this subdirectory subjects that
usage to GPL-3.0; the rest of the package is unaffected.

A copy of the GPL-3.0 text is available at https://www.gnu.org/licenses/gpl-3.0.txt.
