"""MARA automatic IC artifact classifier (Winkler, Haufe & Tangermann 2011).

GPL-3.0 ISOLATED MODULE. Unlike the rest of ``mne-denoise`` (BSD-3-Clause), this
subpackage is licensed GPL-3.0: it is a derivative of the MARA EEGLAB plug-in
(https://github.com/irenne/MARA, (C) 2013 Irene Winkler & Eric Waldburger) and
vendors MARA's GPL leadfield. See ``LICENSE.md`` in this directory. It is not
imported by the top-level ``mne_denoise`` package; it is loaded on demand only by
the optional benchmark comparator ``mara``.

Reference: I. Winkler, S. Haufe, M. Tangermann, "Automatic classification of
artifactual ICA-components for artifact removal in EEG signals," Behav. Brain
Funct. 7:30, 2011. DOI: 10.1186/1744-9081-7-30.
"""

from .core import mara_bad_components

__all__ = ["mara_bad_components"]
