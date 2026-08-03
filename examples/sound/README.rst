SOUND Examples
==============

Overview
--------

Examples demonstrating SOUND (Source-estimate-Utilizing Noise-discarding), which
suppresses channel-specific noise by asking whether each sensor's signal is
consistent with what a forward model says the head could have produced.

Files
-----

- ``plot_01_sound_basics.py``: SOUND on synthetic EEG with deliberately corrupted
  sensors; recovering them, checking the intact channels are not damaged, and
  sweeping ``lambda_``.

Data Requirements
-----------------

- All sections run directly with no external data.
- The lead field is built from the montage with a three-layer spherical head
  model, so no MRI or forward solution is needed.

References
----------

- Mutanen, Metsomaa, Liljander & Ilmoniemi (2018). Automatic and robust noise
  suppression in EEG and MEG: The SOUND algorithm. NeuroImage.
