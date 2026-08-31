===========
mne-denoise
===========

``mne-denoise`` provides artifact-suppression and signal-denoising methods for
EEG and MEG. It includes complementary spatial, spectral, statistical, and
source-informed methods, with NumPy support and optional integration with
MNE-Python containers.

No denoising method is universally best. Choose a method from the artifact,
recording, channel layout, and signal of interest, then evaluate both artifact
attenuation and preservation of the desired signal.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting-started
   evaluation
   dss
   asr
   bss_cca
   icanclean
   sns
   sound
   spectrum_interpolation
   ssa
   sspsir
   zapline
   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Project information

   citing
   development
