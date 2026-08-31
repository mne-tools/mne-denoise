Getting started
===============

Installation
------------

Install the base package and optional integrations as needed:

.. code-block:: console

   pip install mne-denoise
   pip install "mne-denoise[mne]"
   pip install "mne-denoise[viz]"
   pip install "mne-denoise[progress]"

Estimator pattern
-----------------

Most estimators follow the scikit-learn pattern: configure, fit on data used
to learn an operator, then transform compatible data. Use fit_transform when
the method has a fixed fitted operator.

.. code-block:: python

   import numpy as np
   from mne_denoise.dss import BandpassBias, DSS

   data = np.random.default_rng(0).standard_normal((8, 2000))
   bias = BandpassBias((8.0, 12.0), sfreq=250.0)
   clean = DSS(bias=bias, n_components=3).fit_transform(data)

MNE-Python integration
----------------------

Pass supported MNE objects directly; estimators preserve container metadata and
return copies. For example, with a preloaded Raw object named raw:

.. code-block:: python

   from mne_denoise.spectrum_interpolation import SpectrumInterpolation

   clean_raw = SpectrumInterpolation(
       line_freq=60.0, n_harmonics=3
   ).fit_transform(raw)

Choosing a method
-----------------

The method pages summarize assumptions and minimal workflows for:

* ASR for transient, high-variance subspace changes;
* the DSS family for reproducible, spectral, temporal, and lagged-trial
  structure;
* BSS-CCA and iCanClean for lagged or reference-shared components;
* SNS and SOUND for spatially or forward-model-predicted sensor noise;
* spectrum interpolation and ZapLine for line noise;
* SSA for channel-wise delay-coordinate decompositions; and
* SSP-SIR for source-informed TMS-evoked artifact reconstruction.

See the :doc:`api` reference for exact contracts, the
:doc:`auto_examples/index` gallery for complete workflows, and
:doc:`citing` for citation guidance.
