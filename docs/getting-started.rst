Getting started
===============

Install
-------

Install the base package and optional integrations as needed:

.. code-block:: console

   pip install mne-denoise
   pip install "mne-denoise[mne]"
   pip install "mne-denoise[viz]"
   pip install "mne-denoise[progress]"

First NumPy workflow
--------------------

Most estimators follow the scikit-learn pattern: configure, fit on data used
to learn an operator, then transform compatible data. Use fit_transform when
the method has a fixed fitted operator.

.. code-block:: python

   import numpy as np
   from mne_denoise.dss import BandpassBias, DSS

   data = np.random.default_rng(0).standard_normal((8, 2000))
   bias = BandpassBias((8.0, 12.0), sfreq=250.0)
   clean = DSS(bias=bias, n_components=3).fit_transform(data)

First MNE workflow
------------------

Pass supported MNE objects directly; estimators preserve container metadata and
return copies. For example, with a preloaded Raw object named raw:

.. code-block:: python

   from mne_denoise.spectrum_interpolation import SpectrumInterpolation

   clean_raw = SpectrumInterpolation(line_freq=60.0).fit_transform(raw)

Main estimators use ``fit``, ``transform``, and ``fit_transform`` when those
operations match the method. Supported MNE objects are copied rather than
modified in place; exact container and array contracts are documented in the
:doc:`API reference <api>`.

Where to go next
----------------

* :doc:`Choose a method <methods>`
* :doc:`Browse examples <auto_examples/index>`
* :doc:`API reference <api>`
* :doc:`Evaluating denoising <evaluation>`
