Getting started
===============

Installation
------------

Install the package from PyPI:

.. code-block:: console

   pip install mne-denoise

Install the optional MNE-Python integration for ``Raw``, ``Epochs``, and
``Evoked`` inputs:

.. code-block:: console

   pip install "mne-denoise[mne]"

The ``viz`` and ``progress`` extras provide optional plotting and tqdm-based
progress presentation:

.. code-block:: console

   pip install "mne-denoise[viz]"
   pip install "mne-denoise[progress]"

Basic estimator pattern
-----------------------

Most estimators follow the scikit-learn pattern: configure an estimator, call
``fit`` on the data used to learn its operator, and call ``transform`` on the
same or another compatible recording. ``fit_transform`` composes those calls
when the method has a fixed fitted operator.

For example, DSS can extract a narrow-band component from a NumPy array. The
array convention here is ``(n_channels, n_times)``:

.. code-block:: python

   import numpy as np
   from mne_denoise.dss import BandpassBias, DSS

   rng = np.random.default_rng(0)
   data = rng.standard_normal((8, 2000))
   bias = BandpassBias(freq_band=(8.0, 12.0), sfreq=250.0)

   dss = DSS(bias=bias, n_components=3, component_action="extract")
   alpha_sources = dss.fit_transform(data)

The detailed method pages and API reference describe the different array
layouts, fitted attributes, selection rules, and validation requirements for
each estimator. Do not assume that the array convention of one method applies
to another.

MNE-Python integration
----------------------

When an estimator supports MNE containers, pass the object directly. The
estimator uses the object's sampling frequency and returns a copied container;
the method documentation describes channel selection and metadata guarantees.
For example, a preloaded ``raw`` object can be cleaned with spectrum
interpolation without extracting and reconstructing its data manually:

.. code-block:: python

   from mne_denoise.spectrum_interpolation import SpectrumInterpolation

   cleaner = SpectrumInterpolation(line_freq=60.0, n_harmonics=3)
   clean_raw = cleaner.fit_transform(raw)

Here ``raw`` stands for an MNE ``Raw`` object already loaded by the calling
application. The example intentionally does not assume a local data filename.

Choosing a method
-----------------

Use the method pages to identify the assumptions that match the problem:

* **ASR** targets transient, high-variance subspace changes in continuous EEG.
* **DSS** and **TimeShiftDSS** enhance or suppress components defined by a
  reproducibility, spectral, temporal, or lagged-trial bias.
* **BSS-CCA** targets lagged-correlation components associated with muscle
  artifacts without requiring a reference channel.
* **iCanClean** uses physical or pseudo-reference channels to identify shared
  variance.
* **SNS** targets sensor-specific noise under a spatial-redundancy assumption.
* **SOUND** uses a forward model to estimate channel-specific noise.
* **Spectrum interpolation** targets narrow-band line noise in the Fourier
  amplitude spectrum.
* **SSA** provides univariate basic or local delay-coordinate decompositions.
* **SSP-SIR** combines artifact-subspace projection with source-informed
  reconstruction, especially for TMS-evoked EEG workflows.
* **ZapLine** uses DSS to remove power-line noise, with an adaptive extension.

Artifact attenuation is not evidence that the desired neural signal was
preserved. Use held-out data, controls, and domain-appropriate diagnostics to
assess both outcomes; see :doc:`evaluation`.

Next steps
----------

* Read the relevant method pages and their cited primary publications.
* Consult the :doc:`api` reference for exact signatures and fitted attributes.
* Browse the :doc:`auto_examples/index` gallery for complete workflows.
* See :doc:`citing` for software and method citation guidance.
