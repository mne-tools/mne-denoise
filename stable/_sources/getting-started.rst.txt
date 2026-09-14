Getting started
===============

Installation
------------

Install the latest release from PyPI:

.. code-block:: console

   pip install mne-denoise

For development work, clone the repository and install the optional extras:

.. code-block:: console

   pip install -e .[dev,docs]
   pre-commit install

Basic usage
-----------

Here is a simple example of using Linear DSS to enhance an evoked response.

.. code-block:: python

   import numpy as np
   from mne_denoise.dss import compute_dss, compute_covariance

   # Simulate data: (n_channels, n_times, n_trials)
   n_ch, n_times, n_trials = 10, 1000, 50
   data = np.random.randn(n_ch, n_times, n_trials)

   # Add a signal component to the first few channels
   t = np.linspace(0, 1, n_times)
   signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
   data[:3, :, :] += signal[None, :, None] * 0.5

   # 1. Compute Base Variace (Covariance over all data)
   data_2d = data.reshape(n_ch, -1)
   cov_baseline = compute_covariance(data_2d)

   # 2. Compute Biased Variance (Covariance of the trial average)
   #    This 'bias' selects for activity consistent across trials
   data_avg = data.mean(axis=2)
   cov_biased = compute_covariance(data_avg)

   # 3. Compute DSS
   #    Returns filters to extract the signal
   filters, patterns, eigenvalues = compute_dss(
       cov_baseline, cov_biased, n_components=3
   )

   print(f"Top 3 eigenvalues (score): {eigenvalues}")

MNE integration
---------------

You can use the ``DSS`` class directly with MNE objects.

.. code-block:: python

   import mne
   from mne_denoise.dss import DSS, TrialAverageBias

   # Load epochs
   epochs = mne.read_epochs("sample_epochs.fif")

   # Apply DSS to enhance evoked response
   dss = DSS(bias=TrialAverageBias(), n_components=5)
   dss.fit(epochs)

   # Return denoised epochs (projected back to sensor space)
   epochs_clean = dss.transform(epochs, return_type="epochs")

Example gallery
---------------

The repository ships with runnable scripts in ``examples/``. Start with
``examples/dss/plot_01_dss_fundamentals.py`` to see the functional API end-to-end.
