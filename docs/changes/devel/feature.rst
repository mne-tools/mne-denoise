Reconciled structured callbacks around the immutable ``ProgressEvent``
contract across the supported iterative, segmented, and channel-wise methods.
Callbacks are runtime observers and remain independent from package logging.

Added an optional ``TqdmProgress`` adapter. ``tqdm`` remains optional, and the
adapter consumes the existing structured progress callbacks.

The base installation no longer requires MNE-Python or Matplotlib. Install the
``mne`` extra for MNE objects and the ``viz`` extra for plotting; tqdm remains
available through the existing ``progress`` extra.
