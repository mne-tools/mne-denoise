Reconciled structured callbacks around the immutable ``ProgressEvent``
contract across the supported iterative, segmented, and channel-wise methods.
Callbacks are runtime observers and remain independent from package logging.

Added an optional ``TqdmProgress`` adapter. ``tqdm`` remains optional, and the
adapter consumes the existing structured progress callbacks.
