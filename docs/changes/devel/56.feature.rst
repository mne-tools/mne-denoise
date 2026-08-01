**DSS**:
- Added the explicit ``component_action`` (``"extract"``, ``"retain"``, or
  ``"subtract"``), ``component_selection``, and ``whitening_rank`` contract.
  Canonical sensor-space operations preserve NumPy layout or the input MNE
  container, channels, and metadata, and canonical ``fit_transform`` is now
  exactly ``fit(...).transform(...)``.
