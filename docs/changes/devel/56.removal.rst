**DSS**:
- Deprecated ``return_type``, ``n_select``, and ``rank`` in favor of
  ``component_action``, ``component_selection``, and ``whitening_rank``.
  Compatibility names emit ``FutureWarning`` throughout 0.x and are scheduled
  for removal in 1.0. The historical sensor-valued ``return_type``
  ``fit_transform`` behavior remains available only through that deprecated
  compatibility path.
