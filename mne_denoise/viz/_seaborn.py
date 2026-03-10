"""Internal seaborn helpers for viz modules."""

from __future__ import annotations

import contextlib
import warnings


def _try_import_seaborn():
    """Import seaborn or raise a clear error."""
    try:
        import seaborn as sns

        return sns
    except ImportError as err:
        raise ImportError(
            "seaborn is required for this plotting function. "
            "Install it with: pip install seaborn"
        ) from err


@contextlib.contextmanager
def _suppress_seaborn_plot_warnings():
    """Suppress known seaborn/matplotlib warnings only around plot calls."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"Passing `palette` without assigning `hue`",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"set_ticklabels\(\) should only be used",
        )
        yield
