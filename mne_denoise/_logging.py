"""Centralized MNE-style logging for :mod:`mne_denoise`.

``verbose`` is deliberately a logging control, rather than an algorithm
parameter.  ``None`` leaves the configured logger alone, booleans map to the
usual MNE levels, and strings/integers are standard :mod:`logging` levels.
Public operations use :func:`verbose` or :func:`use_log_level` so a per-call
override is always restored, including when the operation raises.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from numbers import Integral
from typing import Any, TypeVar

logger = logging.getLogger("mne_denoise")

_UNSET = object()
_active_verbose_scope = ContextVar(
    "mne_denoise_active_verbose_scope",
    default=_UNSET,
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _level_from_verbose(verbose: bool | str | int | None) -> int | None:
    """Resolve one MNE-style verbosity value to a logging level."""
    if verbose is None:
        return None
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    if isinstance(verbose, str):
        name = verbose.upper()
        try:
            return int(logging._nameToLevel[name])  # noqa: SLF001
        except KeyError as err:
            raise ValueError(
                f"Unknown logging level {verbose!r}; use a standard logging "
                "level name or integer."
            ) from err
    if isinstance(verbose, Integral):
        return int(verbose)
    raise TypeError(
        "verbose must be None, a bool, a standard logging level name, or an "
        f"integer; got {type(verbose).__name__}."
    )


@contextmanager
def use_log_level(verbose: bool | str | int | None) -> Iterator[None]:
    """Temporarily apply an MNE-style verbosity value.

    ``verbose=None`` inherits the existing logger configuration.  A concrete
    value is restored in a ``finally`` block, which makes nested algorithm
    calls and exceptions safe.
    """
    level = _level_from_verbose(verbose)
    previous = logger.level
    token = _active_verbose_scope.set(level)
    try:
        if level is not None:
            logger.setLevel(level)
        yield
    finally:
        if level is not None:
            logger.setLevel(previous)
        _active_verbose_scope.reset(token)


def verbose(function: _F) -> _F:
    """Decorate a public operation with a temporary ``verbose`` override.

    The decorator accepts the same forms as MNE-Python's ``@verbose``. An
    explicit argument, including ``None``, starts a new scope. Otherwise an
    active outer scope is inherited before falling back to ``self.verbose``.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        if "verbose" in bound.arguments:
            with use_log_level(bound.arguments["verbose"]):
                return function(*args, **kwargs)

        if _active_verbose_scope.get() is not _UNSET:
            return function(*args, **kwargs)

        if args and hasattr(args[0], "verbose"):
            with use_log_level(args[0].verbose):
                return function(*args, **kwargs)

        return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
