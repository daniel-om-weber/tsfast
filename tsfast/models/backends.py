"""Shared helpers for models with multiple execution backends (triton/c/compiled/eager)."""

__all__ = [
    "warn_fallback",
]

import warnings

_warned: set[str] = set()


def warn_fallback(key: str, message: str) -> None:
    """Emit ``message`` as a RuntimeWarning exactly once per process per ``key``.

    Backend resolution falls back silently on every call after the first: a long
    training run should mention a missing kernel once, not per batch. Keys should
    identify the (model, backend) pair, e.g. ``"scan.selective.triton"``.
    """
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
