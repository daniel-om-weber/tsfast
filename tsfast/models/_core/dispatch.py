"""Backend preference and fused-kernel resolution shared by all multi-backend models."""

__all__ = [
    "AUTOTUNE_CACHE",
    "BACKENDS",
    "get_backend",
    "set_backend",
    "use_backend",
    "resolve",
    "warn_fallback",
]

import contextvars
import importlib
import os
import warnings
from contextlib import contextmanager
from types import ModuleType

import torch

#: Whether Triton autotune results persist to the on-disk cache across processes
#: (``TSFAST_AUTOTUNE_CACHE=1``). Off by default: a cache hit skips the benchmark
#: sweep entirely, so the first process's pick — possibly measured on a contended
#: GPU — would be reused indefinitely, and the cache key distinguishes GPU
#: architecture but not SKU. Worth enabling where repeated short-lived processes
#: retune the same shapes (test suites), not where the timings matter.
AUTOTUNE_CACHE = os.environ.get("TSFAST_AUTOTUNE_CACHE", "0") == "1"

#: Recognized process-wide backend preferences. ``"auto"`` picks the fastest fused kernel
#: available for the input's device; an explicit kernel family (``"triton"``, ``"c"``,
#: ``"metal"``) forces it where applicable, with a once-warned fallback when unusable;
#: ``"reference"`` disables fused kernels entirely (pure-PyTorch implementations only).
BACKENDS = ("auto", "triton", "c", "metal", "reference")

_default_backend: str = "auto"
_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("tsfast_backend_override", default=None)


def _checked(name: str) -> str:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}, expected one of {BACKENDS}")
    return name


def get_backend() -> str:
    """The backend preference in effect: the innermost ``use_backend`` scope, else the process default.

    Inside a ``torch.compile`` trace only the process default is visible (ContextVar
    reads are untraceable; the global read is guarded, so ``set_backend`` still
    triggers a recompile). Scoped overrides keep working for dispatch that happens
    at runtime inside the custom ops.
    """
    if torch.compiler.is_compiling():
        return _default_backend
    return _override.get() or _default_backend


def set_backend(name: str) -> None:
    """Set the process-wide default backend preference (one of ``BACKENDS``)."""
    global _default_backend
    _default_backend = _checked(name)


@contextmanager
def use_backend(name: str):
    """Scope a backend preference to a ``with`` block (thread- and async-safe).

    Overrides the process default (and any outer scope) for code running inside the
    block; models whose ``backend`` attribute is ``"auto"`` and the scan dispatcher
    both consult it.
    """
    token = _override.set(_checked(name))
    try:
        yield
    finally:
        _override.reset(token)


def resolve(
    op: str,
    registry: dict[str, str | ModuleType],
    auto_order: tuple[str, ...],
    args: tuple,
    requested: str | None = None,
):
    """Pick the fused-kernel backend module for ``op``, or None for the reference path.

    Every backend module speaks the same protocol: ``supports(*args) -> str | None``
    returns a short reason when the backend cannot handle these inputs (device, dtype,
    availability, size limits) and None when it can. A candidate that declines or fails
    to import warns once per process with that reason; the caller then runs its
    reference implementation.

    Args:
        op: dotted identifier used in warning keys, e.g. ``"scan.diagonal"``.
        registry: backend name -> importable module path, or the module object itself.
            Callers whose dispatch runs inside a ``torch.compile`` trace must register
            module objects — Dynamo cannot trace ``importlib``.
        auto_order: candidate names in preference order, tried under ``"auto"``.
        args: arguments handed to each candidate's ``supports``.
        requested: explicit preference; defaults to ``get_backend()``.

    A preference naming a family this op simply does not have (e.g. a process-wide
    ``"c"`` reaching a CUDA-only fused kernel) selects the reference path silently —
    family names themselves are validated at ``set_backend``/``use_backend``.
    """
    requested = get_backend() if requested is None else requested
    if requested == "reference":
        return None
    names = auto_order if requested == "auto" else (requested,)
    for name in names:
        target = registry.get(name)
        if target is None:
            continue
        if isinstance(target, str):
            try:
                mod = importlib.import_module(target)
            except Exception as e:
                warn_fallback(
                    f"{op}.{name}",
                    f"backend {name!r} unusable for {op}: backend import failed ({e!r}); "
                    "using the reference implementation",
                )
                continue
        else:
            mod = target
        reason = mod.supports(*args)
        if reason is None:
            return mod
        warn_fallback(
            f"{op}.{name}",
            f"backend {name!r} unusable for {op}: {reason}; using the reference implementation",
        )
    return None


_warned: set[str] = set()


def warn_fallback(key: str, message: str) -> None:
    """Emit ``message`` as a RuntimeWarning exactly once per process per ``key``.

    Backend resolution falls back silently on every call after the first: a long
    training run should mention a missing kernel once, not per batch. Keys should
    identify the (model, backend) pair, e.g. ``"scan.selective.triton"``. Silent
    inside a ``torch.compile`` trace (warning state is untraceable side effect).
    """
    if torch.compiler.is_compiling():
        return
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
