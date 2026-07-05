"""Shared kernel backends for the scan recurrences, one module per (op, backend) pair.

These are the kernels reached through :mod:`tsfast.models._core.scan`'s dispatcher, so
they are shared across models (``diagonal`` by LRU and S5, ``selective`` by Mamba). A
kernel a single model dispatches to directly instead lives in that model's package
(e.g. dynonet's ``allpole_triton``), not here.

Module naming: ``{op}_{backend}.py`` with ``op`` in ``{"selective", "diagonal"}`` and
``backend`` in ``{"triton", "c"}``. Each module exposes:

- ``supports(lam, v, x0) -> str | None`` — None when the backend can handle these
  tensors (device, dtype, shape limits); otherwise a short reason used in the
  once-per-process fallback warning.
- ``run(lam, v, x0) -> Tensor`` — the recurrence output, autograd-capable
  (custom ``autograd.Function`` with a fused/analytic backward inside).

Dispatch, fallback, and warning policy live in ``tsfast.models._core.scan``; the
reference semantics are the doubling-scan implementations there and the
sequential loop ``_diagonal_recurrence_sequential``.
"""
