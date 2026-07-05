"""Shared machinery for the model architectures — not architectures themselves.

Sequence layers and scalers, stateful-model helpers (state specs, CUDA-graph wrapper),
the backend-dispatch warning helper, the scan-recurrence library and its shared kernel
backends, and the shared C/Triton kernel primitives. Architectures under
:mod:`tsfast.models.architectures` depend on this package; nothing here depends on them.

Public building blocks (layers, scalers) are re-exported through the
:mod:`tsfast.models` facade; the rest is internal and imported by path where needed.
"""
