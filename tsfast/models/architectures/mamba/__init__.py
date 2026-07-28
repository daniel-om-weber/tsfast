"""Mamba selective-SSM layer and deep stack, with its private fused Triton kernels.

The model lives in :mod:`.core`; :mod:`.conv_triton`, :mod:`.mamba_triton` and
:mod:`.backend_c` are the fused causal-conv and selective-SSM kernels it dispatches to
directly — Triton for CUDA, generated C++ for CPU (the generic scan recurrences it shares
with the other scan-family models live in :mod:`tsfast.models._core.scan`).
"""

from .core import *  # noqa: F401,F403
from .core import __all__  # noqa: F401
