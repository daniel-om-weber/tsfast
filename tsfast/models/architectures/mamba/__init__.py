"""Mamba selective-SSM layer and deep stack, with its private fused Triton kernels.

The model lives in :mod:`.core`; :mod:`.conv_triton` and :mod:`.mamba_triton` are the
fused causal-conv and selective-scan kernels it dispatches to directly (the generic
scan recurrences it shares with the other scan-family models live in
:mod:`tsfast.models._core.scan`).
"""

from .core import *  # noqa: F401,F403
from .core import __all__  # noqa: F401
