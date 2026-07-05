"""Shared Triton primitives for the generated-Triton execution backends.

The activation source table, the on-chip padded-width cap, the power-of-two padding
helper, and the CUDA/triton availability probe are common to every generated-Triton
kernel (NeuralStateSpace, NARX, and the diagonal-SSM scan backend). They live here so
those backends depend on a shared kernel toolkit rather than on any one model's module.

Names are kept underscore-prefixed and identical to their original definitions so the
backends import them unchanged (only the module path moved).
"""

__all__ = [
    "is_available",
]

import torch

_ACT_TL = {
    "tanh": ("2.0 * tl.sigmoid(2.0 * ({a})) - 1.0", "(1.0 - {z} * {z})"),
    "sigmoid": ("tl.sigmoid({a})", "({z} * (1.0 - {z}))"),
    "relu": ("tl.maximum({a}, 0.0)", "tl.where({z} > 0.0, 1.0, 0.0)"),
}

_MAX_PADDED_WIDTH = 128


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()
