"""Fused rollout backends for the port-Hamiltonian core (:class:`tsfast.models.phnn.PHNNCore`).

The naive per-step rollout of the OE-pHNN evaluates four component nets four times
per sample (RK4) and computes the Hamiltonian state-gradient in closed form; on GPU
this shatters into ~10^5 tiny kernels per training step and becomes
kernel-granularity bound. These backends fuse the whole section rollout into one
launch (Triton, one program per batch lane) or one batch-parallel C++ call, with a
hand-derived fused BPTT backward (see ``MATH.md``).

- ``c``: generic scalar-templated C++ (float and double), batch-parallel; the fp64
  gradcheck vehicle and the fast CPU path.
- ``triton``: persistent per-lane GPU kernel, float32, within the config caps
  (``supports``); the fast CUDA training path.

Both are exposed through :class:`tsfast.models.phnn.PHNN`'s ``backend`` selector.
"""

__all__ = [
    "PHNNSpec",
    "spec_of",
    "params_of",
    "supports",
]

from .common import PHNNSpec, params_of, spec_of, supports
