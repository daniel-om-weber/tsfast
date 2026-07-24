"""Output-error port-Hamiltonian model (OE-pHNN) and its fused rollout backends.

The model (:class:`HamiltonianMLP`, :class:`PHNNCore`, :class:`PHNN`) lives in
:mod:`.core`. The naive per-step RK4 rollout evaluates four component nets four times
per sample and shatters into ~10^5 tiny GPU kernels per training step; the fused
backends collapse a whole section rollout into one launch with a hand-derived fused
BPTT backward (see ``MATH.md``):

- :mod:`.backend_c`: generic scalar-templated C++ (float and double), batch-parallel;
  the fp64 gradcheck vehicle and the fast CPU path.
- :mod:`.backend_triton`: persistent per-lane GPU kernel, float32, within the config
  caps; the fast CUDA training path.

Both run behind the ``tsfast::phnn_rollout`` / ``tsfast::phnn_rollout_train`` /
``tsfast::phnn_rollout_bwd`` custom ops registered in :mod:`.core` and are selected
through :class:`PHNN`'s ``backend`` argument (or the process-wide preference from
:func:`tsfast.models.set_backend`). The static spec lives in :mod:`.common`; each
backend module reports its own applicability via ``supports(spec, u, x0)``.
"""

from .common import PHNNSpec, params_of, spec_of  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import __all__  # noqa: F401
