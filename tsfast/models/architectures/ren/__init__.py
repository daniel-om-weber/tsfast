"""Sequence models with certificates that hold at every value of their free parameters.

Two architectures from the same lineage, both feedback interconnections of a linear system
with a nonlinearity, and both certified by construction rather than by projection, penalty or
post-hoc verification:

- :mod:`.core` — the recurrent equilibrium network (:class:`~.core.REN`), whose nonlinearity
  is a layer of slope-restricted neurons with feedback onto itself. Contracting, Lipschitz
  and ``(Q,S,R)``-dissipative variants.
- :mod:`.r2dn` — the robust recurrent deep network (:class:`~.r2dn.R2DN`), which drops that
  feedback and replaces the neurons with a 1-Lipschitz network (:mod:`.lbdn`). Same
  certificates, no equilibrium solve, and a certificate matrix whose size no longer grows
  with nonlinear capacity.

The static specs and the explicit-realization containers live in :mod:`.common`.

Both are laid out around one seam: the certificate construction runs once per forward and
produces a bundle of plain tensors, and the sequential rollout reads nothing else. The
``(Q,S,R)`` variants and the fused rollout kernels attach on opposite sides of it without
meeting — a kernel that gets ``∂L/∂A`` right needs no opinion about ``∂L/∂X``, and autograd
carries the rest.

The REN's rollout is sequential twice over, along the sequence and along the ``n_nl`` neurons,
so the naive loop is dispatch-bound by two orders of magnitude. The fused backends collapse a
whole rollout into one launch with a hand-derived BPTT backward (see ``MATH_REN.md``):

- :mod:`.backend_c`: generic scalar-templated C++ (float and double), batch-parallel;
  the fp64 gradcheck vehicle and the fast CPU path.
- :mod:`.backend_triton`: persistent per-trajectory GPU kernel, float32, within the
  config caps; the fast CUDA training path.

Both run behind the ``tsfast::ren_rollout`` / ``tsfast::ren_rollout_train`` /
``tsfast::ren_rollout_bwd`` custom ops registered in :mod:`.core` and are selected through
:class:`~.core.REN`'s ``backend`` argument (or the process-wide preference from
:func:`tsfast.models.set_backend`); each backend reports its own applicability via
``supports(spec, u, x0)``.

The R2DN gets the same treatment in :mod:`.r2dn_backend_triton` (``MATH_R2DN.md``,
``tsfast::r2dn_rollout*``). Deleting the sweep makes its *step* cheap but leaves the launches
untouched, so eager it is dispatch-bound just as the REN is; fused, its cost is flat in
nonlinear capacity where the REN's still grows with ``n_nl``, which is where the
architecture's scalability claim finally shows.
"""

from .common import ExplicitREN, RENSpec  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .lbdn import *  # noqa: F401,F403
from .lbdn import __all__ as _lbdn_all
from .r2dn import *  # noqa: F401,F403
from .r2dn import __all__ as _r2dn_all

__all__ = [*_core_all, *_lbdn_all, *_r2dn_all]
