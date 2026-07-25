"""Static spec, explicit-realization container, and the constructions REN and R2DN share.

The direct parameterization splits in two at :class:`ExplicitREN`: the certificate
construction turns the free parameters into these tensors once per forward, and the
sequential rollout consumes nothing else. A fused rollout backend therefore needs no
knowledge of the LMI at all.
"""

__all__ = [
    "RENSpec",
    "ExplicitREN",
]

import itertools
from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

#: Activations admissible in the nonlinear layer: monotone and slope-restricted to
#: ``[0, 1]``, which is what the certificates need of ``σ`` and nothing more.
_ACTS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}

#: Regularization floor on the certificate matrix ``H``. Load-bearing rather than cosmetic:
#: the REN's ``Λ⁻¹ = 2/diag(H22)`` divides by a diagonal entry that only ``ε`` keeps off zero,
#: and it is the margin by which the Cayley constructions below stay strict.
_EPS = torch.finfo(torch.float32).eps

#: Multiplicative slack applied to a *feedthrough* built by the Cayley transform, i.e. one
#: whose remaining slack a later solve inverts.
#:
#: ``ε`` alone will not do there. The transform gives ``I - NᵀN = 4(I+m)⁻ᵀ(XᵀX + εI)(I+m)⁻¹``,
#: so the margin shrinks quadratically as the free skew part of ``m`` grows and is at best
#: ``~4ε`` to begin with — below float32 resolution. Training walks straight into it: a gain
#: budget is useful, so the optimizer drives ``X → 0``, ``‖N‖ → 1``, and the supply-rate weight
#: the construction inverts (``γI - DᵀD``, which *is* that slack) rounds to singular. Observed
#: on a Lipschitz ``R2DN`` after ~20 epochs, as a ``linalg.solve`` failure.
#:
#: A relative margin does not depend on ``‖m‖``, keeps the attainable set closed, and costs
#: 0.1% of the reachable feedthrough. The certificate only needs the bound, never equality,
#: so a shrunken feedthrough certifies the same ``gamma``.
_CONTRACTION_SLACK = 1e-3


def _lecun_normal_(t: Tensor) -> Tensor:
    return nn.init.normal_(t, std=t.shape[-1] ** -0.5) if t.numel() else t


def cayley_contraction(m: Tensor, z: Tensor, tall: bool) -> Tensor:
    """A matrix of spectral norm below one, from an ``m`` whose symmetric part dominates ``zᵀz``.

    Stacks the Cayley transform ``(I - m)(I + m)⁻¹`` on ``-2 z (I + m)⁻¹``. The pair
    satisfies ``NᵀN = I - 4(I + m)⁻ᵀ S (I + m)⁻¹`` with ``S`` the symmetric part of ``m``
    less ``zᵀz``, so any ``m = XᵀX + Y - Yᵀ + zᵀz + εI`` built from free ``X, Y, z`` gives
    ``‖N‖ < 1``, strictly, with the margin set by ``ε``.

    Both blocks are ``(I + m)⁻¹`` acting on a different right-hand side, so they go through
    one solve against the concatenation rather than two against the same factorization — a
    factorization being the expensive part, and on GPU a launch-heavy one. Leading batch
    dimensions are carried through, which is what lets a stack of layers resolve at once.

    Args:
        m: the ``d × d`` transform argument, optionally batched.
        z: the free block stacked under the transform, ``[..., rows, d]``.
        tall: build the ``(d + rows) × d`` orientation; ``False`` transposes it, which is
            how the same construction reaches a wide non-square bound.
    """
    eye = torch.eye(m.shape[-1], dtype=m.dtype, device=m.device)
    if tall:
        rhs = torch.cat((eye - m, -2 * z), dim=-2)
        return torch.linalg.solve((eye + m).mH, rhs.mH).mH
    return torch.linalg.solve(eye + m, torch.cat((eye - m, -2 * z.mH), dim=-1))


def parameter_cache_key(module: nn.Module, *extra) -> tuple:
    """Identity of a module's current parameter values, for the explicit-realization caches.

    ``Tensor._version`` bumps on every in-place write, so an optimizer step invalidates the
    key without anything having to notify the model.
    """
    tensors = itertools.chain(module.parameters(), module.buffers())
    return (*extra, *((t._version, t.device, t.dtype) for t in tensors))


@dataclass(frozen=True)
class RENSpec:
    """Static description of a REN, used to specialize/cache kernels.

    The certified gain ``gamma`` is deliberately not a field: like ``dt`` for the PHNN it
    is a runtime scalar, so retuning it must not invalidate a compiled rollout.

    Args:
        n_state: state dimension ``nx``.
        n_input: exogenous input dimension ``nu``.
        n_output: observed output dimension ``ny``.
        n_nl: neurons in the equilibrium layer ``nv``.
        variant: ``"contracting"``, ``"lipschitz"`` or ``"dissipative"``.
        alpha: contraction rate ``ᾱ ∈ (0, 1]``.
        act: activation name; must be monotone and slope-restricted to ``[0, 1]``.
    """

    n_state: int
    n_input: int
    n_output: int
    n_nl: int
    variant: str
    alpha: float
    act: str

    @property
    def n_h(self) -> int:
        """Side length of the certificate matrix ``H``: ``2*n_state + n_nl``."""
        return 2 * self.n_state + self.n_nl


@dataclass(frozen=True)
class ExplicitREN:
    """The realization the rollout actually runs, as plain tensors.

    Per sample, with ``σ`` the activation and ``D11`` strictly lower triangular::

        w   = σ(x C1ᵀ + u D12ᵀ + bv + w D11ᵀ)      (one forward substitution over nv)
        x⁺  = x Aᵀ + w B1ᵀ + u B2ᵀ + bx
        y   = x C2ᵀ + w D21ᵀ + u D22ᵀ + by

    ``y`` observes the state *before* the update, so a rollout of length ``L`` from ``x0``
    returns ``y_0 .. y_{L-1}`` and carries ``x_L``.
    """

    A: Tensor
    B1: Tensor
    B2: Tensor
    C1: Tensor
    D11: Tensor
    D12: Tensor
    C2: Tensor
    D21: Tensor
    D22: Tensor
    bx: Tensor
    bv: Tensor
    by: Tensor

    @property
    def tensors(self) -> list[Tensor]:
        """The realization flattened in the canonical order the custom ops and kernels use."""
        return [getattr(self, f.name) for f in fields(self)]
