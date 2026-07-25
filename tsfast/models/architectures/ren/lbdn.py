"""Lipschitz-bounded deep networks: feedforward nets whose gain is fixed by construction.

An :class:`LBDN` is a stack of :class:`SandwichLayer` blocks, each 1-Lipschitz for any value
of its free parameters, closed by a norm-bounded linear map. Composition multiplies the
bounds, so the whole network is ``gamma``-Lipschitz with no projection or penalty — the same
kind of guarantee the REN gives for a recurrence, one step down in structure.

This is the nonlinearity :class:`~.r2dn.R2DN` puts in feedback around a linear system, where
being 1-Lipschitz *as a map* is what replaces the REN's per-neuron slope restriction and
removes the equilibrium solve.

Reference: Wang & Manchester, "Direct Parameterization of Lipschitz-Bounded Deep Networks",
ICML 2023; reference implementations ``acfr/RobustNeuralNetworks.jl`` (Julia) and
``nic-barbara/R2DN`` ``robustnn/lbdn.py`` (JAX), both MIT.
"""

__all__ = [
    "ExplicitSandwich",
    "SandwichLayer",
    "LBDN",
    "lbdn_forward",
    "folded_weights",
]

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .common import _ACTS, _EPS, cayley_contraction

#: ``exp`` argument bounds for the activation scaling, wide enough to be inactive in
#: training and narrow enough that ``psi`` and its reciprocal both stay finite.
_LOG_PSI_CLAMP = 20.0

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class ExplicitSandwich:
    """The tensors a sandwich layer evaluates, once the Cayley transform has been taken.

    ``A`` and ``psi`` are absent on the output layer, which is a plain norm-bounded linear
    map::

        hidden: h ↦ √2 · A Ψ σ(√2 Ψ⁻¹ B h + bias)      with ‖[Aᵀ; Bᵀ]‖ an isometry
        output: h ↦ B h + bias                          with ‖B‖ ≤ 1
    """

    B: Tensor
    bias: Tensor
    A: Tensor | None = None
    psi: Tensor | None = None


class SandwichLayer(nn.Module):
    """A layer that is 1-Lipschitz at every value of its free parameters.

    The weights come from a Cayley transform of one unconstrained matrix, which makes
    ``[Aᵀ; Bᵀ]`` an isometry, and the activation is sandwiched between ``Ψ`` and ``Ψ⁻¹``.
    The bound then follows from the activation being slope-restricted to ``[0, 1]`` rather
    than from any norm product, so ``Ψ`` is free to rescale the units without spending gain
    budget — which is what keeps a deep stack expressive.

    Args:
        n_in: input features.
        n_out: output features.
        act: activation name; must be monotone and slope-restricted to ``[0, 1]``.
        is_output: build the norm-bounded linear form instead, with no activation and no
            ``Ψ``. Used for the last layer of an :class:`LBDN`.
    """

    def __init__(self, n_in: int, n_out: int, act: str = "relu", is_output: bool = False):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"unknown activation {act!r}, expected one of {sorted(_ACTS)}")
        self.n_in, self.n_out, self.is_output = n_in, n_out, is_output
        self._act = _ACTS[act]
        # column-stacked for the Cayley transform, so the fan-in is the row count
        self.XY = nn.Parameter(nn.init.normal_(torch.empty(n_in + n_out, n_out), std=(n_in + n_out) ** -0.5))
        self.a = nn.Parameter(self.XY.detach().pow(2).sum().add(_EPS).sqrt().reshape(1))
        self.b = nn.Parameter(torch.zeros(n_out))
        if not is_output:
            self.d = nn.Parameter(torch.zeros(n_out))

    def explicit(self) -> ExplicitSandwich:
        """Take the Cayley transform of the free parameters."""
        a, b = cayley_blocks(self.XY, self.a, self.n_out)
        if self.is_output:
            return ExplicitSandwich(B=b, bias=self.b)
        return ExplicitSandwich(B=b, bias=self.b, A=a, psi=self.d.clamp(-_LOG_PSI_CLAMP, _LOG_PSI_CLAMP).exp())

    def forward(self, h: Tensor, e: ExplicitSandwich | None = None) -> Tensor:
        """Map ``h [..., n_in]`` to ``[..., n_out]``, from a prebuilt realization if given."""
        return _sandwich_forward(self.explicit() if e is None else e, h, self._act)


def cayley_blocks(xy: Tensor, scale: Tensor, n_out: int) -> tuple[Tensor, Tensor]:
    """The ``(A, Bᵀ)`` blocks a sandwich layer's free parameters build, ``[Aᵀ; Bᵀ]`` an isometry.

    Every operation carries leading batch dimensions, so a stack of identically shaped layers
    resolves in one call — which is the point: the transform costs a matrix factorization, and
    a per-layer loop pays that plus a dozen kernel launches per layer, several times what the
    rollout it feeds costs.

    Args:
        xy: column-stacked free parameters ``[..., n_in + n_out, n_out]``.
        scale: the layer's norm parameter, broadcastable against ``xy``.
        n_out: rows of the leading block.
    """
    w = scale * xy / xy.pow(2).sum((-2, -1), keepdim=True).add(_EPS).sqrt()
    u, v = w[..., :n_out, :], w[..., n_out:, :]
    # the transform argument's symmetric part is exactly vᵀv, so the stacked blocks come out
    # orthonormal rather than merely bounded
    n = cayley_contraction(u - u.mH + v.mH @ v, v, tall=True)
    return n[..., :n_out, :], n[..., n_out:, :].mH


def _sandwich_forward(e: ExplicitSandwich, h: Tensor, act: Callable) -> Tensor:
    if e.A is None:
        return h @ e.B.mH + e.bias
    pre = _SQRT2 * (h @ e.B.mH) / e.psi + e.bias
    return _SQRT2 * (act(pre) * e.psi) @ e.A.mH


def lbdn_forward(layers: tuple[ExplicitSandwich, ...], h: Tensor, act: Callable, gamma: float = 1.0) -> Tensor:
    """Evaluate an LBDN from its explicit realization alone.

    Split out from :class:`LBDN` so a caller that builds the realization once — a rollout
    reusing it at every timestep — never pays for the Cayley transforms again, and so the
    evaluation depends on plain tensors rather than on module state.

    Args:
        layers: per-layer realizations; all but the last are the nonlinear form.
        h: input ``[..., n_input]``.
        act: the activation, monotone and slope-restricted to ``[0, 1]``.
        gamma: Lipschitz bound, applied as ``√gamma`` at each end of the stack.
    """
    scale = math.sqrt(gamma)
    h = scale * h
    for e in layers[:-1]:
        h = _sandwich_forward(e, h, act)
    return _sandwich_forward(layers[-1], scale * h, act)


def folded_weights(layers: tuple[ExplicitSandwich, ...]) -> list[Tensor]:
    """The network collapsed to one matrix and one bias per layer, in fused-kernel order.

    A hidden layer's ``√2 A Ψ σ(√2 Ψ⁻¹ B h + c)`` is ``V σ(W h + c)`` with ``W = √2 Ψ⁻¹ B``
    and ``V = √2 A Ψ``. Every factor there is constant across a rollout, so this folds them
    once per call — and then folds each ``V`` into the *following* layer's ``W``, since
    ``W_{l+1}(V_l a_l) = (W_{l+1} V_l) a_l``. What reaches a kernel is one matrix per layer:
    half the register footprint and, more to the point, half the dependent cross-lane
    reductions per timestep, which is what a sequential rollout is actually bound by
    (``MATH_R2DN.md`` §1). Autograd carries ``∂L/∂B``, ``∂L/∂ψ`` and ``∂L/∂A`` back through
    the composition.

    Returns:
        ``[W_0, c_0, ..., W_out, c_out]`` — two tensors per layer, output layer last.
    """
    out, pending = [], None
    for e in layers:
        w = e.B if e.A is None else _SQRT2 * e.B / e.psi[:, None]
        out += [w if pending is None else w @ pending, e.bias]
        pending = None if e.A is None else _SQRT2 * e.A * e.psi
    return out


class LBDN(nn.Module):
    """Feedforward network with a certified Lipschitz bound of ``gamma``.

    A drop-in replacement for an MLP whose incremental gain is a design parameter:
    ``‖f(u) - f(ũ)‖ ≤ gamma ‖u - ũ‖`` holds for every value of the free parameters, so
    ordinary SGD cannot break it. The bound is on the map, not a bound on how well it fits.

    ``gamma`` is a runtime scalar — reassign it and the next forward re-derives the network,
    since it only rescales the two ends of the stack.

    Args:
        n_input: input features.
        n_output: output features.
        hidden: width of each hidden layer; its length is the number of nonlinear layers.
        act: activation, one of ``tanh``, ``relu``, ``sigmoid``.
        gamma: certified Lipschitz bound.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden: tuple[int, ...] = (64, 64),
        act: str = "relu",
        gamma: float = 1.0,
    ):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        sizes = (n_input, *hidden, n_output)
        self.layers = nn.ModuleList(
            SandwichLayer(i, o, act, is_output=k == len(hidden)) for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:]))
        )
        self.gamma = gamma
        self._act = _ACTS[act]

    def explicit(self) -> tuple[ExplicitSandwich, ...]:
        """Per-layer realizations, in evaluation order.

        Identically shaped hidden layers — the usual case, and the only one a uniform width
        produces — take their Cayley transforms as a single batched call. The output layer
        has a different form and is always built on its own.
        """
        hidden = self.layers[:-1]
        if len(hidden) > 1 and len({tuple(layer.XY.shape) for layer in hidden}) == 1:
            a, b = cayley_blocks(
                torch.stack([layer.XY for layer in hidden]),
                torch.stack([layer.a for layer in hidden]).unsqueeze(-1),
                hidden[0].n_out,
            )
            psi = torch.stack([layer.d for layer in hidden]).clamp(-_LOG_PSI_CLAMP, _LOG_PSI_CLAMP).exp()
            built = tuple(ExplicitSandwich(b[i], layer.b, a[i], psi[i]) for i, layer in enumerate(hidden))
        else:
            built = tuple(layer.explicit() for layer in hidden)
        return (*built, self.layers[-1].explicit())

    def forward(self, h: Tensor, e: tuple[ExplicitSandwich, ...] | None = None) -> Tensor:
        """Map ``h [..., n_input]`` to ``[..., n_output]``, from a prebuilt realization if given."""
        return lbdn_forward(self.explicit() if e is None else e, h, self._act, self.gamma)
