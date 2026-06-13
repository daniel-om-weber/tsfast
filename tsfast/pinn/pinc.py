"""Conventional soft-IC PINN-for-control (PINC) surrogate: generic MLP with an autograd derivative."""

__all__ = [
    "PINC",
    "PINCLearner",
]

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch import Tensor, nn

from ..models.layers import SeqLinear
from .ddpinn import SurrogatePINNLearner


def _dydt_forward(net: nn.Module, feat: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """``y`` and ``dy/dt`` via forward-mode AD — one dual forward, all output channels at once.

    Time is a single scalar input per ``(batch, seq)`` row, so a forward-mode jvp with a unit
    tangent on the time channel yields ``dy/dt`` for *every* state output in a single pass — cost is
    ``O(1)`` in the number of outputs, versus the ``O(n_state)`` backward passes reverse mode needs
    (one seeded output channel each). The tangent stays attached to the network parameters, so the
    training-time backprop through ``dy/dt`` works as reverse-over-forward.
    """
    with fwAD.dual_level():
        dual_y = net(torch.cat([feat, fwAD.make_dual(t, torch.ones_like(t))], dim=-1))
        y, dy_dtau = fwAD.unpack_dual(dual_y)
        if dy_dtau is None:
            raise RuntimeError(
                "forward-mode AD produced no tangent for the time channel; the chosen activation "
                "likely lacks a forward-AD rule. Pass derivative_mode='reverse'."
            )
        return y.clone(), dy_dtau.clone()  # clone: dual views are only valid inside dual_level


def _dydt_reverse(net: nn.Module, feat: Tensor, t: Tensor, n_state: int, create_graph: bool) -> tuple[Tensor, Tensor]:
    """``y`` and ``dy/dt`` via reverse-mode autograd — one backward pass per output channel.

    The diagonal time-Jacobian (each output row depends only on its own scalar ``t``) lets one
    ``grad`` call per channel recover that channel's full derivative. Costs ``O(n_state)`` passes;
    kept as a fallback for activations without a forward-AD rule and for the equivalence test.

    Args:
        create_graph: keep the graph so ``dy/dt`` is itself differentiable (training double-backward).
    """
    t = t.detach().requires_grad_(True)
    y = net(torch.cat([feat, t], dim=-1))
    grads = [
        torch.autograd.grad(y[..., i].sum(), t, create_graph=create_graph, retain_graph=True)[0] for i in range(n_state)
    ]
    return y, torch.cat(grads, dim=-1)


class PINC(nn.Module):
    """Continuous-time PINN-for-control surrogate: a generic MLP over ``[x_k | cond | t]``.

    The conventional PINC architecture (Antonelo et al., 2021): a feed-forward network maps the
    initial state, conditioning, and normalized time directly to the state,
    ``x(t) = NN([x_k, cond, t])``. It is the deliberate foil to
    :class:`~tsfast.pinn.ddpinn.DampedAnsatzPINN` and shares its I/O contract — same normalized
    ``[-1, 1]`` coordinates, same row layout ``[x_k (n_state) | cond (n_cond) | t (1)]``, same
    ``forward(X, derivative_flag)`` signature — so it rides the same collocation sampler, residual,
    and :class:`~tsfast.pinn.ddpinn.DDPINNRollout` unchanged. It differs in exactly two ways:

    * **The initial condition** is, by default (``ic_mode="soft"``), *not* exact — there is no ansatz
      pinning ``x(t_zero) = x_k``; it is enforced softly by :class:`PINCLearner` via a penalty term.
      With ``ic_mode="hard"`` it is made exact by construction —
      ``x(t) = x_k + NN([feat, t]) − NN([feat, t_zero])`` — the DD-PINN's trick on a generic basis,
      which needs no IC loss (train it with the base :class:`SurrogatePINNLearner`). The soft/hard
      pair isolates the "exact IC by construction" advantage; hard-vs-DD-PINN isolates the basis.
    * **The time-derivative comes from autograd**, not a closed form — ``derivative_flag=True``
      differentiates the network output w.r.t. the time channel. (The hard-IC offset terms are
      constant in ``t``, so they do not change ``dx/dt``.)

    The derivative is computed by **forward-mode AD** by default: since time is a single scalar input
    and the state is the (multi-channel) output, one dual forward pass gives ``dx/dt`` for all
    channels at cost ``O(1)`` in the output dimension, versus the ``O(n_state)`` backward passes
    reverse mode needs. The two modes are numerically identical (and both support the training
    double-backward); reverse is roughly break-even at ``n_state = 2`` and forward pulls ahead as the
    state grows.

    Args:
        n_state: state dimension; ``x_k`` occupies the first ``n_state`` input channels and equals
            the output dimension.
        n_cond: conditioning channels (control + collocation vars) between ``x_k`` and ``t``.
        hidden_size: width of the MLP.
        hidden_layer: number of hidden layers in the MLP.
        act: activation function class; defaults to ``Tanh`` — smooth (so the autograd derivative is
            well-behaved), matching the DD-PINN trunk and the standard PINN choice, since
            :class:`~tsfast.models.layers.SeqLinear` otherwise defaults to ``Mish``.
        ic_mode: ``"soft"`` (default; IC enforced by a :class:`PINCLearner` penalty) or ``"hard"``
            (IC exact by construction, no IC loss; train with the base
            :class:`~tsfast.pinn.ddpinn.SurrogatePINNLearner`).
        hard_ic_style: how the hard IC is constructed (``ic_mode="hard"`` only):
            ``"subtract"`` → ``x_k + (NN([feat, t]) − NN([feat, t_zero]))`` (the DD-PINN's ``−sin c``
            trick; leaves ``dx/dt`` unconstrained, costs two net evals), or ``"multiply"`` →
            ``x_k + (t − t_zero)·NN([feat, t])`` (Lagaris trial-function form; one net eval, but pins
            ``dx/dt|_{t_zero} = NN([feat, t_zero])``).
        t_zero: normalized-time value mapped to physical ``t = 0`` (``-1`` for the ``[-1, 1]`` box);
            used only by ``ic_mode="hard"`` and must match the rollout's time map.
        derivative_mode: ``"forward"`` (forward-mode AD, ``O(1)`` in outputs) or ``"reverse"``
            (per-channel backward, ``O(n_state)``). Identical results; ``"forward"`` scales better.
    """

    def __init__(
        self,
        n_state: int,
        n_cond: int,
        hidden_size: int = 64,
        hidden_layer: int = 2,
        act: type = nn.Tanh,
        ic_mode: str = "soft",
        hard_ic_style: str = "subtract",
        t_zero: float = -1.0,
        derivative_mode: str = "forward",
    ):
        super().__init__()
        if derivative_mode not in ("forward", "reverse"):
            raise ValueError(f"derivative_mode must be 'forward' or 'reverse', got {derivative_mode!r}")
        if ic_mode not in ("soft", "hard"):
            raise ValueError(f"ic_mode must be 'soft' or 'hard', got {ic_mode!r}")
        if hard_ic_style not in ("subtract", "multiply"):
            raise ValueError(f"hard_ic_style must be 'subtract' or 'multiply', got {hard_ic_style!r}")
        self.n_state = n_state
        self.n_cond = n_cond
        self.ic_mode = ic_mode
        self.hard_ic_style = hard_ic_style
        self.t_zero = t_zero
        self.derivative_mode = derivative_mode
        self.net = SeqLinear(
            n_state + n_cond + 1,
            n_state,
            hidden_size=hidden_size,
            hidden_layer=hidden_layer,
            act=act,
        )

    def _hard_ic(self, raw: Tensor, draw: Tensor | None, x: Tensor, feat: Tensor, t: Tensor):
        """Apply the hard IC to the raw net output ``raw = NN([feat, t])`` (and its derivative ``draw``).

        Returns the IC-corrected ``x(t)`` (and ``dx/dt`` if ``draw`` is given). Both styles make
        ``x(t_zero) = x_k`` exact: ``"subtract"`` cancels the origin value (offset constant in ``t``,
        so ``dx/dt`` is unchanged); ``"multiply"`` scales by ``(t − t_zero)`` (so ``dx/dt`` picks up a
        product-rule term ``NN + (t − t_zero)·NN'``).
        """
        x_k = x[..., : self.n_state]
        if self.hard_ic_style == "subtract":
            t0 = torch.full_like(t, self.t_zero)
            y = x_k + raw - self.net(torch.cat([feat, t0], dim=-1))
            return y if draw is None else (y, draw)
        dt = t - self.t_zero  # "multiply"
        y = x_k + dt * raw
        return y if draw is None else (y, raw + dt * draw)

    def forward(self, x: Tensor, derivative_flag: bool = False):
        feat, t = x[..., :-1], x[..., -1:]  # (x_k, cond) | t
        if not derivative_flag:
            out = self.net(x)
            return self._hard_ic(out, None, x, feat, t) if self.ic_mode == "hard" else out
        with torch.enable_grad():  # override SurrogatePINNLearner.validate()'s no_grad
            if self.derivative_mode == "forward":
                raw, dy_dtau = _dydt_forward(self.net, feat, t)
            else:
                raw, dy_dtau = _dydt_reverse(self.net, feat, t, self.n_state, create_graph=self.training)
            return self._hard_ic(raw, dy_dtau, x, feat, t) if self.ic_mode == "hard" else (raw, dy_dtau)


class PINCLearner(SurrogatePINNLearner):
    """Physics-only trainer for a :class:`PINC`, adding the soft initial-condition penalty.

    Extends :class:`~tsfast.pinn.ddpinn.SurrogatePINNLearner` with the IC term the DD-PINN gets for
    free by construction. Each collocation point's ``(x_k, cond)`` is re-evaluated at the time origin
    and penalized toward ``x_k``, so the total objective is::

        loss = residual(x_phys, cond_phys, dxdt_phys) + ic_weight · ‖NN([x_k, cond, t_zero]) − x_k‖²

    The residual is the physical-unit ODE residual (unchanged from the base learner); the IC term is
    in normalized coordinates (the model's native space). They therefore live in different units, so
    ``ic_weight`` is a genuine tuning knob — the price of the soft IC, which the hard-IC DD-PINN
    avoids entirely. Everything else (collocation sampler, scalers, chain-rule factor, rollout) is
    inherited unchanged, keeping the PINC and DD-PINN arms comparable.

    Args:
        ic_weight: weight on the soft initial-condition penalty (relative to the physical residual).
        t_zero: normalized-time value mapped to physical ``t = 0`` (``-1`` for the ``[-1, 1]`` time
            box); must match the rollout's time map and the DD-PINN default.
    """

    def __init__(self, *args, ic_weight: float = 1.0, t_zero: float = -1.0, **kw):
        super().__init__(*args, **kw)
        self.ic_weight = ic_weight
        self.t_zero = t_zero

    def physics_loss(self, X: Tensor) -> Tensor:
        """ODE residual plus the soft IC penalty on a batch of normalized collocation points."""
        residual = super().physics_loss(X)
        x_k = X[..., : self.n_state]
        cond = X[..., self.n_state : self.n_state + self.n_cond]
        t0 = torch.full_like(X[..., -1:], self.t_zero)
        ic_pred = self.model(torch.cat([x_k, cond, t0], dim=-1))  # derivative_flag=False
        ic_loss = F.mse_loss(ic_pred, x_k)
        return residual + self.ic_weight * ic_loss
