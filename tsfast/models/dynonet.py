"""Block-oriented dynoNet models built from learnable linear transfer functions and static nonlinearities."""

__all__ = [
    "linear_recurrence",
    "LinearDynamicalOperator",
    "DynoNet",
]

import torch
import torch.nn.functional as F
from torch import nn

from .layers import SeqLinear


def linear_recurrence(A: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the linear recurrence ``x_t = A x_{t-1} + v_t`` with constant ``A`` via a log-doubling scan.

    Because ``A`` is constant along the sequence, the recurrence is a prefix sum
    ``x_t = A^t x_0 + sum_k A^(t-k) v_k`` that parallelizes exactly: each doubling step extends
    the summation window by a factor of two using one batched matmul over the whole sequence,
    so the sequential depth is ``ceil(log2(L))`` instead of ``L``. Exact for any spectral
    radius of ``A`` and differentiable by plain autograd on any device.

    Args:
        A: transition matrices ``[..., n, n]``, broadcast against the leading dims of ``v``.
        v: input sequence ``[..., L, n]``.
        x0: initial state ``[..., n]``; zeros if None.

    Returns:
        States ``x_1 .. x_L`` as ``[..., L, n]``.
    """
    L = v.shape[-2]
    if x0 is not None:
        v = torch.cat((v[..., :1, :] + x0.unsqueeze(-2) @ A.transpose(-1, -2), v[..., 1:, :]), dim=-2)
    x, Ap, s = v, A, 1
    while s < L:
        shifted = F.pad(x[..., :-s, :], (0, 0, s, 0))
        x = x + shifted @ Ap.transpose(-1, -2)
        Ap = Ap @ Ap
        s *= 2
    return x


def _linear_recurrence_sequential(A: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Reference implementation of ``linear_recurrence`` as a per-timestep Python loop."""
    if x0 is None:
        x = v.new_zeros(v.shape[:-2] + v.shape[-1:])
    else:
        x = x0
    At = A.transpose(-1, -2)
    outs = []
    for t in range(v.shape[-2]):
        x = (x.unsqueeze(-2) @ At).squeeze(-2) + v[..., t, :]
        outs.append(x)
    return torch.stack(outs, dim=-2)


class LinearDynamicalOperator(nn.Module):
    """MIMO bank of learnable rational transfer functions ``G(q) = B(q) / A(q)`` (dynoNet G-block).

    Each (output, input) channel pair owns an independent SISO filter with ``nb`` numerator
    taps ``b_0 .. b_{nb-1}`` and ``na`` monic-denominator coefficients ``a_1 .. a_na``; output
    channels sum the filtered contributions of all inputs. The numerator is a grouped causal
    convolution; the denominator recurrence runs in state-space (companion) form through
    ``linear_recurrence``, so the whole operator is exact and sequence-parallel.

    Coefficients are unconstrained as in Forgione & Piga (2021, arXiv:2006.02250; full
    citation on ``DynoNet``): ``b`` starts small and random, ``a`` starts at zero (all poles
    at the origin — a pure FIR filter), so the operator is stable at initialization but
    poles may leave the unit circle during training.

    The internal pair flattening is input-major (``index = j_in * out_channels + i_out``),
    forced by ``conv1d`` group semantics; every reshape below relies on this ordering.

    Args:
        in_channels: number of input signals.
        out_channels: number of output signals.
        nb: number of numerator (FIR) taps per filter.
        na: denominator order per filter; ``0`` gives a pure FIR operator.
        backend: ``"scan"`` (log-doubling, parallel) or ``"eager"`` (sequential loop).
    """

    def __init__(self, in_channels: int, out_channels: int, nb: int = 8, na: int = 2, backend: str = "scan"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb = nb
        self.na = na
        self.backend = backend
        self.b_coeff = nn.Parameter(torch.randn(out_channels, in_channels, nb) * 0.01)
        self.a_coeff = nn.Parameter(torch.zeros(out_channels, in_channels, na))

    @property
    def n_pairs(self) -> int:
        return self.in_channels * self.out_channels

    def _companion(self) -> torch.Tensor:
        """Companion transition matrices ``[n_pairs, na, na]`` of the monic denominators."""
        a = self.a_coeff.permute(1, 0, 2).reshape(self.n_pairs, self.na)
        shift = torch.eye(self.na, dtype=a.dtype, device=a.device)[:-1].expand(self.n_pairs, self.na - 1, self.na)
        return torch.cat((-a.unsqueeze(1), shift), dim=1)

    def forward(self, u: torch.Tensor, state: dict | None = None, return_state: bool = False):
        """Filter the input sequence through all channel pairs and sum over inputs.

        Args:
            u: input sequence ``[batch, seq, in_channels]``.
            state: carried filter state ``{"u": FIR tail, "x": IIR states}`` from a previous
                chunk; zero initial conditions if None.
            return_state: if ``True``, return ``(output, new_state)``.

        Returns:
            Output sequence ``[batch, seq, out_channels]``, optionally with the new state.
        """
        B, L, _ = u.shape
        match state:
            case {"u": u_tail, "x": x0}:
                pass
            case None:
                u_tail = u.new_zeros(B, self.nb - 1, self.in_channels)
                x0 = u.new_zeros(B, self.n_pairs, self.na)
            case _:
                raise TypeError(f"expected state dict {{'u': tensor, 'x': tensor}}, got {type(state)}")

        u_buf = torch.cat((u_tail, u), dim=1)
        # conv1d computes cross-correlation, so the taps are flipped to realize b_0 u_t + ... + b_{nb-1} u_{t-nb+1};
        # the carried tail replaces the zero left-padding of a cold-started causal convolution.
        weight = self.b_coeff.permute(1, 0, 2).reshape(self.n_pairs, 1, self.nb).flip(-1)
        w = F.conv1d(u_buf.transpose(1, 2), weight, groups=self.in_channels)

        if self.na > 0:
            v = F.pad(w.unsqueeze(-1), (0, self.na - 1))
            match self.backend:
                case "scan":
                    x = linear_recurrence(self._companion(), v, x0)
                case "eager":
                    x = _linear_recurrence_sequential(self._companion(), v, x0)
                case unknown:
                    raise ValueError(f"unknown backend {unknown!r}, expected 'scan' or 'eager'")
            y_pairs = x[..., 0]
            x_last = x[..., -1, :]
        else:
            y_pairs = w
            x_last = u.new_zeros(B, self.n_pairs, 0)

        y = y_pairs.view(B, self.in_channels, self.out_channels, L).sum(1).transpose(1, 2)
        if not return_state:
            return y
        new_state = {"u": u_buf[:, u_buf.shape[1] - (self.nb - 1) :], "x": x_last}
        return y, new_state


class DynoNet(nn.Module):
    """dynoNet: linear transfer-function blocks G interconnected with a static nonlinearity F.

    Wiener-Hammerstein-like structure ``G1 -> F -> G2`` with an optional parallel linear
    bypass path, the canonical architecture of Forgione & Piga (2021). ``F`` is a pointwise
    MLP (memoryless), so all dynamics live in the ``LinearDynamicalOperator`` blocks.

    With ``return_state=True`` the model follows the stateful-model protocol
    (``forward(u, state=...) -> (out, state)``); the carried state holds each G-block's FIR
    tail and IIR states, so chunked rollouts are exactly equivalent to the full sequence and
    ``TbpttLearner`` works unchanged. Initial conditions are zero unless ``state`` is passed.

    References:
        M. Forgione and D. Piga, "dynoNet: A neural network architecture for learning
        dynamical systems," International Journal of Adaptive Control and Signal
        Processing, 35(4):612-626, 2021. arXiv:2006.02250.

    Args:
        input_size: number of input signals.
        output_size: number of output signals.
        n_channels: signal width between the blocks.
        nb: numerator taps per filter in every G-block.
        na: denominator order per filter in every G-block.
        hidden_size: hidden width of the static nonlinearity MLP.
        hidden_layers: number of hidden layers of the static nonlinearity MLP.
        act: activation class of the static nonlinearity MLP.
        bypass: add a parallel linear path ``G_lin`` from input to output.
        backend: execution backend of the G-blocks, see ``LinearDynamicalOperator``.
        return_state: if ``True``, return ``(output, state)`` tuple.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_channels: int = 8,
        nb: int = 8,
        na: int = 2,
        hidden_size: int = 32,
        hidden_layers: int = 1,
        act: type[nn.Module] = nn.Tanh,
        bypass: bool = True,
        backend: str = "scan",
        return_state: bool = False,
    ):
        super().__init__()
        self.return_state = return_state
        self.g1 = LinearDynamicalOperator(input_size, n_channels, nb, na, backend)
        self.f = SeqLinear(n_channels, n_channels, hidden_size, hidden_layer=hidden_layers, act=act)
        self.g2 = LinearDynamicalOperator(n_channels, output_size, nb, na, backend)
        self.g_lin = LinearDynamicalOperator(input_size, output_size, nb, na, backend) if bypass else None

    @property
    def backend(self) -> str:
        return self.g1.backend

    @backend.setter
    def backend(self, value: str):
        for m in self.modules():
            if isinstance(m, LinearDynamicalOperator):
                m.backend = value

    def forward(self, u: torch.Tensor, state: dict | None = None):
        """Run the block interconnection over the input sequence.

        Args:
            u: input sequence ``[batch, seq, input_size]``.
            state: carried state ``{"g1": ..., "g2": ..., "lin": ...}`` from a previous chunk.

        Returns:
            Output sequence ``[batch, seq, output_size]``, or ``(sequence, state)`` when
            ``return_state`` is set.
        """
        match state:
            case None:
                s1 = s2 = s_lin = None
            case dict():
                s1, s2, s_lin = state.get("g1"), state.get("g2"), state.get("lin")
            case _:
                raise TypeError(f"expected state dict, got {type(state)}")
        y1, s1 = self.g1(u, state=s1, return_state=True)
        y, s2 = self.g2(self.f(y1), state=s2, return_state=True)
        new_state = {"g1": s1, "g2": s2}
        if self.g_lin is not None:
            y_lin, s_lin = self.g_lin(u, state=s_lin, return_state=True)
            y = y + y_lin
            new_state["lin"] = s_lin
        if self.return_state:
            return y, new_state
        return y
