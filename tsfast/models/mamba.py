"""Mamba models: selective state-space layers with input-dependent dynamics (Gu & Dao 2023)."""

__all__ = [
    "MambaLayer",
    "DeepMamba",
]

import math

import torch
import torch.nn.functional as F
from torch import nn

from .scan import _diagonal_recurrence_sequential, selective_recurrence


class MambaLayer(nn.Module):
    """Mamba-1 mixer: gated depthwise convolution feeding a selective (S6) state-space scan.

    The recurrence ``h_t = exp(Delta_t A) h_{t-1} + Delta_t B_t u_t``, ``y_t = C_t h_t`` has
    input-dependent ``Delta_t``, ``B_t``, ``C_t`` (computed from the convolved signal by
    ``x_proj``/``dt_proj``), so the dynamics change with the signal — in identification terms
    a structured nonlinear state-space model, unlike the time-invariant LRU/S5 layers.
    Discretization follows the official simplification: zero-order hold for ``A`` but the
    Euler-like ``Delta_t B_t u_t`` for the input path. ``A`` is real and negative
    (``-exp(A_log)``, S4D-real initialized), so no complex arithmetic is needed.

    Initialization matches the reference: ``dt_proj`` weight uniform ``+-dt_rank^-0.5``,
    its bias the softplus-inverse of ``Delta ~ LogUniform[dt_min, dt_max]``; ``D`` starts
    at ones.

    References:
        A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
        Spaces," COLM 2024. arXiv:2312.00752.

    Args:
        d_model: input/output signal dimension.
        d_state: SSM state dimension per channel N.
        d_conv: depthwise convolution kernel width.
        expand: width multiplier of the inner channel dimension.
        dt_rank: rank of the ``Delta`` projection bottleneck; ``ceil(d_model / 16)`` if None.
        dt_min: lower bound of the timestep initialization.
        dt_max: upper bound of the timestep initialization.
        backend: ``"scan"`` (parallel Hillis-Steele) or ``"eager"`` (sequential loop).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        backend: str = "scan",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.backend = backend

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=1e-4
        )
        with torch.no_grad():
            # softplus inverse, so softplus(bias) reproduces the sampled timesteps
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        # S4D-real initialization: A[d, n] = -(n + 1), stored in log space
        A = torch.arange(1, d_state + 1, dtype=torch.float32).expand(self.d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A).contiguous())
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def _ssm(self, x: torch.Tensor, h0: torch.Tensor | None):
        """Selective scan over the convolved signal ``x [batch, seq, d_inner]``."""
        B_sz, L, _ = x.shape
        dt, B_t, C_t = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # [batch, seq, d_inner]
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        lam = torch.exp(delta.unsqueeze(-1) * A)  # [batch, seq, d_inner, d_state]
        v = (delta * x).unsqueeze(-1) * B_t.unsqueeze(-2)  # [batch, seq, d_inner, d_state]
        match self.backend:
            case "scan":
                h = selective_recurrence(lam.flatten(-2), v.flatten(-2), h0)
            case "eager":
                h = _diagonal_recurrence_sequential(lam.flatten(-2), v.flatten(-2), h0)
            case unknown:
                raise ValueError(f"unknown backend {unknown!r}, expected 'scan' or 'eager'")
        y = (h.view(B_sz, L, self.d_inner, self.d_state) @ C_t.unsqueeze(-1)).squeeze(-1)
        return y + self.D * x, h[..., -1, :]

    def forward(self, u: torch.Tensor, state: dict | None = None, return_state: bool = False):
        """Run the mixer over an input sequence.

        Args:
            u: input sequence ``[batch, seq, d_model]``.
            state: carried state ``{"conv": pre-conv tail [batch, d_inner, d_conv - 1],
                "ssm": flat scan state [batch, d_inner * d_state]}``; zero initial
                conditions if None.
            return_state: if ``True``, return ``(output, new_state)``.

        Returns:
            Output sequence ``[batch, seq, d_model]``, optionally with the new state.
        """
        B_sz, L, _ = u.shape
        match state:
            case {"conv": conv_tail, "ssm": h0}:
                pass
            case None:
                conv_tail = u.new_zeros(B_sz, self.d_inner, self.d_conv - 1)
                h0 = None
            case _:
                raise TypeError(f"expected state dict {{'conv': tensor, 'ssm': tensor}}, got {type(state)}")

        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = x.transpose(1, 2)  # [batch, d_inner, seq]
        # the carried tail replaces the zero left-padding of a cold-started causal convolution
        x_buf = torch.cat((conv_tail, x), dim=-1)
        x_conv = F.silu(F.conv1d(x_buf, self.conv1d.weight, self.conv1d.bias, groups=self.d_inner))
        y, h_last = self._ssm(x_conv.transpose(1, 2), h0)
        out = self.out_proj(y * F.silu(z))
        if not return_state:
            return out
        new_state = {"conv": x_buf[..., x_buf.shape[-1] - (self.d_conv - 1) :], "ssm": h_last}
        return out, new_state


class MambaResidualBlock(nn.Module):
    """Pre-norm residual block ``x + MambaLayer(RMSNorm(x))`` as in the reference stack."""

    def __init__(self, d_model: int, **mixer_kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mixer = MambaLayer(d_model, **mixer_kwargs)

    def forward(self, u: torch.Tensor, state: dict | None = None, return_state: bool = False):
        h, new_state = self.mixer(self.norm(u), state, return_state=True)
        y = u + h
        if not return_state:
            return y
        return y, new_state


class DeepMamba(nn.Module):
    """Deep Mamba network: linear encoder, stacked pre-norm Mamba blocks, final norm, linear decoder.

    The reference language-model stack with the embedding and LM head replaced by linear
    input/output maps for real-valued signals. Following the reference initialization, every
    block's ``out_proj`` is scaled down by ``sqrt(n_layers)`` to keep the residual stream
    variance flat at initialization.

    With ``return_state=True`` the model follows the stateful-model protocol
    (``forward(u, state=...) -> (out, state)``); the carried state holds each block's
    convolution tail and scan state, so chunked rollouts are exactly equivalent to the full
    sequence and ``TbpttLearner`` works unchanged.

    Args:
        input_size: number of input signals.
        output_size: number of output signals.
        d_model: signal width between the blocks.
        d_state: SSM state dimension per channel.
        n_layers: number of Mamba blocks.
        d_conv: depthwise convolution kernel width.
        expand: inner width multiplier.
        dt_min: timestep initialization lower bound, see ``MambaLayer``.
        dt_max: timestep initialization upper bound, see ``MambaLayer``.
        backend: execution backend of the scans, see ``MambaLayer``.
        return_state: if ``True``, return ``(output, state)`` tuple.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 32,
        d_state: int = 16,
        n_layers: int = 3,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        backend: str = "scan",
        return_state: bool = False,
    ):
        super().__init__()
        self.return_state = return_state
        self.encoder = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList(
            MambaResidualBlock(
                d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_min=dt_min,
                dt_max=dt_max,
                backend=backend,
            )
            for _ in range(n_layers)
        )
        self.norm_f = nn.RMSNorm(d_model, eps=1e-5)
        self.decoder = nn.Linear(d_model, output_size)
        with torch.no_grad():
            for block in self.blocks:
                block.mixer.out_proj.weight /= math.sqrt(n_layers)

    @property
    def backend(self) -> str:
        return self.blocks[0].mixer.backend

    @backend.setter
    def backend(self, value: str):
        for m in self.modules():
            if isinstance(m, MambaLayer):
                m.backend = value

    def forward(self, u: torch.Tensor, state: list | None = None):
        """Run the block stack over the input sequence.

        Args:
            u: input sequence ``[batch, seq, input_size]``.
            state: list of per-block state dicts from a previous chunk.

        Returns:
            Output sequence ``[batch, seq, output_size]``, or ``(sequence, state)`` when
            ``return_state`` is set.
        """
        if state is None:
            state = [None] * len(self.blocks)
        h = self.encoder(u)
        new_state = []
        for block, s in zip(self.blocks, state):
            h, s_new = block(h, s, return_state=True)
            new_state.append(s_new)
        y = self.decoder(self.norm_f(h))
        if self.return_state:
            return y, new_state
        return y
