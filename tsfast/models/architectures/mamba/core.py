"""Mamba models: selective state-space layers with input-dependent dynamics (Gu & Dao 2023)."""

__all__ = [
    "MambaLayer",
    "DeepMamba",
]

import importlib
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..._core import scan
from ..._core.dispatch import warn_fallback
from ..._core.scan import _diagonal_recurrence_sequential, selective_recurrence


def _fused_conv(x, tail, weight, bias):
    """Dispatch the fused causal conv + SiLU kernel; None means run the eager conv path.

    Same backend policy as ``_fused_ssm``: serves ``scan.backend`` "auto" (CUDA only)
    and "triton"; silent on non-CUDA devices under "auto", warns once per process
    otherwise.
    """
    if scan.backend not in ("auto", "triton"):
        return None
    if scan.backend == "auto" and x.device.type != "cuda":
        return None
    try:
        mod = importlib.import_module(".conv_triton", __package__)
    except Exception as e:  # pragma: no cover - triton import failure
        reason = f"backend import failed ({e!r})"
    else:
        reason = mod.supports(x, tail, weight, bias)
        if reason is None:
            return mod.run(x, tail, weight, bias)
    warn_fallback(
        "mamba.conv.triton",
        f"fused conv triton kernel unusable: {reason}; falling back to the eager convolution",
    )
    return None


def _fused_ssm(draw, A, B_t, C_t, u, z, Dp, h0):
    """Dispatch the fused Mamba SSM kernel; None means run the generic scan path.

    Honors ``tsfast.models._core.scan.backend``: the fused kernel serves "auto" (CUDA only)
    and "triton"; "doubling"/"c" force the generic path. Missing module or unsupported
    inputs warn once per process, except on non-CUDA devices under "auto", where the
    generic C/doubling path is the intended backend and silence is correct.
    """
    if scan.backend not in ("auto", "triton"):
        return None
    if scan.backend == "auto" and draw.device.type != "cuda":
        return None
    try:
        mod = importlib.import_module(".mamba_triton", __package__)
    except Exception as e:  # pragma: no cover - triton import failure
        reason = f"backend import failed ({e!r})"
    else:
        reason = mod.supports(draw, A, B_t, C_t, u, z, Dp, h0)
        if reason is None:
            return mod.run(draw, A, B_t, C_t, u, z, Dp, h0)
    warn_fallback(
        "scan.mamba.triton",
        f"fused mamba triton kernel unusable: {reason}; falling back to the generic selective scan",
    )
    return None


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
        backend: ``"scan"`` (fused Triton kernels for the convolution and the selective
            scan on CUDA float32, otherwise the eager convolution and the generic
            parallel scan resolved by ``tsfast.models._core.scan.backend``) or ``"eager"``
            (sequential loop).
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

    def _ssm(self, x: torch.Tensor, z: torch.Tensor, h0: torch.Tensor | None):
        """Selective scan over the convolved signal ``x [batch, seq, d_inner]``, gated by ``z``."""
        B_sz, L, _ = x.shape
        dt, B_t, C_t = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        draw = self.dt_proj(dt)  # [batch, seq, d_inner], pre-softplus
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        if self.backend == "scan":
            h0_dn = h0.view(B_sz, self.d_inner, self.d_state) if h0 is not None else None
            fused = _fused_ssm(draw, A, B_t, C_t, x, z, self.D, h0_dn)
            if fused is not None:
                out, h_last = fused
                return out, h_last.flatten(-2)
        delta = F.softplus(draw)
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
        return (y + self.D * x) * F.silu(z), h[..., -1, :]

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
        # the carried tail replaces the zero left-padding of a cold-started causal convolution
        x_conv = _fused_conv(x, conv_tail, self.conv1d.weight, self.conv1d.bias) if self.backend == "scan" else None
        if x_conv is None:
            x_buf = torch.cat((conv_tail, x.transpose(1, 2)), dim=-1)
            x_conv = F.silu(F.conv1d(x_buf, self.conv1d.weight, self.conv1d.bias, groups=self.d_inner)).transpose(1, 2)
        y, h_last = self._ssm(x_conv, z, h0)
        out = self.out_proj(y)
        if not return_state:
            return out
        tail_len = self.d_conv - 1
        if L >= tail_len:
            # contiguous copy so the carried tail does not pin the in_proj buffer alive
            new_tail = x[:, L - tail_len :].mT.contiguous()
        else:
            new_tail = torch.cat((conv_tail[..., L:], x.mT), dim=-1)
        return out, {"conv": new_tail, "ssm": h_last}


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
