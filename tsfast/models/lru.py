"""Linear Recurrent Unit models: stable diagonal linear state-space layers around static nonlinearities."""

__all__ = [
    "LRU",
    "DeepLRU",
]

import math

import torch
import torch.nn.functional as F
from torch import nn

from .scan import _diagonal_recurrence_sequential, complex_in_proj, diagonal_recurrence, real_out_proj


class LRU(nn.Module):
    """Linear Recurrent Unit: a MIMO diagonal complex linear state-space layer (Orvieto et al. 2023).

    Recurrence ``x_k = diag(lambda) x_{k-1} + exp(gamma_log) * (B u_k)`` with real output
    ``y_k = Re(C x_k) + D u_k``. The eigenvalues are stable by construction through the
    exponential parameterization ``lambda = exp(-exp(nu_log) + i exp(theta_log))`` and are
    initialized uniformly on a ring of the unit disk: ``|lambda|^2 ~ U[r_min^2, r_max^2]``,
    phase ``~ U[0, max_phase]``. The learned ``gamma_log`` normalization starts at
    ``sqrt(1 - |lambda|^2)`` so state and input contributions are balanced at initialization.

    ``B`` and ``C`` are stored as separate real/imaginary parameter pairs (as in the paper's
    pseudocode), avoiding complex-valued optimizer states; ``D`` is a full real feedthrough
    matrix as in the system-identification variant of Forgione et al., which reduces to the
    paper's elementwise skip when diagonal.

    References:
        A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De,
        "Resurrecting Recurrent Neural Networks for Long Sequences," ICML 2023.
        arXiv:2303.06349.

        M. Forgione, M. Mejari, and D. Piga, "Model order reduction of deep structured
        state-space models: A system-theoretic approach," IEEE CDC 2024. arXiv:2403.14833.
        (LRU variant for system identification with a full feedthrough matrix.)

    Args:
        in_features: input signal dimension H.
        out_features: output signal dimension.
        state_features: complex state dimension N.
        r_min: lower bound of the eigenvalue-magnitude ring at initialization.
        r_max: upper bound of the eigenvalue-magnitude ring at initialization.
        max_phase: upper bound of the eigenvalue phase at initialization (radians).
        backend: ``"scan"`` (log-doubling, parallel) or ``"eager"`` (sequential loop).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        state_features: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.283,
        backend: str = "scan",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features
        self.backend = backend

        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lam_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(0.5 * torch.log(1 - lam_abs.square()))
        self.B_re = nn.Parameter(torch.randn(state_features, in_features) / math.sqrt(2 * in_features))
        self.B_im = nn.Parameter(torch.randn(state_features, in_features) / math.sqrt(2 * in_features))
        self.C_re = nn.Parameter(torch.randn(out_features, state_features) / math.sqrt(state_features))
        self.C_im = nn.Parameter(torch.randn(out_features, state_features) / math.sqrt(state_features))
        self.D = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

    def eigenvalues(self) -> torch.Tensor:
        """Complex recurrence eigenvalues ``lambda`` of shape ``[state_features]``."""
        lam_abs = torch.exp(-torch.exp(self.nu_log))
        phase = torch.exp(self.theta_log)
        return torch.complex(lam_abs * torch.cos(phase), lam_abs * torch.sin(phase))

    def forward(self, u: torch.Tensor, state: torch.Tensor | None = None, return_state: bool = False):
        """Run the layer over an input sequence.

        Args:
            u: input sequence ``[batch, seq, in_features]``.
            state: complex recurrence state ``[batch, state_features]`` carried from a
                previous chunk; zero initial conditions if None.
            return_state: if ``True``, return ``(output, new_state)``.

        Returns:
            Output sequence ``[batch, seq, out_features]``, optionally with the new state.
        """
        lam = self.eigenvalues()
        gamma = torch.exp(self.gamma_log).unsqueeze(-1)
        v = complex_in_proj(u, gamma * self.B_re, gamma * self.B_im)
        match self.backend:
            case "scan":
                x = diagonal_recurrence(lam, v, state)
            case "eager":
                x = _diagonal_recurrence_sequential(lam, v, state)
            case unknown:
                raise ValueError(f"unknown backend {unknown!r}, expected 'scan' or 'eager'")
        y = real_out_proj(x, self.C_re, self.C_im) + u @ self.D.mT
        if not return_state:
            return y
        return y, x[..., -1, :]


class LRUBlock(nn.Module):
    """Pre-norm residual block: ``x + GLU(dropout(gelu(LRU(LayerNorm(x)))))``.

    The channel-mixing GLU follows the LRU paper's sequence layer; LayerNorm (not batch
    norm) keeps every operation pointwise in time, so chunked rollouts stay exact.

    Args:
        d_model: signal width of the block.
        d_state: complex state dimension of the LRU layer.
        dropout: dropout probability after the activation and after the GLU.
        r_min: see ``LRU``.
        r_max: see ``LRU``.
        max_phase: see ``LRU``.
        backend: see ``LRU``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.0,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.283,
        backend: str = "scan",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lru = LRU(d_model, d_model, d_state, r_min, r_max, max_phase, backend)
        self.glu = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, state: torch.Tensor | None = None, return_state: bool = False):
        h, new_state = self.lru(self.norm(u), state, return_state=True)
        h = self.dropout(F.gelu(h))
        h = self.dropout(F.glu(self.glu(h), dim=-1))
        y = u + h
        if not return_state:
            return y
        return y, new_state


class DeepLRU(nn.Module):
    """Deep LRU network: linear encoder, stacked LRU blocks, linear decoder.

    The architecture of Orvieto et al. (2023) as used for system identification by
    Forgione et al.: all dynamics live in the diagonal linear recurrences, all
    nonlinearity in the pointwise GELU/GLU mixing between them.

    With ``return_state=True`` the model follows the stateful-model protocol
    (``forward(u, state=...) -> (out, state)``); the carried state is the complex
    recurrence state of every block, so chunked rollouts are exactly equivalent to the
    full sequence and ``TbpttLearner`` works unchanged.

    Args:
        input_size: number of input signals.
        output_size: number of output signals.
        d_model: signal width between the blocks.
        d_state: complex state dimension per block.
        n_layers: number of LRU blocks.
        dropout: dropout probability inside the blocks.
        r_min: eigenvalue-ring lower bound at initialization, see ``LRU``.
        r_max: eigenvalue-ring upper bound at initialization, see ``LRU``.
        max_phase: eigenvalue-phase upper bound at initialization, see ``LRU``.
        backend: execution backend of the recurrences, see ``LRU``.
        return_state: if ``True``, return ``(output, state)`` tuple.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 32,
        d_state: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.283,
        backend: str = "scan",
        return_state: bool = False,
    ):
        super().__init__()
        self.return_state = return_state
        self.encoder = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList(
            LRUBlock(d_model, d_state, dropout, r_min, r_max, max_phase, backend) for _ in range(n_layers)
        )
        self.decoder = nn.Linear(d_model, output_size)

    @property
    def backend(self) -> str:
        return self.blocks[0].lru.backend

    @backend.setter
    def backend(self, value: str):
        for m in self.modules():
            if isinstance(m, LRU):
                m.backend = value

    def forward(self, u: torch.Tensor, state: list | None = None):
        """Run the block stack over the input sequence.

        Args:
            u: input sequence ``[batch, seq, input_size]``.
            state: list of per-block complex recurrence states from a previous chunk.

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
        y = self.decoder(h)
        if self.return_state:
            return y, new_state
        return y
