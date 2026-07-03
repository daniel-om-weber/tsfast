"""S5 models: HiPPO-initialized diagonal continuous-time state-space layers (Smith et al. 2023)."""

__all__ = [
    "make_dplr_hippo",
    "S5",
    "DeepS5",
]

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .scan import _diagonal_recurrence_sequential, diagonal_recurrence


def make_dplr_hippo(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the normal part of the HiPPO-LegS matrix (the S4D/S5 initialization).

    Follows the official S5 reference: the low-rank term of the DPLR decomposition is
    discarded, the real part of every eigenvalue is the mean diagonal of the normal part
    (exactly ``-0.5``), and the imaginary parts/eigenvectors come from an eigendecomposition
    of the skew-symmetric part via the Hermitian trick ``eigh(-1j * S)``.

    References:
        A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Re, "HiPPO: Recurrent Memory with
        Optimal Polynomial Projections," NeurIPS 2020. arXiv:2008.07669.

    Returns:
        ``(Lambda, V)``: eigenvalues ``[n]`` complex128 and eigenvectors ``[n, n]`` complex128.
    """
    p = np.sqrt(1 + 2 * np.arange(n))
    A = np.tril(p[:, None] * p[None, :]) - np.diag(np.arange(n))
    S = -A + np.sqrt(np.arange(n) + 0.5)[:, None] * np.sqrt(np.arange(n) + 0.5)[None, :]
    lambda_real = np.mean(np.diagonal(S)) * np.ones(n)
    lambda_imag, V = np.linalg.eigh(S * -1j)
    return lambda_real + 1j * lambda_imag, V


def _lecun_normal(*shape: int, fan_in: int) -> torch.Tensor:
    """Truncated-normal LeCun initialization matching ``jax.nn.initializers.lecun_normal``."""
    # jax samples a standard normal truncated to [-2, 2] and rescales by std/.879... so the
    # truncated distribution has the requested variance
    std = math.sqrt(1 / fan_in) / 0.87962566103423978
    return torch.nn.init.trunc_normal_(torch.empty(*shape), std=1.0, a=-2.0, b=2.0) * std


class S5(nn.Module):
    """S5 layer: a MIMO diagonal continuous-time linear state-space system, ZOH-discretized.

    The continuous-time diagonal state matrix ``Lambda`` is initialized from the diagonal
    approximation of HiPPO-LegS (block-diagonally for ``blocks > 1``), ``B``/``C`` are
    initialized in the eigenvector basis (``V^{-1}B``, ``CV``), and each state owns a learned
    timestep ``exp(log_step) ~ LogUniform[dt_min, dt_max]``. Discretization runs every
    forward pass: ``Lambda_bar = exp(Lambda * step)``, ``B_bar = (Lambda_bar - 1)/Lambda * B``
    (zero-order hold) or the bilinear transform.

    With ``conj_sym=True`` (default) only one member of each conjugate eigenvalue pair is
    simulated (``d_state / 2`` complex states) and the output takes ``2 * Re(C x)``, which is
    exactly equivalent to the full system for real-valued inputs.

    References:
        J. T. H. Smith, A. Warrington, and S. W. Linderman, "Simplified State Space Layers
        for Sequence Modeling," ICLR 2023. arXiv:2208.04933.

    Args:
        d_model: input/output signal dimension H.
        d_state: full state dimension P of the underlying system (the number of simulated
            complex states is ``P/2`` under conjugate symmetry).
        blocks: number of independent HiPPO blocks the state is split into at initialization.
        conj_sym: simulate half the states and double the real output.
        clip_eigs: clamp ``Re(Lambda) <= -1e-4`` every forward, enforcing stability of the
            continuous-time system (recommended for unbounded rollout lengths).
        discretization: ``"zoh"`` or ``"bilinear"``.
        dt_min: lower bound of the timestep initialization.
        dt_max: upper bound of the timestep initialization.
        C_init: ``"trunc_standard_normal"`` (per-row LeCun in the eigenvector basis),
            ``"lecun_normal"`` (whole-matrix), or ``"complex_normal"`` (no basis transform).
        step_rescale: uniform rescaling of the learned timesteps, e.g. for a different
            sampling rate at inference time.
        backend: ``"scan"`` (log-doubling, parallel) or ``"eager"`` (sequential loop).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        blocks: int = 1,
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretization: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        C_init: str = "trunc_standard_normal",
        step_rescale: float = 1.0,
        backend: str = "scan",
    ):
        super().__init__()
        if d_state % blocks != 0:
            raise ValueError(f"d_state={d_state} must be divisible by blocks={blocks}")
        block_size = d_state // blocks
        if conj_sym and block_size % 2 != 0:
            raise ValueError(f"d_state/blocks={block_size} must be even with conj_sym")
        self.d_model = d_model
        self.d_state = d_state
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.discretization = discretization
        self.step_rescale = step_rescale
        self.backend = backend
        # number of simulated complex states
        self.p = d_state // 2 if conj_sym else d_state

        lam, V = make_dplr_hippo(block_size)
        if conj_sym:
            # eigh returns eigenvalues in ascending imaginary part: the first half holds one
            # member of each conjugate pair
            lam, V = lam[: block_size // 2], V[:, : block_size // 2]
        lam = np.tile(lam, blocks)
        V = np.kron(np.eye(blocks), V)  # block-diagonal [d_state, p]
        Vinv = V.conj().T

        self.Lambda_re = nn.Parameter(torch.tensor(lam.real, dtype=torch.float32))
        self.Lambda_im = nn.Parameter(torch.tensor(lam.imag, dtype=torch.float32))

        local_p = 2 * self.p if conj_sym else self.p
        B = _lecun_normal(local_p, d_model, fan_in=local_p).double().numpy()
        VinvB = Vinv @ B
        self.B_re = nn.Parameter(torch.tensor(VinvB.real, dtype=torch.float32))
        self.B_im = nn.Parameter(torch.tensor(VinvB.imag, dtype=torch.float32))

        match C_init:
            case "trunc_standard_normal":
                C = torch.stack([_lecun_normal(local_p, 2, fan_in=local_p) for _ in range(d_model)])
            case "lecun_normal":
                C = _lecun_normal(d_model, local_p, 2, fan_in=local_p * d_model)
            case "complex_normal":
                C = torch.randn(d_model, self.p, 2) * 0.5**0.5
            case unknown:
                raise ValueError(f"unknown C_init {unknown!r}")
        if C_init == "complex_normal":
            self.C_re = nn.Parameter(C[..., 0])
            self.C_im = nn.Parameter(C[..., 1])
        else:
            CV = (C[..., 0].double().numpy() + 1j * C[..., 1].double().numpy()) @ V
            self.C_re = nn.Parameter(torch.tensor(CV.real, dtype=torch.float32))
            self.C_im = nn.Parameter(torch.tensor(CV.imag, dtype=torch.float32))

        self.D = nn.Parameter(torch.randn(d_model))
        self.log_step = nn.Parameter(torch.rand(self.p) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))

    def discretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretized ``(Lambda_bar [p], B_bar [p, d_model])`` from the current parameters."""
        lam_re = torch.clamp(self.Lambda_re, max=-1e-4) if self.clip_eigs else self.Lambda_re
        lam = torch.complex(lam_re, self.Lambda_im)
        step = self.step_rescale * torch.exp(self.log_step)
        B_tilde = torch.complex(self.B_re, self.B_im)
        match self.discretization:
            case "zoh":
                lam_bar = torch.exp(lam * step)
                B_bar = ((lam_bar - 1) / lam).unsqueeze(-1) * B_tilde
            case "bilinear":
                bl = 1 / (1 - (step / 2) * lam)
                lam_bar = bl * (1 + (step / 2) * lam)
                B_bar = (bl * step).unsqueeze(-1) * B_tilde
            case unknown:
                raise ValueError(f"unknown discretization {unknown!r}")
        return lam_bar, B_bar

    def forward(self, u: torch.Tensor, state: torch.Tensor | None = None, return_state: bool = False):
        """Run the layer over an input sequence.

        Args:
            u: input sequence ``[batch, seq, d_model]``.
            state: complex recurrence state ``[batch, p]`` carried from a previous chunk;
                zero initial conditions if None.
            return_state: if ``True``, return ``(output, new_state)``.

        Returns:
            Output sequence ``[batch, seq, d_model]``, optionally with the new state.
        """
        lam_bar, B_bar = self.discretize()
        v = torch.complex(u @ B_bar.real.mT, u @ B_bar.imag.mT)
        match self.backend:
            case "scan":
                x = diagonal_recurrence(lam_bar, v, state)
            case "eager":
                x = _diagonal_recurrence_sequential(lam_bar, v, state)
            case unknown:
                raise ValueError(f"unknown backend {unknown!r}, expected 'scan' or 'eager'")
        y = x.real @ self.C_re.mT - x.imag @ self.C_im.mT
        if self.conj_sym:
            y = 2 * y
        y = y + self.D * u
        if not return_state:
            return y
        return y, x[..., -1, :]


class S5Block(nn.Module):
    """Pre-norm residual block around an S5 layer, following the official sequence layer.

    ``x + activation(S5(LayerNorm(x)))`` where ``activation`` is one of the official GLU
    variants. LayerNorm is used instead of the reference's batch norm default so every
    operation stays pointwise in time and chunked rollouts remain exact.

    Args:
        d_model: signal width of the block.
        d_state: full state dimension of the S5 layer.
        dropout: dropout probability inside the activation.
        activation: ``"half_glu1"`` (default, gates with a single dense layer),
            ``"half_glu2"`` (GELU only on the gate input), ``"full_glu"``, or ``"gelu"``.
        **ssm_kwargs: forwarded to ``S5``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.0,
        activation: str = "half_glu1",
        **ssm_kwargs,
    ):
        super().__init__()
        self.activation = activation
        self.norm = nn.LayerNorm(d_model)
        self.s5 = S5(d_model, d_state, **ssm_kwargs)
        self.out1 = nn.Linear(d_model, d_model) if activation == "full_glu" else None
        self.out2 = nn.Linear(d_model, d_model) if activation != "gelu" else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, state: torch.Tensor | None = None, return_state: bool = False):
        h, new_state = self.s5(self.norm(u), state, return_state=True)
        match self.activation:
            case "full_glu":
                h = self.dropout(F.gelu(h))
                h = self.out1(h) * torch.sigmoid(self.out2(h))
                h = self.dropout(h)
            case "half_glu1":
                h = self.dropout(F.gelu(h))
                h = h * torch.sigmoid(self.out2(h))
                h = self.dropout(h)
            case "half_glu2":
                h1 = self.dropout(F.gelu(h))
                h = h * torch.sigmoid(self.out2(h1))
                h = self.dropout(h)
            case "gelu":
                h = self.dropout(F.gelu(h))
            case unknown:
                raise ValueError(f"unknown activation {unknown!r}")
        y = u + h
        if not return_state:
            return y
        return y, new_state


class DeepS5(nn.Module):
    """Deep S5 network: linear encoder, stacked S5 blocks, linear decoder.

    With ``return_state=True`` the model follows the stateful-model protocol
    (``forward(u, state=...) -> (out, state)``); the carried state is the complex
    recurrence state of every block, so chunked rollouts are exactly equivalent to the
    full sequence and ``TbpttLearner`` works unchanged.

    Args:
        input_size: number of input signals.
        output_size: number of output signals.
        d_model: signal width between the blocks.
        d_state: full state dimension per block.
        n_layers: number of S5 blocks.
        blocks: HiPPO blocks per layer at initialization, see ``S5``.
        dropout: dropout probability inside the blocks.
        activation: block activation variant, see ``S5Block``.
        dt_min: timestep initialization lower bound, see ``S5``.
        dt_max: timestep initialization upper bound, see ``S5``.
        conj_sym: see ``S5``.
        clip_eigs: see ``S5``.
        backend: execution backend of the recurrences, see ``S5``.
        return_state: if ``True``, return ``(output, state)`` tuple.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 32,
        d_state: int = 64,
        n_layers: int = 3,
        blocks: int = 1,
        dropout: float = 0.0,
        activation: str = "half_glu1",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conj_sym: bool = True,
        clip_eigs: bool = False,
        backend: str = "scan",
        return_state: bool = False,
    ):
        super().__init__()
        self.return_state = return_state
        self.encoder = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList(
            S5Block(
                d_model,
                d_state,
                dropout=dropout,
                activation=activation,
                blocks=blocks,
                dt_min=dt_min,
                dt_max=dt_max,
                conj_sym=conj_sym,
                clip_eigs=clip_eigs,
                backend=backend,
            )
            for _ in range(n_layers)
        )
        self.decoder = nn.Linear(d_model, output_size)

    @property
    def backend(self) -> str:
        return self.blocks[0].s5.backend

    @backend.setter
    def backend(self, value: str):
        for m in self.modules():
            if isinstance(m, S5):
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
