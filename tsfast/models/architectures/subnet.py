"""SUBNET-style subspace encoder models.

A SUBNET model is a discrete-time state-space model whose initial state is
estimated by an encoder network from a window of past inputs and outputs,
trained with a truncated simulation loss over short overlapping sections.
The encoder replaces the zero-initial-state + transient-skip protocol of
``NeuralStateSpace``: the rollout starts from a data-informed state, so every
section contributes a fully warmed-up simulation error.

Reference: Beintema, Schoukens & Tóth, "Deep Subspace Encoders for Nonlinear
System Identification", Automatica 156:111210, 2023 (arXiv:2210.14816);
reference implementation github.com/GerbenBeintema/deepSI (BSD-3-Clause).
The encoder and training scheme follow deepSI exactly; the state transition
deviates by design (plain MLP instead of deepSI's linear + MLP residual nets)
so the fused ``NeuralStateSpace`` rollout backends apply. Numerical agreement
with deepSI is validated in ``comparisons/compare_subnet.py``.
"""

__all__ = [
    "ResMLP",
    "SubnetEncoder",
    "SubnetSSM",
]

import torch
from torch import nn

from .ssm import NeuralStateSpace


def _mlp(n_in: int, n_out: int, hidden_size: int, num_layers: int, act: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """MLP with zero-initialized biases (deepSI ``feed_forward_nn`` convention)."""
    dims = (n_in, *((hidden_size,) * num_layers), n_out)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
    return net


class ResMLP(nn.Module):
    """Linear map plus MLP in parallel (deepSI ``simple_res_net``).

    The linear bypass lets the net represent affine maps exactly and reduces
    the MLP to a nonlinear correction; deepSI uses it for encoders and input
    matrices.
    """

    def __init__(self, n_in: int, n_out: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.mlp = _mlp(n_in, n_out, hidden_size, num_layers) if num_layers > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        return out if self.mlp is None else out + self.mlp(x)


class SubnetEncoder(nn.Module):
    """State encoder ``x0 = psi(u_{t-nb..t-1}, y_{t-na..t-1})`` (deepSI ``default_encoder_net``).

    Histories are flattened time-major and concatenated as ``[u_hist, y_hist]``
    before the shared :class:`ResMLP`, matching the deepSI ordering so weights
    can be transplanted one-to-one.
    """

    def __init__(
        self, n_input: int, n_output: int, n_state: int, na: int, nb: int, hidden_size: int = 64, num_layers: int = 2
    ):
        super().__init__()
        self.na, self.nb = na, nb
        self.net = ResMLP(nb * n_input + na * n_output, n_state, hidden_size, num_layers)

    def forward(self, u_hist: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
        """Map histories ``u_hist [B, nb, nu]`` and ``y_hist [B, na, ny]`` to ``x0 [B, n_state]``."""
        return self.net(torch.cat((u_hist.flatten(1), y_hist.flatten(1)), dim=1))


class SubnetSSM(nn.Module):
    """Encoder-initialized neural state-space model over ``[u, y]`` input channels.

    ``x_{k+1} = f(x_k, u_k)``, ``y_k = C x_k + d``, ``x_{n_init} = psi(u_hist, y_hist)``.

    The input tensor carries ``n_input`` input channels followed by ``n_output``
    measured-output channels (the ``prediction_concat`` layout). Only the first
    ``n_init`` steps of the output channels are ever read — the encoder window —
    so inference can zero-pad them beyond the warm-up. Predictions start at
    ``n_init`` (``y_{n_init}`` observes the encoder state itself); earlier
    positions are zero and must be excluded from the loss via ``n_skip=n_init``.

    The state transition reuses :class:`NeuralStateSpace` including its fused
    ``c``/``triton`` rollout backends, which accept the encoder state as ``x0``
    unchanged. Deviation from the SUBNET paper: the transition is a plain MLP
    rather than a linear + MLP residual pair.

    Args:
        n_input: exogenous input dimension (``u`` channels of the input tensor).
        n_output: observed output dimension (``y`` channels of the input tensor).
        n_state: latent state dimension.
        hidden_size: transition MLP hidden width (or explicit list of widths).
        num_layers: number of transition hidden layers.
        act: transition activation name (``tanh``, ``sigmoid``, ``relu``).
        n_init: encoder warm-up length; predictions and loss start here.
        na: output-history length for the encoder (defaults to ``n_init``).
        nb: input-history length for the encoder (defaults to ``n_init``).
        enc_hidden_size: encoder MLP hidden width.
        enc_num_layers: encoder MLP hidden layers.
        backend: rollout backend, see :class:`NeuralStateSpace`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_state: int = 8,
        hidden_size: int | list[int] = 64,
        num_layers: int = 2,
        act: str = "tanh",
        n_init: int = 50,
        na: int | None = None,
        nb: int | None = None,
        enc_hidden_size: int = 64,
        enc_num_layers: int = 2,
        backend: str = "auto",
    ):
        super().__init__()
        na = n_init if na is None else na
        nb = n_init if nb is None else nb
        if max(na, nb) > n_init:
            raise ValueError(f"encoder windows na={na}, nb={nb} cannot exceed n_init={n_init}")
        self.n_input, self.n_output, self.n_init = n_input, n_output, n_init
        self.core = NeuralStateSpace(n_input, n_output, n_state, hidden_size, num_layers, act, backend)
        self.encoder = SubnetEncoder(n_input, n_output, n_state, na, nb, enc_hidden_size, enc_num_layers)

    @property
    def backend(self) -> str:
        return self.core.backend

    @backend.setter
    def backend(self, value: str):
        self.core.backend = value

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder state from the ``[u, y]`` input tensor at position ``n_init``."""
        u, y = x[..., : self.n_input], x[..., self.n_input :]
        n0 = self.n_init
        return self.encoder(u[:, n0 - self.encoder.nb : n0], y[:, n0 - self.encoder.na : n0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate from the encoder state; returns ``[B, L, n_output]`` with zeros before ``n_init``."""
        if x.shape[1] <= self.n_init:
            raise ValueError(f"sequence length {x.shape[1]} too short for encoder warm-up n_init={self.n_init}")
        x0 = self.encode(x)
        u = x[..., : self.n_input]
        y_first = self.core.output_map(x0).unsqueeze(1)
        warmup = x.new_zeros(x.shape[0], self.n_init, self.n_output)
        u_future = u[:, self.n_init : -1]
        if u_future.shape[1] == 0:
            return torch.cat((warmup, y_first), dim=1)
        # core(u_k..u_{L-2}, x0) observes x_{n_init+1} .. x_{L-1}; y_{n_init} observes x0 itself.
        y_rest = self.core(u_future, x0)
        return torch.cat((warmup, y_first, y_rest), dim=1)
