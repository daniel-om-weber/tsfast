"""NARX models: one-step regressors over explicit lag windows, simulated by output feedback."""

__all__ = [
    "NarxMLP",
]

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cnn import CausalConv1d


class NarxMLP(nn.Module):
    """MLP NARX model ``y[t] = f(y[t-1..t-na], u[t..t-nb+1])``.

    The input sequence carries the concatenated ``[u, y]`` channels (as produced by the
    ``prediction_concat`` transform and by ``InferenceWrapper``); the split into input
    and output lags happens internally. The first hidden layer over the lag window is
    implemented as a pair of causal convolutions, so both execution paths share one set
    of weights:

    - **teacher forcing** (parallel): every timestep reads true outputs from the ``y``
      channel; the whole sequence evaluates as convolutions in one pass. This is the
      classic one-step-ahead NARX training mode.
    - **free run** (sequential): the ``y`` channel is read only for the first
      ``washout`` samples to fill the lag buffer; afterwards the model feeds back its
      own predictions and gradients flow through the feedback. This is simulation
      training, and the only mode that matches free-run evaluation.

    ``forward`` picks the path from the module state: teacher forcing is used only
    while ``self.training`` and ``teacher_forcing=True`` — in eval mode a NARX is
    always a simulator, which keeps validation losses and ``InferenceWrapper`` output
    free-run regardless of the training mode.

    Args:
        u_size: number of input channels ``u``.
        y_size: number of output channels ``y``.
        na: output lags per channel (autoregressive order).
        nb: input lags per channel, including the current sample (exogenous order).
        hidden_size: width of the hidden layers.
        num_layers: number of hidden layers (the lag-window layer is the first).
        act: activation class; bounded activations damp error feedback in free run.
        teacher_forcing: train one-step-ahead instead of free-running.
        washout: initial samples whose true outputs seed the lag buffer in free run.
    """

    def __init__(
        self,
        u_size: int,
        y_size: int,
        na: int = 8,
        nb: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        act: type[nn.Module] = nn.Tanh,
        teacher_forcing: bool = False,
        washout: int | None = None,
    ):
        super().__init__()
        if na < 1 or nb < 1:
            raise ValueError(f"lag orders must be >= 1, got na={na}, nb={nb}")
        self.u_size = u_size
        self.y_size = y_size
        self.na = na
        self.nb = nb
        self.teacher_forcing = teacher_forcing
        self.washout = max(na, nb - 1) if washout is None else washout

        self.conv_u = CausalConv1d(u_size, hidden_size, nb)
        self.conv_y = nn.Conv1d(y_size, hidden_size, na, bias=False)
        self.act = act()
        head = []
        for _ in range(num_layers - 1):
            head += [nn.Conv1d(hidden_size, hidden_size, 1), act()]
        head.append(nn.Conv1d(hidden_size, y_size, 1))
        self.head = nn.Sequential(*head)

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        "Split the concatenated ``[u, y]`` sequence into channel-first tensors."
        if x.shape[-1] != self.u_size + self.y_size:
            raise ValueError(f"expected {self.u_size + self.y_size} input channels ([u, y]), got {x.shape[-1]}")
        return x[..., : self.u_size].transpose(1, 2), x[..., self.u_size :].transpose(1, 2)

    def _forward_teacher_forced(self, x: Tensor) -> Tensor:
        u, y = self._split(x)
        y_lags = F.pad(y, (self.na, 0))[..., :-1]  # window t-na .. t-1
        h = self.act(self.conv_u(u) + self.conv_y(y_lags))
        return self.head(h).transpose(1, 2)

    def _forward_free_run(self, x: Tensor) -> Tensor:
        u, y = self._split(x)
        h_u = self.conv_u(u)  # feedback-independent, evaluated in parallel
        w_y = self.conv_y.weight  # (hidden, y_size, na); index na-1 is lag 1
        history = [x.new_zeros(x.shape[0], self.y_size)] * self.na
        outputs = []
        for t in range(x.shape[1]):
            y_lags = torch.stack(history[-self.na :], dim=-1)
            h = self.act(h_u[..., t] + torch.einsum("hck,bck->bh", w_y, y_lags))
            y_t = self.head(h.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t)
            history.append(y[..., t] if t < self.washout else y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor, ar: bool | None = None) -> Tensor:
        if ar is None:
            ar = not (self.training and self.teacher_forcing)
        return self._forward_free_run(x) if ar else self._forward_teacher_forced(x)
