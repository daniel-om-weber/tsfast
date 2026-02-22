"""Autoregressive progressive prediction model."""

__all__ = ["ARProg"]

import torch
from torch import Tensor, nn

from fastcore.meta import delegates

from ..models.layers import AR_Model
from ..models.rnn import RNN, SimpleRNN


class ARProg(nn.Module):
    """RNN model with teacher-forced initialization and autoregressive prediction.

    Uses an initial segment with teacher forcing to warm up hidden state,
    then switches to autoregressive mode for the remaining sequence.

    Args:
        n_u: number of input signals.
        n_x: number of external state signals.
        n_y: number of output signals.
        init_sz: number of time steps for teacher-forced initialization.
    """

    @delegates(RNN, keep=True)
    def __init__(self, n_u: int, n_x: int, n_y: int, init_sz: int, **kwargs):
        super().__init__()
        self.n_u = n_u
        self.n_x = n_x
        self.n_y = n_y
        self.init_sz = init_sz

        self.rnn_model = AR_Model(
            SimpleRNN(input_size=n_u + n_x + n_y, output_size=n_x + n_y, return_state=True, **kwargs),
            model_has_state=True,
            return_state=True,
            ar=True,
            out_sz=n_x + n_y,
        )

    def forward(self, x: Tensor) -> Tensor:
        y_x = x[..., self.n_u :]
        u = x[..., : self.n_u]

        inp_tf = torch.cat([u[:, : self.init_sz], y_x[:, : self.init_sz]], dim=-1)
        out_init, h = self.rnn_model(inp_tf, ar=False)
        self.rnn_model.y_init = y_x[:, self.init_sz : self.init_sz + 1]
        out_prog, _ = self.rnn_model(u[:, self.init_sz :], h_init=h, ar=True)

        result = torch.cat([out_init, out_prog], 1)

        return result[..., -self.n_y :]
