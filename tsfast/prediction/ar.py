"""Autoregressive progressive prediction model."""

__all__ = ["ARProg"]

import random

import torch
from torch import Tensor, nn

from ..models.layers import AR_Model
from ..models.rnn import SimpleRNN


class ARProg(nn.Module):
    """RNN model with teacher-forced initialization and autoregressive prediction.

    Uses an initial segment with teacher forcing to warm up hidden state,
    then switches to autoregressive mode for the remaining sequence.

    Args:
        n_u: number of input signals.
        n_x: number of external state signals.
        n_y: number of output signals.
        init_sz: number of time steps for teacher-forced initialization.
        init_sz_range: if set, randomize init_sz within (min, max) during training.
    """

    def __init__(
        self, n_u: int, n_x: int, n_y: int, init_sz: int, init_sz_range: tuple[int, int] | None = None, **kwargs
    ):
        super().__init__()
        self.n_u = n_u
        self.n_x = n_x
        self.n_y = n_y
        self.init_sz = init_sz
        self.init_sz_range = init_sz_range

        self.rnn_model = AR_Model(
            SimpleRNN(input_size=n_u + n_x + n_y, output_size=n_x + n_y, return_state=True, **kwargs),
            model_has_state=True,
            return_state=True,
            ar=True,
            out_sz=n_x + n_y,
        )

    def forward(self, x: Tensor) -> Tensor:
        init_sz = random.randint(*self.init_sz_range) if self.training and self.init_sz_range else self.init_sz
        self._effective_init_sz = init_sz

        y_x = x[..., self.n_u :]
        u = x[..., : self.n_u]

        inp_tf = torch.cat([u[:, :init_sz], y_x[:, :init_sz]], dim=-1)
        out_init, returned_state = self.rnn_model(inp_tf, ar=False)

        prog_state = {
            "h": returned_state["h"],
            "y_init": y_x[:, init_sz : init_sz + 1],
        }
        out_prog, _ = self.rnn_model(u[:, init_sz:], state=prog_state, ar=True)

        result = torch.cat([out_init, out_prog], 1)

        return result[..., -self.n_y :]
