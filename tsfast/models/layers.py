"""Reusable model layers and wrappers."""

__all__ = [
    "SeqLinear",
    "AR_Model",
    "SeqAggregation",
]

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import Mish


class SeqLinear(nn.Module):
    """Pointwise MLP applied independently at each sequence position via 1x1 convolutions.

    Args:
        input_size: number of input features
        output_size: number of output features
        hidden_size: number of hidden units per layer
        hidden_layer: number of hidden layers
        act: activation function class
        batch_first: if ``True``, input shape is ``[batch, seq, features]``
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 100,
        hidden_layer: int = 1,
        act=Mish,
        batch_first: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first

        def conv_act(inp, out):
            return nn.Sequential(nn.Conv1d(inp, out, 1), act())

        if hidden_layer < 1:
            self.lin = nn.Conv1d(input_size, output_size, 1)
        else:
            self.lin = nn.Sequential(
                conv_act(input_size, hidden_size),
                *[conv_act(hidden_size, hidden_size) for _ in range(hidden_layer - 1)],
                nn.Conv1d(hidden_size, output_size, 1),
            )

    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        out = self.lin(x.transpose(1, 2)).transpose(1, 2)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out


class AR_Model(nn.Module):
    """Autoregressive model container.

    Runs autoregressively when the output sequence is not provided, otherwise
    uses teacher forcing. Normalization should be handled externally via
    ScaledModel wrapping.

    Args:
        model: inner model to wrap
        ar: if ``True``, default to autoregressive mode in forward
        model_has_state: if ``True``, the inner model accepts and returns hidden state
        return_state: if ``True``, return ``(output, hidden_state)`` tuple
        out_sz: output feature size, used to initialize autoregressive seed
    """

    def __init__(
        self,
        model: nn.Module,
        ar: bool = True,
        model_has_state: bool = False,
        return_state: bool = False,
        out_sz: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.ar = ar
        self.model_has_state = model_has_state
        self.return_state = return_state
        self.out_sz = out_sz
        if return_state and not model_has_state:
            raise ValueError("return_state=True requires model_has_state=True")

    def forward(self, inp: torch.Tensor, state=None, ar: bool | None = None):
        if ar is None:
            ar = self.ar

        # Unpack state — accept dict, list (legacy), or None
        match state:
            case {"h": h_init, **rest}:
                y_prev = rest.get("y_init", None)
            case list():
                h_init = state
                y_prev = None
            case _:
                h_init = None
                y_prev = None

        if ar:
            y_e = []
            y_next = y_prev if y_prev is not None else torch.zeros(inp.shape[0], 1, self.out_sz, device=inp.device)

            if self.model_has_state:
                h0 = h_init
                for u_in in inp.split(1, dim=1):
                    x = torch.cat((u_in, y_next), dim=2)
                    y_next, h0 = self.model(x, h0)
                    y_e.append(y_next)
            else:
                for u_in in inp.split(1, dim=1):
                    x = torch.cat((u_in, y_next), dim=2)
                    y_next = self.model(x)
                    y_e.append(y_next)

            y_e = torch.cat(y_e, dim=1)
            h0 = h0 if self.model_has_state else None
        else:
            if self.model_has_state:
                y_e, h0 = self.model(inp, h_init)
            else:
                y_e = self.model(inp)
                h0 = None

        new_state = {"h": h0, "y_init": y_e[:, -1:].detach()}
        return y_e if not self.return_state else (y_e, new_state)


class SeqAggregation(nn.Module):
    """Aggregation layer that reduces the sequence dimension.

    Args:
        func: aggregation function taking (tensor, dim) and returning reduced tensor
        dim: sequence dimension to aggregate over
    """

    def __init__(
        self,
        func: Callable = lambda x, dim: x.select(dim, -1),
        dim: int = 1,
    ):
        super().__init__()
        self.func = func
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "Apply the aggregation function to the input tensor."
        return self.func(x, dim=self.dim)
