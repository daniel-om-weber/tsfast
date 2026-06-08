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
    """Pointwise MLP applied independently at each sequence position.

    Maps the last (feature) dimension through an MLP while preserving all leading dimensions, so
    ``[batch, seq, features]`` in gives ``[batch, seq, output_size]`` out (2-D and higher-rank
    inputs work too). Implemented with ``nn.Linear``, which broadcasts over leading dims since
    PyTorch 0.4; earlier versions used ``Conv1d(kernel_size=1)`` and those checkpoints still load
    (see ``_load_from_state_dict``).

    Args:
        input_size: number of input features
        output_size: number of output features
        hidden_size: number of hidden units per layer
        hidden_layer: number of hidden layers (``0`` = a single linear map, no activation)
        act: activation function class
        batch_first: retained for API compatibility; ``nn.Linear`` preserves leading-dim order, so
            this no longer changes the result.
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

        def lin_act(inp, out):
            return nn.Sequential(nn.Linear(inp, out), act())

        if hidden_layer < 1:
            self.lin = nn.Linear(input_size, output_size)
        else:
            self.lin = nn.Sequential(
                lin_act(input_size, hidden_size),
                *[lin_act(hidden_size, hidden_size) for _ in range(hidden_layer - 1)],
                nn.Linear(hidden_size, output_size),
            )

    def forward(self, x):
        return self.lin(x)  # nn.Linear maps the last dim, preserving all leading dims

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Back-compat: pre-Linear SeqLinear stored Conv1d(1x1) weights of shape (out, in, 1).
        # Squeeze the trailing singleton so they load into the (out, in) Linear weights. This runs
        # before the child Linear modules load, and keys are unchanged, so only the shape is fixed.
        for key in list(state_dict.keys()):
            if key.startswith(prefix) and key.endswith("weight"):
                w = state_dict[key]
                if w.dim() == 3 and w.shape[-1] == 1:
                    state_dict[key] = w.squeeze(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


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
