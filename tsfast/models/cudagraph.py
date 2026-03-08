"""CUDA-graphed wrapper for stateful models."""

import warnings

import torch
from torch import Tensor, nn

from .state import StateSpec, discover_state_spec, flatten_state, unflatten_state


class _FlatStateBridge(nn.Module):
    """Wraps a stateful model so state is passed as a flat ``[B, D]`` tensor.

    Required for ``make_graphed_callables`` which only accepts Tensor arguments.
    """

    def __init__(self, model: nn.Module, spec: StateSpec):
        super().__init__()
        self.model = model
        self._spec = spec

    def forward(self, x: Tensor, flat_state: Tensor) -> tuple[Tensor, Tensor]:
        state = unflatten_state(flat_state, self._spec)
        pred, new_state = self.model(x, state=state)
        new_flat = flatten_state(new_state, batch_size=x.shape[0])
        return pred, new_flat


class GraphedStatefulModel(nn.Module):
    """Wraps a stateful model with CUDA-graphed forward, same interface.

    The model must return ``(output, state)`` from ``forward()``.
    The CUDA graph is captured lazily on the first forward call.
    When input shapes change (e.g. different batch size at test time),
    falls back to eager execution automatically.

    Args:
        model: stateful model returning ``(output, state)``
        num_warmup_iters: warmup iterations before graph capture
    """

    def __init__(self, model: nn.Module, num_warmup_iters: int = 3):
        super().__init__()
        self.model = model
        self.num_warmup_iters = num_warmup_iters
        self._graphed = None
        self._spec: StateSpec | None = None
        self._zero_flat: Tensor | None = None
        self._graphed_shape: tuple[int, ...] | None = None

    def reset_graph(self):
        """Clear captured graph for re-capture on next forward call."""
        self._graphed = None
        self._spec = None
        self._zero_flat = None
        self._graphed_shape = None

    def _init_graph(self, x: Tensor):
        device = x.device
        assert device.type == "cuda", "GraphedStatefulModel requires a CUDA device"
        spec = discover_state_spec(self.model, x.shape[-1], device)
        self._spec = spec
        dtype = next(self.model.parameters()).dtype
        self._zero_flat = torch.zeros(x.shape[0], spec.state_size, device=device, dtype=dtype)
        wrapper = _FlatStateBridge(self.model, spec)
        sample_x = torch.zeros_like(x)
        sample_state = torch.zeros_like(self._zero_flat)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The AccumulateGrad node's stream")
            self._graphed = torch.cuda.make_graphed_callables(
                wrapper, (sample_x, sample_state), num_warmup_iters=self.num_warmup_iters
            )
        self._graphed_shape = tuple(x.shape)

    def forward(self, x: Tensor, state=None) -> tuple[Tensor, ...]:
        if self._graphed is None:
            self._init_graph(x)
        if tuple(x.shape) != self._graphed_shape:
            return self.model(x, state=state)
        if state is not None:
            flat_state = flatten_state(state, batch_size=x.shape[0])
        else:
            flat_state = torch.zeros_like(self._zero_flat)
        pred, new_flat = self._graphed(x, flat_state)
        new_state = unflatten_state(new_flat, self._spec)
        return pred, new_state
