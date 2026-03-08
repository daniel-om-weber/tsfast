"""Utilities for flattening, unflattening, and detaching nested model state."""

import torch
import torch.utils._pytree as pytree
from torch import Tensor, nn


def detach_state(state):
    """Recursively detach tensors from the computation graph."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (list, tuple)):
        return type(state)(detach_state(s) for s in state)
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    return state


def flatten_state(state, batch_size: int) -> tuple[Tensor, tuple]:
    """Flatten arbitrary nested state to a single ``[B, D]`` tensor.

    Uses ``pytree`` to walk any nested structure (list, tuple, dict) and
    collect leaf tensors.  Each leaf is reshaped to ``[B, -1]`` where
    *B* = *batch_size*, then concatenated along the last dimension.

    Args:
        state: arbitrarily nested container of tensors (list, tuple, dict).
        batch_size: batch dimension size, used to reshape each leaf to ``[B, -1]``.

    Returns:
        ``(flat, spec)`` where *spec* allows reconstruction via
        :func:`unflatten_state`.
    """
    leaves, tree_spec = pytree.tree_flatten(state)
    shapes = tuple(leaf.shape for leaf in leaves)
    flat_leaves = [leaf.reshape(batch_size, -1) for leaf in leaves]
    widths = tuple(fl.shape[-1] for fl in flat_leaves)
    return torch.cat(flat_leaves, dim=-1), (tree_spec, shapes, widths)


def unflatten_state(flat: Tensor, spec: tuple):
    """Reconstruct nested state from a flat ``[B, D]`` tensor + *spec*."""
    tree_spec, shapes, widths = spec
    splits = flat.split(widths, dim=-1)
    leaves = [s.reshape(shape).contiguous() for s, shape in zip(splits, shapes)]
    return pytree.tree_unflatten(leaves, tree_spec)


class FlatStateBridge(nn.Module):
    """Wraps a stateful model so state is passed as a flat ``[B, D]`` tensor.

    Required for ``make_graphed_callables`` which only accepts Tensor arguments.
    """

    def __init__(self, model: nn.Module, spec: tuple):
        super().__init__()
        self.model = model
        self._spec = spec

    def forward(self, x: Tensor, flat_state: Tensor) -> tuple[Tensor, Tensor]:
        state = unflatten_state(flat_state, self._spec)
        pred, new_state = self.model(x, state=state)
        new_flat, _ = flatten_state(new_state, batch_size=x.shape[0])
        return pred, new_flat
