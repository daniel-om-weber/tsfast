"""Utilities for flattening, unflattening, and detaching nested model state."""

import torch
import torch.utils._pytree as pytree
from torch import Tensor, nn
from typing import NamedTuple


class StateSpec(NamedTuple):
    """Batch-agnostic specification for a nested model state structure.

    Stores shape templates with ``-1`` at the batch dimension, enabling
    flatten/unflatten at any batch size.
    """

    tree_spec: pytree.TreeSpec
    templates: tuple[tuple[int, ...], ...]
    widths: tuple[int, ...]

    @property
    def state_size(self) -> int:
        """Total flat state dimension (sum of all leaf widths)."""
        return sum(self.widths)


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


def discover_state_spec(model: nn.Module, n_in: int, device: str | torch.device = "cpu") -> StateSpec:
    """Probe a stateful model to discover its state structure.

    Forwards two batches of different sizes and compares the resulting state
    shapes to identify the batch dimension.  Templates store ``-1`` at the
    batch dim so the spec is batch-agnostic.

    Args:
        model: stateful model returning ``(output, state)``
        n_in: number of input features
        device: device for probe tensors
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, state_a = model(torch.zeros(2, 1, n_in, device=device), state=None)
            _, state_b = model(torch.zeros(3, 1, n_in, device=device), state=None)
    finally:
        if was_training:
            model.train()

    leaves_a, tree_spec = pytree.tree_flatten(state_a)
    leaves_b, _ = pytree.tree_flatten(state_b)

    templates = []
    widths = []
    for la, lb in zip(leaves_a, leaves_b):
        sa, sb = la.shape, lb.shape
        assert len(sa) == len(sb), f"Rank mismatch: {sa} vs {sb}"
        template = []
        width = 1
        batch_found = False
        for da, db in zip(sa, sb):
            if da != db and not batch_found:
                template.append(-1)
                batch_found = True
            else:
                template.append(da)
                width *= da
        assert batch_found, f"No batch dim found: shapes {sa} vs {sb} are identical"
        templates.append(tuple(template))
        widths.append(width)

    return StateSpec(tree_spec, tuple(templates), tuple(widths))


def build_spec_from_state(state, batch_size: int) -> StateSpec:
    """Build a StateSpec from a single state sample by inferring the batch dimension.

    The batch dimension is identified as the first dimension matching ``batch_size``.
    Ambiguous when a non-batch dimension equals ``batch_size``.
    """
    leaves, tree_spec = pytree.tree_flatten(state)
    templates = []
    widths = []
    for leaf in leaves:
        template = []
        width = 1
        batch_found = False
        for d in leaf.shape:
            if d == batch_size and not batch_found:
                template.append(-1)
                batch_found = True
            else:
                template.append(d)
                width *= d
        assert batch_found, f"No dim matching batch_size={batch_size} in shape {leaf.shape}"
        templates.append(tuple(template))
        widths.append(width)
    return StateSpec(tree_spec, tuple(templates), tuple(widths))


def flatten_state(state, batch_size: int) -> Tensor:
    """Flatten arbitrary nested state to a single ``[B, D]`` tensor."""
    leaves, _ = pytree.tree_flatten(state)
    flat_leaves = [leaf.reshape(batch_size, -1) for leaf in leaves]
    return torch.cat(flat_leaves, dim=-1)


def unflatten_state(flat: Tensor, spec: StateSpec):
    """Reconstruct nested state from a flat ``[B, D]`` tensor + StateSpec."""
    B = flat.shape[0]
    splits = flat.split(spec.widths, dim=-1)
    leaves = []
    for s, template in zip(splits, spec.templates):
        shape = tuple(B if d == -1 else d for d in template)
        leaves.append(s.reshape(shape).contiguous())
    return pytree.tree_unflatten(leaves, spec.tree_spec)


class FlatStateBridge(nn.Module):
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
