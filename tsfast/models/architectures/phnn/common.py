"""Spec, parameter extraction, and applicability checks shared by the PHNN backends.

The fused backends receive the component-net weights as flat lists in a fixed
canonical order (``params_of``) and a static :class:`PHNNSpec` describing the shapes
and step structure. The eager and compiled paths in ``phnn.py`` are untouched; these
helpers only read the existing submodules of a :class:`~tsfast.models.architectures.phnn.PHNNCore`.
"""

__all__ = [
    "PHNNSpec",
    "spec_of",
    "params_of",
    "supports",
]

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PHNNSpec:
    """Static description of a PHNN step, used to specialize/cache kernels.

    ``dt`` is deliberately not part of the spec: it is a runtime scalar, so changing
    the time-normalization constant never triggers a recompile.
    """

    n_state: int
    n_input: int
    n_output: int
    hidden: int
    num_layers: int
    rk4_steps: int
    output: str  # "ph" or "linear"
    has_bound: bool

    @property
    def n_linear(self) -> int:
        """Linear layers per component net (``num_layers`` hidden + 1 output)."""
        return self.num_layers + 1


def _linears(seq: nn.Module) -> list[nn.Linear]:
    return [m for m in seq if isinstance(m, nn.Linear)]


def spec_of(core) -> PHNNSpec:
    """Build the :class:`PHNNSpec` for a :class:`~tsfast.models.architectures.phnn.PHNNCore`."""
    lin = _linears(core.hamiltonian.net)
    hidden = lin[0].out_features
    num_layers = len(lin) - 1
    return PHNNSpec(
        n_state=core.n_state,
        n_input=core.n_input,
        n_output=core.n_output,
        hidden=hidden,
        num_layers=num_layers,
        rk4_steps=core.rk4_steps,
        output=core.output,
        has_bound=core.hamiltonian.lower_bound is not None,
    )


def params_of(core) -> dict:
    """Component-net parameters in canonical backend order.

    Returns a dict with, per net, the per-layer weight/bias lists (output layer last),
    plus the g-net linear bypass and the optional linear output map. The tensors are
    the live ``nn.Parameter``s, so backend gradients flow to them through autograd.
    """
    h_lin = _linears(core.hamiltonian.net)
    j_lin = _linears(core.j_net)
    r_lin = _linears(core.r_net)
    g_lin = _linears(core.g_net.mlp)
    out = {
        "hw": [m.weight for m in h_lin],
        "hb": [m.bias for m in h_lin],
        "jw": [m.weight for m in j_lin],
        "jb": [m.bias for m in j_lin],
        "rw": [m.weight for m in r_lin],
        "rb": [m.bias for m in r_lin],
        "glw": core.g_net.lin.weight,
        "glb": core.g_net.lin.bias,
        "gw": [m.weight for m in g_lin],
        "gb": [m.bias for m in g_lin],
        "ow": core.output_map.weight if core.output_map is not None else None,
        "ob": core.output_map.bias if core.output_map is not None else None,
    }
    return out


def flat_params(core) -> list[torch.Tensor]:
    """All backend parameters flattened in a single canonical order (for autograd wiring)."""
    p = params_of(core)
    flat = [*p["hw"], *p["hb"], *p["jw"], *p["jb"], *p["rw"], *p["rb"], p["glw"], p["glb"], *p["gw"], *p["gb"]]
    if p["ow"] is not None:
        flat += [p["ow"], p["ob"]]
    return flat


def bound_value(core) -> float:
    """The ELU shift ``b = lower_bound + 1`` (0.0 when the bound is disabled)."""
    lb = core.hamiltonian.lower_bound
    return 0.0 if lb is None else lb + 1.0


def supports(spec: PHNNSpec, backend: str) -> bool:
    """Whether ``backend`` can run this spec.

    Common caps (both backends): a single RK4 step and at least one hidden layer.
    The Triton persistent kernel additionally requires the weights to fit on-chip:
    ``hidden <= 128``, ``n_state <= 16``, ``num_layers <= 2`` (the benchmark search
    space); outside that envelope callers fall back to the compiled loop.
    """
    if spec.rk4_steps != 1 or spec.num_layers < 1:
        return False
    if backend == "c":
        return True
    if backend == "triton":
        return spec.hidden <= 128 and spec.n_state <= 16 and spec.num_layers <= 2
    return False
