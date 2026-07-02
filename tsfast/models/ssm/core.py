"""Discrete-time neural state space models with interchangeable execution backends."""

__all__ = [
    "NeuralStateSpace",
]

from dataclasses import dataclass

import torch
from torch import nn

_ACTS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
}


@dataclass(frozen=True)
class SSMSpec:
    """Static architecture description of the transition MLP, used to generate backend kernels.

    Args:
        n_state: state dimension (also the model output size)
        n_input: exogenous input dimension
        hidden: widths of the hidden layers (may be empty for a linear state space model)
        act: activation name, a key of ``_ACTS``
    """

    n_state: int
    n_input: int
    hidden: tuple[int, ...]
    act: str

    @property
    def dims(self) -> tuple[int, ...]:
        """Feature sizes through the MLP: ``(n_state + n_input, *hidden, n_state)``."""
        return (self.n_state + self.n_input, *self.hidden, self.n_state)

    @property
    def n_linear(self) -> int:
        return len(self.hidden) + 1


class NeuralStateSpace(nn.Module):
    """Discrete-time neural state space model ``x_{k+1} = f(x_k, u_k)`` with an MLP transition.

    The rollout over the input sequence is irreducibly sequential, so a naive per-step Python
    loop is dispatch-bound rather than FLOP-bound. Several backends implement the identical
    computation:

    - ``"eager"``: plain Python loop — the reference implementation, any device and dtype.
    - ``"compiled"``: ``torch.compile`` over the unrolled loop — any device, slow first call.
    - ``"c"``: generated C++ rollout with a fused BPTT backward, OpenMP-parallel over the
      batch — float32 on CPU, fastest CPU option for small models.
    - ``"triton"``: persistent-GEMV rollout kernel with a fused BPTT backward — float32 on
      CUDA, hidden widths up to 128.
    - ``"auto"``: ``triton`` when it applies, else ``compiled`` on CUDA; ``eager`` on CPU
      (select ``"c"`` explicitly to trade a one-time compilation for much faster CPU training).

    All backends share the same parameters, so the backend can be switched at any time via
    the ``backend`` attribute. The fused backends are loss-agnostic ``autograd.Function``s:
    ``loss.backward()`` and every ``Learner`` feature work unchanged.

    Args:
        n_state: state dimension (model output size).
        n_input: exogenous input dimension.
        hidden_size: hidden width, or an explicit list of hidden widths for arbitrary layers.
        num_layers: number of hidden layers (ignored when ``hidden_size`` is a list).
        act: activation name, one of ``tanh``, ``sigmoid``, ``relu``.
        backend: execution backend, see above.
    """

    def __init__(
        self,
        n_state: int,
        n_input: int,
        hidden_size: int | list[int] = 64,
        num_layers: int = 2,
        act: str = "tanh",
        backend: str = "auto",
    ):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"unknown activation {act!r}, expected one of {sorted(_ACTS)}")
        hidden = tuple(hidden_size) if isinstance(hidden_size, (list, tuple)) else (hidden_size,) * num_layers
        self.spec = SSMSpec(n_state, n_input, hidden, act)
        self.backend = backend

        layers: list[nn.Module] = []
        dims = self.spec.dims
        for i in range(self.spec.n_linear):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < self.spec.n_linear - 1:
                layers.append(_ACTS[act]())
        self.net = nn.Sequential(*layers)
        self._compiled_rollout = None

    @property
    def linears(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def _params_flat(self) -> list[torch.Tensor]:
        return [t for lin in self.linears for t in (lin.weight, lin.bias)]

    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Roll the transition MLP over the input sequence.

        Args:
            u: input sequence ``[batch, seq, n_input]``.
            x0: initial state ``[batch, n_state]`` (or ``[batch, 1, n_state]``); zeros if None.

        Returns:
            State sequence ``[batch, seq, n_state]`` containing ``x_1 .. x_L``.
        """
        if x0 is None:
            x0 = u.new_zeros(u.shape[0], self.spec.n_state)
        elif x0.dim() == 3:
            x0 = x0.squeeze(1)
        match self._resolve_backend(u):
            case "eager":
                return self._rollout_eager(u, x0)
            case "compiled":
                return self._rollout_compiled(u, x0)
            case "c":
                from .backend_c import c_rollout

                return c_rollout(self.spec, u, x0, self._params_flat())
            case "triton":
                from .backend_triton import triton_rollout

                return triton_rollout(self.spec, u, x0, self._params_flat())
            case unknown:
                raise ValueError(f"unknown backend {unknown!r}")

    def _resolve_backend(self, u: torch.Tensor) -> str:
        if self.backend != "auto":
            return self.backend
        if u.is_cuda and u.dtype == torch.float32:
            from . import backend_triton

            if backend_triton.is_available() and backend_triton.fits(self.spec):
                return "triton"
            return "compiled"
        return "eager"

    def _rollout_eager(self, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        outs = []
        for t in range(u.shape[1]):
            x = self.net(torch.cat((x, u[:, t]), dim=1))
            outs.append(x)
        return torch.stack(outs, dim=1)

    def _rollout_compiled(self, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        if self._compiled_rollout is None:
            # The unrolled sequence is one large graph; raise the recompile budget so shape
            # changes (batch/seq) do not silently fall back to eager.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._compiled_rollout = torch.compile(self._rollout_eager, dynamic=False)
        return self._compiled_rollout(u, x0)


def check_rollout_args(spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, device_type: str) -> None:
    """Validate inputs of the fused backends, which are float32-only and device-specific."""
    if u.device.type != device_type:
        raise RuntimeError(f"this backend requires {device_type} tensors, got {u.device.type}")
    if u.dtype != torch.float32 or x0.dtype != torch.float32:
        raise RuntimeError(f"this backend requires float32, got {u.dtype}")
    if u.dim() != 3 or u.shape[-1] != spec.n_input:
        raise RuntimeError(f"expected u of shape [B, L, {spec.n_input}], got {tuple(u.shape)}")
    if x0.shape != (u.shape[0], spec.n_state):
        raise RuntimeError(f"expected x0 of shape [{u.shape[0]}, {spec.n_state}], got {tuple(x0.shape)}")


def mlp_param_grads(
    spec: SSMSpec,
    x0: torch.Tensor,
    u: torch.Tensor,
    out: torch.Tensor,
    zs: list[torch.Tensor],
    gy: torch.Tensor,
    gas: list[torch.Tensor],
    w0: torch.Tensor,
    need_du: bool,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Parameter gradients of the rollout as batched GEMMs over the flattened adjoints.

    The state-adjoint recurrence is the only sequential part of BPTT; the parameter
    gradients ``dW_l = sum_{b,t} ga_l ⊗ z_{l-1}`` are plain reductions over all ``B*L``
    step samples, which is exactly the batched GEMM BLAS is built for. Shared by the C
    and Triton backends.

    Args:
        x0: initial state ``[B, NX]``.
        u: input sequence ``[B, L, NU]``.
        out: forward result ``[B, L, NX]`` (``out[t] = x_{t+1}``).
        zs: stored post-activation hidden sequences, one ``[B, L, h]`` per hidden layer.
        gy: total adjoint of each step output ``[B, L, NX]``.
        gas: pre-activation adjoints of the hidden layers, one ``[B, L, h]`` per layer.
        w0: first-layer weight ``[dims[1], NX+NU]``, needed for the input gradient.
        need_du: also compute the gradient w.r.t. ``u``.

    Returns:
        ``(grads, du)`` where grads is ``[dW_0, db_0, dW_1, db_1, ...]`` in layer order.
    """
    B, L = u.shape[0], u.shape[1]
    BL = B * L
    # state fed INTO each step: x_t = (x0, out[0], ..., out[L-2])
    xt = torch.cat((x0.unsqueeze(1), out[:, :-1]), dim=1)
    inp0 = torch.cat((xt.reshape(BL, spec.n_state), u.reshape(BL, spec.n_input)), dim=1)
    acts = [inp0] + [z.reshape(BL, z.shape[-1]) for z in zs]
    adjoints = [ga.reshape(BL, ga.shape[-1]) for ga in gas] + [gy.reshape(BL, spec.n_state)]
    grads: list[torch.Tensor] = []
    for a_prev, ga in zip(acts, adjoints):
        grads.append(ga.t() @ a_prev)
        grads.append(ga.sum(0))
    du = (adjoints[0] @ w0[:, spec.n_state :]).reshape(B, L, spec.n_input) if need_du else None
    return grads, du
