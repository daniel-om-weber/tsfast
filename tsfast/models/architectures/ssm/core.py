"""Discrete-time neural state space models with interchangeable execution backends."""

__all__ = [
    "NeuralStateSpace",
    "fused_rollout",
]

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..._core.dispatch import get_backend, resolve

_ACTS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
}


@dataclass(frozen=True)
class SSMSpec:
    """Static architecture description of the transition MLP, used to generate backend kernels.

    Args:
        n_state: state dimension
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


# Fused-kernel backends, resolved through dispatch.resolve: each module exposes
# supports(spec, u, x0) -> str | None plus forward_train/forward_infer/backward entry
# points. The custom ops try every family native to the input's device under "auto";
# the model-level dispatch keeps CPU "auto" non-fused (the C backend trades a one-time
# host compilation for speed and stays opt-in).
_FUSED = {
    "triton": "tsfast.models.architectures.ssm.backend_triton",
    "c": "tsfast.models.architectures.ssm.backend_c",
    "metal": "tsfast.models.architectures.ssm.backend_metal",
}
_OP_AUTO = {"cuda": ("triton",), "cpu": ("c",), "mps": ("metal",)}
_CORE_AUTO = {"cuda": ("triton",), "mps": ("metal",)}


@torch._dynamo.assume_constant_result
def _rollout_mode(backend: str, spec: SSMSpec, u: Tensor, x0: Tensor) -> str:
    """Execution mode for this call: ``"eager"``, ``"compiled"``, or ``"fused"``.

    ``backend`` is the instance attribute; ``"auto"`` defers to the process preference
    (``tsfast.models.set_backend``/``use_backend``), and a ``"reference"`` preference
    disables fused kernels even for instances that request one explicitly. An explicit
    fused family that cannot handle the inputs warns once (via ``dispatch.resolve``)
    and falls back to the non-fused policy: compiled on CUDA float32, eager elsewhere.
    """
    pref = get_backend()
    if backend == "auto":
        backend = pref
    elif pref == "reference" and backend in _FUSED:
        backend = "reference"
    fallback = "compiled" if u.device.type == "cuda" and u.dtype == torch.float32 else "eager"
    match backend:
        case "eager" | "compiled":
            return backend
        case "reference":
            return fallback
        case "auto" | "triton" | "c" | "metal":
            mod = resolve("ssm.rollout", _FUSED, _CORE_AUTO.get(u.device.type, ()), (spec, u, x0), requested=backend)
            return "fused" if mod is not None else fallback
        case unknown:
            raise ValueError(f"unknown backend {unknown!r}")


def _fused_module(spec: SSMSpec, u: Tensor, x0: Tensor):
    """The fused backend module serving this op call, honoring the process preference.

    An explicit process preference can name a family foreign to this device (e.g.
    ``use_backend("triton")`` around a ``backend="c"`` model on CPU); the op then falls
    back to the device's own fused family rather than failing, since the model dispatch
    already committed to a fused rollout.
    """
    order = _OP_AUTO.get(u.device.type, ())
    mod = resolve("ssm.rollout", _FUSED, order, (spec, u, x0))
    if mod is None:
        mod = resolve("ssm.rollout", _FUSED, order, (spec, u, x0), requested="auto")
    if mod is None:
        raise RuntimeError(
            f"no fused NeuralStateSpace backend usable for device {u.device.type!r}; "
            "use backend='eager' or backend='compiled'"
        )
    return mod


# ------------------------------------------------------------------------- custom ops
#
# The rollout is exposed as torch.library custom ops so it composes with torch.compile
# (no graph breaks), fake/meta tracing, and export. Spec fields cross the op boundary
# as scalars (frozen dataclasses cannot) and are rebuilt inside the impls. The backward
# is its own registered op so compiled autograd also sees no graph break. Tensor inputs
# may arrive as non-contiguous views (in the backward exactly as the forward received
# them), and the kernels index raw data pointers, so every impl materializes its inputs.


def _spec_from(n_state: int, n_input: int, hidden: list[int], act: str) -> SSMSpec:
    return SSMSpec(n_state, n_input, tuple(hidden), act)


@torch.library.custom_op("tsfast::ssm_rollout", mutates_args=())
def _ssm_rollout(
    u: Tensor, x0: Tensor, params: list[Tensor], n_state: int, n_input: int, hidden: list[int], act: str
) -> Tensor:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, hidden, act)
    return _fused_module(spec, u, x0).forward_infer(spec, u, x0, params)


@_ssm_rollout.register_fake
def _(u, x0, params, n_state, n_input, hidden, act):
    return u.new_empty(u.shape[0], u.shape[1], n_state)


@torch.library.custom_op("tsfast::ssm_rollout_train", mutates_args=())
def _ssm_rollout_train(
    u: Tensor, x0: Tensor, params: list[Tensor], n_state: int, n_input: int, hidden: list[int], act: str
) -> tuple[Tensor, list[Tensor]]:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, hidden, act)
    out, zs = _fused_module(spec, u, x0).forward_train(spec, u, x0, params)
    return out, zs


@_ssm_rollout_train.register_fake
def _(u, x0, params, n_state, n_input, hidden, act):
    B, L = u.shape[0], u.shape[1]
    return u.new_empty(B, L, n_state), [u.new_empty(B, L, h) for h in hidden]


@torch.library.custom_op("tsfast::ssm_rollout_bwd", mutates_args=())
def _ssm_rollout_bwd(
    grad_out: Tensor | None,
    u: Tensor,
    x0: Tensor,
    out: Tensor,
    zs: list[Tensor],
    params: list[Tensor],
    n_state: int,
    n_input: int,
    hidden: list[int],
    act: str,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    u, x0, out = u.contiguous(), x0.contiguous(), out.contiguous()
    g = grad_out.contiguous() if grad_out is not None else torch.zeros_like(out)
    zs = [z.contiguous() for z in zs]
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, hidden, act)
    weights = params[0::2]
    gy, gas, gx0 = _fused_module(spec, u, x0).backward(spec, g, zs, weights)
    grads, du = mlp_param_grads(spec, x0, u, out, zs, gy, gas, w0=weights[0])
    return du, gx0, grads


@_ssm_rollout_bwd.register_fake
def _(grad_out, u, x0, out, zs, params, n_state, n_input, hidden, act):
    return torch.empty_like(u), torch.empty_like(x0), [torch.empty_like(p) for p in params]


def _train_setup(ctx, inputs, output):
    u, x0, params, n_state, n_input, hidden, act = inputs
    out, zs = output
    ctx.fields = (n_state, n_input, list(hidden), act)
    ctx.save_for_backward(u, x0, out, *zs, *params)


def _train_backward(ctx, grad_out, grad_zs):
    n_state, n_input, hidden, act = ctx.fields
    saved = ctx.saved_tensors
    u, x0, out = saved[0], saved[1], saved[2]
    zs = list(saved[3 : 3 + len(hidden)])
    params = list(saved[3 + len(hidden) :])
    du, dx0, dparams = _ssm_rollout_bwd(grad_out, u, x0, out, zs, params, n_state, n_input, hidden, act)
    return du, dx0, list(dparams), None, None, ([] if not hidden else None), None


_ssm_rollout_train.register_autograd(_train_backward, setup_context=_train_setup)


def fused_rollout(spec: SSMSpec, u: Tensor, x0: Tensor, params: list[Tensor]) -> Tensor:
    """Run the state rollout through the fused-kernel custom ops (autograd-capable).

    Picks the training op (stores the hidden activations for the analytic BPTT backward)
    when gradients are live, else the inference op, which keeps no intermediates.

    Args:
        spec: transition-MLP architecture.
        u: input sequence ``[B, L, n_input]``, float32 on the fused backend's device.
        x0: initial state ``[B, n_state]``, float32.
        params: transition parameters ``[W_0, b_0, W_1, b_1, ...]`` in layer order.

    Returns:
        States ``x_1 .. x_L`` as ``[B, L, n_state]``.
    """
    fields = (spec.n_state, spec.n_input, list(spec.hidden), spec.act)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (u, x0, *params)):
        return _ssm_rollout_train(u, x0, list(params), *fields)[0]
    return _ssm_rollout(u, x0, list(params), *fields)


class NeuralStateSpace(nn.Module):
    """Discrete-time neural state space model with an MLP transition and a linear observation.

    ``x_{k+1} = f(x_k, u_k)``, ``y_k = C x_k + d``. The latent state dimension is independent
    of the output dimension, so the model can carry more internal dynamics (e.g. velocities,
    phases) than the measured signal exposes — ``n_state`` must be at least the order of the
    system being identified.

    The rollout over the input sequence is irreducibly sequential, so a naive per-step Python
    loop is dispatch-bound rather than FLOP-bound. Several backends implement the identical
    state rollout (the observation map is a single batched matmul applied on top and works
    with every backend unchanged):

    - ``"eager"``: plain Python loop — the reference implementation, any device and dtype.
    - ``"compiled"``: ``torch.compile`` over the unrolled loop — any device, slow first call.
    - ``"c"``: generated C++ rollout with a fused BPTT backward, batch-parallel via the
      ATen thread pool — float32 on CPU, fastest CPU option for small models.
    - ``"triton"``: persistent-GEMV rollout kernel with a fused BPTT backward — float32 on
      CUDA, hidden widths up to 128.
    - ``"metal"``: persistent register-resident rollout kernel with a fused BPTT backward —
      float32 on MPS (Apple GPUs), layer widths up to 128.
    - ``"auto"``: defers to the process-wide preference (``tsfast.models.set_backend`` /
      ``use_backend``); under an ``"auto"`` preference picks ``triton`` when it applies,
      else ``compiled`` on CUDA; ``metal`` when it applies on MPS; ``eager`` on CPU
      (select ``"c"`` explicitly to trade a one-time compilation for much faster CPU
      training). A ``"reference"`` preference forces the non-fused path everywhere.

    An explicit backend that cannot handle the inputs warns once per process and falls
    back to the non-fused path instead of raising.

    All backends share the same parameters, so the backend can be switched at any time via
    the ``backend`` attribute. The fused backends run as registered ``torch.library`` custom
    ops with analytic-BPTT backward ops, so they are loss-agnostic and compose with
    ``torch.compile``: ``loss.backward()`` and every ``Learner`` feature work unchanged.

    With ``return_state=True`` the model follows the stateful-model protocol
    (``forward(u, state=...) -> (out, {"x": x_last})``), so ``TbpttLearner`` state carrying
    and ``GraphedStatefulModel`` CUDA-graph capture work like for the RNNs. The carried
    state is the physical state itself, so chunked rollouts are exactly equivalent to the
    full sequence.

    Args:
        n_input: exogenous input dimension.
        n_output: observed output dimension.
        n_state: latent state dimension.
        hidden_size: hidden width, or an explicit list of hidden widths for arbitrary layers.
        num_layers: number of hidden layers (ignored when ``hidden_size`` is a list).
        act: activation name, one of ``tanh``, ``sigmoid``, ``relu``.
        backend: execution backend, see above.
        return_state: if ``True``, return ``(output, state)`` tuple.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_state: int = 8,
        hidden_size: int | list[int] = 64,
        num_layers: int = 2,
        act: str = "tanh",
        backend: str = "auto",
        return_state: bool = False,
    ):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"unknown activation {act!r}, expected one of {sorted(_ACTS)}")
        hidden = tuple(hidden_size) if isinstance(hidden_size, (list, tuple)) else (hidden_size,) * num_layers
        self.spec = SSMSpec(n_state, n_input, hidden, act)
        self.n_output = n_output
        self.backend = backend
        self.return_state = return_state

        layers: list[nn.Module] = []
        dims = self.spec.dims
        for i in range(self.spec.n_linear):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < self.spec.n_linear - 1:
                layers.append(_ACTS[act]())
        self.net = nn.Sequential(*layers)
        self.output_map = nn.Linear(n_state, n_output)
        self._compiled_rollout = None

    @property
    def linears(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def _params_flat(self) -> list[torch.Tensor]:
        """Transition MLP parameters in backend order (the observation map is applied outside)."""
        return [t for lin in self.linears for t in (lin.weight, lin.bias)]

    def forward(self, u: torch.Tensor, x0: torch.Tensor | None = None, state: dict | None = None):
        """Roll the transition MLP over the input sequence and observe the states.

        Args:
            u: input sequence ``[batch, seq, n_input]``.
            x0: initial state ``[batch, n_state]`` (or ``[batch, 1, n_state]``); zeros if None.
            state: carried state ``{"x": x_last}`` from a previous chunk; overrides ``x0``.

        Returns:
            Output sequence ``[batch, seq, n_output]`` observing the states ``x_1 .. x_L``,
            or ``(sequence, {"x": x_L})`` when ``return_state`` is set.
        """
        match state:
            case {"x": x_carry}:
                x0 = x_carry
            case None:
                pass
            case _:
                raise TypeError(f"expected state dict {{'x': tensor}}, got {type(state)}")
        if x0 is None:
            x0 = u.new_zeros(u.shape[0], self.spec.n_state)
        elif x0.dim() == 3:
            x0 = x0.squeeze(1)
        match _rollout_mode(self.backend, self.spec, u, x0):
            case "eager":
                out = self._rollout_eager(u, x0)
            case "compiled":
                out = self._rollout_compiled(u, x0)
            case _:
                out = fused_rollout(self.spec, u, x0, self._params_flat())
        y = self.output_map(out)
        if self.return_state:
            return y, {"x": out[:, -1]}
        return y

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


def rollout_unsupported(spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, device_type: str) -> str | None:
    """Device/dtype/shape screen shared by the fused backends' ``supports``: reason or None."""
    if u.device.type != device_type:
        return f"input on {u.device.type}, this backend requires {device_type}"
    if u.dtype != torch.float32 or x0.dtype != torch.float32:
        return f"requires float32, got u={u.dtype}, x0={x0.dtype}"
    if u.dim() != 3 or u.shape[-1] != spec.n_input:
        return f"expected u of shape [B, L, {spec.n_input}], got {tuple(u.shape)}"
    if x0.shape != (u.shape[0], spec.n_state):
        return f"expected x0 of shape [{u.shape[0]}, {spec.n_state}], got {tuple(x0.shape)}"
    return None


def mlp_param_grads(
    spec: SSMSpec,
    x0: torch.Tensor,
    u: torch.Tensor,
    out: torch.Tensor,
    zs: list[torch.Tensor],
    gy: torch.Tensor,
    gas: list[torch.Tensor],
    w0: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Parameter and input gradients of the rollout as batched GEMMs over the flattened adjoints.

    The state-adjoint recurrence is the only sequential part of BPTT; the parameter
    gradients ``dW_l = sum_{b,t} ga_l ⊗ z_{l-1}`` are plain reductions over all ``B*L``
    step samples, which is exactly the batched GEMM BLAS is built for. Shared by every
    fused backend through the backward op.

    Args:
        x0: initial state ``[B, NX]``.
        u: input sequence ``[B, L, NU]``.
        out: forward result ``[B, L, NX]`` (``out[t] = x_{t+1}``).
        zs: stored post-activation hidden sequences, one ``[B, L, h]`` per hidden layer.
        gy: total adjoint of each step output ``[B, L, NX]``.
        gas: pre-activation adjoints of the hidden layers, one ``[B, L, h]`` per layer.
        w0: first-layer weight ``[dims[1], NX+NU]``, needed for the input gradient.

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
    du = (adjoints[0] @ w0[:, spec.n_state :]).reshape(B, L, spec.n_input)
    return grads, du
