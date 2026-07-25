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

# Gates whose pre-activation is emitted by the transition MLP itself, doubling the width of
# its final layer. "leak" instead carries a standalone per-state parameter, and "none"
# leaves the transition ungated.
_INPUT_GATES = ("gru", "residual")
_GATES = ("none", "leak", *_INPUT_GATES)


@dataclass(frozen=True)
class SSMSpec:
    """Static architecture description of the transition MLP, used to generate backend kernels.

    Args:
        n_state: state dimension
        n_input: exogenous input dimension
        hidden: widths of the hidden layers (may be empty for a linear state space model)
        act: activation name, a key of ``_ACTS``
        gate: gating mode of the state update, one of ``_GATES``
        eps: step size of the ``residual`` gate, ignored by the other modes
    """

    n_state: int
    n_input: int
    hidden: tuple[int, ...]
    act: str
    gate: str = "none"
    eps: float = 1.0

    @property
    def out_width(self) -> int:
        """Final linear width: the candidate state, plus a gate pre-activation when gated."""
        return self.n_state * (2 if self.gate in _INPUT_GATES else 1)

    @property
    def dims(self) -> tuple[int, ...]:
        """Feature sizes through the MLP: ``(n_state + n_input, *hidden, out_width)``."""
        return (self.n_state + self.n_input, *self.hidden, self.out_width)

    @property
    def n_linear(self) -> int:
        return len(self.hidden) + 1


def gate_step(spec: SSMSpec, x: Tensor, y: Tensor, a: Tensor | None) -> Tensor:
    """Combine the current state with the transition MLP output into the next state.

    The ``leak`` and ``gru`` modes interpolate rather than overwrite, so ``1 - z`` is the
    per-channel discrete-time pole and the state Jacobian gains a ``diag(1 - z)`` path that
    BPTT can carry over long rollouts. ``residual`` instead adds a gated increment, leaving
    an exactly unit carry path.

    Args:
        x: current state ``[B, n_state]``.
        y: final linear output ``[B, out_width]``.
        a: leak ``sigmoid(lambda)`` of shape ``[n_state]`` for ``gate="leak"``, else None.
    """
    nx = spec.n_state
    match spec.gate:
        case "none":
            return y
        case "leak":
            return x + a * (y - x)
        case "gru":
            return x + torch.sigmoid(y[..., nx:]) * (y[..., :nx] - x)
        case "residual":
            return x + spec.eps * torch.sigmoid(y[..., nx:]) * y[..., :nx]
        case unknown:
            raise ValueError(f"unknown gate {unknown!r}")


def chrono_bias(n_state: int, tmax: float) -> Tensor:
    """Update-gate bias placing the initial time constants uniformly in ``[1, tmax - 1]``.

    Following Tallec & Ollivier (arXiv:1804.11188), whose retention gate starts at
    ``T / (T + 1)``; the update gate is its complement, so the bias is ``-log(T)``.
    """
    if tmax <= 2.0:
        raise ValueError(f"gate_tmax must exceed 2, got {tmax}")
    return -torch.log(torch.rand(n_state) * (tmax - 2.0) + 1.0)


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


def _spec_from(n_state: int, n_input: int, hidden: list[int], act: str, gate: str, eps: float) -> SSMSpec:
    return SSMSpec(n_state, n_input, tuple(hidden), act, gate, eps)


def saved_widths(spec: SSMSpec) -> tuple[int, ...]:
    """Per-step widths the training forward stores for the backward, in order.

    The hidden post-activations, then for a gated transition the two gate tensors: the gate
    pre-activation ``s``, and whichever vector the gate scales by ``z`` — the candidate
    offset ``d = c - x`` for the lerp gates, the candidate ``c`` itself for ``residual``.
    Storing that vector rather than the candidate unconditionally is what keeps the reverse
    sweep from needing ``out``/``x0``, and it costs the forward nothing: each gate's update
    forms its own vector anyway. ``leak`` stores only that vector — its gate is a parameter,
    so there is no per-step pre-activation to keep.
    """
    match spec.gate:
        case "none":
            return spec.hidden
        case "leak":
            return (*spec.hidden, spec.n_state)
        case _:
            return (*spec.hidden, spec.n_state, spec.n_state)


@torch.library.custom_op("tsfast::ssm_rollout", mutates_args=())
def _ssm_rollout(
    u: Tensor,
    x0: Tensor,
    params: list[Tensor],
    leak: Tensor | None,
    n_state: int,
    n_input: int,
    hidden: list[int],
    act: str,
    gate: str,
    eps: float,
) -> Tensor:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, hidden, act, gate, eps)
    return _fused_module(spec, u, x0).forward_infer(spec, u, x0, params, leak)


@_ssm_rollout.register_fake
def _(u, x0, params, leak, n_state, n_input, hidden, act, gate, eps):
    return u.new_empty(u.shape[0], u.shape[1], n_state)


@torch.library.custom_op("tsfast::ssm_rollout_train", mutates_args=())
def _ssm_rollout_train(
    u: Tensor,
    x0: Tensor,
    params: list[Tensor],
    leak: Tensor | None,
    n_state: int,
    n_input: int,
    hidden: list[int],
    act: str,
    gate: str,
    eps: float,
) -> tuple[Tensor, list[Tensor]]:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, hidden, act, gate, eps)
    out, zs = _fused_module(spec, u, x0).forward_train(spec, u, x0, params, leak)
    return out, zs


@_ssm_rollout_train.register_fake
def _(u, x0, params, leak, n_state, n_input, hidden, act, gate, eps):
    B, L = u.shape[0], u.shape[1]
    spec = _spec_from(n_state, n_input, hidden, act, gate, eps)
    return u.new_empty(B, L, n_state), [u.new_empty(B, L, w) for w in saved_widths(spec)]


@torch.library.custom_op("tsfast::ssm_rollout_bwd", mutates_args=())
def _ssm_rollout_bwd(
    grad_out: Tensor | None,
    u: Tensor,
    x0: Tensor,
    out: Tensor,
    zs: list[Tensor],
    params: list[Tensor],
    leak: Tensor | None,
    n_state: int,
    n_input: int,
    hidden: list[int],
    act: str,
    gate: str,
    eps: float,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
    u, x0, out = u.contiguous(), x0.contiguous(), out.contiguous()
    g = grad_out.contiguous() if grad_out is not None else torch.zeros_like(out)
    zs = [z.contiguous() for z in zs]
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, hidden, act, gate, eps)
    weights = params[0::2]
    gy, gas, gx0, gleak = _fused_module(spec, u, x0).backward(spec, g, zs, weights, leak)
    grads, du = mlp_param_grads(spec, x0, u, out, zs[: len(hidden)], gy, gas, w0=weights[0])
    return du, gx0, grads, gleak if gleak is not None else u.new_empty(0)


@_ssm_rollout_bwd.register_fake
def _(grad_out, u, x0, out, zs, params, leak, n_state, n_input, hidden, act, gate, eps):
    dleak = torch.empty_like(leak) if leak is not None else u.new_empty(0)
    return torch.empty_like(u), torch.empty_like(x0), [torch.empty_like(p) for p in params], dleak


def _train_setup(ctx, inputs, output):
    u, x0, params, leak, n_state, n_input, hidden, act, gate, eps = inputs
    out, zs = output
    ctx.fields = (n_state, n_input, list(hidden), act, gate, eps)
    ctx.n_saved = len(zs)
    ctx.has_leak = leak is not None
    ctx.save_for_backward(u, x0, out, *zs, *params, *([leak] if leak is not None else []))


def _train_backward(ctx, grad_out, grad_zs):
    n_state, n_input, hidden, act, gate, eps = ctx.fields
    saved = ctx.saved_tensors
    u, x0, out = saved[0], saved[1], saved[2]
    zs = list(saved[3 : 3 + ctx.n_saved])
    rest = list(saved[3 + ctx.n_saved :])
    params, leak = (rest[:-1], rest[-1]) if ctx.has_leak else (rest, None)
    du, dx0, dparams, dleak = _ssm_rollout_bwd(
        grad_out, u, x0, out, zs, params, leak, n_state, n_input, hidden, act, gate, eps
    )
    return (
        du,
        dx0,
        list(dparams),
        dleak if ctx.has_leak else None,
        None,
        None,
        ([] if not hidden else None),
        None,
        None,
        None,
    )


_ssm_rollout_train.register_autograd(_train_backward, setup_context=_train_setup)


def fused_rollout(spec: SSMSpec, u: Tensor, x0: Tensor, params: list[Tensor], leak: Tensor | None = None) -> Tensor:
    """Run the state rollout through the fused-kernel custom ops (autograd-capable).

    Picks the training op (stores the hidden activations for the analytic BPTT backward)
    when gradients are live, else the inference op, which keeps no intermediates.

    Args:
        spec: transition-MLP architecture.
        u: input sequence ``[B, L, n_input]``, float32 on the fused backend's device.
        x0: initial state ``[B, n_state]``, float32.
        params: transition parameters ``[W_0, b_0, W_1, b_1, ...]`` in layer order.
        leak: the ``gate="leak"`` rate ``sigmoid(lambda)`` as ``[n_state]``, else None. It
            crosses the boundary already squashed, so autograd differentiates the sigmoid
            outside the op and the kernels only accumulate ``dL/da``.

    Returns:
        States ``x_1 .. x_L`` as ``[B, L, n_state]``.
    """
    fields = (spec.n_state, spec.n_input, list(spec.hidden), spec.act, spec.gate, spec.eps)
    live = (u, x0, *params) if leak is None else (u, x0, *params, leak)
    if torch.is_grad_enabled() and any(t.requires_grad for t in live):
        return _ssm_rollout_train(u, x0, list(params), leak, *fields)[0]
    return _ssm_rollout(u, x0, list(params), leak, *fields)


class NeuralStateSpace(nn.Module):
    """Discrete-time neural state space model with an MLP transition and a linear observation.

    ``x_{k+1} = f(x_k, u_k)``, ``y_k = C x_k + d``. The latent state dimension is independent
    of the output dimension, so the model can carry more internal dynamics (e.g. velocities,
    phases) than the measured signal exposes — ``n_state`` must be at least the order of the
    system being identified.

    The transition Jacobian of the ungated model is a product of the MLP's dense weights, so
    over a long rollout BPTT multiplies ``L`` unconstrained matrices and the gradient that
    reaches ``x_0`` typically vanishes. ``gate`` interpolates instead of overwriting, which
    puts a tunable near-identity path in the Jacobian — with ``c = f(x_k, u_k)`` the candidate
    state and ``z`` the gate:

    - ``"none"``: ``x_{k+1} = c``.
    - ``"leak"``: ``x_{k+1} = x_k + a (c - x_k)`` with ``a = sigmoid(lambda)`` a learned
      per-state parameter. ``1 - a`` is the channel's discrete-time pole, so this is a
      learned time constant and nothing more.
    - ``"gru"``: as ``leak``, but ``z`` is emitted per step by the transition MLP alongside
      the candidate, so the time constant adapts to the operating point.
    - ``"residual"``: ``x_{k+1} = x_k + eps * z * c``, an unbounded gated increment whose
      carry path is exactly the identity.

    The gated modes are chrono-initialized (``gate_tmax``), spreading the initial time
    constants over ``[1, gate_tmax - 1]`` steps.

    Prefer ``"gru"`` when reaching for a gate. It is the only mode whose gradient horizon
    actually tracks ``gate_tmax`` — the fitted time constant stays within 22% of it out to
    ``gate_tmax=300``, against 41% for ``"leak"`` — and it subsumes ``"leak"``, which is what
    it reduces to when the gate weight rows are zero. ``"residual"`` keeps an adjoint that
    never decays, so its memory horizon is not a parameter of the model at all, which is the
    wrong prior for a dissipative plant. The three are not separable on fit quality:
    ``benchmarks/gate_ssm_gating.py`` measures both criteria and reproduces that result.

    The rollout over the input sequence is irreducibly sequential, so a naive per-step Python
    loop is dispatch-bound rather than FLOP-bound. Several backends implement the identical
    state rollout (the observation map is a single batched matmul applied on top and works
    with every backend unchanged):

    - ``"eager"``: plain Python loop — the reference implementation, any device and dtype.
    - ``"compiled"``: ``torch.compile`` over the unrolled loop — any device, slow first call.
    - ``"c"``: generated C++ rollout with a fused BPTT backward, batch-parallel via the
      ATen thread pool — float32 on CPU, fastest CPU option for small models. Serves every
      gate.
    - ``"triton"``: persistent-GEMV rollout kernel with a fused BPTT backward — float32 on
      CUDA, hidden widths up to 128. Serves every gate.
    - ``"metal"``: persistent register-resident rollout kernel with a fused BPTT backward —
      float32 on MPS (Apple GPUs), layer widths up to 128. Ungated transitions only.
    - ``"auto"``: defers to the process-wide preference (``tsfast.models.set_backend`` /
      ``use_backend``); under an ``"auto"`` preference picks ``triton`` when it applies,
      else ``compiled`` on CUDA; ``metal`` when it applies on MPS; ``eager`` on CPU
      (select ``"c"`` explicitly to trade a one-time compilation for much faster CPU
      training). A ``"reference"`` preference forces the non-fused path everywhere.

    An explicit backend that cannot handle the inputs warns once per process and falls
    back to the non-fused path instead of raising. The ``c`` and ``triton`` kernels cover
    every gate; ``metal`` serves the ungated transition only, so a gated model on MPS
    falls back to the reference path.

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
        gate: gating mode of the state update, one of ``none``, ``leak``, ``gru``,
            ``residual``.
        gate_tmax: longest initial time constant, in steps, of the chrono initialization.
        eps: step size of the ``residual`` gate.
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
        gate: str = "none",
        gate_tmax: float = 100.0,
        eps: float = 1.0,
        backend: str = "auto",
        return_state: bool = False,
    ):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"unknown activation {act!r}, expected one of {sorted(_ACTS)}")
        if gate not in _GATES:
            raise ValueError(f"unknown gate {gate!r}, expected one of {sorted(_GATES)}")
        hidden = tuple(hidden_size) if isinstance(hidden_size, (list, tuple)) else (hidden_size,) * num_layers
        # Only "residual" reads eps, and the spec keys the compiled-kernel caches: pinning it
        # elsewhere keeps two numerically identical models from compiling separate kernels.
        self.spec = SSMSpec(n_state, n_input, hidden, act, gate, eps if gate == "residual" else 1.0)
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

        # Chrono init. The input-dependent gates start state-independent — zeroed weight rows
        # leave sigmoid(bias) as the whole gate — so they and "leak" open on identical
        # dynamics and differ only in what training makes of the extra inputs.
        if gate == "leak":
            self.leak_logit = nn.Parameter(chrono_bias(n_state, gate_tmax))
        elif gate in _INPUT_GATES:
            last = self.linears[-1]
            with torch.no_grad():
                last.weight[n_state:].zero_()
                last.bias[n_state:] = chrono_bias(n_state, gate_tmax)

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
                leak = torch.sigmoid(self.leak_logit) if self.spec.gate == "leak" else None
                out = fused_rollout(self.spec, u, x0, self._params_flat(), leak)
        y = self.output_map(out)
        if self.return_state:
            return y, {"x": out[:, -1]}
        return y

    def _rollout_eager(self, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.leak_logit) if self.spec.gate == "leak" else None
        x = x0
        outs = []
        for t in range(u.shape[1]):
            x = gate_step(self.spec, x, self.net(torch.cat((x, u[:, t]), dim=1)), a)
            outs.append(x)
        return torch.stack(outs, dim=1)

    def _rollout_compiled(self, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        if self._compiled_rollout is None:
            # The unrolled sequence is one large graph; raise the recompile budget so shape
            # changes (batch/seq) do not silently fall back to eager.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._compiled_rollout = torch.compile(self._rollout_eager, dynamic=False)
        return self._compiled_rollout(u, x0)


def rollout_unsupported(
    spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, device_type: str, gates: tuple[str, ...]
) -> str | None:
    """Device/dtype/shape screen shared by the fused backends' ``supports``: reason or None.

    ``gates`` is the set of ``SSMSpec.gate`` values the calling backend's generator actually
    emits. It is explicit per backend rather than defaulted: a backend that silently inherits
    a wider set would accept a spec whose final layer is twice as wide as its kernel expects
    and return wrong results instead of declining.
    """
    if spec.gate not in gates:
        return f"gate={spec.gate!r} has no {device_type} kernel"
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
        gy: adjoint of the final linear's output, ``[B, L, out_width]`` — the state adjoint
            itself when ungated, and ``[gc; gs]`` when the final layer also emits a gate.
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
    adjoints = [ga.reshape(BL, ga.shape[-1]) for ga in gas] + [gy.reshape(BL, spec.out_width)]
    grads: list[torch.Tensor] = []
    for a_prev, ga in zip(acts, adjoints):
        grads.append(ga.t() @ a_prev)
        grads.append(ga.sum(0))
    du = (adjoints[0] @ w0[:, spec.n_state :]).reshape(B, L, spec.n_input)
    return grads, du
