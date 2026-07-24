"""Output-error port-Hamiltonian neural networks (OE-pHNN, Moradi et al. 2026).

Continuous-time model with port-Hamiltonian structure, discretized by one
explicit RK4 step per sample under zero-order hold:

    dx/dt = (J(x) - R(x)) dH/dx(x) + G(x) u,    y = G(x)^T dH/dx(x)

with J skew-symmetric, R positive semidefinite, and H a scalar network, so the
energy balance dH/dt <= y^T u holds for any weights (cyclo-passivity). The
collocated output map forces n_input == n_output; the ``output="linear"``
variant replaces it with a learned linear observation for non-square systems,
trading away the passivity guarantee with respect to the measured output.

Reference: Moradi, Beintema, Jaensson, Tóth & Schoukens, "Port-Hamiltonian
Neural Networks with Output Error Noise Models", Automatica 2026
(arXiv:2502.14432); reference implementation github.com/sarvin90/OE-pHNN
(no license file). Faithful reimplementation: same parametrization, scaling
factors, and integrator. Two implementation differences that leave the
function identical but remove the per-step autograd overhead of the
reference: dH/dx is computed in closed form instead of
``torch.autograd.grad(create_graph=True)``, and the RK4 stage at the current
state shares its network evaluations with the output computation. Numerical
agreement is validated in ``comparisons/compare_phnn.py``.
"""

__all__ = [
    "HamiltonianMLP",
    "PHNNCore",
    "PHNN",
]

import torch
from torch import nn

from ..._core.dispatch import BACKENDS, get_backend, resolve
from ..subnet import ResMLP, SubnetEncoder, _mlp
from .common import PHNNSpec, bound_value, flat_params, spec_of

# Fused rollout backends, resolved through dispatch.resolve: each module exposes
# supports(spec, u, x0) -> str | None plus forward/forward_train/backward entry points
# on the flat_params parameter layout.
_BACKENDS = {
    "triton": "tsfast.models.architectures.phnn.backend_triton",
    "c": "tsfast.models.architectures.phnn.backend_c",
}
_AUTO_ORDER = {"cuda": ("triton",), "cpu": ("c",)}


class HamiltonianMLP(nn.Module):
    """Scalar Hamiltonian network with closed-form gradient.

    A tanh MLP followed by an optional ELU lower bound
    ``H_b = elu(H - (c+1)) + (c+1) >= c`` (cyclo-passivity requires H bounded
    from below; the reference's cascaded-tanks model omits the bound, so it is
    optional here). ``forward`` returns ``(H, dH/dx)`` with the gradient built
    by explicit backpropagation — an expression of the weights that autograd
    can differentiate again for training, equivalent to
    ``torch.autograd.grad(H.sum(), x, create_graph=True)`` but cheaper and
    ``torch.compile``-friendly.
    """

    def __init__(self, n_state: int, hidden_size: int = 64, num_layers: int = 2, lower_bound: float | None = 0.0):
        super().__init__()
        self.net = _mlp(n_state, 1, hidden_size, num_layers)
        self.lower_bound = lower_bound

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hamiltonian ``[B]`` and its state gradient ``[B, n_state]``."""
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        z = x
        hiddens = []
        for lin in linears[:-1]:
            z = torch.tanh(lin(z))
            hiddens.append(z)
        h = linears[-1](z)[:, 0]

        g = linears[-1].weight[0].expand_as(z)
        for lin, z_i in zip(reversed(linears[:-1]), reversed(hiddens)):
            g = (g * (1.0 - z_i * z_i)) @ lin.weight

        if self.lower_bound is None:
            return h, g
        b = self.lower_bound + 1.0
        s = h - b
        h = torch.nn.functional.elu(s) + b
        # d elu(s)/ds is exp(s) for s < 0 and 1 otherwise.
        return h, g * torch.where(s > 0, torch.ones_like(s), s.exp()).unsqueeze(-1)


class PHNNCore(nn.Module):
    """One-sample pH step: output at the current state, RK4 state update.

    ``step(x, u) -> (y, x_next)`` matching the reference ``hf_net_pHNN.forward``
    exactly: ``y_k`` observes ``x_k`` before the update, the input is held
    constant over the RK4 stages (ZOH), and ``dt`` scales the vector field.

    Parametrization (reference defaults): J and R are built from plain-MLP
    matrix nets ``B(x)`` with scale ``((2+n)n)^-0.25`` as ``J = B - B^T`` and
    ``R = A A^T``; G is a linear + MLP residual net with scale ``nu^-0.5``.

    Args:
        n_state: state dimension.
        n_input: input dimension.
        n_output: output dimension (must equal ``n_input`` for ``output="ph"``).
        hidden_size: hidden width of all component nets.
        num_layers: hidden layers of all component nets.
        dt: integrator step size in the model's time unit. The reference scales
            time so that ``dt`` is O(0.1) (e.g. 0.04 instead of the true 4 s for
            cascaded tanks); treat it as a tunable time-normalization constant.
        rk4_steps: RK4 substeps per sample.
        h_lower_bound: ELU lower bound of the Hamiltonian, or None to disable.
        output: ``"ph"`` for the collocated map ``G^T dH/dx``; ``"linear"`` for
            a learned ``nn.Linear`` observation (non-square systems, forfeits
            the output passivity structure).
    """

    def __init__(
        self,
        n_state: int,
        n_input: int,
        n_output: int | None = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dt: float = 0.1,
        rk4_steps: int = 1,
        h_lower_bound: float | None = 0.0,
        output: str = "ph",
    ):
        super().__init__()
        n_output = n_input if n_output is None else n_output
        if output not in ("ph", "linear"):
            raise ValueError(f"output must be 'ph' or 'linear', got {output!r}")
        if output == "ph" and n_input != n_output:
            raise ValueError(
                f"the collocated pH output map requires n_input == n_output, got {n_input} != {n_output}; "
                "use output='linear' for non-square systems"
            )
        self.n_state, self.n_input, self.n_output = n_state, n_input, n_output
        self.dt, self.rk4_steps, self.output = dt, rk4_steps, output
        self.jr_scale = ((2.0 + n_state) * n_state) ** -0.25
        self.g_scale = n_input**-0.5

        self.hamiltonian = HamiltonianMLP(n_state, hidden_size, num_layers, h_lower_bound)
        self.j_net = _mlp(n_state, n_state * n_state, hidden_size, num_layers)
        self.r_net = _mlp(n_state, n_state * n_state, hidden_size, num_layers)
        self.g_net = ResMLP(n_state, n_state * n_input, hidden_size, num_layers)
        self.output_map = nn.Linear(n_state, n_output) if output == "linear" else None

    def _fields(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate all component nets at ``x``: returns ``(G, dH/dx, (J-R) dH/dx)``.

        One fused evaluation shared between the output map and the first RK4
        stage; the reference evaluates the nets once for the output and once
        more inside the first stage at the identical state.
        """
        n = self.n_state
        _, dhdx = self.hamiltonian(x)
        b = self.j_net(x).view(-1, n, n) * self.jr_scale
        a = self.r_net(x).view(-1, n, n) * self.jr_scale
        jr = b - b.transpose(1, 2) - a @ a.transpose(1, 2)
        g = self.g_net(x).view(-1, n, self.n_input) * self.g_scale
        drift = (jr @ dhdx.unsqueeze(-1)).squeeze(-1)
        return g, dhdx, drift

    def _rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        g, _, drift = self._fields(x)
        return drift + (g @ u.unsqueeze(-1)).squeeze(-1)

    def step(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Output ``y_k [B, n_output]`` at ``x_k`` and the next state ``x_{k+1} [B, n_state]``."""
        g, dhdx, drift = self._fields(x)
        if self.output_map is not None:
            y = self.output_map(x)
        else:
            y = (g.transpose(1, 2) @ dhdx.unsqueeze(-1)).squeeze(-1)

        h = self.dt / self.rk4_steps
        gu = (g @ u.unsqueeze(-1)).squeeze(-1)
        k1 = h * (drift + gu)  # stage 1 reuses the output evaluation
        for i in range(self.rk4_steps):
            if i > 0:
                k1 = h * self._rhs(x, u)
            k2 = h * self._rhs(x + k1 / 2, u)
            k3 = h * self._rhs(x + k2 / 2, u)
            k4 = h * self._rhs(x + k3, u)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y, x


class PHNN(nn.Module):
    """OE-pHNN sequence model: SUBNET encoder + pH rollout over ``[u, y]`` input channels.

    Same input contract as :class:`~tsfast.models.architectures.subnet.SubnetSSM`: the input
    tensor carries ``n_input`` input channels then ``n_output`` measured-output
    channels, of which only the first ``n_init`` steps are read (encoder
    warm-up). Predictions start at ``n_init``; earlier positions are zero and
    must be excluded from the loss via ``n_skip=n_init``.

    Backends: ``"eager"`` is a plain Python loop; ``"compiled"`` keeps the loop
    but routes each transition through a ``torch.compile``d ``core.step`` — the
    traced graph covers a single step, so graph size and compile time are
    independent of sequence length (compiling the whole rollout unrolled every
    step into one graph, which exhausted memory on long free runs); ``"c"`` and
    ``"triton"`` fuse the whole section rollout and its BPTT into one call (batch-
    parallel C++ on CPU, a persistent per-lane kernel on CUDA — see
    :mod:`tsfast.models.architectures.phnn`); ``"reference"`` forces the non-fused
    policy (``compiled`` on CUDA, ``eager`` on CPU); ``"auto"`` defers to the process
    preference (:func:`tsfast.models.set_backend` / ``use_backend``), under which it
    picks the fused kernel for the input's device when usable, else the non-fused
    policy. An explicit fused backend that cannot run warns once per process and
    falls back the same way.

    Args:
        n_input: exogenous input dimension.
        n_output: observed output dimension.
        n_state: state dimension.
        hidden_size: hidden width of all pH component nets.
        num_layers: hidden layers of all pH component nets.
        dt: RK4 step size (time-normalization constant, see :class:`PHNNCore`).
        n_init: encoder warm-up length.
        na: encoder output-history length (defaults to ``n_init``).
        nb: encoder input-history length (defaults to ``n_init``).
        enc_hidden_size: encoder MLP hidden width.
        enc_num_layers: encoder MLP hidden layers.
        rk4_steps: RK4 substeps per sample.
        h_lower_bound: ELU lower bound of the Hamiltonian, or None to disable.
        output: ``"ph"`` or ``"linear"``, see :class:`PHNNCore`.
        backend: ``"eager"``, ``"compiled"``, or ``"auto"``.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_state: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dt: float = 0.1,
        n_init: int = 50,
        na: int | None = None,
        nb: int | None = None,
        enc_hidden_size: int = 64,
        enc_num_layers: int = 2,
        rk4_steps: int = 1,
        h_lower_bound: float | None = 0.0,
        output: str = "ph",
        backend: str = "auto",
    ):
        super().__init__()
        na = n_init if na is None else na
        nb = n_init if nb is None else nb
        if max(na, nb) > n_init:
            raise ValueError(f"encoder windows na={na}, nb={nb} cannot exceed n_init={n_init}")
        self.n_input, self.n_output, self.n_init = n_input, n_output, n_init
        self.backend = backend
        self.core = PHNNCore(n_state, n_input, n_output, hidden_size, num_layers, dt, rk4_steps, h_lower_bound, output)
        self.encoder = SubnetEncoder(n_input, n_output, n_state, na, nb, enc_hidden_size, enc_num_layers)
        self._compiled_step = None

    @property
    def dt(self) -> float:
        return self.core.dt

    @dt.setter
    def dt(self, value: float):
        self.core.dt = value

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder state from the ``[u, y]`` input tensor at position ``n_init``."""
        u, y = x[..., : self.n_input], x[..., self.n_input :]
        n0 = self.n_init
        return self.encoder(u[:, n0 - self.encoder.nb : n0], y[:, n0 - self.encoder.na : n0])

    def _rollout(self, u_future: torch.Tensor, x0: torch.Tensor, step=None) -> torch.Tensor:
        step = self.core.step if step is None else step
        x = x0
        outs = []
        for t in range(u_future.shape[1]):
            y, x = step(x, u_future[:, t])
            outs.append(y)
        return torch.stack(outs, dim=1)

    def _rollout_compiled(self, u_future: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        if self._compiled_step is None:
            # The bump covers per-batch-shape recompiles (e.g. the last, smaller
            # batch of an epoch); graph size no longer depends on sequence length.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._compiled_step = torch.compile(self.core.step, dynamic=False)
        return self._rollout(u_future, x0, step=self._compiled_step)

    def _route(self, u_future: torch.Tensor) -> str:
        """Concrete execution route for this call: ``"eager"``, ``"compiled"``, or ``"fused"``."""
        spec = spec_of(self.core)
        fields = (
            spec.n_state,
            spec.n_input,
            spec.n_output,
            spec.hidden,
            spec.num_layers,
            spec.rk4_steps,
            spec.output,
            spec.has_bound,
        )
        return _route_for(
            self.backend, fields, u_future.device.type, u_future.dtype, u_future.dim(), u_future.shape[-1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate from the encoder state; returns ``[B, L, n_output]`` with zeros before ``n_init``."""
        if x.shape[1] <= self.n_init:
            raise ValueError(f"sequence length {x.shape[1]} too short for encoder warm-up n_init={self.n_init}")
        x0 = self.encode(x)
        u_future = x[:, self.n_init :, : self.n_input]
        match self._route(u_future):
            case "eager":
                out = self._rollout(u_future, x0)
            case "compiled":
                out = self._rollout_compiled(u_future, x0)
            case _:
                out = fused_rollout(self.core, u_future, x0)
        warmup = x.new_zeros(x.shape[0], self.n_init, self.n_output)
        return torch.cat((warmup, out), dim=1)


@torch._dynamo.assume_constant_result
def _route_for(backend: str, spec_fields: tuple, device_type: str, dtype: torch.dtype, u_dim: int, u_last: int) -> str:
    """Map the instance ``backend`` to ``"eager"``, ``"compiled"``, or ``"fused"``.

    ``"auto"`` defers to the process preference; ``"reference"`` and any fused family
    that declines (via the once-warned ``dispatch.resolve``) map to the non-fused
    policy: ``compiled`` on CUDA, ``eager`` on CPU. Takes only hashables so
    ``torch.compile`` can fold the decision into a trace-time constant — the import
    and availability probing inside ``resolve`` never enter the graph.
    """
    requested = get_backend() if backend == "auto" else backend
    if requested in ("eager", "compiled"):
        return requested
    if requested not in BACKENDS:
        raise ValueError(f"unknown backend {requested!r}, expected 'eager', 'compiled', or one of {BACKENDS}")
    fallback = "compiled" if device_type == "cuda" else "eager"
    if requested == "reference":
        return fallback
    spec = PHNNSpec(*spec_fields)
    probe = torch.empty(*([0] * (u_dim - 1)), u_last, dtype=dtype, device=device_type)
    mod = resolve("phnn.rollout", _BACKENDS, _AUTO_ORDER.get(device_type, ()), (spec, probe, None), requested=requested)
    return "fused" if mod is not None else fallback


# ------------------------------------------------------------------------- custom ops
#
# The fused rollout is exposed as forward/backward custom-op pairs so it composes with
# torch.compile (no graph breaks), fake/meta tracing, and export. The frozen PHNNSpec
# cannot cross the op boundary, so its fields travel as scalars and are rebuilt inside
# the impls; the component-net parameters travel as one flat list in flat_params order
# and receive gradients through the registered autograd bridge. Inside the op, dispatch
# picks the fused backend for the input's device (triton on CUDA, c on CPU); the
# non-fused eager/compiled paths stay outside the ops so torch.compile can trace them.


def _resolve_fused(spec: PHNNSpec, u: torch.Tensor, x0: torch.Tensor | None):
    mod = resolve("phnn.rollout", _BACKENDS, _AUTO_ORDER.get(u.device.type, ()), (spec, u, x0), requested="auto")
    if mod is None:
        raise RuntimeError(f"no fused PHNN backend usable for device {u.device.type} / dtype {u.dtype} / spec {spec}")
    return mod


@torch.library.custom_op("tsfast::phnn_rollout", mutates_args=())
def _rollout_op(
    u: torch.Tensor,
    x0: torch.Tensor,
    params: list[torch.Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    hidden: int,
    num_layers: int,
    rk4_steps: int,
    output: str,
    has_bound: bool,
    dt: float,
    bound: float,
    jr_scale: float,
    g_scale: float,
) -> torch.Tensor:
    spec = PHNNSpec(n_state, n_input, n_output, hidden, num_layers, rk4_steps, output, has_bound)
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    mod = _resolve_fused(spec, u, x0)
    return mod.forward(u, x0, params, spec, dt, bound, jr_scale, g_scale)


@_rollout_op.register_fake
def _(
    u,
    x0,
    params,
    n_state,
    n_input,
    n_output,
    hidden,
    num_layers,
    rk4_steps,
    output,
    has_bound,
    dt,
    bound,
    jr_scale,
    g_scale,
):
    return u.new_empty(u.shape[0], u.shape[1], n_output)


@torch.library.custom_op("tsfast::phnn_rollout_train", mutates_args=())
def _rollout_train_op(
    u: torch.Tensor,
    x0: torch.Tensor,
    params: list[torch.Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    hidden: int,
    num_layers: int,
    rk4_steps: int,
    output: str,
    has_bound: bool,
    dt: float,
    bound: float,
    jr_scale: float,
    g_scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    spec = PHNNSpec(n_state, n_input, n_output, hidden, num_layers, rk4_steps, output, has_bound)
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    mod = _resolve_fused(spec, u, x0)
    return mod.forward_train(u, x0, params, spec, dt, bound, jr_scale, g_scale)


@_rollout_train_op.register_fake
def _(
    u,
    x0,
    params,
    n_state,
    n_input,
    n_output,
    hidden,
    num_layers,
    rk4_steps,
    output,
    has_bound,
    dt,
    bound,
    jr_scale,
    g_scale,
):
    spec = PHNNSpec(n_state, n_input, n_output, hidden, num_layers, rk4_steps, output, has_bound)
    if u.device.type == "cuda":
        from .backend_triton import fake_saved
    else:
        from .backend_c import fake_saved
    return u.new_empty(u.shape[0], u.shape[1], n_output), fake_saved(u, spec)


@torch.library.custom_op("tsfast::phnn_rollout_bwd", mutates_args=())
def _rollout_bwd_op(
    grad_out: torch.Tensor,
    u: torch.Tensor,
    saved: list[torch.Tensor],
    params: list[torch.Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    hidden: int,
    num_layers: int,
    rk4_steps: int,
    output: str,
    has_bound: bool,
    dt: float,
    bound: float,
    jr_scale: float,
    g_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    spec = PHNNSpec(n_state, n_input, n_output, hidden, num_layers, rk4_steps, output, has_bound)
    grad_out, u = grad_out.contiguous(), u.contiguous()
    saved = [t.contiguous() for t in saved]
    params = [p.contiguous() for p in params]
    mod = _resolve_fused(spec, u, None)
    return mod.backward(grad_out, u, saved, params, spec, dt, bound, jr_scale, g_scale)


@_rollout_bwd_op.register_fake
def _(
    grad_out,
    u,
    saved,
    params,
    n_state,
    n_input,
    n_output,
    hidden,
    num_layers,
    rk4_steps,
    output,
    has_bound,
    dt,
    bound,
    jr_scale,
    g_scale,
):
    return torch.empty_like(u), u.new_empty(u.shape[0], n_state), [torch.empty_like(p) for p in params]


def _train_setup(ctx, inputs, output):
    u, x0, params, *scalars = inputs
    _, saved = output
    ctx.n_params = len(params)
    ctx.scalars = scalars
    ctx.save_for_backward(u, *params, *saved)


def _train_backward(ctx, grad_out, grad_saved):
    u, *rest = ctx.saved_tensors
    params = list(rest[: ctx.n_params])
    saved = list(rest[ctx.n_params :])
    du, gx0, gparams = _rollout_bwd_op(grad_out, u, saved, params, *ctx.scalars)
    return (du, gx0, gparams, *([None] * len(ctx.scalars)))


_rollout_train_op.register_autograd(_train_backward, setup_context=_train_setup)


def fused_rollout(core: PHNNCore, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Run the section rollout through the fused custom ops (autograd-capable).

    ``u`` is the future input ``[B, L, n_input]`` and ``x0`` the encoder state
    ``[B, n_state]``; returns the output sequence ``[B, L, n_output]`` for the L future
    steps (the encoder warm-up is prepended by the caller). Picks the intermediate-storing
    train op only when gradients can flow; raises RuntimeError when no fused backend is
    usable for the input's device.
    """
    spec = spec_of(core)
    params = flat_params(core)
    args = (
        u,
        x0,
        params,
        spec.n_state,
        spec.n_input,
        spec.n_output,
        spec.hidden,
        spec.num_layers,
        spec.rk4_steps,
        spec.output,
        spec.has_bound,
        float(core.dt),
        bound_value(core),
        float(core.jr_scale),
        float(core.g_scale),
    )
    if torch.is_grad_enabled() and any(t.requires_grad for t in (u, x0, *params)):
        return _rollout_train_op(*args)[0]
    return _rollout_op(*args)
