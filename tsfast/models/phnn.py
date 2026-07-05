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

from .subnet import ResMLP, SubnetEncoder, _mlp


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

    Same input contract as :class:`~tsfast.models.subnet.SubnetSSM`: the input
    tensor carries ``n_input`` input channels then ``n_output`` measured-output
    channels, of which only the first ``n_init`` steps are read (encoder
    warm-up). Predictions start at ``n_init``; earlier positions are zero and
    must be excluded from the loss via ``n_skip=n_init``.

    Backends: ``"eager"`` is a plain Python loop; ``"compiled"`` keeps the loop
    but routes each transition through a ``torch.compile``d ``core.step`` — the
    traced graph covers a single step, so graph size and compile time are
    independent of sequence length (compiling the whole rollout unrolled every
    step into one graph, which exhausted memory on long free runs); ``"auto"``
    picks ``compiled`` on CUDA and ``eager`` otherwise.

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

    def _resolve_backend(self, u_future: torch.Tensor) -> str:
        """Map ``"auto"`` to a concrete backend and downgrade unavailable fused backends.

        ``auto`` picks the fused ``triton`` kernel on CUDA and the fused ``c`` kernel on
        CPU when they apply (float32/float64, config within caps), otherwise ``compiled``
        on CUDA and ``eager`` on CPU. An explicit fused backend that does not apply falls
        back the same way with a once-per-process warning.
        """
        from . import backends as _b
        from .phnn_backends import spec_of, supports

        backend = self.backend
        spec = spec_of(self.core)
        if backend in ("auto", "triton") and u_future.is_cuda:
            from .phnn_backends import backend_triton

            if u_future.dtype == torch.float32 and backend_triton.is_available() and supports(spec, "triton"):
                return "triton"
            if backend == "triton":
                _b.warn_fallback("phnn.triton", "PHNN triton backend unavailable for this config; using compiled")
            return "compiled"
        if backend in ("auto", "c") and not u_future.is_cuda:
            from .phnn_backends import backend_c

            if u_future.dtype in (torch.float32, torch.float64) and backend_c.is_available() and supports(spec, "c"):
                return "c"
            if backend == "c":
                _b.warn_fallback("phnn.c", "PHNN c backend unavailable for this config; using eager")
            return "eager"
        if backend == "auto":
            return "compiled" if u_future.is_cuda else "eager"
        if backend == "triton":  # requested on CPU
            _b.warn_fallback("phnn.triton", "PHNN triton backend requires CUDA; using eager")
            return "eager"
        if backend == "c":  # requested on CUDA
            _b.warn_fallback("phnn.c", "PHNN c backend requires CPU; using compiled")
            return "compiled"
        return backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate from the encoder state; returns ``[B, L, n_output]`` with zeros before ``n_init``."""
        if x.shape[1] <= self.n_init:
            raise ValueError(f"sequence length {x.shape[1]} too short for encoder warm-up n_init={self.n_init}")
        x0 = self.encode(x)
        u_future = x[:, self.n_init :, : self.n_input]
        from .phnn_backends import spec_of

        match self._resolve_backend(u_future):
            case "eager":
                out = self._rollout(u_future, x0)
            case "compiled":
                out = self._rollout_compiled(u_future, x0)
            case "c":
                from .phnn_backends.backend_c import c_rollout

                out = c_rollout(self.core, spec_of(self.core), u_future.contiguous(), x0.contiguous())
            case "triton":
                from .phnn_backends.backend_triton import triton_rollout

                out = triton_rollout(self.core, spec_of(self.core), u_future.contiguous(), x0.contiguous())
            case unknown:
                raise ValueError(f"unknown backend {unknown!r}")
        warmup = x.new_zeros(x.shape[0], self.n_init, self.n_output)
        return torch.cat((warmup, out), dim=1)
