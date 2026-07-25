"""Robust recurrent deep networks (R2DN, Barbara, Wang & Manchester 2025).

A sequence model with the REN's certificates and none of its equilibrium layer. The
structure is the same Lur'e interconnection — a linear system wrapped around a static
nonlinearity — but the feedthrough from the nonlinearity's output back to its input is
dropped (``D11 = 0``) and the ``n_nl`` scalar activations are replaced by one
:class:`~.lbdn.LBDN`, a feedforward network that is 1-Lipschitz by construction::

    x⁺ = A x + B1 w + B2 u + bx
    v  = C1 x + D12 u + bv
    w  = φ(v)                       φ 1-Lipschitz, any depth
    y  = C2 x + D21 w + D22 u + by

The certificate needs of ``φ`` only that it is 1-Lipschitz *as a map*, which a deep stack
satisfies just as well as a single layer of slope-restricted neurons. That is the whole
trade: the REN's ``w = σ(D11 w + ...)`` needs a solve (a sequential sweep over the neurons
even in the acyclic case), while ``w = φ(v)`` is a handful of GEMMs. Nonlinear capacity then
costs depth rather than width, so at matched parameter count the model is markedly cheaper
to evaluate — the paper reports up to an order of magnitude on GPU.

That saving is only visible once the launches are gone, since both rollouts are dispatch-bound
in eager: :mod:`.r2dn_backend_triton` collapses the whole rollout into one kernel, and the
comparison against the REN's equally fused sweep is what ``benchmarks/benchmark_r2dn.py``
measures.

The direct parameterization splits at the same seam as the REN's:
:meth:`R2DNParameterization.forward` builds an :class:`ExplicitR2DN` once per forward and
:meth:`R2DNCore.rollout` reads nothing else.

Reference: Barbara, Wang & Manchester, "R2DN: Scalable Parameterization of Contracting and
Lipschitz Recurrent Deep Networks" (arXiv:2504.01250), §IV-V; reference implementation
github.com/nic-barbara/R2DN ``robustnn/r2dn.py`` (MIT, JAX), which covers the contracting
case. Numerical agreement is validated in ``comparisons/compare_r2dn.py``.
"""

__all__ = [
    "R2DNSpec",
    "ExplicitR2DN",
    "R2DNParameterization",
    "R2DNCore",
    "R2DN",
]

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from ..._core.dispatch import get_backend, resolve
from .common import _ACTS, _CONTRACTION_SLACK, _EPS, _lecun_normal_, cayley_contraction, parameter_cache_key
from .lbdn import LBDN, ExplicitSandwich, folded_weights, lbdn_forward


@dataclass(frozen=True)
class R2DNSpec:
    """Static description of an R2DN.

    The certified gain ``gamma`` is deliberately not a field: like the REN's it is a runtime
    scalar, so retuning it must not invalidate anything specialized on the spec.

    Args:
        n_state: state dimension ``nx``.
        n_input: exogenous input dimension ``nu``.
        n_output: observed output dimension ``ny``.
        n_nl: width of the interconnection, i.e. of both ``v`` and ``w``.
        hidden: hidden widths of the 1-Lipschitz network; its length is the network's depth.
        variant: ``"contracting"`` or ``"lipschitz"``.
        alpha: contraction rate ``ᾱ ∈ (0, 1]``.
        act: activation name; must be monotone and slope-restricted to ``[0, 1]``.
    """

    n_state: int
    n_input: int
    n_output: int
    n_nl: int
    hidden: tuple[int, ...]
    variant: str
    alpha: float
    act: str

    @property
    def n_h(self) -> int:
        """Side length of the certificate matrix ``H``: ``2*n_state``.

        Independent of ``n_nl`` and of the network's depth, which is the scalability claim —
        the REN's ``H`` is ``(2*n_state + n_nl)²`` and grows with nonlinear capacity.
        """
        return 2 * self.n_state


@dataclass(frozen=True)
class ExplicitR2DN:
    """The realization the rollout actually runs, as plain tensors.

    Per sample, with ``φ`` the 1-Lipschitz network held in :attr:`net`::

        w   = φ(x C1ᵀ + u D12ᵀ + bv)
        x⁺  = x Aᵀ + w B1ᵀ + u B2ᵀ + bx
        y   = x C2ᵀ + w D21ᵀ + u D22ᵀ + by

    ``y`` observes the state *before* the update, so a rollout of length ``L`` from ``x0``
    returns ``y_0 .. y_{L-1}`` and carries ``x_L``.
    """

    A: Tensor
    B1: Tensor
    B2: Tensor
    C1: Tensor
    C2: Tensor
    D12: Tensor
    D21: Tensor
    D22: Tensor
    bx: Tensor
    bv: Tensor
    by: Tensor
    net: tuple[ExplicitSandwich, ...]

    @property
    def tensors(self) -> list[Tensor]:
        """The linear part, flattened in field order; excludes :attr:`net`."""
        return [getattr(self, f.name) for f in fields(self) if f.name != "net"]


class R2DNParameterization(nn.Module):
    """Free parameters of an R2DN and the direct construction of its explicit realization.

    Every parameter is unconstrained. The construction places the certificate's ``given``
    terms *into* ``H = XᵀX + εI + (given)`` so that the dissipation LMI it has to satisfy
    reduces to ``XᵀX + εI ≻ 0`` — true for any ``X`` — and then reads the realization off the
    blocks of ``H``. The free ``B1``, ``B2`` are the *implicit* input maps ``E B1``, ``E B2``,
    since the LMI constrains those rather than the explicit ones.

    Args:
        spec: static architecture description.
        gamma: certified incremental ``ℓ2`` gain for ``variant="lipschitz"``. A runtime
            scalar: reassigning it re-derives the certificate on the next forward.
        eps: regularization floor on ``H``.
        polar: use the polar parameterization ``H = (p²/‖X‖²) XᵀX + εI + ...``, which
            decouples the scale of ``H`` from the direction of ``X``.
        init: ``"long_memory"`` builds ``X`` so the explicit ``A`` starts near the identity —
            random ``X`` yields fast-forgetting models that will not fit long horizons.
            ``"random"`` samples ``X`` directly.
    """

    def __init__(
        self,
        spec: R2DNSpec,
        gamma: float = 1.0,
        eps: float = _EPS,
        polar: bool = True,
        init: str = "long_memory",
    ):
        super().__init__()
        if init not in ("long_memory", "random"):
            raise ValueError(f"unknown init {init!r}, expected 'long_memory' or 'random'")
        nx, nu, ny, nv = spec.n_state, spec.n_input, spec.n_output, spec.n_nl
        self.spec = spec
        self.gamma = gamma
        self.eps = eps
        self.polar = polar

        # gamma = 1 is not a default here but the certificate's premise: the LMI below is
        # built around ‖Δw‖ ≤ ‖Δv‖, and a network of any other gain would break it
        self.net = LBDN(nv, nv, spec.hidden, act=spec.act, gamma=1.0)

        self.B2 = nn.Parameter(_lecun_normal_(torch.empty(nx, nu)))
        self.C2 = nn.Parameter(_lecun_normal_(torch.empty(ny, nx)))
        self.bx = nn.Parameter(torch.zeros(nx))
        self.bv = nn.Parameter(torch.zeros(nv))
        self.by = nn.Parameter(torch.zeros(ny))

        long_memory = init == "long_memory"
        x = _long_memory_x(spec, eps) if long_memory else _lecun_normal_(torch.empty(spec.n_h, spec.n_h))
        self.X = nn.Parameter(x)
        self.p = nn.Parameter(x.pow(2).sum().add(eps).sqrt().reshape(1))
        # the long-memory target realization assumes a dead nonlinear coupling and Y = E = I
        self.Y = nn.Parameter(torch.eye(nx) if long_memory else _lecun_normal_(torch.empty(nx, nx)))
        self.B1 = nn.Parameter(torch.zeros(nx, nv) if long_memory else _lecun_normal_(torch.empty(nx, nv)))
        self.C1 = nn.Parameter(torch.zeros(nv, nx) if long_memory else _lecun_normal_(torch.empty(nv, nx)))

        if spec.variant == "contracting":
            self.D12 = nn.Parameter(_lecun_normal_(torch.empty(nv, nu)))
            self.D21 = nn.Parameter(_lecun_normal_(torch.empty(ny, nv)))
            self.D22 = nn.Parameter(torch.zeros(ny, nu))
        else:
            # D12 and D21 stop being free: the supply rate's input weight R is only positive
            # definite while both stay below √gamma in norm, so each is built from a
            # Cayley transform of these. D22 is dropped entirely (see the class docstring).
            self._cayley_params("12", nv, nu)
            self._cayley_params("21", ny, nv)

    def forward(self) -> ExplicitR2DN:
        """Build the explicit realization from the current free parameters."""
        h, d12, d21, d22 = self._construct()
        nx, nv, nu = self.spec.n_state, self.spec.n_nl, self.spec.n_input
        h11, h21, h22 = h[:nx, :nx], h[nx:, :nx], h[nx:, nx:]

        e = (h11 + h22 / self.spec.alpha**2 + self.Y - self.Y.mH) / 2
        # One solve for every column block that E⁻¹ acts on; never form the inverse.
        a, b1, b2 = torch.linalg.solve(e, torch.cat((h21, self.B1, self.B2), dim=1)).split((nx, nv, nu), dim=1)
        return ExplicitR2DN(
            A=a,
            B1=b1,
            B2=b2,
            C1=self.C1,
            C2=self.C2,
            D12=d12,
            D21=d21,
            D22=d22,
            bx=self.bx,
            bv=self.bv,
            by=self.by,
            net=self.net.explicit(),
        )

    def hmatrix(self) -> Tensor:
        """The certificate matrix ``H``, positive definite for any parameter values.

        Exposed because it is the guarantee itself: the storage matrix ``P = H22`` and the
        dissipation LMI are read off its blocks.
        """
        return self._construct()[0]

    def cache_key(self) -> tuple:
        """Identity of the current parameter values, for the inference-mode explicit cache."""
        return parameter_cache_key(self, self.gamma)

    def _cayley_params(self, name: str, rows: int, cols: int) -> None:
        """Register the free parameters of a ``rows × cols`` matrix of norm below one."""
        d = min(rows, cols)
        for prefix, shape in (("X", (d, d)), ("Y", (d, d)), ("Z", (abs(rows - cols), d))):
            self.register_parameter(prefix + name, nn.Parameter(_lecun_normal_(torch.empty(*shape))))

    def _bounded(self, name: str, tall: bool) -> Tensor:
        """The matrix of spectral norm below one that ``name``'s free parameters build.

        Held a further :data:`~.common._CONTRACTION_SLACK` inside the unit ball, because the
        slack left over here is exactly what ``_construct`` inverts.
        """
        x, y, z = (getattr(self, prefix + name) for prefix in ("X", "Y", "Z"))
        eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
        m = x.mH @ x + y - y.mH + z.mH @ z + self.eps * eye
        return (1.0 - _CONTRACTION_SLACK) * cayley_contraction(m, z, tall)

    def _construct(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(H, D12, D21, D22)`` — the certificate matrix and the maps it is built around."""
        spec = self.spec
        nx, nv, nu, ny = spec.n_state, spec.n_nl, spec.n_input, spec.n_output
        h = self.X.mH @ self.X
        if self.polar:
            h = h * self.p.pow(2) / (self.X.pow(2).sum() + self.eps)
        h = h + self.eps * torch.eye(spec.n_h, dtype=h.dtype, device=h.device)
        c1c1 = self.C1.mH @ self.C1

        if spec.variant == "contracting":
            # Eq. 20-21: the storage matrix absorbs B1B1ᵀ and E absorbs C1ᵀC1, which is
            # exactly what turns the contraction LMI into H ≻ 0.
            return h + torch.block_diag(c1c1, self.B1 @ self.B1.mH), self.D12, self.D21, self.D22

        gamma = self.gamma
        d12 = gamma**0.5 * self._bounded("12", nv >= nu)
        d21 = gamma**0.5 * self._bounded("21", ny >= nv)
        eye_v = torch.eye(nv, dtype=h.dtype, device=h.device)
        eye_u = torch.eye(nu, dtype=h.dtype, device=h.device)
        # Eq. 24-26 with D22 = 0, for the supply rate Q = -I/gamma, S = 0, R = gamma·I
        # stacked on the (-I, 0, I) constraint the 1-Lipschitz network satisfies. R is block
        # diagonal only because D22 vanishes; both blocks are positive definite by the norm
        # bounds above, so the correction terms below are positive semidefinite and H ≻ 0.
        r = torch.block_diag(eye_v - d21.mH @ d21 / gamma, gamma * eye_u - d12.mH @ d12)
        top = torch.cat((-self.C2.mH @ d21 / gamma, -self.C1.mH @ d12), dim=1)
        mul = torch.cat((top, torch.cat((self.B1, self.B2), dim=1)), dim=0)
        given = torch.block_diag(c1c1 + self.C2.mH @ self.C2 / gamma, h.new_zeros(nx, nx))
        return h + mul @ torch.linalg.solve(r, mul.mH) + given, d12, d21, h.new_zeros(ny, nu)


def _long_memory_x(spec: R2DNSpec, eps: float) -> Tensor:
    """``X`` factoring an ``H`` whose explicit ``A`` is the identity minus small eigenvalues.

    Random ``X`` gives a model that forgets its state within a few samples, which no
    optimizer recovers from on long horizons. Here the target realization is written down
    first (``E = P = I``, ``A = I - diag(eigs)``, dead nonlinear coupling) and ``X`` falls out
    of its Cholesky factor.
    """
    nx = spec.n_state
    eye = torch.eye(nx)
    a = eye - torch.diag(0.05 * torch.rand(nx))
    h = torch.cat((torch.cat((eye, a.mH), dim=1), torch.cat((a, eye), dim=1)), dim=0)
    return torch.linalg.cholesky(h + eps * torch.eye(spec.n_h)).mH


# Fused-kernel backends, resolved through dispatch.resolve exactly as the REN's are: each
# module exposes supports(spec, u, x0) -> str | None plus forward_infer/forward_train/backward
# over the flattened parameter order of MATH_R2DN.md §4. A preference naming a family this
# model has no kernel for (``"c"``, ``"metal"``) resolves to the reference path silently.
_FUSED = {"triton": "tsfast.models.architectures.ren.r2dn_backend_triton"}
_OP_AUTO = {"cuda": ("triton",)}
_CORE_AUTO = {"cuda": ("triton",)}

#: Count of explicit tensors ahead of the network's, i.e. ``len(ExplicitR2DN.tensors)``.
_N_LINEAR = 11


@torch._dynamo.assume_constant_result
def _rollout_mode(backend: str, spec: R2DNSpec, u: Tensor, x0: Tensor) -> str:
    """Execution mode for this call: ``"eager"``, ``"compiled"``, or ``"fused"``.

    ``"auto"`` defers to the process preference, and a ``"reference"`` preference disables
    fused kernels even for instances that request one. The non-fused fallback is always
    eager: ``torch.compile`` unrolls the whole sequence, which is only viable on short ones,
    so it never gets picked implicitly.
    """
    pref = get_backend()
    if backend == "auto":
        backend = pref
    elif pref == "reference" and backend in _FUSED:
        backend = "reference"
    match backend:
        case "eager" | "compiled":
            return backend
        case "reference":
            return "eager"
        case "auto" | "triton" | "c" | "metal":
            mod = resolve("r2dn.rollout", _FUSED, _CORE_AUTO.get(u.device.type, ()), (spec, u, x0), requested=backend)
            return "fused" if mod is not None else "eager"
        case unknown:
            raise ValueError(f"unknown backend {unknown!r}")


def _fused_module(spec: R2DNSpec, u: Tensor, x0: Tensor):
    """The fused backend module serving this op call, honoring the process preference."""
    order = _OP_AUTO.get(u.device.type, ())
    mod = resolve("r2dn.rollout", _FUSED, order, (spec, u, x0))
    if mod is None:
        mod = resolve("r2dn.rollout", _FUSED, order, (spec, u, x0), requested="auto")
    if mod is None:
        raise RuntimeError(f"no fused R2DN backend usable for device {u.device.type!r}; use backend='eager'")
    return mod


def split_params(params: list[Tensor]) -> tuple[list, list]:
    """Cut the flat parameter list into ``(linear tensors, per-layer (W, c) pairs)``.

    The flattening is ``MATH_R2DN.md`` §4: the eleven explicit tensors of
    :class:`ExplicitR2DN`, then one ``(W, c)`` pair per layer with the output layer last. The
    layer count follows from the length, so nothing else has to carry it.
    """
    pairs = [params[i : i + 2] for i in range(_N_LINEAR, len(params), 2)]
    return params[:_N_LINEAR], pairs


def hidden_widths(params: list[Tensor]) -> tuple[int, ...]:
    """The network's hidden widths, read off the weight shapes.

    Every layer's ``W`` is ``[out, in]``, so the hidden widths are the row counts of all but
    the output layer's. This is why the op signature carries no width list: the shapes it
    already ships are the same information, and a redundant argument is one more thing a
    backend could disagree with.
    """
    return tuple(w.shape[0] for w, _ in split_params(params)[1][:-1])


# ------------------------------------------------------------------------- custom ops
#
# As in ``core.py``: torch.library custom ops so the rollout composes with torch.compile,
# fake/meta tracing and export. Spec fields cross the boundary as scalars and are rebuilt
# inside the impls; the backward is its own op so compiled autograd sees no graph break.


def _spec_from(n_state: int, n_input: int, n_output: int, n_nl: int, act: str, params: list[Tensor]) -> R2DNSpec:
    # variant and alpha are certificate properties; nothing below this boundary reads them
    return R2DNSpec(n_state, n_input, n_output, n_nl, hidden_widths(params), "contracting", 1.0, act)


@torch.library.custom_op("tsfast::r2dn_rollout", mutates_args=())
def _r2dn_rollout(
    u: Tensor,
    x0: Tensor,
    params: list[Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    n_nl: int,
    act: str,
) -> tuple[Tensor, Tensor]:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, n_output, n_nl, act, params)
    return _fused_module(spec, u, x0).forward_infer(spec, u, x0, params)


@_r2dn_rollout.register_fake
def _(u, x0, params, n_state, n_input, n_output, n_nl, act):
    return u.new_empty(u.shape[0], u.shape[1], n_output), u.new_empty(u.shape[0], n_state)


@torch.library.custom_op("tsfast::r2dn_rollout_train", mutates_args=())
def _r2dn_rollout_train(
    u: Tensor,
    x0: Tensor,
    params: list[Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    n_nl: int,
    act: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, n_output, n_nl, act, params)
    return _fused_module(spec, u, x0).forward_train(spec, u, x0, params)


@_r2dn_rollout_train.register_fake
def _(u, x0, params, n_state, n_input, n_output, n_nl, act):
    b, ln = u.shape[0], u.shape[1]
    return (
        u.new_empty(b, ln, n_output),
        u.new_empty(b, n_state),
        u.new_empty(b, ln, n_state),
        u.new_empty(b, ln, n_nl),
        u.new_empty(b, ln, n_nl),
        [u.new_empty(b, ln, m) for m in hidden_widths(params)],
    )


@torch.library.custom_op("tsfast::r2dn_rollout_bwd", mutates_args=())
def _r2dn_rollout_bwd(
    grad_y: Tensor | None,
    grad_xl: Tensor | None,
    u: Tensor,
    xs: Tensor,
    vs: Tensor,
    ws: Tensor,
    acts: list[Tensor],
    params: list[Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    n_nl: int,
    act: str,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    u, xs, vs, ws = u.contiguous(), xs.contiguous(), vs.contiguous(), ws.contiguous()
    acts = [a.contiguous() for a in acts]
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, n_output, n_nl, act, params)
    gy = grad_y.contiguous() if grad_y is not None else u.new_zeros(u.shape[0], u.shape[1], n_output)
    gxl = grad_xl.contiguous() if grad_xl is not None else u.new_zeros(u.shape[0], n_state)
    x0 = xs[:, 0].contiguous()
    lam, gv, gps, gx0 = _fused_module(spec, u, x0).backward(spec, gy, gxl, acts, params)
    grads, du = r2dn_param_grads(spec, u, xs, vs, ws, acts, gy, lam, gv, gps, params)
    # BLAS is free to hand back a transposed view of a GEMM result; the fake kernel below
    # promises contiguous outputs, and inductor asserts on the mismatch.
    return du.contiguous(), gx0.contiguous(), [g.contiguous() for g in grads]


@_r2dn_rollout_bwd.register_fake
def _(grad_y, grad_xl, u, xs, vs, ws, acts, params, n_state, n_input, n_output, n_nl, act):
    # new_empty, not empty_like: the explicit matrices reach the op as strided views of the
    # solve that produced them, while the gradients the impl returns are contiguous
    return u.new_empty(u.shape), u.new_empty(u.shape[0], n_state), [p.new_empty(p.shape) for p in params]


def _train_setup(ctx, inputs, output):
    u, x0, params, n_state, n_input, n_output, n_nl, act = inputs
    _, _, xs, vs, ws, acts = output
    ctx.fields = (n_state, n_input, n_output, n_nl, act)
    ctx.n_acts = len(acts)
    ctx.save_for_backward(u, xs, vs, ws, *acts, *params)


def _train_backward(ctx, grad_y, grad_xl, grad_xs, grad_vs, grad_ws, grad_acts):
    saved = ctx.saved_tensors
    u, xs, vs, ws = saved[:4]
    acts = list(saved[4 : 4 + ctx.n_acts])
    params = list(saved[4 + ctx.n_acts :])
    du, dx0, dparams = _r2dn_rollout_bwd(grad_y, grad_xl, u, xs, vs, ws, acts, params, *ctx.fields)
    return du, dx0, list(dparams), None, None, None, None, None


_r2dn_rollout_train.register_autograd(_train_backward, setup_context=_train_setup)


def r2dn_param_grads(
    spec: R2DNSpec,
    u: Tensor,
    xs: Tensor,
    vs: Tensor,
    ws: Tensor,
    acts: list[Tensor],
    gy: Tensor,
    lam: Tensor,
    gv: Tensor,
    gps: list[Tensor],
    params: list[Tensor],
) -> tuple[list[Tensor], Tensor]:
    """Parameter and input gradients of the rollout as batched GEMMs over the step samples.

    The adjoint recurrence (``lam``, ``gv``, ``gps``) is the only sequential part of BPTT and
    is what a backend computes; every parameter gradient is then a reduction over all ``B*L``
    step samples. Folding the layers leaves each one a plain outer product of its
    pre-activation adjoint with its input, the input being the previous layer's activation —
    see ``MATH_R2DN.md`` §3.2.

    Args:
        u: input sequence ``[B, L, n_input]``.
        xs: states fed into each step ``[B, L, n_state]`` (``xs[t] = x_t``).
        vs: network inputs ``[B, L, n_nl]``.
        ws: network outputs ``[B, L, n_nl]``.
        acts: post-activations of each hidden layer, ``[B, L, m_l]``.
        gy: output adjoints ``[B, L, n_output]``.
        lam: next-state adjoints ``[B, L, n_state]`` (``lam[t] = ∂L/∂x_{t+1}``).
        gv: adjoints of the network input ``[B, L, n_nl]``.
        gps: pre-activation adjoints of each hidden layer, ``[B, L, m_l]``.
        params: the flattened realization, see :func:`split_params`.

    Returns:
        ``(grads, du)`` with grads in the same order as ``params``.
    """
    lin, _ = split_params(params)
    b1, b2, d12, d21, d22 = lin[1], lin[2], lin[5], lin[6], lin[7]
    n = u.shape[0] * u.shape[1]
    flat = lambda t: t.reshape(n, t.shape[-1])  # noqa: E731
    uf, xf, vf, wf = flat(u), flat(xs), flat(vs), flat(ws)
    gyf, lf, gvf = flat(gy), flat(lam), flat(gv)
    af, gpf = [flat(a) for a in acts], [flat(g) for g in gps]
    gwf = gyf @ d21 + lf @ b1  # one GEMM, so the kernel need not emit it

    grads = [
        lf.t() @ xf,  # A
        lf.t() @ wf,  # B1
        lf.t() @ uf,  # B2
        gvf.t() @ xf,  # C1
        gyf.t() @ xf,  # C2
        gvf.t() @ uf,  # D12
        gyf.t() @ wf,  # D21
        gyf.t() @ uf,  # D22
        lf.sum(0),  # bx
        gvf.sum(0),  # bv
        gyf.sum(0),  # by
    ]
    for gp, h in zip([*gpf, gwf], [vf, *af]):  # (W, c) of every layer, output layer last
        grads += [gp.t() @ h, gp.sum(0)]
    du = (lf @ b2 + gvf @ d12 + gyf @ d22).reshape(u.shape)
    return grads, du


def fused_rollout(spec: R2DNSpec, u: Tensor, x0: Tensor, e: ExplicitR2DN) -> tuple[Tensor, Tensor]:
    """Run the rollout through the fused-kernel custom ops (autograd-capable).

    Picks the training op (which stores the BPTT tape) when gradients are live, else the
    inference op, which keeps no intermediates.

    Returns:
        ``(y, x_L)`` with ``y`` shaped ``[B, L, n_output]``.
    """
    params = [*e.tensors, *folded_weights(e.net)]
    fields = (spec.n_state, spec.n_input, spec.n_output, spec.n_nl, spec.act)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (u, x0, *params)):
        y, x_last, *_ = _r2dn_rollout_train(u, x0, params, *fields)
        return y, x_last
    return _r2dn_rollout(u, x0, params, *fields)


class R2DNCore(nn.Module):
    """Explicit realization plus the sequential rollout over an input sequence.

    Holds the free parameters (in :attr:`parameterization`) but evaluates only through
    :class:`ExplicitR2DN` tensors: the rollout has no opinion about how the matrices were
    certified, and the 1-Lipschitz network's Cayley transforms are taken once per rollout
    rather than once per timestep.

    Args:
        spec: static architecture description.
        **kwargs: forwarded to :class:`R2DNParameterization`.
    """

    def __init__(self, spec: R2DNSpec, **kwargs):
        super().__init__()
        if spec.act not in _ACTS:
            raise ValueError(f"unknown activation {spec.act!r}, expected one of {sorted(_ACTS)}")
        if not 0.0 < spec.alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0, 1], got {spec.alpha}")
        self.spec = spec
        self.parameterization = R2DNParameterization(spec, **kwargs)
        self._act = _ACTS[spec.act]
        self._cache: tuple[tuple, ExplicitR2DN] | None = None

    def explicit(self) -> ExplicitR2DN:
        """The explicit realization, rebuilt on demand and cached while gradients are off.

        The construction costs a few matrix products, one solve on a ``2*n_state`` matrix and
        one Cayley transform per network layer — irrelevant next to an ``L``-step rollout
        during training, but worth caching for repeated inference from fixed weights.
        """
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return self.parameterization()
        key = self.parameterization.cache_key()
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.parameterization())
        return self._cache[1]

    def rollout(self, e: ExplicitR2DN, u: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        """Run the realization over ``u [B, L, n_input]`` from ``x0 [B, n_state]``.

        Returns:
            ``(y, x_L)`` with ``y`` shaped ``[B, L, n_output]``.
        """
        bv = u @ e.D12.mH + e.bv
        bx = u @ e.B2.mH + e.bx
        by = u @ e.D22.mH + e.by
        x = x0
        ys = []
        for t in range(u.shape[1]):
            w = lbdn_forward(e.net, x @ e.C1.mH + bv[:, t], self._act)
            ys.append(x @ e.C2.mH + w @ e.D21.mH + by[:, t])
            x = x @ e.A.mH + w @ e.B1.mH + bx[:, t]
        return torch.stack(ys, dim=1), x


class R2DN(nn.Module):
    """Robust recurrent deep network: contracting by construction, with a deep nonlinearity.

    Trained with plain SGD from any initialization, the model satisfies its certificate at
    every step of training, because the certificate is built into the map from free
    parameters to model matrices rather than enforced on top of it.

    Two variants, differing only in how the certificate matrix is assembled:

    - ``"contracting"``: two trajectories under the same input converge at rate ``alpha``.
    - ``"lipschitz"``: additionally ``‖y(u) - y(ũ)‖ ≤ gamma ‖u - ũ‖`` in truncated ``ℓ2``
      **from a common initial state**. ``gamma`` is a runtime scalar and may be reassigned.
      The paper's parameterization of this case covers the subset with ``D22 = 0``, so there
      is no direct feedthrough from input to output; the nonlinearity still provides a
      static path through ``D12`` and ``D21``.

    What the Lipschitz certificate buys, precisely: an input perturbation of energy ``δ``
    moves the output by at most ``gamma·δ``, measured in ``ℓ2`` over the horizon and starting
    from the same state. It says nothing about model-vs-plant error — an R2DN with
    ``gamma = 1`` can be an arbitrarily bad model of a system, certified smooth and stable
    rather than correct.

    Against :class:`~.core.REN`, which certifies the same properties: nonlinear capacity
    here is depth in a 1-Lipschitz network instead of width in an equilibrium layer, so a
    step costs ``depth`` small GEMMs rather than a sequential sweep over ``n_nl`` neurons, and
    the certificate matrix stays ``2·n_state`` square however large the nonlinearity grows.
    The REN's parameterization is the more general one — its equilibrium layer contains
    multi-layer networks as special cases — but the sweep is what makes it expensive.

    Both models are dispatch-bound before they are fused, and both have a kernel that fixes
    that, so the comparison is between the fused paths. There the scalability claim holds:
    this rollout's cost is flat in nonlinear capacity while the REN's grows with ``n_nl``, so
    at matched parameter count the two cross over around ``n_nl ≈ 24`` and the R2DN is several
    times faster beyond it — and slower below, where the REN's sweep is only a few neurons
    long. ``benchmarks/benchmark_r2dn.py`` times both at matched parameter count.

    The rollout is sequential along the sequence, so a Python loop is dispatch-bound at short
    ``n_state``:

    - ``"eager"``: the loop — any device and dtype.
    - ``"triton"``: persistent per-trajectory GPU kernel with a fused BPTT backward
      (``MATH_R2DN.md``) — float32 on CUDA, within the size caps its ``fits`` reports.
    - ``"compiled"``: ``torch.compile`` over the unrolled loop. Only usable on short
      sequences (the graph holds ``seq`` copies of the network), so it is never selected
      implicitly.
    - ``"auto"``: defers to the process-wide preference (``tsfast.models.set_backend`` /
      ``use_backend``); under an ``"auto"`` preference picks ``triton`` where it applies and
      eager elsewhere. A ``"reference"`` preference forces the eager path everywhere.

    All backends share the same parameters, so the backend can be switched at any time via
    the ``backend`` attribute. There is no CPU kernel: a ``"c"`` request selects eager.

    Contraction is a prior about the plant, not free insurance, and it fits some systems
    badly: on friction-dominated plants the stick-slip phases are exactly what a contraction
    certificate excludes, since trajectories in a stick phase do not converge. On
    ``benchmarks/gate_ren.py`` that shows up on ``EMPS``, where the best certified model
    trails a GRU by 2.3x under ``FranSys`` — better than the REN manages there, but the same
    verdict. **Do not reach for either model on stick-slip or hysteretic plants.**

    Against the REN at matched parameter count the two are at parity overall, which is what
    the paper reports: this model wins on ``WH`` (0.037 NRMSE against 0.045, closing the one
    benchmark the REN loses to a GRU) and on ``EMPS`` under ``FranSys`` (0.058 against 0.106),
    ties on ``Silverbox`` and on ``CascadedTanks`` under ``FranSys`` (0.097 both), and loses
    on both integrating plants standalone (0.43 against 0.37, 0.36 against 0.28). Prefer it
    when the nonlinearity has to be large, where its cost is flat in capacity and the REN's
    is not; prefer the REN at small ``n_nl``.

    ``gamma`` is not a free addition. It costs nothing on ``WH`` (0.038 against 0.037
    contracting) but a great deal on the integrating plants (0.61 against 0.43 on
    ``CascadedTanks``), where the budget binds against dynamics that need the range. Prescribe
    it when something downstream consumes the bound, not by default. What it does buy is
    honest: on trained models the certificate is tight, ``gamma_empirical/gamma_certified``
    landing in 0.48-0.83 across the suite — the same range the REN reaches, and nothing like
    the orders-of-magnitude slack typical of post-hoc bounds on freely-trained networks.

    Contraction makes the initial state self-correcting at rate ``alpha``, so ``x0=None``
    (zeros) plus ``n_skip`` is usually enough. But the forgetting time ``≈ 1/(1-alpha)`` *is*
    the longest time constant the model can represent, so no ``alpha`` both forgets ``x0``
    quickly and represents an integrator. For integrating plants (position from velocity,
    tank level, thermal accumulation) use ``return_state=True`` and compose with
    :class:`~tsfast.prediction.fransys.FranSys`, which estimates ``x0`` from an ``(u, y)``
    window instead of asking the dynamics to forget it. Note that the Lipschitz bound is
    stated for a fixed initial state and does not survive that composition.

    Args:
        n_input: exogenous input dimension.
        n_output: observed output dimension.
        n_state: state dimension.
        n_nl: width of the interconnection with the nonlinearity, and of its hidden layers.
        depth: number of nonlinear layers in the 1-Lipschitz network; at least ``1``.
        variant: ``"contracting"`` or ``"lipschitz"``.
        alpha: contraction rate in ``(0, 1]``. ``1.0`` admits arbitrarily long memory.
        gamma: certified incremental ``ℓ2`` gain, for ``variant="lipschitz"``.
        act: activation of the 1-Lipschitz network, one of ``tanh``, ``relu``, ``sigmoid``;
            must be monotone and slope-restricted to ``[0, 1]``, which all three are.
        eps: regularization floor on the certificate matrix.
        polar: use the polar parameterization of ``H``.
        init: ``"long_memory"`` (default) or ``"random"``; see :class:`R2DNParameterization`.
        backend: execution backend, see above.
        return_state: if ``True``, return ``(output, state)`` following the stateful-model
            protocol, so ``TbpttLearner`` state carrying and ``FranSys`` both work.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_state: int = 8,
        n_nl: int = 32,
        depth: int = 2,
        variant: str = "contracting",
        alpha: float = 1.0,
        gamma: float = 1.0,
        act: str = "relu",
        eps: float = _EPS,
        polar: bool = True,
        init: str = "long_memory",
        backend: str = "auto",
        return_state: bool = False,
    ):
        super().__init__()
        if variant not in ("contracting", "lipschitz"):
            raise ValueError(f"unknown variant {variant!r}, expected 'contracting' or 'lipschitz'")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        spec = R2DNSpec(n_state, n_input, n_output, n_nl, (n_nl,) * depth, variant, alpha, act)
        self.core = R2DNCore(spec, gamma=gamma, eps=eps, polar=polar, init=init)
        self.backend = backend
        self.return_state = return_state
        self._compiled_rollout = None

    @property
    def spec(self) -> R2DNSpec:
        return self.core.spec

    @property
    def gamma(self) -> float:
        """Certified incremental ``ℓ2`` gain; reassign to retune the certificate."""
        return self.core.parameterization.gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self.core.parameterization.gamma = value

    def forward(self, u: Tensor, x0: Tensor | None = None, state: dict | None = None):
        """Roll the certified dynamics over the input sequence.

        Args:
            u: input sequence ``[batch, seq, n_input]``.
            x0: initial state ``[batch, n_state]`` (or ``[batch, 1, n_state]``); zeros if None.
            state: carried state ``{"x": x_last}`` from a previous chunk; overrides ``x0``.

        Returns:
            Output sequence ``[batch, seq, n_output]`` observing the states ``x_0 .. x_{L-1}``,
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
                y, x_last = self.core.rollout(self.core.explicit(), u, x0)
            case "compiled":
                y, x_last = self._rollout_compiled(u, x0)
            case _:
                y, x_last = fused_rollout(self.spec, u, x0, self.core.explicit())
        if self.return_state:
            return y, {"x": x_last}
        return y

    def _rollout_compiled(self, u: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        if self._compiled_rollout is None:
            # The unrolled sequence is one large graph; raise the recompile budget so shape
            # changes (batch/seq) do not silently fall back to eager.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._compiled_rollout = torch.compile(self._rollout_eager, dynamic=False)
        return self._compiled_rollout(u, x0)

    def _rollout_eager(self, u: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        return self.core.rollout(self.core.explicit(), u, x0)
