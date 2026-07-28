"""Recurrent equilibrium networks (REN, Revay, Wang & Manchester 2023).

A sequence model whose contraction and incremental-Lipschitz certificates hold at
*every* value of its free parameters, so it trains under ordinary unconstrained SGD with
no projection, no penalty and no post-hoc verification.

The construction is direct: a free matrix ``X`` builds ``H = XᵀX + εI ≻ 0``, and the
blocks of ``H`` are read off as the implicit realization ``E x⁺ = F x + B1 w + B2 u``,
``Λ v = C1 x + D11 w + D12 u``, ``w = σ(v)``. Positive-definiteness of ``H`` *is* the
contraction LMI, and ``D11`` comes out strictly lower triangular, so the equilibrium
layer resolves by forward substitution over the ``n_nl`` neurons rather than a fixed-point
solve (the "acyclic" REN class).

Two steps, at a seam that matters: :meth:`RENParameterization.forward` turns the free
parameters into an :class:`~.common.ExplicitREN` bundle once per forward, and
:meth:`RENCore.rollout` consumes nothing but those tensors. The certificate variants
(Lipschitz, ``(Q,S,R)``-dissipative) change only the first step.

Reference: Revay, Wang & Manchester, "Recurrent Equilibrium Networks: Flexible Dynamic
Models with Guaranteed Stability and Robustness", IEEE TAC 2023 (arXiv:2104.05942),
§V; reference implementation github.com/nic-barbara/R2DN ``robustnn/ren.py`` (MIT, JAX).
Faithful reimplementation: same construction (eqs. 28-33), same ``long_memory``
initialization, same explicit realization. Numerical agreement is validated in
``comparisons/compare_ren.py``.
"""

__all__ = [
    "equilibrium_sweep",
    "RENParameterization",
    "RENCore",
    "REN",
    "fused_rollout",
]

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from ..._core.dispatch import get_backend, resolve
from .common import _ACTS, _EPS, ExplicitREN, RENSpec, _lecun_normal_, cayley_contraction, parameter_cache_key


def equilibrium_sweep(b: Tensor, d11: Tensor, act: Callable[[Tensor], Tensor]) -> Tensor:
    """Solve ``w = act(w D11ᵀ + b)`` by forward substitution over the neurons.

    ``D11`` is strictly lower triangular by construction, so neuron ``i`` depends only on
    ``0 .. i-1`` and the equilibrium resolves exactly in one sweep. Each step is a rank-1
    update of the pending pre-activations, which is why the loop costs two tensor ops per
    neuron rather than a growing matmul.

    Args:
        b: input-and-state part of the pre-activation ``[..., n_nl]``.
        d11: strictly lower triangular feedback ``[n_nl, n_nl]``.
        act: the activation, monotone and slope-restricted to ``[0, 1]``.
    """
    acc = b
    ws = []
    for i in range(d11.shape[0]):
        w_i = act(acc[..., i])
        ws.append(w_i)
        if i + 1 < d11.shape[0]:
            acc = acc + w_i.unsqueeze(-1) * d11[:, i]
    return torch.stack(ws, dim=-1)


def _is_posdef(a: Tensor) -> bool:
    return bool(torch.allclose(a, a.mH)) and bool((torch.linalg.eigvalsh(a) > 0).all())


class RENParameterization(nn.Module):
    """Free parameters of a REN and the direct construction of its explicit realization.

    Every parameter here is unconstrained: the certificate is a property of the
    *construction*, not of where the optimizer happens to be. ``forward`` runs the whole
    construction — the ``XᵀX`` product, the partition of ``H``, and the ``E⁻¹``/``Λ⁻¹``
    solves — and returns plain tensors.

    Args:
        spec: static architecture description.
        gamma: certified incremental ``ℓ2`` gain for ``variant="lipschitz"``. A runtime
            scalar: reassigning it re-derives the certificate on the next forward.
        qsr: ``(Q, S, R)`` supply-rate matrices for ``variant="dissipative"``, shaped
            ``(ny,ny)``, ``(nu,ny)``, ``(nu,nu)``, with ``Q ≺ 0`` and ``R - S Q⁻¹ Sᵀ ≻ 0``.
        eps: regularization floor on ``H``.
        polar: use the polar parameterization ``H = (p²/‖X‖²) XᵀX + εI``, which decouples
            the scale of ``H`` from the direction of ``X``.
        init: ``"long_memory"`` builds ``X`` so the explicit ``A`` starts near the
            identity — random ``X`` yields fast-forgetting models that will not fit long
            horizons. ``"random"`` samples ``X`` directly.
    """

    def __init__(
        self,
        spec: RENSpec,
        gamma: float = 1.0,
        qsr: tuple[Tensor, Tensor, Tensor] | None = None,
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

        self.B2 = nn.Parameter(_lecun_normal_(torch.empty(nx, nu)))
        self.D12 = nn.Parameter(_lecun_normal_(torch.empty(nv, nu)))
        self.C2 = nn.Parameter(_lecun_normal_(torch.empty(ny, nx)))
        self.D21 = nn.Parameter(_lecun_normal_(torch.empty(ny, nv)))
        self.bx = nn.Parameter(torch.zeros(nx))
        self.bv = nn.Parameter(torch.zeros(nv))
        self.by = nn.Parameter(torch.zeros(ny))

        x = _long_memory_x(spec, eps) if init == "long_memory" else _lecun_normal_(torch.empty(spec.n_h, spec.n_h))
        self.X = nn.Parameter(x)
        self.Y1 = nn.Parameter(torch.eye(nx) if init == "long_memory" else _lecun_normal_(torch.empty(nx, nx)))
        self.p = nn.Parameter(x.pow(2).sum().add(eps).sqrt().reshape(1))

        if spec.variant == "contracting":
            self.D22 = nn.Parameter(torch.zeros(ny, nu))
        else:
            # D22 is no longer free: it is built from these through a nonsquare Cayley
            # transform so that ‖N‖ ≤ 1, which is what makes the supply rate's input
            # weight invertible. The values below give D22 = 0 at initialization.
            d = min(nu, ny)
            self.X3 = nn.Parameter(torch.eye(d))
            self.Y3 = nn.Parameter(torch.zeros(d, d))
            self.Z3 = nn.Parameter(torch.zeros(abs(ny - nu), d))

        if spec.variant == "dissipative":
            if qsr is None:
                raise ValueError("variant='dissipative' requires qsr=(Q, S, R)")
            q, s, r = (torch.as_tensor(m, dtype=torch.get_default_dtype()) for m in qsr)
            _check_qsr(q, s, r, nu, ny)
            self.register_buffer("Q", q)
            self.register_buffer("S", s)
            self.register_buffer("R", r)

    def forward(self) -> ExplicitREN:
        """Build the explicit realization from the current free parameters."""
        h, d22 = self._construct()
        spec = self.spec
        nx, nv, nu = spec.n_state, spec.n_nl, spec.n_input
        h11, h21, h22 = h[:nx, :nx], h[nx : nx + nv, :nx], h[nx : nx + nv, nx : nx + nv]
        h31, h32, h33 = h[nx + nv :, :nx], h[nx + nv :, nx : nx + nv], h[nx + nv :, nx + nv :]

        e = (h11 + h33 / spec.alpha**2 + self.Y1 - self.Y1.mH) / 2
        # One solve for every column block that E⁻¹ acts on; never form the inverse.
        a, b1, b2 = torch.linalg.solve(e, torch.cat((h31, h32, self.B2), dim=1)).split((nx, nv, nu), dim=1)
        lam_inv = (2.0 / torch.diagonal(h22)).unsqueeze(1)
        return ExplicitREN(
            A=a,
            B1=b1,
            B2=b2,
            C1=-lam_inv * h21,
            D11=-lam_inv * torch.tril(h22, -1),
            D12=lam_inv * self.D12,
            C2=self.C2,
            D21=self.D21,
            D22=d22,
            bx=self.bx,
            bv=self.bv,
            by=self.by,
        )

    def hmatrix(self) -> Tensor:
        """The certificate matrix ``H``, positive definite for any parameter values.

        Exposed because it is the guarantee itself: the storage matrix ``P = H33`` and the
        contraction LMI are read off its blocks.
        """
        return self._construct()[0]

    def qsr(self) -> tuple[Tensor, Tensor, Tensor]:
        """Supply-rate matrices ``(Q, S, R)`` in effect.

        ``variant="lipschitz"`` is the special case ``Q = -I/γ``, ``S = 0``, ``R = γI``, so
        the certified gain is a runtime scalar rather than a stored matrix.
        """
        ny, nu = self.spec.n_output, self.spec.n_input
        if self.spec.variant == "lipschitz":
            eye_y = torch.eye(ny, dtype=self.X.dtype, device=self.X.device)
            eye_u = torch.eye(nu, dtype=self.X.dtype, device=self.X.device)
            return -eye_y / self.gamma, self.X.new_zeros(nu, ny), self.gamma * eye_u
        # User-supplied matrices may sit right on the definiteness boundary; nudge them
        # inside it so the Cholesky factors the construction takes stay well conditioned.
        eye_q = torch.eye(ny, dtype=self.Q.dtype, device=self.Q.device)
        eye_r = torch.eye(nu, dtype=self.R.dtype, device=self.R.device)
        return self.Q - self.eps * eye_q, self.S, self.R + self.eps * eye_r

    def cache_key(self) -> tuple:
        """Identity of the current parameter values, for the inference-mode explicit cache."""
        return parameter_cache_key(self, self.gamma)

    def _construct(self) -> tuple[Tensor, Tensor]:
        """``(H, D22)`` — the certificate matrix and the feedthrough it is built around."""
        h = self.X.mH @ self.X
        if self.polar:
            h = h * self.p.pow(2) / (self.X.pow(2).sum() + self.eps)
        h = h + self.eps * torch.eye(self.spec.n_h, dtype=h.dtype, device=h.device)
        if self.spec.variant == "contracting":
            return h, self.D22

        q, s, r = self.qsr()
        d22 = self._d22(q, s, r)
        # Eq. 28: the dissipation LMI after eliminating the input block by a Schur
        # complement on R̃. Q ⪯ 0 makes both correction terms positive semidefinite, so H
        # stays positive definite whatever the free parameters do.
        sq = d22.mH @ q + s
        r_tilde = r + s @ d22 + d22.mH @ s.mH + d22.mH @ q @ d22
        mul_r = torch.cat((sq @ self.C2, sq @ self.D21 - self.D12.mH, self.B2.mH), dim=1)
        mul_q = torch.cat((self.C2, self.D21, self.C2.new_zeros(self.spec.n_output, self.spec.n_state)), dim=1)
        h = h + mul_r.mH @ torch.linalg.solve(r_tilde, mul_r) - mul_q.mH @ q @ mul_q
        return h, d22

    def _d22(self, q: Tensor, s: Tensor, r: Tensor) -> Tensor:
        """Feedthrough consistent with the supply rate (eqs. 31-33)."""
        ny, nu = self.spec.n_output, self.spec.n_input
        eye = torch.eye(min(ny, nu), dtype=q.dtype, device=q.device)
        m = self.X3.mH @ self.X3 + self.Y3 - self.Y3.mH + self.Z3.mH @ self.Z3 + self.eps * eye
        n = cayley_contraction(m, self.Z3, ny >= nu)
        lq = torch.linalg.cholesky(-q).mH
        lr = torch.linalg.cholesky(r - s @ torch.linalg.solve(q, s.mH)).mH
        return torch.linalg.solve(-q, s.mH) + torch.linalg.solve(lq, n) @ lr


def _long_memory_x(spec: RENSpec, eps: float) -> Tensor:
    """``X`` factoring an ``H`` whose explicit ``A`` is the identity minus small eigenvalues.

    Random ``X`` gives a model that forgets its state within a few samples, which no
    optimizer recovers from on long horizons. Here the target realization is written down
    first (``E = P = I``, ``F = I - diag(eigs)``, dead nonlinear coupling) and ``X`` falls
    out of its Cholesky factor; ``Λ`` is raised just far enough to keep ``H22`` positive
    definite for the sampled ``D11``.
    """
    nx, nv = spec.n_state, spec.n_nl
    eye_x = torch.eye(nx)
    f = eye_x - torch.diag(0.05 * torch.rand(nx))
    d11 = _lecun_normal_(torch.empty(nv, nv))
    lam = torch.linalg.eigvalsh(d11 + d11.mH).max() / 2 + 1e-4
    h22 = 2 * lam * torch.eye(nv) - d11 - d11.mH
    zeros_xv = torch.zeros(nx, nv)
    h = torch.cat(
        (
            torch.cat((eye_x, zeros_xv, f.mH), dim=1),
            torch.cat((zeros_xv.mH, h22, zeros_xv.mH), dim=1),
            torch.cat((f, zeros_xv, eye_x), dim=1),
        ),
        dim=0,
    )
    return torch.linalg.cholesky(h + eps * torch.eye(spec.n_h)).mH


def _check_qsr(q: Tensor, s: Tensor, r: Tensor, nu: int, ny: int) -> None:
    for name, m, shape in (("Q", q, (ny, ny)), ("S", s, (nu, ny)), ("R", r, (nu, nu))):
        if m.shape != shape:
            raise ValueError(f"expected {name} of shape {shape}, got {tuple(m.shape)}")
    if not _is_posdef(-q):
        raise ValueError("Q must be symmetric negative definite")
    if not _is_posdef(r - s @ torch.linalg.solve(q, s.mH)):
        raise ValueError("R - S Q⁻¹ Sᵀ must be symmetric positive definite")


# Fused-kernel backends, resolved through dispatch.resolve: each module exposes
# supports(spec, u, x0) -> str | None plus forward_infer/forward_train/backward on the
# ExplicitREN.tensors parameter order. The kernels never see the certificate construction.
_FUSED = {
    "triton": "tsfast.models.architectures.ren.backend_triton",
    "c": "tsfast.models.architectures.ren.backend_c",
}
_OP_AUTO = {"cuda": ("triton",), "cpu": ("c",)}
_CORE_AUTO = {"cuda": ("triton",)}


@torch._dynamo.assume_constant_result
def _rollout_mode(backend: str, spec: RENSpec, u: Tensor, x0: Tensor) -> str:
    """Execution mode for this call: ``"eager"``, ``"compiled"``, or ``"fused"``.

    ``"auto"`` defers to the process preference, and a ``"reference"`` preference disables
    fused kernels even for instances that request one. The non-fused fallback is always
    eager: ``torch.compile`` unrolls ``seq * n_nl`` nodes, which is only viable on short
    sequences, so it never gets picked implicitly.
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
            mod = resolve("ren.rollout", _FUSED, _CORE_AUTO.get(u.device.type, ()), (spec, u, x0), requested=backend)
            return "fused" if mod is not None else "eager"
        case unknown:
            raise ValueError(f"unknown backend {unknown!r}")


def _fused_module(spec: RENSpec, u: Tensor, x0: Tensor):
    """The fused backend module serving this op call, honoring the process preference."""
    order = _OP_AUTO.get(u.device.type, ())
    mod = resolve("ren.rollout", _FUSED, order, (spec, u, x0))
    if mod is None:
        mod = resolve("ren.rollout", _FUSED, order, (spec, u, x0), requested="auto")
    if mod is None:
        raise RuntimeError(f"no fused REN backend usable for device {u.device.type!r}; use backend='eager'")
    return mod


def rollout_unsupported(spec: RENSpec, u: Tensor, x0: Tensor, device_type: str, dtypes: tuple) -> str | None:
    """Device/dtype/shape screen shared by the fused backends' ``supports``: reason or None."""
    if u.device.type != device_type:
        return f"input on {u.device.type}, this backend requires {device_type}"
    if u.dtype not in dtypes or x0.dtype != u.dtype:
        return f"requires {'/'.join(str(d).removeprefix('torch.') for d in dtypes)}, got u={u.dtype}, x0={x0.dtype}"
    if u.dim() != 3 or u.shape[-1] != spec.n_input:
        return f"expected u of shape [B, L, {spec.n_input}], got {tuple(u.shape)}"
    if x0.shape != (u.shape[0], spec.n_state):
        return f"expected x0 of shape [{u.shape[0]}, {spec.n_state}], got {tuple(x0.shape)}"
    if spec.act not in _ACTS:
        return f"activation {spec.act!r} not implemented in the fused kernels"
    return None


# ------------------------------------------------------------------------- custom ops
#
# The rollout is exposed as torch.library custom ops so it composes with torch.compile
# (no graph breaks), fake/meta tracing, and export. Spec fields cross the op boundary as
# scalars (frozen dataclasses cannot) and are rebuilt inside the impls. The backward is
# its own registered op so compiled autograd also sees no graph break. Tensor inputs may
# arrive as non-contiguous views, and the kernels index raw data pointers, so every impl
# materializes its inputs.


def _spec_from(n_state: int, n_input: int, n_output: int, n_nl: int, act: str) -> RENSpec:
    # variant and alpha are certificate properties; nothing below this boundary reads them
    return RENSpec(n_state, n_input, n_output, n_nl, "contracting", 1.0, act)


@torch.library.custom_op("tsfast::ren_rollout", mutates_args=())
def _ren_rollout(
    u: Tensor, x0: Tensor, params: list[Tensor], n_state: int, n_input: int, n_output: int, n_nl: int, act: str
) -> tuple[Tensor, Tensor]:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, n_output, n_nl, act)
    return _fused_module(spec, u, x0).forward_infer(spec, u, x0, params)


@_ren_rollout.register_fake
def _(u, x0, params, n_state, n_input, n_output, n_nl, act):
    return u.new_empty(u.shape[0], u.shape[1], n_output), u.new_empty(u.shape[0], n_state)


@torch.library.custom_op("tsfast::ren_rollout_train", mutates_args=())
def _ren_rollout_train(
    u: Tensor, x0: Tensor, params: list[Tensor], n_state: int, n_input: int, n_output: int, n_nl: int, act: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    u, x0 = u.contiguous(), x0.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, n_output, n_nl, act)
    return _fused_module(spec, u, x0).forward_train(spec, u, x0, params)


@_ren_rollout_train.register_fake
def _(u, x0, params, n_state, n_input, n_output, n_nl, act):
    b, ln = u.shape[0], u.shape[1]
    return (
        u.new_empty(b, ln, n_output),
        u.new_empty(b, n_state),
        u.new_empty(b, ln, n_state),
        u.new_empty(b, ln, n_nl),
    )


@torch.library.custom_op("tsfast::ren_rollout_bwd", mutates_args=())
def _ren_rollout_bwd(
    grad_y: Tensor | None,
    grad_xl: Tensor | None,
    u: Tensor,
    xs: Tensor,
    ws: Tensor,
    params: list[Tensor],
    n_state: int,
    n_input: int,
    n_output: int,
    n_nl: int,
    act: str,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    u, xs, ws = u.contiguous(), xs.contiguous(), ws.contiguous()
    params = [p.contiguous() for p in params]
    spec = _spec_from(n_state, n_input, n_output, n_nl, act)
    gy = grad_y.contiguous() if grad_y is not None else u.new_zeros(u.shape[0], u.shape[1], n_output)
    gxl = grad_xl.contiguous() if grad_xl is not None else u.new_zeros(u.shape[0], n_state)
    x0 = xs[:, 0].contiguous()
    lam, gv, gx0 = _fused_module(spec, u, x0).backward(spec, gy, gxl, xs, ws, params)
    grads, du = ren_param_grads(spec, u, xs, ws, gy, lam, gv, params)
    # BLAS is free to hand back a transposed view of a GEMM result; the fake kernel below
    # promises contiguous outputs, and inductor asserts on the mismatch.
    return du.contiguous(), gx0.contiguous(), [g.contiguous() for g in grads]


@_ren_rollout_bwd.register_fake
def _(grad_y, grad_xl, u, xs, ws, params, n_state, n_input, n_output, n_nl, act):
    # new_empty, not empty_like: the explicit matrices reach the op as strided views of the
    # solve that produced them, while the gradients the impl returns are contiguous
    return u.new_empty(u.shape), u.new_empty(u.shape[0], n_state), [p.new_empty(p.shape) for p in params]


def _train_setup(ctx, inputs, output):
    u, x0, params, n_state, n_input, n_output, n_nl, act = inputs
    _, _, xs, ws = output
    ctx.fields = (n_state, n_input, n_output, n_nl, act)
    ctx.save_for_backward(u, xs, ws, *params)


def _train_backward(ctx, grad_y, grad_xl, grad_xs, grad_ws):
    saved = ctx.saved_tensors
    u, xs, ws = saved[0], saved[1], saved[2]
    params = list(saved[3:])
    du, dx0, dparams = _ren_rollout_bwd(grad_y, grad_xl, u, xs, ws, params, *ctx.fields)
    return du, dx0, list(dparams), None, None, None, None, None


_ren_rollout_train.register_autograd(_train_backward, setup_context=_train_setup)


def ren_param_grads(
    spec: RENSpec, u: Tensor, xs: Tensor, ws: Tensor, gy: Tensor, lam: Tensor, gv: Tensor, params: list[Tensor]
) -> tuple[list[Tensor], Tensor]:
    """Parameter and input gradients of the rollout as batched GEMMs over the step samples.

    The state-adjoint recurrence (``lam``, ``gv``) is the only sequential part of BPTT and
    is what a backend computes; every parameter gradient is then a plain reduction over all
    ``B*L`` step samples, which is exactly the batched GEMM BLAS is built for. See
    ``MATH_REN.md`` §2.2.

    Args:
        u: input sequence ``[B, L, n_input]``.
        xs: states fed into each step ``[B, L, n_state]`` (``xs[t] = x_t``).
        ws: equilibrium-layer activations ``[B, L, n_nl]``.
        gy: output adjoints ``[B, L, n_output]``.
        lam: next-state adjoints ``[B, L, n_state]`` (``lam[t] = ∂L/∂x_{t+1}``).
        gv: pre-activation adjoints of the equilibrium layer ``[B, L, n_nl]``.
        params: the explicit realization in ``ExplicitREN.tensors`` order.

    Returns:
        ``(grads, du)`` with grads in the same order as ``params``.
    """
    b, ln = u.shape[0], u.shape[1]
    uf = u.reshape(b * ln, spec.n_input)
    xf = xs.reshape(b * ln, spec.n_state)
    wf = ws.reshape(b * ln, spec.n_nl)
    gyf = gy.reshape(b * ln, spec.n_output)
    lf = lam.reshape(b * ln, spec.n_state)
    gvf = gv.reshape(b * ln, spec.n_nl)
    grads = [
        lf.t() @ xf,  # A
        lf.t() @ wf,  # B1
        lf.t() @ uf,  # B2
        gvf.t() @ xf,  # C1
        # entries on and above the diagonal are structurally absent from the forward
        torch.tril(gvf.t() @ wf, -1),  # D11
        gvf.t() @ uf,  # D12
        gyf.t() @ xf,  # C2
        gyf.t() @ wf,  # D21
        gyf.t() @ uf,  # D22
        lf.sum(0),  # bx
        gvf.sum(0),  # bv
        gyf.sum(0),  # by
    ]
    du = (lf @ params[2] + gvf @ params[5] + gyf @ params[8]).reshape(b, ln, spec.n_input)
    return grads, du


def fused_rollout(spec: RENSpec, u: Tensor, x0: Tensor, e: ExplicitREN) -> tuple[Tensor, Tensor]:
    """Run the rollout through the fused-kernel custom ops (autograd-capable).

    Picks the training op (which stores the states and equilibrium activations for the
    analytic BPTT backward) when gradients are live, else the inference op, which keeps no
    intermediates.

    Returns:
        ``(y, x_L)`` with ``y`` shaped ``[B, L, n_output]``.
    """
    params = e.tensors
    fields = (spec.n_state, spec.n_input, spec.n_output, spec.n_nl, spec.act)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (u, x0, *params)):
        y, x_last, _, _ = _ren_rollout_train(u, x0, params, *fields)
        return y, x_last
    return _ren_rollout(u, x0, params, *fields)


class RENCore(nn.Module):
    """Explicit realization plus the sequential rollout over an input sequence.

    Holds the free parameters (in :attr:`parameterization`) but evaluates only through
    :class:`~.common.ExplicitREN` tensors, which is the seam a fused kernel would attach
    to: the rollout has no opinion about how the matrices were certified.

    Args:
        spec: static architecture description.
        **kwargs: forwarded to :class:`RENParameterization`.
    """

    def __init__(self, spec: RENSpec, **kwargs: Any):
        super().__init__()
        if spec.act not in _ACTS:
            raise ValueError(f"unknown activation {spec.act!r}, expected one of {sorted(_ACTS)}")
        if not 0.0 < spec.alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0, 1], got {spec.alpha}")
        self.spec = spec
        self.parameterization = RENParameterization(spec, **kwargs)
        self._act = _ACTS[spec.act]
        self._cache: tuple[tuple, ExplicitREN] | None = None

    def explicit(self) -> ExplicitREN:
        """The explicit realization, rebuilt on demand and cached while gradients are off.

        The construction costs a few matrix products and two solves on ``(2nx+nv)``-sized
        matrices — irrelevant next to an ``L``-step rollout during training, but worth
        caching for repeated inference from fixed weights.
        """
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return self.parameterization()
        key = self.parameterization.cache_key()
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.parameterization())
        return self._cache[1]

    def rollout(self, e: ExplicitREN, u: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
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
            w = equilibrium_sweep(x @ e.C1.mH + bv[:, t], e.D11, self._act)
            ys.append(x @ e.C2.mH + w @ e.D21.mH + by[:, t])
            x = x @ e.A.mH + w @ e.B1.mH + bx[:, t]
        return torch.stack(ys, dim=1), x


class REN(nn.Module):
    """Recurrent equilibrium network: a sequence model that is contracting by construction.

    Trained with plain SGD from any initialization, the model satisfies its certificate at
    every step of training, because the certificate is built into the map from free
    parameters to model matrices rather than enforced on top of it.

    Three variants, differing only in how the certificate matrix is assembled:

    - ``"contracting"``: two trajectories under the same input converge at rate ``alpha``.
    - ``"lipschitz"``: additionally ``‖y(u) - y(ũ)‖ ≤ gamma ‖u - ũ‖`` in truncated ``ℓ2``
      **from a common initial state**. ``gamma`` is a runtime scalar and may be reassigned.
    - ``"dissipative"``: additionally satisfies the incremental IQC given by ``qsr``.

    What the Lipschitz certificate buys, precisely: an input perturbation of energy ``δ``
    moves the output by at most ``gamma·δ``, measured in ``ℓ2`` over the horizon and
    starting from the same state. It says nothing about model-vs-plant error — a REN with
    ``gamma = 1`` can be an arbitrarily bad model of a system, certified smooth and stable
    rather than correct.

    The rollout is irreducibly sequential twice over — once along the sequence, once along
    the ``n_nl`` neurons of the equilibrium layer — so a naive Python loop is dispatch-bound
    by a wide margin. Several backends implement the identical recurrence:

    - ``"eager"``: nested Python loop — the reference implementation, any device and dtype.
    - ``"c"``: generated C++ rollout with a fused BPTT backward, batch-parallel via the
      ATen thread pool — float32 and float64 on CPU, and the ``gradcheck`` vehicle.
    - ``"triton"``: persistent per-trajectory GPU kernel with a fused BPTT backward —
      float32 on CUDA, within the size caps its ``fits`` reports.
    - ``"compiled"``: ``torch.compile`` over the unrolled loop. Only usable on short
      sequences — the graph has ``seq * n_nl`` nodes and compiles at roughly 0.2 s per
      node — so it is never selected implicitly.
    - ``"auto"``: defers to the process-wide preference (``tsfast.models.set_backend`` /
      ``use_backend``); under an ``"auto"`` preference picks ``triton`` where it applies and
      eager elsewhere (select ``"c"`` explicitly to trade a one-time compilation for much
      faster CPU training). A ``"reference"`` preference forces the eager path everywhere.

    All backends share the same parameters, so the backend can be switched at any time via
    the ``backend`` attribute. The fused backends run as registered ``torch.library`` custom
    ops with analytic-BPTT backward ops (``MATH_REN.md``), so they are loss-agnostic and
    compose with ``torch.compile``. They consume the explicit realization as plain tensors
    and know nothing about the certificate, which is built in ordinary autograd above them.

    Contraction is a prior about the plant, not free insurance, and it fits some systems
    badly. On ``benchmarks/gate_ren.py`` at matched parameter count it wins clearly on
    ``CascadedTanks`` (0.37 NRMSE against 0.60 for a GRU, and 0.10 against 0.23 with both
    under ``FranSys``), ties on ``Silverbox``, and loses by roughly a quarter on ``WH`` —
    where :class:`~.r2dn.R2DN` reaches 0.037 at the same budget and closes the gap, so that
    loss is the equilibrium layer's rather than contraction's. It loses badly on ``EMPS``, by
    4.2x under ``FranSys`` — that plant is friction-dominated, and stick-slip is exactly the
    behaviour a contraction certificate excludes, since trajectories in a stick phase do not
    converge. **Do not reach for this model on stick-slip or hysteretic systems**; the
    guarantee it offers is one those dynamics cannot satisfy in the first place.

    ``gamma`` is not a free addition either. It costs nothing on ``WH`` (0.037, matching the
    contracting model) but a third of the accuracy on ``CascadedTanks`` (0.43 against 0.37),
    where the gain budget binds against dynamics that need the range. What it buys is a
    certificate that is tight on trained models — ``gamma_empirical/gamma_certified`` in
    0.37-0.84 across the suite — rather than the orders-of-magnitude slack typical of
    post-hoc bounds on freely-trained networks.

    Contraction makes the initial state self-correcting at rate ``alpha``, so ``x0=None``
    (zeros) plus ``n_skip`` is usually enough. But the forgetting time ``≈ 1/(1-alpha)``
    *is* the longest time constant the model can represent, so no ``alpha`` both forgets
    ``x0`` quickly and represents an integrator. For integrating plants (position from
    velocity, tank level, thermal accumulation) use ``return_state=True`` and compose with
    :class:`~tsfast.prediction.fransys.FranSys`, which estimates ``x0`` from an ``(u, y)``
    window instead of asking the dynamics to forget it. Note that the Lipschitz bound is
    stated for a fixed initial state and does not survive that composition.

    Args:
        n_input: exogenous input dimension.
        n_output: observed output dimension.
        n_state: state dimension.
        n_nl: neurons in the equilibrium layer; the model's nonlinear capacity.
        variant: ``"contracting"``, ``"lipschitz"`` or ``"dissipative"``.
        alpha: contraction rate in ``(0, 1]``. ``1.0`` admits arbitrarily long memory.
        gamma: certified incremental ``ℓ2`` gain, for ``variant="lipschitz"``.
        qsr: ``(Q, S, R)`` supply-rate matrices, required for ``variant="dissipative"``.
        act: equilibrium-layer activation, one of ``tanh``, ``relu``, ``sigmoid``; must be
            monotone and slope-restricted to ``[0, 1]``, which all three are.
        eps: regularization floor on the certificate matrix.
        polar: use the polar parameterization of ``H``.
        init: ``"long_memory"`` (default) or ``"random"``; see :class:`RENParameterization`.
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
        variant: str = "contracting",
        alpha: float = 1.0,
        gamma: float = 1.0,
        qsr: tuple[Tensor, Tensor, Tensor] | None = None,
        act: str = "tanh",
        eps: float = _EPS,
        polar: bool = True,
        init: str = "long_memory",
        backend: str = "auto",
        return_state: bool = False,
    ):
        super().__init__()
        if variant not in ("contracting", "lipschitz", "dissipative"):
            raise ValueError(f"unknown variant {variant!r}")
        spec = RENSpec(n_state, n_input, n_output, n_nl, variant, alpha, act)
        self.core = RENCore(spec, gamma=gamma, qsr=qsr, eps=eps, polar=polar, init=init)
        self.backend = backend
        self.return_state = return_state
        self._compiled_rollout = None

    @property
    def spec(self) -> RENSpec:
        return self.core.spec

    @property
    def gamma(self) -> float:
        """Certified incremental ``ℓ2`` gain; reassign to retune the certificate."""
        return self.core.parameterization.gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self.core.parameterization.gamma = value

    def forward(self, u: Tensor, x0: Tensor | None = None, state: dict | None = None) -> Tensor | tuple[Tensor, dict]:
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
