"""Tests for the fused R2DN rollout backend (triton).

Correctness contract in ``tsfast/models/architectures/ren/MATH_R2DN.md``. There is no
fp64-capable R2DN kernel to gradcheck against — the eager path carries that duty in
``test_r2dn.py`` — so parity against eager autograd in float32 is what validates the analytic
BPTT here, across every activation and depth, plus the certificate itself measured through
the fused rollout.

The kernels consume only the explicit realization, so these tests deliberately feed them
matrices and layers that no certificate produced — the rollout must be right on its own
terms, not merely on the ones the parameterization happens to hand it.
"""

import pytest
import torch

from tsfast.models.architectures.ren import R2DN
from tsfast.models.architectures.ren.lbdn import ExplicitSandwich, lbdn_forward
from tsfast.models.architectures.ren.r2dn import ExplicitR2DN, R2DNSpec, fused_rollout

#: ``(nu, ny, nx, nv, hidden, act)``
CONFIGS = [
    (2, 3, 4, 8, (8, 8), "tanh"),
    (1, 1, 3, 5, (6,), "relu"),
    (3, 2, 6, 16, (12, 10, 8), "sigmoid"),
    (2, 2, 8, 32, (32, 32), "relu"),
    (4, 1, 2, 7, (5, 9), "tanh"),
    (2, 2, 4, 6, (), "tanh"),  # depthless: the network is the affine output layer alone
]


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _random_explicit(nu, ny, nx, nv, hidden, device="cuda", dtype=torch.float32, seed=0):
    """An explicit realization drawn at random, not built by the certificate construction.

    The sandwich layers get nonzero biases on purpose: with the zero bias the constructor
    starts from, ``relu``'s positive homogeneity makes ``Ψ relu(Ψ⁻¹ z) = relu(z)``, so the
    layer is genuinely independent of ``psi`` and the ``psi`` gradient no longer tests
    anything.

    Entries are scaled by fan-in so every matvec has gain ``~0.4`` whatever the width. No
    certificate is involved — the point is only that a rollout of an *arbitrary* realization
    with ``relu`` and loop gain above one overflows float32 within a few steps, and two paths
    that agree exactly up to the overflow then compare ``nan`` against ``nan``.
    """
    torch.manual_seed(seed)

    def r(*shape):
        scale = 0.4 / (shape[-1] ** 0.5) if len(shape) > 1 else 0.4
        return (torch.randn(*shape, dtype=dtype, device=device) * scale).requires_grad_()

    net, widths = [], [nv, *hidden, nv]
    for k, (i, o) in enumerate(zip(widths[:-1], widths[1:])):
        if k == len(hidden):
            net.append(ExplicitSandwich(B=r(o, i), bias=r(o)))
        else:
            net.append(
                ExplicitSandwich(
                    B=r(o, i), bias=r(o), A=r(o, o), psi=(0.5 + torch.rand(o, device=device)).requires_grad_()
                )
            )
    return ExplicitR2DN(
        A=r(nx, nx), B1=r(nx, nv), B2=r(nx, nu), C1=r(nv, nx), C2=r(ny, nx), D12=r(nv, nu),
        D21=r(ny, nv), D22=r(ny, nu), bx=r(nx), bv=r(nv), by=r(ny), net=tuple(net),
    )  # fmt: skip


def _eager_roll(spec, e, u, x0):
    """The reference rollout, written out here so the test does not lean on R2DNCore."""
    act = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[spec.act]
    x, outs = x0, []
    for t in range(u.shape[1]):
        w = lbdn_forward(e.net, x @ e.C1.mH + u[:, t] @ e.D12.mH + e.bv, act)
        outs.append(x @ e.C2.mH + w @ e.D21.mH + u[:, t] @ e.D22.mH + e.by)
        x = x @ e.A.mH + w @ e.B1.mH + u[:, t] @ e.B2.mH + e.bx
    return torch.stack(outs, dim=1), x


def _leaves(e: ExplicitR2DN):
    """Fresh differentiable copies of every tensor in a realization, and the realization."""
    lin = [t.detach().clone().requires_grad_() for t in e.tensors]
    net, flat = [], []
    for layer in e.net:
        parts = [
            t.detach().clone().requires_grad_() for t in (layer.B, layer.bias, layer.A, layer.psi) if t is not None
        ]
        flat += parts
        net.append(
            ExplicitSandwich(*parts) if len(parts) == 2 else ExplicitSandwich(parts[0], parts[1], parts[2], parts[3])
        )
    return ExplicitR2DN(*lin, tuple(net)), [*lin, *flat]


def _run(spec, e, u, x0, fused: bool):
    """Forward + backward through both outputs; returns (y, x_L, all grads)."""
    uu, xx = u.clone().requires_grad_(), x0.clone().requires_grad_()
    ex, params = _leaves(e)
    y, x_last = fused_rollout(spec, uu, xx, ex) if fused else _eager_roll(spec, ex, uu, xx)
    ((y**2).mean() + y.abs().sum() * 0.01 + (x_last**2).sum() * 0.03).backward()
    return y.detach(), x_last.detach(), [t.grad.clone() for t in (uu, xx, *params)]


def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    return prev


def _requires_triton():
    from tsfast.models.architectures.ren import r2dn_backend_triton

    if not r2dn_backend_triton.is_available():
        pytest.skip("no CUDA / triton")


class TestTritonBackend:
    @pytest.mark.parametrize("nu,ny,nx,nv,hidden,act", CONFIGS)
    def test_parity_float32(self, nu, ny, nx, nv, hidden, act):
        _requires_triton()
        from tsfast.models import use_backend

        spec = R2DNSpec(nx, nu, ny, nv, hidden, "contracting", 1.0, act)
        e = _random_explicit(nu, ny, nx, nv, hidden)
        u = torch.randn(5, 23, nu, device="cuda")
        x0 = torch.randn(5, nx, device="cuda")
        prev = _no_tf32()
        try:
            with use_backend("reference"):
                y_e, xl_e, g_e = _run(spec, e, u, x0, fused=False)
            with use_backend("triton"):
                y_f, xl_f, g_f = _run(spec, e, u, x0, fused=True)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        assert _rel(y_f, y_e) < 5e-5
        assert _rel(xl_f, xl_e) < 5e-5
        # gradients are compared against the whole realization's scale: a layer whose true
        # gradient is zero would otherwise be judged against its own rounding noise
        scale = max(g.abs().max().item() for g in g_e)
        assert max((a - b).abs().max().item() for a, b in zip(g_f, g_e)) < 5e-5 * scale

    def test_inference_path_matches(self):
        """The no-grad op keeps no tape; it must still roll the same trajectory."""
        _requires_triton()
        from tsfast.models import use_backend

        spec = R2DNSpec(4, 2, 3, 8, (8, 8), "contracting", 1.0, "tanh")
        e = _random_explicit(2, 3, 4, 8, (8, 8))
        u = torch.randn(3, 11, 2, device="cuda")
        x0 = torch.randn(3, 4, device="cuda")
        with torch.no_grad():
            ref = _eager_roll(spec, e, u, x0)
            with use_backend("triton"):
                got = fused_rollout(spec, u, x0, e)
        assert _rel(got[0], ref[0]) < 5e-5 and _rel(got[1], ref[1]) < 5e-5

    def test_fit_envelope(self):
        from tsfast.models.architectures.ren.r2dn_backend_triton import fits

        base = dict(variant="contracting", alpha=1.0, act="tanh")
        assert fits(R2DNSpec(8, 2, 2, 32, (32, 32), **base))
        assert not fits(R2DNSpec(8, 2, 2, 256, (8,), **base))  # padded n_nl over cap
        assert not fits(R2DNSpec(128, 2, 2, 8, (8,), **base))  # padded n_state over cap
        assert not fits(R2DNSpec(8, 2, 2, 64, (128, 128, 128), **base))  # total tile area over cap
        assert not fits(R2DNSpec(8, 2, 2, 8, (8,), variant="contracting", alpha=1.0, act="elu"))


class TestModelIntegration:
    """The paths a user actually hits: R2DN.forward picking a backend, and training through it."""

    def test_model_parity(self):
        _requires_triton()
        torch.manual_seed(0)
        m = R2DN(2, 2, n_state=6, n_nl=16, depth=2, variant="lipschitz", gamma=2.0, backend="eager", return_state=True)
        m = m.cuda()
        for layer in m.core.parameterization.net.layers:
            torch.nn.init.normal_(layer.b, std=0.3)
        u = torch.randn(4, 30, 2, device="cuda")
        x0 = torch.randn(4, 6, device="cuda")

        def run(bk):
            m.backend = bk
            for p in m.parameters():
                p.grad = None
            uu, xx = u.clone().requires_grad_(), x0.clone().requires_grad_()
            y, st = m(uu, xx)
            ((y**2).mean() + (st["x"] ** 2).sum() * 0.03).backward()
            return y.detach(), st["x"].detach(), [p.grad.clone() for p in m.parameters()], uu.grad, xx.grad

        prev = _no_tf32()
        try:
            y_e, s_e, g_e, du_e, dx_e = run("eager")
            y_f, s_f, g_f, du_f, dx_f = run("triton")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        assert _rel(y_f, y_e) < 5e-5 and _rel(s_f, s_e) < 5e-5
        scale = max(g.abs().max().item() for g in g_e)
        assert max((a - b).abs().max().item() for a, b in zip(g_f, g_e)) < 5e-5 * scale
        assert _rel(du_f, du_e) < 5e-5 and _rel(dx_f, dx_e) < 5e-5

    def test_stateful_chunked_equivalence(self):
        _requires_triton()
        torch.manual_seed(0)
        m = R2DN(2, 1, n_state=3, n_nl=6, depth=2, backend="triton", return_state=True).cuda()
        u = torch.randn(4, 30, 2, device="cuda")
        full, state_full = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:25], state=state)
        out3, state = m(u[:, 25:], state=state)
        assert _rel(torch.cat((out1, out2, out3), dim=1), full) < 5e-6
        assert _rel(state["x"], state_full["x"]) < 5e-6

    def test_certificate_holds_through_the_fused_path(self):
        """A kernel that mis-rolled the dynamics would break the contraction it is blind to."""
        _requires_triton()
        torch.manual_seed(0)
        m = R2DN(2, 3, n_state=4, n_nl=8, depth=2, backend="triton", return_state=True).cuda()
        with torch.no_grad():
            for p in m.core.parameterization.parameters():
                p.copy_(torch.randn_like(p) * 3)
        u = torch.randn(6, 40, 2, device="cuda")
        xa, xb = torch.randn(6, 4, device="cuda") * 3, torch.randn(6, 4, device="cuda") * 3
        p = m.core.parameterization
        h = p.hmatrix().detach().double()
        e = (h[:4, :4] + h[4:, 4:] + p.Y.detach().double() - p.Y.detach().double().mH) / 2
        metric = e.mH @ torch.linalg.solve(h[4:, 4:], e)
        lyap = lambda d: torch.einsum("bi,ij,bj->b", d.double(), metric, d.double())  # noqa: E731
        # the floor is set once, against the *initial* separation: these parameters contract
        # hard, and once V has fallen ~10 orders it is float32 rounding rather than dynamics
        floor = lyap(xa - xb).max().item() * 1e-10
        with torch.no_grad():
            for t in range(u.shape[1]):
                _, na = m(u[:, t : t + 1], xa)
                _, nb = m(u[:, t : t + 1], xb)
                v0, v1 = lyap(xa - xb), lyap(na["x"] - nb["x"])
                live = v0 > floor
                if not live.any():
                    break
                assert (v1[live] / v0[live]).max().item() <= 1.0 + 1e-4
                xa, xb = na["x"], nb["x"]

    def test_compile_fullgraph_parity(self):
        _requires_triton()
        torch.manual_seed(0)
        m = R2DN(2, 2, n_state=4, n_nl=8, depth=2, backend="triton").cuda()
        u, x0 = torch.randn(3, 12, 2, device="cuda"), torch.randn(3, 4, device="cuda")
        eager = m(u, x0)
        compiled = torch.compile(m, fullgraph=True)(u, x0)
        assert _rel(compiled, eager) < 5e-6

    def test_learner_fit(self, dls_simulation):
        _requires_triton()
        from tsfast.training import R2DNLearner

        lrn = R2DNLearner(dls_simulation, n_state=4, n_nl=8, depth=2, backend="triton", show_bar=False)
        lrn.fit(1, 3e-3)
        assert len(lrn.recorder) == 1
