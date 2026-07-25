"""Tests for the fused REN rollout backends (c, triton).

Correctness contract in ``tsfast/models/architectures/ren/MATH_REN.md``. The C backend is fp64-gradcheckable and is the
reference the Triton backend is validated against on GPU. Both run behind the
``tsfast::ren_rollout*`` custom ops, entered through ``fused_rollout``.

The kernels consume only the explicit realization, so these tests deliberately feed them
matrices that no certificate produced — the rollout must be right on its own terms, not
merely on the ones the parameterization happens to hand it.
"""

import pytest
import torch

from tsfast.models.architectures.ren import REN, RENSpec, equilibrium_sweep
from tsfast.models.architectures.ren.common import ExplicitREN
from tsfast.models.architectures.ren.core import fused_rollout

CONFIGS = [
    (2, 3, 4, 8, "tanh"),
    (1, 1, 3, 5, "tanh"),
    (3, 2, 6, 16, "relu"),
    (2, 2, 8, 32, "sigmoid"),
    (4, 1, 2, 7, "relu"),
]


def _rel(a, b):
    if b.numel() == 0:  # Z3 is empty whenever n_input == n_output
        return 0.0
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _random_explicit(nu, ny, nx, nv, dtype=torch.float64, device="cpu", seed=0):
    """An explicit realization drawn at random, not built by the certificate construction."""
    torch.manual_seed(seed)

    def r(*shape):
        return (torch.randn(*shape, dtype=dtype, device=device) * 0.4).requires_grad_()

    d11 = torch.tril(torch.randn(nv, nv, dtype=dtype, device=device) * 0.4, -1).requires_grad_()
    return ExplicitREN(
        A=r(nx, nx), B1=r(nx, nv), B2=r(nx, nu), C1=r(nv, nx), D11=d11, D12=r(nv, nu),
        C2=r(ny, nx), D21=r(ny, nv), D22=r(ny, nu), bx=r(nx), bv=r(nv), by=r(ny),
    )  # fmt: skip


def _eager_roll(spec, e, u, x0):
    """The reference rollout, written out here so the test does not lean on RENCore."""
    act = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[spec.act]
    x, outs = x0, []
    for t in range(u.shape[1]):
        w = equilibrium_sweep(x @ e.C1.mH + u[:, t] @ e.D12.mH + e.bv, e.D11, act)
        outs.append(x @ e.C2.mH + w @ e.D21.mH + u[:, t] @ e.D22.mH + e.by)
        x = x @ e.A.mH + w @ e.B1.mH + u[:, t] @ e.B2.mH + e.bx
    return torch.stack(outs, dim=1), x


def _run(spec, e, u, x0, fused: bool):
    """Forward + backward through both outputs; returns (y, x_L, all grads)."""
    leaves = [
        u.clone().requires_grad_(),
        x0.clone().requires_grad_(),
        *(t.detach().clone().requires_grad_() for t in e.tensors),
    ]
    ex = ExplicitREN(*leaves[2:])
    if fused:
        y, x_last = fused_rollout(spec, leaves[0], leaves[1], ex)
    else:
        y, x_last = _eager_roll(spec, ex, leaves[0], leaves[1])
    loss = (y**2).mean() + y.abs().sum() * 0.01 + (x_last**2).sum() * 0.03
    loss.backward()
    return y.detach(), x_last.detach(), [t.grad.clone() for t in leaves]


def _assert_parity(nu, ny, nx, nv, act, backend, device, dtype, tol):
    spec = RENSpec(nx, nu, ny, nv, "contracting", 1.0, act)
    e = _random_explicit(nu, ny, nx, nv, dtype=dtype, device=device)
    u = torch.randn(5, 23, nu, dtype=dtype, device=device)
    x0 = torch.randn(5, nx, dtype=dtype, device=device)
    from tsfast.models import use_backend

    with use_backend("reference"):
        y_e, xl_e, g_e = _run(spec, e, u, x0, fused=False)
    with use_backend(backend):
        y_f, xl_f, g_f = _run(spec, e, u, x0, fused=True)
    assert _rel(y_f, y_e) < tol
    assert _rel(xl_f, xl_e) < tol
    assert max(_rel(a, b) for a, b in zip(g_f, g_e)) < tol


class TestCBackend:
    @pytest.mark.parametrize("nu,ny,nx,nv,act", CONFIGS)
    def test_parity_float64(self, nu, ny, nx, nv, act):
        from tsfast.models.architectures.ren import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        _assert_parity(nu, ny, nx, nv, act, "c", "cpu", torch.float64, 1e-12)

    def test_parity_float32(self):
        from tsfast.models.architectures.ren import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        _assert_parity(2, 3, 4, 8, "tanh", "c", "cpu", torch.float32, 5e-5)

    def test_gradcheck(self):
        """fp64 gradcheck straight through the custom op, the point of a double-capable kernel."""
        from tsfast.models.architectures.ren import backend_c
        from tsfast.models import use_backend

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        spec = RENSpec(3, 1, 2, 4, "contracting", 1.0, "tanh")
        e = _random_explicit(1, 2, 3, 4)
        u = torch.randn(2, 5, 1, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

        def f(u, x0, *params):
            with use_backend("c"):
                return fused_rollout(spec, u, x0, ExplicitREN(*params))

        assert torch.autograd.gradcheck(f, (u, x0, *e.tensors), eps=1e-6, atol=1e-7)

    def test_inference_path_matches(self):
        from tsfast.models.architectures.ren import backend_c
        from tsfast.models import use_backend

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        spec = RENSpec(4, 2, 3, 8, "contracting", 1.0, "tanh")
        e = _random_explicit(2, 3, 4, 8)
        u = torch.randn(3, 11, 2, dtype=torch.float64)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        with torch.no_grad():
            ref = _eager_roll(spec, e, u, x0)
            with use_backend("c"):
                got = fused_rollout(spec, u, x0, e)
        assert _rel(got[0], ref[0]) < 1e-12 and _rel(got[1], ref[1]) < 1e-12


class TestTritonBackend:
    @pytest.mark.parametrize("nu,ny,nx,nv,act", CONFIGS)
    def test_parity_float32(self, nu, ny, nx, nv, act):
        from tsfast.models.architectures.ren import backend_triton

        if not backend_triton.is_available():
            pytest.skip("no CUDA / triton")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            _assert_parity(nu, ny, nx, nv, act, "triton", "cuda", torch.float32, 5e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def test_fit_envelope(self):
        from tsfast.models.architectures.ren.backend_triton import fits

        assert fits(RENSpec(8, 2, 2, 64, "contracting", 1.0, "tanh"))
        assert not fits(RENSpec(8, 2, 2, 256, "contracting", 1.0, "tanh"))  # padded n_nl over cap
        assert not fits(RENSpec(128, 2, 2, 8, "contracting", 1.0, "tanh"))  # padded n_state over cap
        assert not fits(RENSpec(64, 2, 2, 128, "contracting", 1.0, "tanh"))  # product over cap


class TestModelIntegration:
    """The paths a user actually hits: REN.forward picking a backend, and training through it."""

    def _model_parity(self, backend, device, tol):
        torch.manual_seed(0)
        m = REN(2, 2, n_state=6, n_nl=16, variant="lipschitz", gamma=2.0, backend="eager", return_state=True).to(device)
        u = torch.randn(4, 30, 2, device=device)
        x0 = torch.randn(4, 6, device=device)

        def run(bk):
            m.backend = bk
            for p in m.parameters():
                p.grad = None
            uu, xx = u.clone().requires_grad_(), x0.clone().requires_grad_()
            y, st = m(uu, xx)
            ((y**2).mean() + (st["x"] ** 2).sum() * 0.03).backward()
            return y.detach(), st["x"].detach(), [p.grad.clone() for p in m.parameters()], uu.grad, xx.grad

        y_e, s_e, g_e, du_e, dx_e = run("eager")
        y_f, s_f, g_f, du_f, dx_f = run(backend)
        assert _rel(y_f, y_e) < tol and _rel(s_f, s_e) < tol
        assert max(_rel(a, b) for a, b in zip(g_f, g_e)) < tol
        assert _rel(du_f, du_e) < tol and _rel(dx_f, dx_e) < tol

    def test_c_model_parity(self):
        from tsfast.models.architectures.ren import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        self._model_parity("c", "cpu", 5e-5)

    def test_triton_model_parity(self):
        from tsfast.models.architectures.ren import backend_triton

        if not backend_triton.is_available():
            pytest.skip("no CUDA / triton")
        self._model_parity("triton", "cuda", 5e-5)

    def test_certificate_holds_through_the_fused_path(self):
        """A kernel that mis-rolled the dynamics would break the contraction it is blind to."""
        from tsfast.models.architectures.ren import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        from test_ren import _worst_contraction_ratio

        torch.manual_seed(0)
        m = REN(2, 3, n_state=4, n_nl=8, backend="c").double()
        with torch.no_grad():
            for p in m.core.parameterization.parameters():
                p.copy_(torch.randn_like(p) * 3)
        assert _worst_contraction_ratio(m) <= 1.0 + 1e-9

    def test_stateful_chunked_equivalence(self):
        from tsfast.models.architectures.ren import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        torch.manual_seed(0)
        m = REN(2, 1, n_state=3, n_nl=6, backend="c", return_state=True).double()
        u = torch.randn(4, 30, 2, dtype=torch.float64)
        full, state_full = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:25], state=state)
        out3, state = m(u[:, 25:], state=state)
        assert _rel(torch.cat((out1, out2, out3), dim=1), full) < 1e-12
        assert _rel(state["x"], state_full["x"]) < 1e-12

    def test_compile_fullgraph_parity(self):
        from tsfast.models.architectures.ren import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        torch.manual_seed(0)
        m = REN(2, 2, n_state=4, n_nl=8, backend="c")
        u, x0 = torch.randn(3, 12, 2), torch.randn(3, 4)

        def run(model):
            for p in m.parameters():
                p.grad = None
            uu, xx = u.clone().requires_grad_(), x0.clone().requires_grad_()
            y = model(uu, xx)
            (y**2).mean().backward()
            return y.detach(), [p.grad.clone() for p in m.parameters()], uu.grad.clone(), xx.grad.clone()

        out_e, g_e, du_e, dx_e = run(m)
        out_c, g_c, du_c, dx_c = run(torch.compile(m, fullgraph=True))
        assert _rel(out_c, out_e) < 1e-5
        assert max(_rel(a, b) for a, b in zip(g_c, g_e)) < 1e-5
        assert _rel(du_c, du_e) < 1e-5 and _rel(dx_c, dx_e) < 1e-5
