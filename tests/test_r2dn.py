"""Tests for tsfast.models.architectures.ren.r2dn (R2DN, its LBDN, and their certificates).

The certificate tests are the distinctive ones. A direct parameterization claims its
guarantee holds at *every* parameter value, not merely at initialization or after a
well-behaved training run, so they check it at randomly drawn and deliberately badly-scaled
parameters — which is what an unconstrained optimizer eventually produces.
"""

import pytest
import torch


def _rel(a, b):
    if b.numel() == 0:  # the Cayley Z block is empty whenever the matrix is square
        return 0.0
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _model(nu=2, ny=3, nx=4, nv=6, depth=2, **kwargs):
    """A float64 R2DN; the certificate margins are far below float32 resolution."""
    from tsfast.models.architectures.ren import R2DN

    kwargs.setdefault("backend", "eager")
    return R2DN(nu, ny, n_state=nx, n_nl=nv, depth=depth, **kwargs).double()


def _perturb(m, dist, scale):
    """Move every free parameter, the way an optimizer eventually would."""
    with torch.no_grad():
        for p in m.core.parameterization.parameters():
            match dist:
                case "normal":
                    p.copy_(torch.randn_like(p) * scale)
                case "uniform":
                    p.copy_((torch.rand_like(p) * 2 - 1) * scale)
                case "rescale":
                    p.mul_(scale)
    return m


def _lyapunov_metric(m):
    """``M = Eᵀ P⁻¹ E``, the metric in which the contraction rate is exactly ``alpha``.

    The incremental step is ``E Δx⁺ = H21 Δx + B1 Δw`` with ``‖Δw‖ ≤ ‖Δv‖ = ‖C1 Δx‖``, since
    the network in the loop is 1-Lipschitz. Taking the Schur complement of ``H ≻ 0`` on its
    ``P = H22`` block — and using that ``P`` carries ``B1B1ᵀ`` and ``H11`` carries ``C1ᵀC1``,
    which is what the construction puts there — gives
    ``(EΔx⁺)ᵀP⁻¹(EΔx⁺) < ΔxᵀH11Δx + ‖Δv‖² - ‖Δw‖²``. Since ``EᵀP̃⁻¹E ⪰ E + Eᵀ - P̃`` for any
    ``P̃ ≻ 0``, taking ``P̃ = P/alpha²`` bounds the first term by ``alpha²·ΔxᵀMΔx``, so
    ``V = ΔxᵀMΔx`` decays at exactly ``alpha`` per step.
    """
    p = m.core.parameterization
    h = p.hmatrix()
    nx = m.spec.n_state
    h11, h22 = h[:nx, :nx], h[nx:, nx:]
    e = (h11 + h22 / m.spec.alpha**2 + p.Y - p.Y.mH) / 2
    return e.mH @ torch.linalg.solve(h22, e)


def _worst_contraction_ratio(m, steps=40, batch=6):
    """Largest per-step decay of the Lyapunov metric, relative to ``alpha``.

    Trajectories that have already merged carry no information about the rate — their
    separation is rounding noise — so they drop out of the measurement.
    """
    with torch.no_grad():
        metric = _lyapunov_metric(m)
        e = m.core.explicit()
        u = torch.randn(batch, steps, m.spec.n_input, dtype=torch.float64)
        xa = torch.randn(batch, m.spec.n_state, dtype=torch.float64) * 3
        xb = torch.randn(batch, m.spec.n_state, dtype=torch.float64) * 3
        v = lambda d: torch.einsum("bi,ij,bj->b", d, metric, d)  # noqa: E731
        floor = v(xa - xb).max() * 1e-18
        worst = 0.0
        for t in range(steps):
            _, xa_next = m.core.rollout(e, u[:, t : t + 1], xa)
            _, xb_next = m.core.rollout(e, u[:, t : t + 1], xb)
            v0, v1 = v(xa - xb), v(xa_next - xb_next)
            live = v0 > floor
            if live.any():
                worst = max(worst, (v1[live] / v0[live]).max().sqrt().item() / m.spec.alpha)
            xa, xb = xa_next, xb_next
        return worst


def _dissipation_residual(m):
    """Largest eigenvalue of the quadratic form the certificate needs negative semidefinite.

    Checks the guarantee from first principles, on the explicit realization alone. With
    ``V(Δx) = Δxᵀ M Δx`` the Lyapunov function and the network in the loop 1-Lipschitz
    (``‖Δw‖ ≤ ‖Δv‖``), the certificate is exactly

        V(Δx⁺) - alpha²V(Δx) - s(Δu, Δy) + ‖Δv‖² - ‖Δw‖² ≤ 0   for *all* (Δx, Δw, Δu)

    with the supply rate ``s = gamma‖Δu‖² - ‖Δy‖²/gamma`` (dropped, along with ``Δu``, in the
    contracting case where both trajectories see the same input). Everything below the
    quantifier is a quadratic form in ``[Δx; Δw; Δu]``, so one eigenvalue decides it — no
    sampling, and nothing here reads the construction that produced ``H``.
    """
    p = m.core.parameterization
    h = p.hmatrix()
    nx, nv, nu = m.spec.n_state, m.spec.n_nl, m.spec.n_input
    e = m.core.explicit()
    metric = _lyapunov_metric(m)
    zeros = lambda r, c: torch.zeros(r, c, dtype=h.dtype)  # noqa: E731
    eye_v = torch.eye(nv, dtype=h.dtype)
    if m.spec.variant == "contracting":
        blocks = ((e.A, e.B1), (e.C1, zeros(nv, nv)), (zeros(nv, nx), eye_v))
        pad = (metric, zeros(nv, nv))
    else:
        blocks = (
            (e.A, e.B1, e.B2),
            (e.C1, zeros(nv, nv), e.D12),
            (zeros(nv, nx), eye_v, zeros(nv, nu)),
        )
        pad = (metric, zeros(nv, nv), zeros(nu, nu))
    step, to_v, to_w = (torch.cat(b, dim=1) for b in blocks)
    form = step.mH @ metric @ step - m.spec.alpha**2 * torch.block_diag(*pad) + to_v.mH @ to_v - to_w.mH @ to_w
    if m.spec.variant == "lipschitz":
        to_y = torch.cat((e.C2, e.D21, e.D22), dim=1)
        to_u = torch.cat((zeros(nu, nx), zeros(nu, nv), torch.eye(nu, dtype=h.dtype)), dim=1)
        form = form + to_y.mH @ to_y / m.gamma - m.gamma * to_u.mH @ to_u
    return torch.linalg.eigvalsh((form + form.mH) / 2).max().item()


def _power_iterate(f, ref, d, iters):
    """Lower bound on the gain of ``f`` around ``ref``, by power iteration on its Jacobian.

    Each step measures the gain along the current perturbation direction, then replaces it
    with the direction the adjoint points at.
    """
    best = 0.0
    for _ in range(iters):
        d = d / (d.flatten(1).norm(dim=1).view(-1, *(1,) * (d.dim() - 1)) + 1e-30)
        pert = (ref + d).requires_grad_()
        dy = f(pert) - f(ref)
        norm = dy.detach().flatten(1).norm(dim=1)
        best = max(best, norm.max().item())
        weight = (dy / norm.view(-1, *(1,) * (dy.dim() - 1)) + 1e-30).detach()
        d = torch.autograd.grad((dy * weight).sum(), pert)[0].detach()
    return best


def _empirical_gain(m, length=60, iters=25, batch=8):
    """Lower bound on ``sup ‖y(u) - y(ũ)‖_ℓ2 / ‖u - ũ‖_ℓ2`` from a common initial state.

    The bound is stated for a fixed initial state, so both trajectories start from zeros.
    """
    e = m.core.explicit()
    x0 = torch.zeros(batch, m.spec.n_state, dtype=torch.float64)
    u = torch.randn(batch, length, m.spec.n_input, dtype=torch.float64)
    d = torch.randn(batch, length, m.spec.n_input, dtype=torch.float64)
    return _power_iterate(lambda uu: m.core.rollout(e, uu, x0)[0], u, d, iters)


class TestCertificates:
    @pytest.mark.parametrize("dist,scale", [("normal", 1.0), ("normal", 1e3), ("uniform", 10.0), ("rescale", 1e3)])
    @pytest.mark.parametrize("alpha", [1.0, 0.7])
    def test_contraction_at_random_parameters(self, dist, scale, alpha):
        torch.manual_seed(0)
        m = _perturb(_model(alpha=alpha, init="random"), dist, scale)
        assert _worst_contraction_ratio(m) <= 1.0 + 1e-9

    @pytest.mark.parametrize("act", ["tanh", "relu", "sigmoid"])
    def test_contraction_after_a_parameter_step(self, act):
        """The property a direct parameterization has and a projected one does not."""
        torch.manual_seed(1)
        m = _model(act=act, alpha=0.9)
        assert _worst_contraction_ratio(m) <= 1.0 + 1e-9
        with torch.no_grad():
            for p in m.core.parameterization.parameters():
                p.add_(torch.randn_like(p) * 50)
        assert _worst_contraction_ratio(m) <= 1.0 + 1e-9

    @pytest.mark.parametrize("gamma", [0.5, 2.0])
    @pytest.mark.parametrize("nu,ny", [(2, 3), (3, 2), (2, 2)])
    def test_incremental_gain_below_certificate(self, gamma, nu, ny, capsys):
        torch.manual_seed(0)
        m = _perturb(_model(nu=nu, ny=ny, variant="lipschitz", gamma=gamma), "rescale", 3.0)
        empirical = _empirical_gain(m)
        # the ratio is what says whether the certificate is tight or vacuous, so report it
        with capsys.disabled():
            print(f"\n  gamma={gamma} nu={nu} ny={ny}: empirical/certified = {empirical / gamma:.3f}")
        assert empirical <= gamma * (1 + 1e-6)

    @pytest.mark.parametrize("scale", [1.0, 5.0, 50.0])
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(),
            dict(alpha=0.7),
            dict(act="tanh", init="random"),
            dict(variant="lipschitz", gamma=0.5),
            dict(variant="lipschitz", gamma=7.0, alpha=0.8),
            dict(variant="lipschitz", gamma=2.0, init="random"),
        ],
    )
    def test_dissipation_inequality(self, kwargs, scale):
        """The certificate itself, as an eigenvalue rather than as a sampled trajectory."""
        torch.manual_seed(0)
        m = _perturb(_model(**kwargs), "rescale", scale)
        assert _dissipation_residual(m) <= 1e-9 * max(1.0, _lyapunov_metric(m).abs().max().item())

    def test_gain_certificate_tracks_reassigned_gamma(self):
        torch.manual_seed(0)
        m = _perturb(_model(variant="lipschitz", gamma=4.0), "rescale", 2.0)
        assert _empirical_gain(m, length=40, iters=15) <= 4.0 * (1 + 1e-6)
        m.gamma = 0.25
        assert _empirical_gain(m, length=40, iters=15) <= 0.25 * (1 + 1e-6)

    def test_saturated_feedthrough_stays_invertible(self):
        """The supply-rate weight must stay definite where the optimizer actually goes.

        Consuming the gain budget is useful, so training drives the free ``X`` of each bounded
        feedthrough toward zero: the Cayley argument becomes skew, its transform becomes
        *exactly* orthogonal, and ``γI - D12ᵀD12`` — which the construction inverts — collapses.
        Reached here directly rather than by training, since ``X = Z = 0`` with a large skew
        part is the limit that ~20 epochs of Adam walked into as a ``linalg.solve`` failure.
        """
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(0)
            m = _model(variant="lipschitz", gamma=5.0).to(dtype)
            p = m.core.parameterization
            with torch.no_grad():
                for name in ("12", "21"):
                    getattr(p, "X" + name).zero_()
                    getattr(p, "Z" + name).zero_()
                    getattr(p, "Y" + name).mul_(50.0)
            e = m.core.explicit()  # must not raise
            gamma, nv, nu = 5.0, m.spec.n_nl, m.spec.n_input
            r = torch.block_diag(
                torch.eye(nv, dtype=dtype) - e.D21.mH @ e.D21 / gamma,
                gamma * torch.eye(nu, dtype=dtype) - e.D12.mH @ e.D12,
            )
            assert torch.linalg.eigvalsh((r + r.mH) / 2).min() > 0
            assert torch.linalg.matrix_norm(e.D12, 2) < gamma**0.5
            assert torch.linalg.matrix_norm(e.D21, 2) < gamma**0.5
            assert torch.isfinite(p.hmatrix()).all()

    def test_lipschitz_construction_reduces_to_contracting(self):
        """A model with an unspent gain budget must land on the contracting construction.

        The contracting ``H`` carries ``blkdiag(C1ᵀC1, B1B1ᵀ)``; the Lipschitz one carries
        ``Γ R⁻¹ Γᵀ`` plus a ``C2ᵀC2/gamma`` term. Letting ``gamma`` grow with the two bounded
        feedthroughs held at zero collapses the second expression onto the first, term by
        term. Nothing else in the suite ties the two branches together.
        """
        torch.manual_seed(0)
        gamma = 1e8
        lip = _model(nu=2, ny=2, variant="lipschitz", gamma=gamma, init="random")
        con = _model(nu=2, ny=2, init="random")
        pl, pc = lip.core.parameterization, con.core.parameterization
        with torch.no_grad():
            for name in ("X", "Y", "p", "B1", "B2", "C1", "C2", "bx", "bv", "by"):
                getattr(pc, name).copy_(getattr(pl, name))
            pc.D12.zero_()
            pc.D21.zero_()
            for name in ("12", "21"):
                # the transform sends m = I to the zero matrix, so D12 = D21 = 0
                x, y, z = (getattr(pl, prefix + name) for prefix in ("X", "Y", "Z"))
                x.copy_(torch.eye(x.shape[0], dtype=x.dtype))
                y.zero_()
                z.zero_()
        # zero up to the eps floor the transform keeps between itself and the boundary
        assert lip.core.explicit().D12.abs().max() < gamma**0.5 * 1e-6
        assert _rel(pl.hmatrix(), pc.hmatrix()) < 1e-6

    def test_dissipative_variant_rejected(self):
        from tsfast.models.architectures.ren import R2DN

        with pytest.raises(ValueError, match="variant"):
            R2DN(2, 2, variant="dissipative")


class TestLBDN:
    @pytest.mark.parametrize("gamma", [0.5, 1.0, 5.0])
    @pytest.mark.parametrize("act", ["relu", "tanh"])
    def test_lipschitz_bound_at_badly_scaled_parameters(self, gamma, act, capsys):
        from tsfast.models.architectures.ren import LBDN

        torch.manual_seed(0)
        net = LBDN(5, 3, (8, 8), act=act, gamma=gamma).double()
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(20.0).add_(torch.randn_like(p) * 5)
        v = torch.randn(32, 5, dtype=torch.float64)
        empirical = _power_iterate(net, v, torch.randn(32, 5, dtype=torch.float64) * 1e-4, 30)
        with capsys.disabled():
            print(f"\n  gamma={gamma} act={act}: empirical/certified = {empirical / gamma:.3f}")
        assert empirical <= gamma * (1 + 1e-6)

    def test_layer_realizations_are_norm_bounded(self):
        """``[Aᵀ; Bᵀ]`` an isometry on the hidden layers, ``‖B‖ ≤ 1`` on the output layer."""
        from tsfast.models.architectures.ren import LBDN

        torch.manual_seed(0)
        net = LBDN(4, 6, (5, 5)).double()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 3)
        for layer in net.layers:
            e = layer.explicit()
            if e.A is None:
                assert torch.linalg.matrix_norm(e.B, 2) <= 1.0 + 1e-9
            else:
                gram = e.A.mH @ e.A + e.B @ e.B.mH
                assert _rel(gram, torch.eye(gram.shape[0], dtype=gram.dtype)) < 1e-12

    def test_output_layer_is_affine(self):
        """No activation on the output layer, so a depthless LBDN is an affine map."""
        from tsfast.models.architectures.ren import LBDN

        torch.manual_seed(0)
        net = LBDN(3, 2, ()).double()
        a, b = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
        assert _rel(net(a + b), net(a) + net(b) - net(torch.zeros_like(a))) < 1e-12

    def test_realization_reuse_matches_rebuilding(self):
        from tsfast.models.architectures.ren import LBDN

        torch.manual_seed(0)
        net = LBDN(4, 4, (6,)).double()
        v = torch.randn(3, 4, dtype=torch.float64)
        assert _rel(net(v, net.explicit()), net(v)) < 1e-14

    def test_rejects_unknown_configuration(self):
        from tsfast.models.architectures.ren import LBDN

        with pytest.raises(ValueError, match="activation"):
            LBDN(2, 2, (4,), act="gelu")
        with pytest.raises(ValueError, match="gamma"):
            LBDN(2, 2, (4,), gamma=0.0)


class TestWellPosedness:
    def test_explicit_invariants(self):
        torch.manual_seed(0)
        nx, nv, nu, ny, depth = 4, 6, 2, 3, 2
        m = _perturb(_model(nu, ny, nx, nv, depth), "normal", 20.0)
        p = m.core.parameterization
        h = p.hmatrix()
        assert h.shape == (2 * nx, 2 * nx) == (m.spec.n_h, m.spec.n_h)
        assert torch.linalg.eigvalsh(h).min() > 0
        e_imp = (h[:nx, :nx] + h[nx:, nx:] / m.spec.alpha**2 + p.Y - p.Y.mH) / 2
        assert torch.linalg.eigvalsh(e_imp + e_imp.mH).min() > 0
        ex = m.core.explicit()
        shapes = {
            "A": (nx, nx), "B1": (nx, nv), "B2": (nx, nu), "C1": (nv, nx), "C2": (ny, nx),
            "D12": (nv, nu), "D21": (ny, nv), "D22": (ny, nu),
            "bx": (nx,), "bv": (nv,), "by": (ny,),
        }  # fmt: skip
        assert {k: tuple(getattr(ex, k).shape) for k in shapes} == shapes
        assert len(ex.net) == depth + 1  # the nonlinear layers plus the linear output

    @pytest.mark.parametrize("variant", ["contracting", "lipschitz"])
    def test_explicit_is_differentiable(self, variant):
        """Every free parameter reaches the explicit realization; nothing is a dead end."""
        torch.manual_seed(0)
        m = _model(variant=variant, gamma=2.0)
        e = m.core.explicit()
        total = sum(t.sum() for t in e.tensors)
        for layer in e.net:
            total = total + sum(t.sum() for t in (layer.B, layer.bias, layer.A, layer.psi) if t is not None)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

    def test_feedthrough_absent_under_the_lipschitz_certificate(self):
        """``D22 = 0`` is what the paper's Lipschitz parameterization covers."""
        torch.manual_seed(0)
        m = _perturb(_model(variant="lipschitz", gamma=3.0), "normal", 5.0)
        e = m.core.explicit()
        assert (e.D22 == 0).all()
        assert torch.linalg.matrix_norm(e.D12, 2) <= 3.0**0.5 + 1e-9
        assert torch.linalg.matrix_norm(e.D21, 2) <= 3.0**0.5 + 1e-9


class TestR2DN:
    def test_shapes_and_zero_x0_default(self):
        from tsfast.models.architectures.ren import R2DN

        m = R2DN(3, 2, n_state=5, n_nl=6, depth=1, backend="eager")
        u = torch.randn(4, 25, 3)
        assert m(u).shape == (4, 25, 2)
        assert m(u, torch.randn(4, 5)).shape == (4, 25, 2)
        assert m(u, torch.randn(4, 1, 5)).shape == (4, 25, 2)  # [B,1,NX] x0 accepted
        assert _rel(m(u), m(u, torch.zeros(4, 5))) < 1e-12

    def test_rejects_unknown_configuration(self):
        from tsfast.models.architectures.ren import R2DN

        with pytest.raises(ValueError, match="activation"):
            R2DN(1, 1, act="gelu")
        with pytest.raises(ValueError, match="variant"):
            R2DN(1, 1, variant="passive")
        with pytest.raises(ValueError, match="alpha"):
            R2DN(1, 1, alpha=1.5)
        with pytest.raises(ValueError, match="depth"):
            R2DN(1, 1, depth=0)
        with pytest.raises(ValueError, match="init"):
            R2DN(1, 1, init="glorot")
        with pytest.raises(ValueError, match="unknown backend"):
            R2DN(1, 1, backend="cuda")(torch.randn(2, 3, 1))

    def test_stateful_chunked_equivalence(self):
        from tsfast.models.architectures.ren import R2DN

        torch.manual_seed(0)
        m = R2DN(2, 1, n_state=3, n_nl=6, depth=2, backend="eager", return_state=True).double()
        u = torch.randn(4, 30, 2, dtype=torch.float64)
        full, state_full = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:25], state=state)
        out3, state = m(u[:, 25:], state=state)
        assert _rel(torch.cat((out1, out2, out3), dim=1), full) < 1e-12
        assert _rel(state["x"], state_full["x"]) < 1e-12

    def test_rejects_malformed_state(self):
        from tsfast.models.architectures.ren import R2DN

        m = R2DN(2, 1, backend="eager", return_state=True)
        with pytest.raises(TypeError):
            m(torch.randn(2, 4, 2), state=torch.zeros(2, 8))

    def test_explicit_cache_tracks_parameter_updates(self):
        torch.manual_seed(0)
        m = _model()
        m.eval()
        with torch.no_grad():
            first = m.core.explicit()
            assert m.core.explicit() is first
            m.core.parameterization.net.layers[0].XY.mul_(1.3)
            assert m.core.explicit() is not first
        assert m.core.explicit() is not first  # grad enabled: never served from cache

    def test_gradcheck(self):
        torch.manual_seed(0)
        m = _model(nu=1, ny=1, nx=2, nv=3, depth=1, act="tanh")
        names = ["core.parameterization." + n for n in ("X", "Y", "B1", "B2", "C1", "p", "net.layers.0.XY")]
        params = [dict(m.named_parameters())[n].detach().clone().requires_grad_() for n in names]

        def f(u, x0, *ps):
            return torch.func.functional_call(m, dict(zip(names, ps)), (u, x0))

        u = torch.randn(2, 5, 1, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (u, x0, *params), eps=1e-6, atol=1e-7)

    def test_compile_fullgraph_parity(self):
        from tsfast.models.architectures.ren import R2DN

        torch.manual_seed(0)
        m = R2DN(2, 2, n_state=3, n_nl=4, depth=1, backend="eager")
        u, x0 = torch.randn(2, 8, 2), torch.randn(2, 3)

        def run(model):
            for p in m.parameters():
                p.grad = None
            uu, xx = u.clone().requires_grad_(), x0.clone().requires_grad_()
            y = model(uu, xx)
            (y**2).mean().backward()
            return y.detach(), [p.grad.clone() for p in m.parameters()], uu.grad.clone(), xx.grad.clone()

        out_e, g_e, du_e, dx0_e = run(m)
        out_c, g_c, du_c, dx0_c = run(torch.compile(m, fullgraph=True))
        assert _rel(out_c, out_e) < 1e-5
        assert max(_rel(a, b) for a, b in zip(g_c, g_e)) < 1e-5
        assert _rel(du_c, du_e) < 1e-5 and _rel(dx0_c, dx0_e) < 1e-5

    def test_compiled_backend_parity(self):
        from tsfast.models.architectures.ren import R2DN

        torch.manual_seed(0)
        m = R2DN(2, 2, n_state=3, n_nl=4, depth=1, backend="eager").double()
        u = torch.randn(2, 6, 2, dtype=torch.float64)
        expected = m(u)
        m.backend = "compiled"
        assert _rel(m(u), expected) < 1e-10

    def test_cuda(self):
        from tsfast.models.architectures.ren import R2DN

        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        torch.manual_seed(0)
        m = R2DN(2, 2, n_state=4, n_nl=8, depth=2, backend="eager")
        u = torch.randn(3, 12, 2)
        expected = m(u)
        got = m.cuda()(u.cuda()).cpu()
        assert _rel(got, expected) < 1e-4

    def test_fransys_composition(self):
        """The supported answer for integrating plants: estimate x0 instead of forgetting it."""
        from tsfast.models.architectures.ren import R2DN
        from tsfast.prediction import FranSys

        torch.manual_seed(0)
        prognosis = R2DN(2, 1, n_state=5, n_nl=8, depth=2, backend="eager", return_state=True)
        model = FranSys(n_u=2, n_y=1, init_sz=10, prognosis=prognosis, hidden_size=16)
        assert model._state_spec.state_size == 5  # the physical state, discovered by pytree
        out = model(torch.randn(4, 40, 3))
        assert out.shape == (4, 40, 1)
        assert (out[:, :10] == 0).all()  # diagnosis window is not predicted
        out.pow(2).mean().backward()
        assert all(p.grad is not None for p in prognosis.parameters())

    @pytest.mark.slow
    def test_r2dn_learner_fit(self, dls_simulation):
        from tsfast.training import R2DNLearner

        lrn = R2DNLearner(dls_simulation, n_state=4, n_nl=8, depth=1, backend="eager", n_skip=5)
        lrn.fit(1, 1e-3)
        assert not torch.isnan(torch.tensor(lrn.recorder[-1][1]))

    @pytest.mark.slow
    def test_r2dn_learner_tbptt(self, dls_simulation):
        from tsfast.training import R2DNLearner

        lrn = R2DNLearner(dls_simulation, n_state=4, n_nl=8, depth=1, backend="eager", n_skip=5, sub_seq_len=50)
        lrn.fit(1, 1e-3)
        assert not torch.isnan(torch.tensor(lrn.recorder[-1][1]))

    @pytest.mark.slow
    def test_certificate_survives_training(self):
        """Training in float32 must not walk the model out of its own guarantee."""
        from tsfast.models.architectures.ren import R2DN

        torch.manual_seed(0)
        m = R2DN(2, 2, n_state=6, n_nl=16, depth=2, variant="lipschitz", gamma=3.0, backend="eager")
        opt = torch.optim.Adam(m.parameters(), 1e-2)
        u, target = torch.randn(8, 40, 2), torch.randn(8, 40, 2)
        for _ in range(30):
            loss = ((m(u) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)
        assert torch.linalg.eigvalsh(m.core.parameterization.hmatrix().double()).min() > 0
        assert _empirical_gain(m.double(), length=40, iters=15) <= 3.0 * (1 + 1e-6)
