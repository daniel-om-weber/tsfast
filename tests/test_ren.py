"""Tests for tsfast.models.architectures.ren (REN and its certificates).

The certificate tests are the distinctive ones. A direct parameterization claims its
guarantee holds at *every* parameter value, not merely at initialization or after a
well-behaved training run, so they check it at randomly drawn and deliberately
badly-scaled parameters — which is what an unconstrained optimizer eventually produces.
"""

import pytest
import torch


def _rel(a, b):
    if b.numel() == 0:  # Z3 is empty whenever n_input == n_output
        return 0.0
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _model(nu=2, ny=3, nx=4, nv=8, **kwargs):
    """A float64 REN; the certificate margins are far below float32 resolution."""
    from tsfast.models.architectures.ren import REN

    kwargs.setdefault("backend", "eager")
    return REN(nu, ny, n_state=nx, n_nl=nv, **kwargs).double()


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

    Writing the incremental step as ``E Δx⁺ = F Δx + B1 Δw`` and taking the Schur
    complement of ``H ≻ 0`` on its ``P = H33`` block gives
    ``(EΔx⁺)ᵀP⁻¹(EΔx⁺) < ΔxᵀH11Δx + 2ΔwᵀΛ(Δw - Δv)``, whose second term is ``≤ 0`` for a
    slope-``[0,1]`` activation. Since ``EᵀP̃⁻¹E ⪰ E + Eᵀ - P̃`` for any ``P̃ ≻ 0``, taking
    ``P̃ = P/alpha²`` bounds the first term by ``alpha²·ΔxᵀMΔx``. So ``V = ΔxᵀMΔx`` decays
    at exactly ``alpha`` per step, which is the sharp statement and the one worth asserting.

    Squeezing ``V`` between its eigenvalues turns it into the paper's Euclidean form
    ``‖Δx_t‖ ≤ K alphaᵗ ‖Δx_0‖`` with ``K = cond(M)^½`` — note ``M``, not the storage matrix
    ``P``: the ``E`` factor does not cancel. That constant is deliberately not exposed by
    the model, since ``n_skip`` is a training decision and a bound this loose is a poor way
    to make it.
    """
    p = m.core.parameterization
    h = p.hmatrix()
    nx, nv = m.spec.n_state, m.spec.n_nl
    h11, h33 = h[:nx, :nx], h[nx + nv :, nx + nv :]
    e = (h11 + h33 / m.spec.alpha**2 + p.Y1 - p.Y1.mH) / 2
    return e.mH @ torch.linalg.solve(h33, e)


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


def _empirical_gain(m, length=60, iters=25, batch=8):
    """Lower bound on ``sup ‖y(u) - y(ũ)‖_ℓ2 / ‖u - ũ‖_ℓ2`` from a common initial state.

    Power iteration on the Jacobian of the input-to-output map: each step measures the
    gain along the current direction, then replaces it with the direction the adjoint
    points at. The bound is stated for a fixed initial state, so both trajectories start
    from zeros.
    """
    e = m.core.explicit()
    x0 = torch.zeros(batch, m.spec.n_state, dtype=torch.float64)
    u = torch.randn(batch, length, m.spec.n_input, dtype=torch.float64)
    d = torch.randn(batch, length, m.spec.n_input, dtype=torch.float64)
    best = 0.0
    for _ in range(iters):
        d = d / (d.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-30)
        ud = (u + d).requires_grad_()
        y_pert, _ = m.core.rollout(e, ud, x0)
        with torch.no_grad():
            y_ref, _ = m.core.rollout(e, u, x0)
        dy = y_pert - y_ref
        norm = dy.detach().flatten(1).norm(dim=1)
        best = max(best, norm.max().item())
        d = torch.autograd.grad((dy * (dy / (norm.view(-1, 1, 1) + 1e-30)).detach()).sum(), ud)[0].detach()
    return best


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
        # Print the ratio as well as asserting it: how tight the certificate is matters
        # as much as that it holds.
        with capsys.disabled():
            print(f"\n  gamma={gamma} nu={nu} ny={ny}: empirical/certified = {empirical / gamma:.3f}")
        assert empirical <= gamma * (1 + 1e-6)

    def test_dissipativity_supply_rate(self):
        """``Σ_t s(Δu_t, Δy_t) ≥ 0`` for perturbation pairs from a common initial state."""
        torch.manual_seed(0)
        nu, ny = 3, 2
        a = torch.randn(ny, ny, dtype=torch.float64)
        q = -(a.mH @ a + 0.3 * torch.eye(ny, dtype=torch.float64))
        s = torch.randn(nu, ny, dtype=torch.float64) * 0.5
        b = torch.randn(nu, nu, dtype=torch.float64)
        r = s @ torch.linalg.solve(q, s.mH) + b.mH @ b + 0.5 * torch.eye(nu, dtype=torch.float64)
        m = _perturb(_model(nu=nu, ny=ny, variant="dissipative", qsr=(q, s, r)), "rescale", 2.0)
        with torch.no_grad():
            e = m.core.explicit()
            worst = float("inf")
            for k in range(20):
                x0 = torch.randn(16, m.spec.n_state, dtype=torch.float64) * 2
                ua = torch.randn(16, 50, nu, dtype=torch.float64) * (0.1 if k % 2 else 3.0)
                ub = ua + torch.randn(16, 50, nu, dtype=torch.float64) * 2
                ya, _ = m.core.rollout(e, ua, x0)
                yb, _ = m.core.rollout(e, ub, x0)
                dy, du = ya - yb, ua - ub
                supply = (dy @ q * dy).sum(-1) + 2 * (dy @ s.mH * du).sum(-1) + (du @ r * du).sum(-1)
                worst = min(worst, supply.sum(1).min().item())
        assert worst >= -1e-9

    def test_invalid_qsr_rejected(self):
        from tsfast.models.architectures.ren import REN

        eye2, zero2 = torch.eye(2), torch.zeros(2, 2)
        with pytest.raises(ValueError, match="negative definite"):
            REN(2, 2, variant="dissipative", qsr=(eye2, zero2, eye2))
        with pytest.raises(ValueError, match="requires qsr"):
            REN(2, 2, variant="dissipative")
        with pytest.raises(ValueError, match="shape"):
            REN(2, 3, variant="dissipative", qsr=(-eye2, zero2, eye2))


class TestWellPosedness:
    @pytest.mark.parametrize("variant", ["contracting", "lipschitz"])
    def test_d11_strictly_lower_triangular(self, variant):
        torch.manual_seed(0)
        m = _perturb(_model(variant=variant), "normal", 5.0)
        assert (m.core.explicit().D11.triu(0) == 0).all()

    def test_sweep_solves_the_equilibrium(self):
        from tsfast.models.architectures.ren import equilibrium_sweep

        torch.manual_seed(0)
        m = _perturb(_model(nv=16), "normal", 3.0)
        e = m.core.explicit()
        x = torch.randn(5, m.spec.n_state, dtype=torch.float64)
        u = torch.randn(5, m.spec.n_input, dtype=torch.float64)
        b = x @ e.C1.mH + u @ e.D12.mH + e.bv
        w = equilibrium_sweep(b, e.D11, torch.tanh)
        assert _rel(w, torch.tanh(b + w @ e.D11.mH)) < 1e-12

    def test_explicit_invariants(self):
        torch.manual_seed(0)
        nx, nv, nu, ny = 4, 8, 2, 3
        m = _perturb(_model(nu, ny, nx, nv), "normal", 20.0)
        p = m.core.parameterization
        h = p.hmatrix()
        assert torch.linalg.eigvalsh(h).min() > 0
        e_imp = (h[:nx, :nx] + h[nx + nv :, nx + nv :] / m.spec.alpha**2 + p.Y1 - p.Y1.mH) / 2
        assert torch.linalg.eigvalsh(e_imp + e_imp.mH).min() > 0
        assert (torch.diagonal(h[nx : nx + nv, nx : nx + nv]) > 0).all()  # Lambda positive
        ex = m.core.explicit()
        shapes = {
            "A": (nx, nx), "B1": (nx, nv), "B2": (nx, nu), "C1": (nv, nx), "D11": (nv, nv),
            "D12": (nv, nu), "C2": (ny, nx), "D21": (ny, nv), "D22": (ny, nu),
            "bx": (nx,), "bv": (nv,), "by": (ny,),
        }  # fmt: skip
        assert {k: tuple(getattr(ex, k).shape) for k in shapes} == shapes

    def test_explicit_is_differentiable(self):
        """Every free parameter reaches the explicit realization; nothing is a dead end."""
        import dataclasses

        torch.manual_seed(0)
        m = _model(variant="lipschitz", gamma=2.0)
        e = m.core.explicit()
        sum(getattr(e, f.name).sum() for f in dataclasses.fields(e)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


class TestREN:
    def test_shapes_and_zero_x0_default(self):
        from tsfast.models.architectures.ren import REN

        m = REN(3, 2, n_state=5, n_nl=6, backend="eager")
        u = torch.randn(4, 25, 3)
        assert m(u).shape == (4, 25, 2)
        assert m(u, torch.randn(4, 5)).shape == (4, 25, 2)
        assert m(u, torch.randn(4, 1, 5)).shape == (4, 25, 2)  # [B,1,NX] x0 accepted
        assert _rel(m(u), m(u, torch.zeros(4, 5))) < 1e-12

    def test_rejects_unknown_configuration(self):
        from tsfast.models.architectures.ren import REN

        with pytest.raises(ValueError, match="activation"):
            REN(1, 1, act="gelu")
        with pytest.raises(ValueError, match="variant"):
            REN(1, 1, variant="passive")
        with pytest.raises(ValueError, match="alpha"):
            REN(1, 1, alpha=1.5)
        with pytest.raises(ValueError, match="init"):
            REN(1, 1, init="glorot")
        with pytest.raises(ValueError, match="unknown backend"):
            REN(1, 1, backend="cuda")(torch.randn(2, 3, 1))

    def test_stateful_chunked_equivalence(self):
        from tsfast.models.architectures.ren import REN

        torch.manual_seed(0)
        m = REN(2, 1, n_state=3, n_nl=6, backend="eager", return_state=True).double()
        u = torch.randn(4, 30, 2, dtype=torch.float64)
        full, state_full = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:25], state=state)
        out3, state = m(u[:, 25:], state=state)
        assert _rel(torch.cat((out1, out2, out3), dim=1), full) < 1e-12
        assert _rel(state["x"], state_full["x"]) < 1e-12

    def test_rejects_malformed_state(self):
        from tsfast.models.architectures.ren import REN

        m = REN(2, 1, backend="eager", return_state=True)
        with pytest.raises(TypeError):
            m(torch.randn(2, 4, 2), state=torch.zeros(2, 8))

    def test_explicit_cache_tracks_parameter_updates(self):
        torch.manual_seed(0)
        m = _model()
        m.eval()
        with torch.no_grad():
            first = m.core.explicit()
            assert m.core.explicit() is first
            m.core.parameterization.X.mul_(1.3)
            assert m.core.explicit() is not first
        assert m.core.explicit() is not first  # grad enabled: never served from cache

    def test_gradcheck(self):
        torch.manual_seed(0)
        m = _model(nu=1, ny=1, nx=2, nv=3)
        names = ["core.parameterization." + n for n in ("X", "Y1", "B2", "D12", "p")]
        params = [dict(m.named_parameters())[n].detach().clone().requires_grad_() for n in names]

        def f(u, x0, *ps):
            return torch.func.functional_call(m, dict(zip(names, ps)), (u, x0))

        u = torch.randn(2, 5, 1, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (u, x0, *params), eps=1e-6, atol=1e-7)

    def test_compile_fullgraph_parity(self):
        from tsfast.models.architectures.ren import REN

        torch.manual_seed(0)
        m = REN(2, 2, n_state=3, n_nl=4, backend="eager")
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

    def test_cuda(self):
        from tsfast.models.architectures.ren import REN

        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        torch.manual_seed(0)
        m = REN(2, 2, n_state=4, n_nl=8, backend="eager")
        u = torch.randn(3, 12, 2)
        expected = m(u)
        got = m.cuda()(u.cuda()).cpu()
        assert _rel(got, expected) < 1e-4

    def test_fransys_composition(self):
        """The supported answer for integrating plants: estimate x0 instead of forgetting it."""
        from tsfast.models.architectures.ren import REN
        from tsfast.prediction import FranSys

        torch.manual_seed(0)
        prognosis = REN(2, 1, n_state=5, n_nl=8, backend="eager", return_state=True)
        model = FranSys(n_u=2, n_y=1, init_sz=10, prognosis=prognosis, hidden_size=16)
        assert model._state_spec.state_size == 5  # the physical state, discovered by pytree
        out = model(torch.randn(4, 40, 3))
        assert out.shape == (4, 40, 1)
        assert (out[:, :10] == 0).all()  # diagnosis window is not predicted
        out.pow(2).mean().backward()
        assert all(p.grad is not None for p in prognosis.parameters())

    @pytest.mark.slow
    def test_ren_learner_fit(self, dls_simulation):
        from tsfast.training import RENLearner

        lrn = RENLearner(dls_simulation, n_state=4, n_nl=8, backend="eager", n_skip=5)
        lrn.fit(1, 1e-3)
        assert not torch.isnan(torch.tensor(lrn.recorder[-1][1]))

    @pytest.mark.slow
    def test_ren_learner_tbptt(self, dls_simulation):
        from tsfast.training import RENLearner

        lrn = RENLearner(dls_simulation, n_state=4, n_nl=8, backend="eager", n_skip=5, sub_seq_len=50)
        lrn.fit(1, 1e-3)
        assert not torch.isnan(torch.tensor(lrn.recorder[-1][1]))

    @pytest.mark.slow
    def test_certificate_survives_training(self):
        """Training in float32 must not walk the model out of its own guarantee."""
        from tsfast.models.architectures.ren import REN

        torch.manual_seed(0)
        m = REN(2, 2, n_state=6, n_nl=16, variant="lipschitz", gamma=3.0, backend="eager")
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
