"""Tests for tsfast.models.phnn (HamiltonianMLP, PHNNCore, PHNN)."""

import pytest
import torch
from torch import nn

from tsfast.models.phnn import PHNN, HamiltonianMLP, PHNNCore


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _randomize(module, std=0.4):
    for p in module.parameters():
        nn.init.normal_(p, std=std)


class TestHamiltonianMLP:
    @pytest.mark.parametrize("lower_bound", [0.0, None, 2.5])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_closed_form_gradient_matches_autograd(self, lower_bound, num_layers):
        torch.manual_seed(0)
        ham = HamiltonianMLP(4, hidden_size=8, num_layers=num_layers, lower_bound=lower_bound).double()
        _randomize(ham)
        x = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        h, g = ham(x)
        g_auto = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
        assert _rel(g, g_auto) < 1e-14
        # second order: gradients of a loss through dH/dx w.r.t. the weights must match too
        for pa, pb in zip(
            torch.autograd.grad((g**2).sum(), list(ham.parameters()), retain_graph=True, allow_unused=True),
            torch.autograd.grad((g_auto**2).sum(), list(ham.parameters()), allow_unused=True),
        ):
            assert (pa is None) == (pb is None)
            if pa is not None:
                assert _rel(pa, pb) < 1e-12

    def test_lower_bound_holds(self):
        torch.manual_seed(1)
        ham = HamiltonianMLP(3, lower_bound=1.5)
        _randomize(ham, std=2.0)
        h, _ = ham(torch.randn(512, 3) * 5)
        # elu saturates to -1, so H approaches the bound from above and reaches it in float32
        assert h.min() >= 1.5 - 1e-6


class TestPHNNCore:
    def test_energy_balance(self):
        """Passivity structure: with u = 0 the Hamiltonian never increases along the flow."""
        torch.manual_seed(2)
        core = PHNNCore(4, 1, dt=0.01, h_lower_bound=0.0).double()
        _randomize(core)
        x = torch.randn(64, 4, dtype=torch.float64)
        u = torch.zeros(64, 1, dtype=torch.float64)
        with torch.no_grad():
            h_before, _ = core.hamiltonian(x)
            for _ in range(20):
                _, x = core.step(x, u)
            h_after, _ = core.hamiltonian(x)
        assert (h_after - h_before).max() < 1e-6

    def test_structure_matrices(self):
        torch.manual_seed(3)
        core = PHNNCore(5, 2, dt=0.1)
        _randomize(core)
        x = torch.randn(9, 5)
        n = core.n_state
        b = core.j_net(x).view(-1, n, n) * core.jr_scale
        a = core.r_net(x).view(-1, n, n) * core.jr_scale
        j = b - b.transpose(1, 2)
        r = a @ a.transpose(1, 2)
        assert (j + j.transpose(1, 2)).abs().max() < 1e-6  # skew-symmetric
        eigvals = torch.linalg.eigvalsh(r)
        assert eigvals.min() > -1e-6  # positive semidefinite

    def test_square_requirement(self):
        with pytest.raises(ValueError, match="requires n_input == n_output"):
            PHNNCore(4, 2, n_output=3)
        core = PHNNCore(4, 2, n_output=3, output="linear")
        y, _ = core.step(torch.randn(5, 4), torch.randn(5, 2))
        assert y.shape == (5, 3)


class TestPHNN:
    def test_matches_reference_rk4_rollout(self):
        """Fused step (shared first RK4 stage, closed-form dH/dx) equals the naive formulation."""
        torch.manual_seed(4)
        m = PHNN(1, 1, n_state=4, hidden_size=8, dt=0.13, n_init=5, backend="eager").double()
        _randomize(m)
        xin = torch.randn(3, 25, 2, dtype=torch.float64, requires_grad=True)
        out = m(xin)

        core = m.core

        def rhs(x, u):
            assert x.requires_grad  # non-leaf graph node: gradient flows through the trajectory
            h = core.hamiltonian.net(x)[:, 0]
            b0 = core.hamiltonian.lower_bound
            if b0 is not None:
                h = torch.nn.functional.elu(h - (b0 + 1.0)) + (b0 + 1.0)
            dhdx = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
            bmat = core.j_net(x).view(-1, 4, 4) * core.jr_scale
            amat = core.r_net(x).view(-1, 4, 4) * core.jr_scale
            g = core.g_net(x).view(-1, 4, 1) * core.g_scale
            return torch.einsum("bij,bj->bi", bmat - bmat.transpose(1, 2) - amat @ amat.transpose(1, 2), dhdx) + \
                torch.einsum("bij,bj->bi", g, u), g, dhdx

        u = xin[..., :1]
        x = m.encoder(u[:, :5], xin[:, :5, 1:])
        ys = []
        for t in range(5, 25):
            f0, g, dhdx = rhs(x, u[:, t])
            ys.append(torch.einsum("bij,bi->bj", g, dhdx))
            k1 = core.dt * f0
            k2 = core.dt * rhs(x + k1 / 2, u[:, t])[0]
            k3 = core.dt * rhs(x + k2 / 2, u[:, t])[0]
            k4 = core.dt * rhs(x + k3, u[:, t])[0]
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ref = torch.stack(ys, dim=1)
        assert _rel(out[:, 5:], ref) < 1e-13

        # end-to-end parameter gradients through both formulations
        g_ours = torch.autograd.grad(out.pow(2).mean(), list(m.parameters()), retain_graph=True)
        g_ref = torch.autograd.grad(
            torch.cat((torch.zeros_like(out[:, :5]), ref), 1).pow(2).mean(), list(m.parameters())
        )
        for a, b in zip(g_ours, g_ref):
            assert _rel(a, b) < 1e-12

    def test_only_warmup_y_is_read(self):
        torch.manual_seed(5)
        m = PHNN(1, 1, n_state=3, n_init=5, dt=0.1)
        xin = torch.randn(2, 20, 2)
        perturbed = xin.clone()
        perturbed[:, 5:, 1] = torch.randn(2, 15)
        assert torch.equal(m(xin), m(perturbed))

    def test_short_sequence_rejected(self):
        m = PHNN(1, 1, n_init=8, dt=0.1)
        with pytest.raises(ValueError, match="too short"):
            m(torch.randn(2, 8, 2))

    @pytest.mark.slow
    def test_compiled_backend_matches_eager(self):
        torch.manual_seed(6)
        m = PHNN(1, 1, n_state=3, hidden_size=8, dt=0.1, n_init=4, backend="eager")
        xin = torch.randn(2, 20, 2)
        out_eager = m(xin)
        m.backend = "compiled"
        out_compiled = m(xin)
        assert _rel(out_compiled, out_eager) < 1e-5
