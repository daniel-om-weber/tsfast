"""Tests for the damped-ansatz DD-PINN surrogate (tsfast.pinn.ddpinn)."""

import math

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def fast_cpu():
    """Cap intra-op threads: tiny PINN tensors are thread-dispatch bound, not compute bound."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 0 — ansatz unit tests (zero physics)
# ──────────────────────────────────────────────────────────────────────────────


class TestDampedAnsatz:
    def test_initial_condition_exact(self):
        from tsfast.pinn.ddpinn import DampedAnsatzPINN

        torch.manual_seed(0)
        m = DampedAnsatzPINN(n_state=3, n_cond=2, n_ansatz=8)
        x = torch.randn(8, 1, 6)
        x[..., -1] = m.t_zero  # t -> tau = 0
        assert torch.allclose(m(x), x[..., :3], atol=1e-6)  # x(0) == x_k

    def test_analytic_derivative_matches_autograd(self):
        from tsfast.pinn.ddpinn import DampedAnsatzPINN

        torch.manual_seed(0)
        m = DampedAnsatzPINN(n_state=2, n_cond=1, n_ansatz=8)
        x = torch.randn(4, 1, 4, requires_grad=True)
        y, dy = m(x, derivative_flag=True)
        for i in range(2):
            (g,) = torch.autograd.grad(y[..., i].sum(), x, retain_graph=True)
            assert torch.allclose(dy[..., i], g[..., -1], rtol=1e-4, atol=1e-5)

    def test_fully_batched(self):
        from tsfast.pinn.ddpinn import DampedAnsatzPINN

        torch.manual_seed(0)
        m = DampedAnsatzPINN(n_state=2, n_cond=1, n_ansatz=8)
        x = torch.randn(5, 1, 4)
        stacked = torch.cat([m(x[i : i + 1]) for i in range(5)])
        assert torch.allclose(m(x), stacked, atol=1e-6)

    def test_derivative_flag_shapes(self):
        from tsfast.pinn.ddpinn import DampedAnsatzPINN

        m = DampedAnsatzPINN(n_state=2, n_cond=3, n_ansatz=8)
        x = torch.randn(6, 4, 6)  # [B, S, n_state + n_cond + 1]
        y = m(x)
        assert y.shape == (6, 4, 2)
        y2, dy = m(x, derivative_flag=True)
        assert y2.shape == (6, 4, 2) and dy.shape == (6, 4, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 1 — linear decay  dx/dt = -lambda*x  (analytic ground truth)
# ──────────────────────────────────────────────────────────────────────────────


class TestLinearDecay:
    def test_single_step_and_rollout(self, fast_cpu):
        from tsfast.pinn.ddpinn import DampedAnsatzPINN, SurrogatePINNLearner

        torch.manual_seed(0)
        LAM, T = 2.0, 0.5

        def residual(x, cond, dxdt):
            return F.mse_loss(dxdt + LAM * x, torch.zeros_like(x))

        def gen(bs, seq_len, dev):
            xk = torch.empty(bs, seq_len, 1, device=dev).uniform_(-1, 1)
            t = torch.empty(bs, seq_len, 1, device=dev).uniform_(-1, 1)
            return torch.cat([xk, t], dim=-1)

        model = DampedAnsatzPINN(1, 0, n_ansatz=16, hidden_size=32, hidden_layer=3)
        lrn = SurrogatePINNLearner(
            model,
            gen,
            residual,
            state_range=[(-1, 1)],
            cond_range=[],
            t_max=T,
            steps_per_epoch=50,
            bs=1024,
            val_steps=5,
            device=torch.device("cpu"),
            show_bar=False,
        )
        assert len(lrn.dls.train) == 50  # scheduler total-step count
        lrn.fit_flat_cos(8, lr=3e-3)

        x0 = torch.linspace(-0.9, 0.9, 50)[:, None, None]
        with torch.no_grad():
            pred = lrn.model(torch.cat([x0, torch.ones_like(x0)], dim=-1))  # t_scaled=+1 == t=T
        assert (pred - x0 * math.exp(-LAM * T)).abs().max() < 5e-2

        roll = lrn.as_rollout(t_sample=T)  # physical-in/physical-out, both scalers bundled
        with torch.no_grad():
            traj = roll(x0[:, 0], torch.empty(50, 20, 0))
        assert (traj[:, 0] - x0[:, 0] * math.exp(-LAM * T)).abs().max() < 5e-2
        assert (traj[:, -1] - x0[:, 0] * math.exp(-LAM * T * 20)).abs().max() < 1e-1


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 2 — damped oscillator (2nd-order, the soft-robot analog)
#  State [q, qd], control u, conditioning omega; reference = scipy.solve_ivp.
# ──────────────────────────────────────────────────────────────────────────────

ZETA, T_OSC = 0.2, 0.1
SR = [(-1.0, 1.0), (-5.0, 5.0)]  # q, qd
CR = [(-2.0, 2.0), (2.0, 4.0)]  # u, omega


def _osc_gen(bs, seq_len, dev):
    return torch.empty(bs, seq_len, 5, device=dev).uniform_(-1, 1)


def _osc_residual(x, cond, dxdt):
    q, qd = x[..., 0:1], x[..., 1:2]
    dq, dqd = dxdt[..., 0:1], dxdt[..., 1:2]
    u, om = cond[..., 0:1], cond[..., 1:2]
    res = torch.cat([dq - qd, dqd - (-(om**2) * q - 2 * ZETA * om * qd + u)], dim=-1)
    return F.mse_loss(res, torch.zeros_like(res))


def _train_oscillator(residual, n_epoch=10, seed=0):
    from tsfast.pinn.ddpinn import DampedAnsatzPINN, SurrogatePINNLearner

    torch.manual_seed(seed)
    model = DampedAnsatzPINN(2, 2, n_ansatz=16, hidden_size=48, hidden_layer=2)
    lrn = SurrogatePINNLearner(
        model,
        _osc_gen,
        residual,
        state_range=SR,
        cond_range=CR,
        t_max=T_OSC,
        steps_per_epoch=40,
        bs=512,
        val_steps=4,
        device=torch.device("cpu"),
        show_bar=False,
    )
    lrn.fit_flat_cos(n_epoch, lr=3e-3)
    return lrn


def _norm(v, r):
    return 2 * (v - r[0]) / (r[1] - r[0]) - 1


def _denorm(v, r):
    return (v + 1) / 2 * (r[1] - r[0]) + r[0]


class TestDampedOscillator:
    def test_single_step_and_rollout_vs_solve_ivp(self, fast_cpu):
        import numpy as np
        from scipy.integrate import solve_ivp

        from tsfast.pinn import DDPINNRollout

        lrn = _train_oscillator(_osc_residual)

        # single-step accuracy on random points within the box
        rng = np.random.default_rng(1)
        n = 24
        q0, qd0 = rng.uniform(-1, 1, n), rng.uniform(-5, 5, n)
        u, om = rng.uniform(-2, 2, n), rng.uniform(2, 4, n)

        def rhs(t, Y):
            Y = Y.reshape(n, 2)
            return np.stack([Y[:, 1], -(om**2) * Y[:, 0] - 2 * ZETA * om * Y[:, 1] + u], 1).flatten()

        ref = (
            solve_ivp(rhs, [0, T_OSC], np.stack([q0, qd0], 1).flatten(), t_eval=[T_OSC], rtol=1e-9, atol=1e-12)
            .y[:, -1]
            .reshape(n, 2)
        )
        X = torch.tensor(
            np.stack([_norm(q0, SR[0]), _norm(qd0, SR[1]), _norm(u, CR[0]), _norm(om, CR[1]), np.ones(n)], 1),
            dtype=torch.float32,
        )[:, None, :]
        with torch.no_grad():
            y = lrn.model(X)[:, 0, :].numpy()
        pred = np.stack([_denorm(y[:, 0], SR[0]), _denorm(y[:, 1], SR[1])], 1)
        ss_rmse = np.sqrt(np.mean((pred - ref) ** 2, axis=0))
        assert ss_rmse[0] < 0.05 and ss_rmse[1] < 0.05

        # autoregressive rollout vs segment-wise solve_ivp (piecewise-constant u)
        rng = np.random.default_rng(2)
        B, Ns = 4, 10
        om_r = rng.uniform(2, 4, B)
        q0r, qd0r = rng.uniform(-0.5, 0.5, B), rng.uniform(-2, 2, B)
        u_seq = rng.uniform(-1, 1, (B, Ns))
        cond = torch.tensor(  # physical [B, Ns, 2]: (u, omega)
            np.stack([u_seq, np.repeat(om_r[:, None], Ns, 1)], -1), dtype=torch.float32
        )
        x0 = torch.tensor(np.stack([q0r, qd0r], 1), dtype=torch.float32)  # physical [B, 2]
        roll = DDPINNRollout(lrn.model, lrn.state_scaler, lrn.cond_scaler, t_sample=T_OSC, t_max=T_OSC)
        with torch.no_grad():
            traj = roll(x0, cond).numpy()

        rt = np.zeros((B, Ns, 2))
        for bi in range(B):
            st = np.array([q0r[bi], qd0r[bi]])
            for k in range(Ns):
                st = solve_ivp(
                    lambda t, Y, uk=u_seq[bi, k], omk=om_r[bi]: [Y[1], -(omk**2) * Y[0] - 2 * ZETA * omk * Y[1] + uk],
                    [0, T_OSC],
                    st,
                    t_eval=[T_OSC],
                    rtol=1e-9,
                    atol=1e-12,
                ).y[:, -1]
                rt[bi, k] = st
        assert np.sqrt(np.mean((traj[..., 0] - rt[..., 0]) ** 2)) < 0.05  # q
        assert np.sqrt(np.mean((traj[..., 1] - rt[..., 1]) ** 2)) < 0.08  # qd
        assert np.abs(traj[:, -1] - rt[:, -1]).max() < 0.2

    def test_mass_matrix_residual_form_matches(self):
        """The M(q)·qdd residual form (M = I here) must flow through the seam identically.

        Rehearses the mass-matrix indexing of the real soft-robot ODE without retraining.
        """
        from tsfast.pinn.ddpinn import DampedAnsatzPINN, SurrogatePINNLearner

        def mass_matrix_residual(x, cond, dxdt):
            q, qd = x[..., 0:1], x[..., 1:2]
            dq, dqd = dxdt[..., 0:1], dxdt[..., 1:2]
            u, om = cond[..., 0:1], cond[..., 1:2]
            M = torch.eye(1, device=x.device).reshape(1, 1, 1, 1).expand(x.shape[0], x.shape[1], 1, 1)
            Mqdd = torch.einsum("...ij,...j->...i", M, dqd)
            res = torch.cat([dq - qd, Mqdd - (-(om**2) * q - 2 * ZETA * om * qd + u)], dim=-1)
            return F.mse_loss(res, torch.zeros_like(res))

        torch.manual_seed(0)
        model = DampedAnsatzPINN(2, 2, n_ansatz=8)
        lrn = SurrogatePINNLearner(
            model,
            _osc_gen,
            _osc_residual,
            state_range=SR,
            cond_range=CR,
            t_max=T_OSC,
            steps_per_epoch=2,
            bs=16,
            val_steps=1,
            device=torch.device("cpu"),
            show_bar=False,
        )
        X = _osc_gen(16, 1, "cpu")
        loss_plain = lrn.physics_loss(X)
        lrn.residual_func = mass_matrix_residual
        loss_mass = lrn.physics_loss(X)
        assert torch.allclose(loss_plain, loss_mass)
