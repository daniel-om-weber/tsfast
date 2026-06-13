"""Tests for the conventional soft-IC PINN-for-control surrogate (tsfast.pinn.pinc)."""

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
#  Layer 0 — model unit tests (zero physics)
# ──────────────────────────────────────────────────────────────────────────────


class TestPINC:
    def test_forward_and_derivative_shapes(self):
        from tsfast.pinn.pinc import PINC

        m = PINC(n_state=2, n_cond=3, hidden_size=32, hidden_layer=2)
        x = torch.randn(6, 4, 6)  # [B, S, n_state + n_cond + 1]
        y = m(x)
        assert y.shape == (6, 4, 2)
        y2, dy = m(x, derivative_flag=True)
        assert y2.shape == (6, 4, 2) and dy.shape == (6, 4, 2)
        assert torch.allclose(y, y2)  # both branches share the same forward value

    def test_initial_condition_is_not_exact(self):
        """The PINC's IC is *not* pinned by construction (that is the whole point vs the DD-PINN)."""
        from tsfast.pinn.pinc import PINC

        torch.manual_seed(0)
        m = PINC(n_state=3, n_cond=2, hidden_size=16)
        x = torch.randn(8, 1, 6)
        x[..., -1] = -1.0  # t_zero
        assert not torch.allclose(m(x), x[..., :3], atol=1e-3)  # untrained: x(t0) != x_k

    @pytest.mark.parametrize("dmode", ["forward", "reverse"])
    @pytest.mark.parametrize("ic_mode,style", [("soft", "subtract"), ("hard", "subtract"), ("hard", "multiply")])
    def test_returned_derivative_matches_fd_of_output(self, ic_mode, style, dmode):
        """``derivative_flag=True`` returns the true time-derivative of the *returned* ``x(t)``.

        Covers the IC transform's effect on the derivative — trivial for ``subtract`` (offset constant
        in t) but the product-rule term for ``multiply`` — across both AD modes.
        """
        from tsfast.pinn.pinc import PINC

        torch.manual_seed(0)
        m = PINC(2, 1, hidden_size=32, hidden_layer=2, ic_mode=ic_mode, hard_ic_style=style, derivative_mode=dmode).eval()
        x = torch.randn(4, 3, 4)
        _, dy = m(x, derivative_flag=True)

        eps = 1e-4
        xp, xm = x.clone(), x.clone()
        xp[..., -1] += eps
        xm[..., -1] -= eps
        with torch.no_grad():
            fd = (m(xp) - m(xm)) / (2 * eps)
        assert torch.allclose(dy.detach(), fd, rtol=1e-2, atol=1e-3)

    def test_forward_and_reverse_modes_agree(self):
        """Forward-mode (dual) and reverse-mode (per-channel backward) derivatives are identical."""
        from tsfast.pinn.pinc import PINC

        torch.manual_seed(0)
        mf = PINC(n_state=3, n_cond=2, hidden_size=32, hidden_layer=2, derivative_mode="forward")
        mr = PINC(n_state=3, n_cond=2, hidden_size=32, hidden_layer=2, derivative_mode="reverse")
        mr.load_state_dict(mf.state_dict())  # same weights
        x = torch.randn(8, 4, 6)
        yf, dyf = mf(x, derivative_flag=True)
        yr, dyr = mr(x, derivative_flag=True)
        assert torch.allclose(yf, yr, atol=1e-6)
        assert torch.allclose(dyf, dyr, atol=1e-5)

    def test_rejects_unknown_derivative_mode(self):
        from tsfast.pinn.pinc import PINC

        with pytest.raises(ValueError, match="derivative_mode"):
            PINC(n_state=2, n_cond=1, derivative_mode="sideways")

    def test_rejects_unknown_ic_mode(self):
        from tsfast.pinn.pinc import PINC

        with pytest.raises(ValueError, match="ic_mode"):
            PINC(n_state=2, n_cond=1, ic_mode="medium")

    def test_rejects_unknown_hard_ic_style(self):
        from tsfast.pinn.pinc import PINC

        with pytest.raises(ValueError, match="hard_ic_style"):
            PINC(n_state=2, n_cond=1, ic_mode="hard", hard_ic_style="divide")

    @pytest.mark.parametrize("style", ["subtract", "multiply"])
    def test_hard_ic_is_exact_by_construction(self, style):
        """Both hard-IC styles pin ``x(t_zero) = x_k`` for any (untrained) weights."""
        from tsfast.pinn.pinc import PINC

        torch.manual_seed(0)
        m = PINC(n_state=3, n_cond=2, hidden_size=16, ic_mode="hard", hard_ic_style=style)
        x = torch.randn(8, 1, 6)
        x[..., -1] = m.t_zero  # t -> t_zero
        assert torch.allclose(m(x), x[..., :3], atol=1e-6)  # x(t_zero) == x_k

    def test_hard_ic_leaves_derivative_unchanged_and_modes_agree(self):
        """The hard-IC offsets are constant in t (dx/dt unchanged), and both AD modes agree."""
        from tsfast.pinn.pinc import PINC

        torch.manual_seed(0)
        mf = PINC(n_state=2, n_cond=1, hidden_size=32, ic_mode="hard", derivative_mode="forward")
        mr = PINC(n_state=2, n_cond=1, hidden_size=32, ic_mode="hard", derivative_mode="reverse")
        mr.load_state_dict(mf.state_dict())
        x = torch.randn(6, 3, 4)
        yf, dyf = mf(x, derivative_flag=True)
        yr, dyr = mr(x, derivative_flag=True)
        assert torch.allclose(yf, yr, atol=1e-6) and torch.allclose(dyf, dyr, atol=1e-5)
        # derivative equals that of the plain (soft) net — offsets drop out
        msoft = PINC(n_state=2, n_cond=1, hidden_size=32, ic_mode="soft", derivative_mode="forward")
        msoft.load_state_dict(mf.state_dict())
        _, dy_soft = msoft(x, derivative_flag=True)
        assert torch.allclose(dyf, dy_soft, atol=1e-5)

    def test_fully_batched(self):
        from tsfast.pinn.pinc import PINC

        torch.manual_seed(0)
        m = PINC(n_state=2, n_cond=1, hidden_size=16)
        x = torch.randn(5, 1, 4)
        stacked = torch.cat([m(x[i : i + 1]) for i in range(5)])
        assert torch.allclose(m(x), stacked, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 1 — linear decay  dx/dt = -lambda*x  (analytic ground truth)
#  End-to-end: PINCLearner physics-only training + soft IC + autoregressive rollout.
# ──────────────────────────────────────────────────────────────────────────────


class TestLinearDecay:
    def test_soft_ic_and_rollout(self, fast_cpu):
        from tsfast.pinn.pinc import PINC, PINCLearner

        torch.manual_seed(0)
        LAM, T = 2.0, 0.5

        def residual(x, cond, dxdt):
            return F.mse_loss(dxdt + LAM * x, torch.zeros_like(x))

        def gen(bs, seq_len, dev):
            xk = torch.empty(bs, seq_len, 1, device=dev).uniform_(-1, 1)
            t = torch.empty(bs, seq_len, 1, device=dev).uniform_(-1, 1)
            return torch.cat([xk, t], dim=-1)

        model = PINC(1, 0, hidden_size=32, hidden_layer=2)
        lrn = PINCLearner(
            model,
            gen,
            residual,
            state_range=[(-1, 1)],
            cond_range=[],
            t_max=T,
            ic_weight=1.0,
            steps_per_epoch=50,
            bs=1024,
            val_steps=5,
            device=torch.device("cpu"),
            show_bar=False,
        )
        assert len(lrn.dls.train) == 50  # scheduler total-step count
        lrn.fit_flat_cos(15, lr=3e-3)

        # soft IC: the trained net reproduces x_k at the time origin to good (not exact) accuracy
        xk = torch.linspace(-0.9, 0.9, 50)[:, None, None]
        t0 = torch.full_like(xk, -1.0)  # normalized t -> physical t=0
        with torch.no_grad():
            ic = lrn.model(torch.cat([xk, t0], dim=-1))
        assert (ic - xk).abs().max() < 5e-2

        # single-step accuracy: x(T) == x0 * exp(-lambda*T)
        with torch.no_grad():
            pred = lrn.model(torch.cat([xk, torch.ones_like(xk)], dim=-1))  # t_scaled=+1 == t=T
        assert (pred - xk * math.exp(-LAM * T)).abs().max() < 8e-2

        # autoregressive rollout compounds the per-step map over 20 steps
        roll = lrn.as_rollout(t_sample=T)
        with torch.no_grad():
            traj = roll(xk[:, 0], torch.empty(50, 20, 0))
        assert (traj[:, 0] - xk[:, 0] * math.exp(-LAM * T)).abs().max() < 8e-2
        assert (traj[:, -1] - xk[:, 0] * math.exp(-LAM * T * 20)).abs().max() < 1e-1

    def test_ic_weight_zero_lets_ic_drift(self, fast_cpu):
        """With ``ic_weight=0`` nothing anchors the IC, so it is free to drift — the knob bites."""
        from tsfast.pinn.pinc import PINC, PINCLearner

        torch.manual_seed(0)

        def residual(x, cond, dxdt):
            return F.mse_loss(dxdt + 2.0 * x, torch.zeros_like(x))

        def gen(bs, seq_len, dev):
            return torch.cat(
                [
                    torch.empty(bs, seq_len, 1, device=dev).uniform_(-1, 1),
                    torch.empty(bs, seq_len, 1, device=dev).uniform_(-1, 1),
                ],
                dim=-1,
            )

        def make(ic_weight):
            torch.manual_seed(0)
            m = PINC(1, 0, hidden_size=32, hidden_layer=2)
            lrn = PINCLearner(
                m, gen, residual,
                state_range=[(-1, 1)], cond_range=[], t_max=0.5, ic_weight=ic_weight,
                steps_per_epoch=50, bs=512, val_steps=4,
                device=torch.device("cpu"), show_bar=False,
            )
            lrn.fit_flat_cos(10, lr=3e-3)
            xk = torch.linspace(-0.9, 0.9, 50)[:, None, None]
            t0 = torch.full_like(xk, -1.0)
            with torch.no_grad():
                return (lrn.model(torch.cat([xk, t0], dim=-1)) - xk).abs().max().item()

        ic_err_on = make(1.0)
        ic_err_off = make(0.0)
        assert ic_err_on < ic_err_off  # the soft IC penalty measurably improves the IC
