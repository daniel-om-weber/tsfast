"""Tests for tsfast.models.architectures.dynonet (linear_recurrence, LinearDynamicalOperator, DynoNet)."""

import numpy as np
import pytest
import torch


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _stable_matrices(n_batch, n, radius=0.95):
    A = torch.randn(n_batch, n, n, dtype=torch.float64)
    return A / torch.linalg.matrix_norm(A, 2).view(n_batch, 1, 1) * radius


def _run(m, backend, u):
    """Forward + backward on a cloned leaf; returns (out, param grads, du)."""
    m.backend = backend
    for p in m.parameters():
        p.grad = None
    u = u.clone().requires_grad_()
    out = m(u)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    return out, [p.grad.clone() for p in m.parameters()], u.grad.clone()


def _lfilter_reference(op, u):
    """Sum of per-pair scipy filters, the ground truth for a MIMO operator."""
    from scipy import signal

    B, L, _ = u.shape
    ref = np.zeros((B, L, op.out_channels))
    for i in range(op.out_channels):
        for j in range(op.in_channels):
            b = op.b_coeff[i, j].detach().numpy()
            a = np.concatenate(([1.0], op.a_coeff[i, j].detach().numpy()))
            ref[:, :, i] += signal.lfilter(b, a, u[:, :, j].numpy(), axis=-1)
    return ref


class TestLinearRecurrence:
    def test_scan_matches_sequential(self):
        from tsfast.models.architectures.dynonet import _linear_recurrence_sequential, linear_recurrence

        torch.manual_seed(0)
        for L in (1, 7, 100):
            A = _stable_matrices(5, 3)
            v = torch.randn(2, 5, L, 3, dtype=torch.float64)
            x0 = torch.randn(2, 5, 3, dtype=torch.float64)
            for init in (None, x0):
                x_scan = linear_recurrence(A, v, init)
                x_seq = _linear_recurrence_sequential(A, v, init)
                assert (x_scan - x_seq).abs().max() < 1e-10

    def test_scan_gradcheck(self):
        from tsfast.models.architectures.dynonet import linear_recurrence

        torch.manual_seed(0)
        A = (torch.randn(2, 2, 2, dtype=torch.float64) * 0.5).requires_grad_()
        v = torch.randn(2, 2, 5, 2, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(2, 2, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(linear_recurrence, (A, v, x0))


class TestLinearDynamicalOperator:
    def test_matches_lfilter_siso(self):
        from tsfast.models.architectures.dynonet import LinearDynamicalOperator

        torch.manual_seed(0)
        op = LinearDynamicalOperator(1, 1, nb=4, na=3).double()
        with torch.no_grad():
            op.b_coeff.copy_(torch.randn(1, 1, 4, dtype=torch.float64))
            op.a_coeff.copy_(torch.tensor([[[-0.5, 0.2, -0.1]]], dtype=torch.float64))
        u = torch.randn(2, 50, 1, dtype=torch.float64)
        ref = _lfilter_reference(op, u)
        assert np.abs(op(u).detach().numpy() - ref).max() < 1e-12

    def test_matches_lfilter_mimo(self):
        from tsfast.models.architectures.dynonet import LinearDynamicalOperator

        torch.manual_seed(1)
        op = LinearDynamicalOperator(2, 3, nb=3, na=2).double()
        with torch.no_grad():
            op.b_coeff.copy_(torch.randn(3, 2, 3, dtype=torch.float64) * 0.5)
            op.a_coeff.copy_(torch.randn(3, 2, 2, dtype=torch.float64) * 0.3)
        u = torch.randn(2, 40, 2, dtype=torch.float64)
        ref = _lfilter_reference(op, u)
        assert np.abs(op(u).detach().numpy() - ref).max() < 1e-12

    def test_shapes_and_edges(self):
        from tsfast.models.architectures.dynonet import LinearDynamicalOperator

        torch.manual_seed(0)
        op = LinearDynamicalOperator(2, 3, nb=4, na=2)
        y, state = op(torch.randn(5, 25, 2), return_state=True)
        assert y.shape == (5, 25, 3)
        assert state["u"].shape == (5, 3, 2) and state["x"].shape == (5, 6, 2)
        # nb=1 keeps an empty FIR tail, na=0 keeps an empty IIR state (pure FIR path)
        op = LinearDynamicalOperator(1, 2, nb=1, na=0).double()
        u = torch.randn(2, 9, 1, dtype=torch.float64)
        y, state = op(u, return_state=True)
        assert state["u"].shape == (2, 0, 1) and state["x"].shape == (2, 2, 0)
        assert np.abs(y.detach().numpy() - _lfilter_reference(op, u)).max() < 1e-12

    def test_matches_reference_dynonet_package(self):
        """Outputs and gradients agree with the authors' reference implementation (if installed)."""
        pytest.importorskip("dynonet")
        from dynonet.lti import MimoLinearDynamicalOperator

        from tsfast.models.architectures.dynonet import LinearDynamicalOperator

        torch.manual_seed(0)
        np.random.seed(0)
        in_ch, out_ch, nb, na, L = 2, 3, 4, 2, 200
        b = torch.randn(out_ch, in_ch, nb, dtype=torch.float64) * 0.5
        # sample coefficients from poles inside the unit circle so both filters stay stable
        a = np.zeros((out_ch, in_ch, na))
        for i in range(out_ch):
            for j in range(in_ch):
                r, th = np.random.uniform(0.3, 0.9), np.random.uniform(0, np.pi)
                p = r * np.exp(1j * th)
                a[i, j] = np.real(np.poly([p, p.conj()]))[1:]
        a = torch.tensor(a)

        ref = MimoLinearDynamicalOperator(in_ch, out_ch, n_b=nb, n_a=na).double()
        ours = LinearDynamicalOperator(in_ch, out_ch, nb=nb, na=na).double()
        with torch.no_grad():
            ref.b_coeff.copy_(b), ours.b_coeff.copy_(b)
            ref.a_coeff.copy_(a), ours.a_coeff.copy_(a)
        u = torch.randn(4, L, in_ch, dtype=torch.float64)
        u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
        y_ref, y_ours = ref(u_ref), ours(u_ours)
        w = torch.randn_like(y_ref)
        (y_ref * w).sum().backward()
        (y_ours * w).sum().backward()
        assert _rel(y_ours, y_ref) < 1e-12
        assert _rel(ours.b_coeff.grad, ref.b_coeff.grad) < 1e-12
        assert _rel(ours.a_coeff.grad, ref.a_coeff.grad) < 1e-12
        assert _rel(u_ours.grad, u_ref.grad) < 1e-12

    def test_backend_parity(self):
        from tsfast.models.architectures.dynonet import DynoNet

        torch.manual_seed(0)
        m = DynoNet(3, 2, n_channels=4, nb=4, na=2)
        with torch.no_grad():
            for op in (m.g1, m.g2, m.g_lin):
                op.a_coeff.uniform_(-0.3, 0.3)
        u = torch.randn(5, 64, 3)
        out_e, g_e, du_e = _run(m, "eager", u)
        out_s, g_s, du_s = _run(m, "scan", u)
        assert _rel(out_s, out_e) < 5e-5
        assert max(_rel(a, b) for a, b in zip(g_s, g_e)) < 5e-5
        assert _rel(du_s, du_e) < 5e-5

    def test_unknown_backend_raises(self):
        from tsfast.models.architectures.dynonet import LinearDynamicalOperator

        op = LinearDynamicalOperator(1, 1, backend="fft")
        with pytest.raises(ValueError):
            op(torch.randn(1, 10, 1))

    def test_unstable_poles_overflow_to_nonfinite(self):
        from tsfast.models.architectures.dynonet import LinearDynamicalOperator

        op = LinearDynamicalOperator(1, 1, nb=2, na=1)
        with torch.no_grad():
            op.a_coeff.fill_(-1.1)  # pole at 1.1, outside the unit circle
            op.b_coeff.fill_(1.0)
        y = op(torch.ones(1, 4096, 1))
        # the forward must complete; the Learner's NaN guard handles the resulting loss
        assert not torch.isfinite(y).all()


class TestDynoNet:
    def test_shapes(self):
        from tsfast.models.architectures.dynonet import DynoNet

        u = torch.randn(4, 25, 3)
        assert DynoNet(3, 2)(u).shape == (4, 25, 2)
        assert DynoNet(3, 2, bypass=False)(u).shape == (4, 25, 2)
        assert DynoNet(3, 2, hidden_layers=0)(u).shape == (4, 25, 2)

    def test_stateful_chunked_equivalence(self):
        from tsfast.models.architectures.dynonet import DynoNet

        torch.manual_seed(0)
        # chunk lengths indivisible by nb, one chunk (2) shorter than the FIR tail nb-1
        m = DynoNet(2, 1, n_channels=4, nb=4, na=2, return_state=True).double()
        with torch.no_grad():
            for op in (m.g1, m.g2, m.g_lin):
                op.a_coeff.uniform_(-0.3, 0.3)
        u = torch.randn(4, 30, 2, dtype=torch.float64)
        full, _ = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:12], state=state)
        out3, state = m(u[:, 12:25], state=state)
        out4, _ = m(u[:, 25:], state=state)
        chunked = torch.cat((out1, out2, out3, out4), dim=1)
        assert _rel(chunked, full) < 1e-12  # FIR tail + IIR states fully capture the dynamics

    def test_cuda_parity(self):
        from tsfast.models.architectures.dynonet import DynoNet

        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(0)
            m = DynoNet(3, 2, n_channels=4, nb=4, na=2)
            with torch.no_grad():
                for op in (m.g1, m.g2, m.g_lin):
                    op.a_coeff.uniform_(-0.3, 0.3)
            u = torch.randn(5, 64, 3)
            out_c, g_c, du_c = _run(m, "scan", u)
            m = m.cuda()
            out_g, g_g, du_g = _run(m, "scan", u.cuda())
            assert _rel(out_g.cpu(), out_c) < 5e-5
            assert max(_rel(a.cpu(), b) for a, b in zip(g_g, g_c)) < 5e-5
            assert _rel(du_g.cpu(), du_c) < 5e-5
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    @pytest.mark.slow
    def test_dynonet_learner_fit(self, dls_simulation):
        from tsfast.training import DynoNetLearner

        lrn = DynoNetLearner(dls_simulation, n_channels=4, nb=4, na=2, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))

    @pytest.mark.slow
    def test_dynonet_learner_tbptt_fit(self, dls_simulation):
        from tsfast.training import DynoNetLearner

        lrn = DynoNetLearner(dls_simulation, n_channels=4, nb=4, na=2, sub_seq_len=50, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))
