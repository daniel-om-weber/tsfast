"""Tests for the analytic matrix adjoint of ``linear_recurrence`` (custom autograd.Function).

The forward is the log-doubling matmul scan; the backward is the reverse-time scan
``G_t = A^T G_{t+1} + g_t`` with ``grad_v = G``, ``grad_A = sum_t outer(G_t, x_{t-1})`` and
``grad_x0 = A^T G_1``. gradcheck (fp64) pins the adjoint against numerical jacobians; the
model-level check pins scan-backend parameter grads against the sequential eager backend.
"""

import pytest
import torch


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class TestAdjointGradcheck:
    @pytest.mark.parametrize("L", [1, 7, 64])
    @pytest.mark.parametrize("with_x0", [False, True])
    def test_batched(self, L, with_x0):
        from tsfast.models.dynonet import linear_recurrence

        torch.manual_seed(L + int(with_x0))
        n_batch, n = 3, 2
        A = (torch.randn(n_batch, n, n, dtype=torch.float64) * 0.4).requires_grad_()
        v = torch.randn(n_batch, L, n, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(n_batch, n, dtype=torch.float64, requires_grad=True) if with_x0 else None
        inputs = (A, v, x0) if with_x0 else (A, v)
        assert torch.autograd.gradcheck(lambda *a: linear_recurrence(*a), inputs)

    @pytest.mark.parametrize("L", [1, 7, 64])
    @pytest.mark.parametrize("with_x0", [False, True])
    def test_broadcast_A_no_batch(self, L, with_x0):
        """A carries no batch dims and must broadcast against a batched v (grad_A folds the batch)."""
        from tsfast.models.dynonet import linear_recurrence

        torch.manual_seed(100 + L + int(with_x0))
        n_batch, n = 4, 3
        A = (torch.randn(n, n, dtype=torch.float64) * 0.3).requires_grad_()
        v = torch.randn(n_batch, L, n, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(n_batch, n, dtype=torch.float64, requires_grad=True) if with_x0 else None
        inputs = (A, v, x0) if with_x0 else (A, v)
        assert torch.autograd.gradcheck(lambda *a: linear_recurrence(*a), inputs)

    def test_broadcast_x0_no_batch(self):
        """x0 without batch dims broadcasts across a batched v; grad_x0 folds the batch."""
        from tsfast.models.dynonet import linear_recurrence

        torch.manual_seed(7)
        n_batch, n, L = 4, 2, 9
        A = (torch.randn(n_batch, n, n, dtype=torch.float64) * 0.4).requires_grad_()
        v = torch.randn(n_batch, L, n, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(n, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda *a: linear_recurrence(*a), (A, v, x0))


class TestAdjointMatchesEager:
    def test_recurrence_grads_match_sequential(self):
        """Param/input grads of the scan adjoint match autograd through the sequential loop."""
        from tsfast.models.dynonet import _linear_recurrence_sequential, linear_recurrence

        torch.manual_seed(0)
        n_batch, n, L = 5, 3, 64
        A0 = torch.randn(n_batch, n, n, dtype=torch.float64) * 0.4
        v0 = torch.randn(n_batch, L, n, dtype=torch.float64)
        x00 = torch.randn(n_batch, n, dtype=torch.float64)
        w = torch.randn(n_batch, L, n, dtype=torch.float64)

        grads = {}
        for name, fn in (("scan", linear_recurrence), ("eager", _linear_recurrence_sequential)):
            A = A0.clone().requires_grad_()
            v = v0.clone().requires_grad_()
            x0 = x00.clone().requires_grad_()
            (fn(A, v, x0) * w).sum().backward()
            grads[name] = (A.grad, v.grad, x0.grad)
        for gs, ge in zip(grads["scan"], grads["eager"]):
            assert _rel(gs, ge) < 1e-10

    def test_model_param_grads_scan_vs_eager(self):
        """Full DynoNet: every parameter grad from the scan backend matches the eager backend (fp64)."""
        from tsfast.models.dynonet import DynoNet

        torch.manual_seed(0)
        m = DynoNet(3, 2, n_channels=4, nb=4, na=2).double()
        with torch.no_grad():
            for op in (m.g1, m.g2, m.g_lin):
                op.a_coeff.uniform_(-0.3, 0.3)
        u = torch.randn(5, 64, 3, dtype=torch.float64)
        w = torch.randn(5, 64, 2, dtype=torch.float64)

        grads = {}
        for backend in ("scan", "eager"):
            m.backend = backend
            for p in m.parameters():
                p.grad = None
            (m(u) * w).sum().backward()
            grads[backend] = [p.grad.clone() for p in m.parameters()]
        rels = [_rel(gs, ge) for gs, ge in zip(grads["scan"], grads["eager"])]
        assert max(rels) < 1e-10, f"max relative param-grad diff {max(rels):.2e}"
