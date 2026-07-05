"""Tests for tsfast.models._core.scan (diagonal_recurrence, selective_recurrence)."""

import pytest
import torch

from tsfast.models._core.scan import (
    _diagonal_recurrence_sequential,
    diagonal_recurrence,
    selective_recurrence,
)


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class TestDiagonalRecurrence:
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_matches_sequential(self, dtype):
        torch.manual_seed(0)
        for L in (1, 7, 100):
            lam = torch.rand(5, 3).to(dtype) * 0.9
            if dtype.is_complex:
                lam = lam * torch.exp(1j * torch.rand(5, 3, dtype=dtype) * 3.0)
            v = torch.randn(2, 5, L, 3).to(dtype)
            x0 = torch.randn(2, 5, 3).to(dtype)
            for init in (None, x0):
                x_scan = diagonal_recurrence(lam, v, init)
                x_seq = _diagonal_recurrence_sequential(lam, v, init)
                assert (x_scan - x_seq).abs().max() < 1e-10

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("with_x0", [True, False])
    def test_gradcheck(self, dtype, with_x0):
        torch.manual_seed(0)
        lam = (torch.rand(2, 3).to(dtype) * 0.8).requires_grad_()
        v = torch.randn(2, 9, 3, dtype=dtype, requires_grad=True)
        x0 = torch.randn(2, 3, dtype=dtype, requires_grad=True) if with_x0 else None
        assert torch.autograd.gradcheck(diagonal_recurrence, (lam, v, x0))

    def test_gradcheck_broadcast_lam(self):
        # lam without batch dims, broadcast against a batched v: grads must reduce correctly
        torch.manual_seed(0)
        lam = (torch.rand(3, dtype=torch.complex128) * 0.8).requires_grad_()
        v = torch.randn(2, 5, 9, 3, dtype=torch.complex128, requires_grad=True)
        assert torch.autograd.gradcheck(diagonal_recurrence, (lam, v, None))


class TestSelectiveRecurrence:
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_matches_sequential(self, dtype):
        torch.manual_seed(0)
        for L in (1, 7, 100):
            lam = torch.rand(2, 5, L, 3).to(dtype) * 0.9
            v = torch.randn(2, 5, L, 3).to(dtype)
            x0 = torch.randn(2, 5, 3).to(dtype)
            for init in (None, x0):
                x_scan = selective_recurrence(lam, v, init)
                x_seq = _diagonal_recurrence_sequential(lam, v, init)
                assert (x_scan - x_seq).abs().max() < 1e-10

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("with_x0", [True, False])
    def test_gradcheck(self, dtype, with_x0):
        torch.manual_seed(0)
        lam = (torch.rand(2, 9, 3).to(dtype) * 0.9).requires_grad_()
        v = torch.randn(2, 9, 3, dtype=dtype, requires_grad=True)
        x0 = torch.randn(2, 3, dtype=dtype, requires_grad=True) if with_x0 else None
        assert torch.autograd.gradcheck(selective_recurrence, (lam, v, x0))

    def test_gradcheck_broadcast_lam(self):
        # lam missing the leading batch dim, as in the constant-specialization test
        torch.manual_seed(0)
        lam = (torch.rand(5, 9, 3, dtype=torch.complex128) * 0.9).requires_grad_()
        v = torch.randn(2, 5, 9, 3, dtype=torch.complex128, requires_grad=True)
        assert torch.autograd.gradcheck(selective_recurrence, (lam, v, None))

    def test_matches_constant_specialization(self):
        torch.manual_seed(0)
        lam = torch.rand(5, 3, dtype=torch.complex128) * 0.9
        v = torch.randn(2, 5, 40, 3, dtype=torch.complex128)
        x_const = diagonal_recurrence(lam, v)
        x_var = selective_recurrence(lam.unsqueeze(-2).expand(5, 40, 3), v)
        assert (x_const - x_var).abs().max() < 1e-10


@pytest.mark.parametrize("fn_kind", ["diagonal", "selective"])
def test_cuda_parity(fn_kind):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    B, L, N = 4, 257, 8
    lam_shape = (B, L, N) if fn_kind == "selective" else (B, N)
    fn = selective_recurrence if fn_kind == "selective" else diagonal_recurrence
    lam = torch.rand(*lam_shape, dtype=torch.cfloat) * 0.9
    v = torch.randn(B, L, N, dtype=torch.cfloat)
    x0 = torch.randn(B, N, dtype=torch.cfloat)

    def run(lam, v, x0):
        lam, v, x0 = lam.clone().requires_grad_(), v.clone().requires_grad_(), x0.clone().requires_grad_()
        x = fn(lam, v, x0)
        x.abs().sum().backward()
        return x, lam.grad, v.grad, x0.grad

    outs_cpu = run(lam, v, x0)
    outs_gpu = run(lam.cuda(), v.cuda(), x0.cuda())
    for a, b in zip(outs_gpu, outs_cpu):
        assert _rel(a.cpu(), b) < 1e-4
