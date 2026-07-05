"""Equivalence tests for the fused selective-scan backends (Triton GPU, C++ CPU).

Each backend's forward and full gradient set (lam, v, x0) are checked against the
pure-PyTorch doubling scan (the reference semantics), in float32, across several shapes
including the L=1 edge case, with and without an initial state. A final test compares the
whole DeepMamba parameter-gradient set backend-vs-doubling on the real model.
"""

import pytest
import torch

import tsfast.models._core.scan as scan
from tsfast.models.architectures.mamba import DeepMamba
from tsfast.models._core.scan_backends import selective_c, selective_triton

SHAPES = [(2, 7, 3), (4, 257, 8), (2, 1, 3)]

_C_OK = selective_c.is_available()
_CUDA_OK = torch.cuda.is_available() and selective_triton._HAVE_TRITON

_BACKENDS = [
    pytest.param(selective_c, "cpu", marks=pytest.mark.skipif(not _C_OK, reason="no C++ toolchain/ninja")),
    pytest.param(selective_triton, "cuda", marks=pytest.mark.skipif(not _CUDA_OK, reason="no CUDA/triton")),
]


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


@pytest.fixture(autouse=True)
def _reset_backend(monkeypatch):
    monkeypatch.setattr(scan, "backend", "auto")
    yield


def _reference(lam, v, x0, g):
    """Forward + backward through the forced doubling scan on cloned leaves."""
    scan.backend = "doubling"
    li, vi = lam.clone().requires_grad_(), v.clone().requires_grad_()
    x0i = x0.clone().requires_grad_() if x0 is not None else None
    out = scan.selective_recurrence(li, vi, x0i)
    out.backward(g)
    scan.backend = "auto"
    return out, li.grad, vi.grad, (x0i.grad if x0 is not None else None)


@pytest.mark.parametrize("mod, dev", _BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_x0", [False, True])
def test_forward_and_grads_match_doubling(mod, dev, shape, with_x0):
    B, L, N = shape
    torch.manual_seed(0)
    lam = torch.rand(B, L, N, device=dev) * 0.9 + 0.05
    v = torch.randn(B, L, N, device=dev)
    x0 = torch.randn(B, N, device=dev) if with_x0 else None
    g = torch.randn(B, L, N, device=dev)

    assert mod.supports(lam, v, x0) is None

    li, vi = lam.clone().requires_grad_(), v.clone().requires_grad_()
    x0i = x0.clone().requires_grad_() if with_x0 else None
    out = mod.run(li, vi, x0i)
    out.backward(g)

    ref_out, ref_glam, ref_gv, ref_gx0 = _reference(lam, v, x0, g)
    assert _rel(out, ref_out) < 1e-4
    assert _rel(li.grad, ref_glam) < 1e-4
    assert _rel(vi.grad, ref_gv) < 1e-4
    if with_x0:
        assert _rel(x0i.grad, ref_gx0) < 1e-4


@pytest.mark.parametrize("mod, dev", _BACKENDS)
def test_deepmamba_param_grads_match_doubling(mod, dev):
    """Max relative parameter-grad diff backend-vs-doubling on the real DeepMamba model."""
    torch.manual_seed(0)
    m = DeepMamba(3, 2, d_model=32, d_state=16, n_layers=2, return_state=True).to(dev)
    u = torch.randn(4, 48, 3, device=dev)

    def run(scan_backend):
        # DeepMamba always runs the parallel "scan" path; the module-level scan.backend
        # override picks the fused kernel ("auto" -> this backend on its device) or the
        # pure-PyTorch reference ("doubling").
        scan.backend = scan_backend
        for p in m.parameters():
            p.grad = None
        # chunk handoff exactly as tests/test_mamba.py exercises the stateful path
        out1, state = m(u[:, :20])
        out2, state = m(u[:, 20:22], state=state)
        out3, _ = m(u[:, 22:], state=state)
        out = torch.cat((out1, out2, out3), dim=1)
        (out**2).mean().backward()
        return out, [p.grad.clone() for p in m.parameters()]

    out_ref, g_ref = run("doubling")
    out_k, g_k = run("auto")  # resolves to this backend on its device
    assert _rel(out_k, out_ref) < 1e-3
    worst = max(_rel(a, b) for a, b in zip(g_k, g_ref))
    assert worst < 1e-3, f"worst param-grad rel diff {worst:.2e}"
