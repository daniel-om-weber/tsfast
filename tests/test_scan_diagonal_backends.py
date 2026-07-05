"""Equivalence tests for the diagonal-scan kernel backends (triton/c) vs the doubling scan.

Each backend's forward output and the three gradients (lam, v, x0) must match the pure-PyTorch
doubling implementation for float32 and complex64, across model-realistic and edge shapes, with
and without an initial state. The doubling backend is the reference oracle (its own equivalence
to the sequential loop is covered in test_scan.py).
"""

import pytest
import torch

import tsfast.models.scan as scan
from tsfast.models.scan_backends import diagonal_c, diagonal_triton

# (v_shape, lam_shape): model-realistic (LRU d_state=64 -> n=64; lam broadcast over batch/time),
# a per-batch lam, tiny shapes, and the L=1 edge.
_SHAPES = [
    ((32, 200, 64), (64,)),  # LRU-like
    ((16, 200, 32), (32,)),  # S5-like (conj_sym halves the state)
    ((4, 257, 8), (4, 8)),  # lam carries a batch axis
    ((2, 7, 3), (3,)),
    ((3, 1, 6), (6,)),  # L = 1 edge
    ((1, 8192, 24), (24,)),  # single row, long L: exercises the C backend's time-chunked passes
    ((2, 300, 100), (100,)),  # state dim above the C backend's block width: ragged tail block
]


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _make(v_shape, lam_shape, dtype, device):
    lam = torch.rand(*lam_shape) * 0.85
    v = torch.randn(*v_shape)
    x0 = torch.randn(v_shape[0], v_shape[-1])
    if dtype.is_complex:
        lam = lam * torch.exp(1j * torch.rand(*lam_shape) * 3.0)
        v = v + 1j * torch.randn(*v_shape)
        x0 = x0 + 1j * torch.randn_like(x0)
    to = lambda t: t.to(dtype).to(device)  # noqa: E731
    return to(lam), to(v), to(x0)


def _fwd_bwd(backend, lam, v, x0):
    lam = lam.clone().requires_grad_()
    v = v.clone().requires_grad_()
    x0 = None if x0 is None else x0.clone().requires_grad_()
    scan.backend = backend
    try:
        out = scan.diagonal_recurrence(lam, v, x0)
        (out.abs() ** 2).sum().backward()
    finally:
        scan.backend = "auto"
    return out, lam.grad, v.grad, (None if x0 is None else x0.grad)


def _check(backend_name, device, dtype, v_shape, lam_shape, with_x0):
    torch.manual_seed(0)
    lam, v, x0 = _make(v_shape, lam_shape, dtype, device)
    init = x0 if with_x0 else None
    ref = _fwd_bwd("doubling", lam, v, init)
    got = _fwd_bwd(backend_name, lam, v, init)
    labels = ["out", "grad_lam", "grad_v", "grad_x0"]
    for name, a, b in zip(labels, got, ref):
        if a is None:
            continue
        assert _rel(a, b) < 1e-4, f"{backend_name} {dtype} {v_shape} {name}: rel {_rel(a, b):.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("v_shape,lam_shape", _SHAPES)
@pytest.mark.parametrize("with_x0", [True, False])
def test_c_matches_doubling(dtype, v_shape, lam_shape, with_x0):
    if diagonal_c.supports(torch.zeros(lam_shape[-1], dtype=dtype), torch.zeros(v_shape, dtype=dtype), None):
        pytest.skip("c backend unavailable")
    _check("c", "cpu", dtype, v_shape, lam_shape, with_x0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("v_shape,lam_shape", _SHAPES)
@pytest.mark.parametrize("with_x0", [True, False])
def test_triton_matches_doubling(dtype, v_shape, lam_shape, with_x0):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    dev = torch.zeros(v_shape, dtype=dtype, device="cuda")
    if diagonal_triton.supports(torch.zeros(lam_shape[-1], dtype=dtype, device="cuda"), dev, None):
        pytest.skip("triton backend unavailable")
    _check("triton", "cuda", dtype, v_shape, lam_shape, with_x0)


def test_inference_path_matches_grad_path():
    """run() under no_grad must return the same forward output as the autograd path (both backends)."""
    torch.manual_seed(0)
    for name, mod, device in [("c", diagonal_c, "cpu"), ("triton", diagonal_triton, "cuda")]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        lam, v, x0 = _make((8, 64, 16), (16,), torch.complex64, device)
        if mod.supports(lam, v, x0):
            continue
        with torch.no_grad():
            out_nograd = mod.run(lam, v, x0)
        out_grad = mod.run(lam.requires_grad_(), v.requires_grad_(), x0.requires_grad_())
        assert _rel(out_nograd, out_grad) < 1e-5, name
