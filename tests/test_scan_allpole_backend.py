"""Equivalence tests for the fused all-pole Triton backend vs the matrix doubling scan.

The all-pole kernel computes ``y_t = w_t - sum_i a_i y_{t-i}``, the scalar form of the
companion-matrix recurrence of ``LinearDynamicalOperator``. The reference oracle is
``linear_recurrence`` over the equivalent companion matrix (its own adjoint is pinned by
gradcheck in test_dynonet_adjoint.py). Forward output and all three gradients (a, w, y0)
must match across model-realistic and edge shapes, with and without an initial state,
and the fused DynoNet forward must stay equivalent to the doubling path, including
chunked state carry with chunks shorter than the filter order.
"""

import pytest
import torch
import torch.nn.functional as F

import tsfast.models._core.scan as scan
from tsfast.models.architectures.dynonet import DynoNet, linear_recurrence
from tsfast.models.architectures.dynonet import allpole_triton

# (B, n_pairs, L, na): dynoNet-realistic lane counts and orders, plus edges
_SHAPES = [
    (32, 8, 200, 2),  # WH-like G-block
    (4, 32, 100, 8),  # largest searched order
    (3, 5, 7, 3),
    (2, 4, 1, 2),  # L = 1 edge
    (2, 4, 2, 4),  # L < na edge
    (1, 1, 513, 1),  # single lane, first-order, odd length
]


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _skip_unless_supported(shape=(2, 3, 4, 2)):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    B, P, L, na = shape
    reason = allpole_triton.supports(torch.zeros(P, na, device="cuda"), torch.zeros(B, P, L, device="cuda"), None)
    if reason:
        pytest.skip(f"allpole triton backend unavailable: {reason}")


def _companion(a):
    na = a.shape[-1]
    shift = torch.eye(na, device=a.device, dtype=a.dtype)[:-1].expand(a.shape[0], na - 1, na)
    return torch.cat((-a.unsqueeze(1), shift), dim=1)


def _reference(a, w, y0):
    """Companion-matrix doubling scan of the same filter (y = first state component)."""
    x = linear_recurrence(_companion(a), F.pad(w.unsqueeze(-1), (0, a.shape[-1] - 1)), y0)
    return x[..., 0]


@pytest.mark.parametrize("B,P,L,na", _SHAPES)
@pytest.mark.parametrize("with_y0", [True, False])
def test_matches_matrix_scan(B, P, L, na, with_y0):
    _skip_unless_supported()
    torch.manual_seed(B * L + na)
    a0 = torch.randn(P, na, device="cuda") * 0.25
    w0 = torch.randn(B, P, L, device="cuda")
    y00 = torch.randn(B, P, na, device="cuda") if with_y0 else None
    gy = torch.randn(B, P, L, device="cuda")

    results = []
    for fn in (allpole_triton.run, _reference):
        a = a0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        y0 = y00.clone().requires_grad_() if with_y0 else None
        y = fn(a, w, y0)
        grads = torch.autograd.grad(y, [t for t in (a, w, y0) if t is not None], gy)
        results.append((y.detach(), *[g.detach() for g in grads]))
    labels = ["out", "grad_a", "grad_w", "grad_y0"]
    for name, got, ref in zip(labels, results[0], results[1]):
        assert _rel(got, ref) < 1e-4, f"({B},{P},{L},{na}) y0={with_y0} {name}: rel {_rel(got, ref):.2e}"


def test_a_broadcast_over_full_batch():
    """a carrying explicit batch dims must produce per-batch grad_a (no fold)."""
    _skip_unless_supported()
    torch.manual_seed(0)
    B, P, L, na = 4, 3, 50, 2
    a0 = torch.randn(B, P, na, device="cuda") * 0.25
    w0 = torch.randn(B, P, L, device="cuda")
    gy = torch.randn(B, P, L, device="cuda")
    grads = {}
    for name, fn in (
        ("triton", allpole_triton.run),
        ("ref", lambda a, w, y0: _reference(a.reshape(B * P, na), w.reshape(B * P, L), None).reshape(B, P, L)),
    ):
        a = a0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        y = fn(a, w, None)
        grads[name] = torch.autograd.grad(y, (a, w), gy)
    for got, ref in zip(grads["triton"], grads["ref"]):
        assert _rel(got, ref) < 1e-4


def test_inference_path_matches_grad_path():
    """run() under no_grad must return the same forward output as the autograd path."""
    _skip_unless_supported()
    torch.manual_seed(0)
    a = torch.randn(6, 3, device="cuda") * 0.25
    w = torch.randn(4, 6, 128, device="cuda")
    y0 = torch.randn(4, 6, 3, device="cuda")
    with torch.no_grad():
        out_nograd = allpole_triton.run(a, w, y0)
    out_grad = allpole_triton.run(a.requires_grad_(), w.requires_grad_(), y0.requires_grad_())
    assert _rel(out_nograd, out_grad) < 1e-6


class TestDynoNetFusedPath:
    def _prep_model(self, **kw):
        torch.manual_seed(0)
        m = DynoNet(3, 2, n_channels=4, nb=4, na=kw.pop("na", 2), **kw).cuda()
        with torch.no_grad():
            for op in (m.g1, m.g2, m.g_lin):
                op.a_coeff.uniform_(-0.3, 0.3)
        return m

    def test_model_matches_doubling(self):
        """Fused DynoNet forward and every parameter grad match the doubling path."""
        _skip_unless_supported()
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            m = self._prep_model()
            u = torch.randn(5, 64, 3, device="cuda")
            res = {}
            for backend in ("auto", "doubling"):
                scan.backend = backend
                try:
                    for p in m.parameters():
                        p.grad = None
                    out = m(u)
                    out.square().mean().backward()
                finally:
                    scan.backend = "auto"
                res[backend] = (out.detach().clone(), [p.grad.clone() for p in m.parameters()])
            assert _rel(res["auto"][0], res["doubling"][0]) < 1e-5
            rels = [_rel(g, r) for g, r in zip(res["auto"][1], res["doubling"][1])]
            assert max(rels) < 1e-4, f"max relative param-grad diff {max(rels):.2e}"
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def test_stateful_chunked_equivalence(self):
        """Chunked rollout through the fused kernel equals the full sequence, chunks < na included."""
        _skip_unless_supported()
        m = self._prep_model(na=4, return_state=True)
        u = torch.randn(3, 40, 3, device="cuda")
        full, _ = m(u)
        o1, st = m(u[:, :10])
        o2, st = m(u[:, 10:12], state=st)  # 2-step chunk, shorter than na=4 and nb-1
        o3, st = m(u[:, 12:14], state=st)
        o4, _ = m(u[:, 14:], state=st)
        chunked = torch.cat((o1, o2, o3, o4), dim=1)
        assert _rel(chunked, full) < 1e-5
