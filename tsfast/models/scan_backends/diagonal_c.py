"""Generated-C++ execution backend for the constant-coefficient diagonal scan (LRU/S5 core).

The doubling scan of ``scan.diagonal_recurrence`` sweeps the sequence ``ceil(log2(L))`` times;
this backend runs the recurrence ``x_t = lam * x_{t-1} + v_t`` as a single sequential pass and
its analytic adjoint as a single reverse pass, parallelizing over the ``batch x n`` independent
lanes (each lane owns one constant coefficient and streams its column of ``v`` once). The
generated source is dtype-generic through a ``c10::complex<float>`` template instantiation, so
the same reverse recurrence carries the conjugates ``conj(lam)`` (in the gradient scan) and
``conj(x_{t-1})`` (in the ``grad_lam`` reduction) that the complex case needs. Toolchain probing,
compile flags, and the batch-parallel scaffolding are shared with the SSM/NARX C backends.

Only the coefficients and the forward output are saved for backward (matching the doubling
implementation's O(L) memory contract); the per-lane ``grad_lam`` partials are reduced to the
broadcast shape of ``lam`` on the torch side, as are ``grad_v`` and ``grad_x0``.
"""

__all__ = [
    "supports",
    "run",
]

import hashlib
import sys

import torch

from ..ssm.backend_c import (
    _BATCH_PARALLEL_ATEN,
    _BATCH_PARALLEL_GCD,
    _build_flags,
    is_available,
)

_DTYPES = (torch.float32, torch.complex64)
_EXT = None


# ----------------------------------------------------------------------- shared torch-side glue
# _prep / _reduce are the broadcast bookkeeping both the C and the Triton backend share: flatten
# the recurrence to M = prod(batch dims) independent lanes of N states over L steps, materialize
# lam and x0 in that [M, N] lane layout, and reduce the per-lane gradients back to the (possibly
# broadcast) shapes of the original lam / v / x0. The Triton backend imports both from here.


def _prep(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None):
    """Broadcast to the output shape and flatten to lane layout.

    Returns ``(lam_lane [M, N], v_flat [M, L, N], x0_lane [M, N] | None, meta)`` all contiguous,
    where ``meta = (out_shape, batch_dims, M, L, N)``.
    """
    out_shape = torch.broadcast_shapes(lam.unsqueeze(-2).shape, v.shape)
    bdims = tuple(out_shape[:-2])
    L, n = out_shape[-2], out_shape[-1]
    m = 1
    for d in bdims:
        m *= d
    lam_lane = lam.broadcast_to(bdims + (n,)).reshape(m, n).contiguous()
    v_flat = v.broadcast_to(out_shape).reshape(m, L, n).contiguous()
    x0_lane = None if x0 is None else x0.broadcast_to(bdims + (n,)).reshape(m, n).contiguous()
    return lam_lane, v_flat, x0_lane, (out_shape, bdims, m, L, n)


def _reduce(grad_v_flat, grad_lam_lane, grad_x0_lane, lam, v, x0, meta, needs):
    """Reduce per-lane gradients to the broadcast shapes of ``lam`` / ``v`` / ``x0``.

    ``needs`` is ``(need_lam, need_v, need_x0)``; grads for absent needs are None.
    """
    out_shape, bdims, _m, _l, n = meta
    grad_lam = grad_v = grad_x0 = None
    if needs[0]:
        grad_lam = grad_lam_lane.reshape(bdims + (n,)).sum_to_size(lam.shape)
    if needs[1]:
        grad_v = grad_v_flat.reshape(out_shape).sum_to_size(v.shape)
    if x0 is not None and needs[2]:
        grad_x0 = grad_x0_lane.reshape(bdims + (n,)).sum_to_size(x0.shape)
    return grad_lam, grad_v, grad_x0


# ------------------------------------------------------------------------------ generated source


def _source() -> str:
    darwin = sys.platform == "darwin"
    return "\n".join(
        [
            "#include <torch/extension.h>",
            "#include <ATen/Parallel.h>",
            "#include <c10/util/complex.h>",
            "#include <algorithm>",
            "",
            _BATCH_PARALLEL_GCD if darwin else _BATCH_PARALLEL_ATEN,
            "using cf = c10::complex<float>;",
            "static inline float cconj(float x) { return x; }",
            "static inline cf cconj(cf z) { return cf(z.real(), -z.imag()); }",
            "",
            "template <typename T>",
            "static void fwd_impl(const T* lam, const T* v, const T* x0, T* out,",
            "                     bool has_x0, int64_t M, int64_t L, int64_t N) {",
            "    batch_parallel(M * N, [&](int64_t l0, int64_t l1) {",
            "    for (int64_t lane = l0; lane < l1; ++lane) {",
            "        const int64_t m = lane / N, c = lane % N;",
            "        const T a = lam[lane];",
            "        T x = has_x0 ? x0[lane] : T(0);",
            "        const int64_t base = m * L * N + c;",
            "        for (int64_t t = 0; t < L; ++t) {",
            "            const int64_t idx = base + t * N;",
            "            x = a * x + v[idx];",
            "            out[idx] = x;",
            "        }",
            "    }",
            "    });",
            "}",
            "",
            "template <typename T>",
            "static void bwd_impl(const T* g, const T* lam, const T* out, const T* x0, bool has_x0,",
            "                     T* gv, T* glam, T* gx0, int64_t M, int64_t L, int64_t N) {",
            "    batch_parallel(M * N, [&](int64_t l0, int64_t l1) {",
            "    for (int64_t lane = l0; lane < l1; ++lane) {",
            "        const int64_t m = lane / N, c = lane % N;",
            "        const T a = lam[lane];",
            "        const T ac = cconj(a);",
            "        const int64_t base = m * L * N + c;",
            "        T G = T(0), gl = T(0);",
            "        for (int64_t t = L - 1; t >= 0; --t) {",
            "            const int64_t idx = base + t * N;",
            "            G = g[idx] + ac * G;",
            "            gv[idx] = G;",
            "            const T xp = (t > 0) ? out[base + (t - 1) * N] : (has_x0 ? x0[lane] : T(0));",
            "            gl += G * cconj(xp);",
            "        }",
            "        glam[lane] = gl;",
            "        if (has_x0) gx0[lane] = ac * G;",  # G holds G_1 after the loop
            "    }",
            "    });",
            "}",
            "",
            "void diag_fwd(torch::Tensor lam, torch::Tensor v, torch::Tensor x0, torch::Tensor out,",
            "              bool has_x0, bool is_complex, int64_t M, int64_t L, int64_t N) {",
            "    if (is_complex)",
            "        fwd_impl<cf>(lam.data_ptr<cf>(), v.data_ptr<cf>(), has_x0 ? x0.data_ptr<cf>() : nullptr,",
            "                     out.data_ptr<cf>(), has_x0, M, L, N);",
            "    else",
            "        fwd_impl<float>(lam.data_ptr<float>(), v.data_ptr<float>(),",
            "                        has_x0 ? x0.data_ptr<float>() : nullptr, out.data_ptr<float>(), has_x0, M, L, N);",
            "}",
            "",
            "void diag_bwd(torch::Tensor g, torch::Tensor lam, torch::Tensor out, torch::Tensor x0,",
            "              torch::Tensor gv, torch::Tensor glam, torch::Tensor gx0, bool has_x0,",
            "              bool is_complex, int64_t M, int64_t L, int64_t N) {",
            "    if (is_complex)",
            "        bwd_impl<cf>(g.data_ptr<cf>(), lam.data_ptr<cf>(), out.data_ptr<cf>(),",
            "                     has_x0 ? x0.data_ptr<cf>() : nullptr, has_x0, gv.data_ptr<cf>(),",
            "                     glam.data_ptr<cf>(), has_x0 ? gx0.data_ptr<cf>() : nullptr, M, L, N);",
            "    else",
            "        bwd_impl<float>(g.data_ptr<float>(), lam.data_ptr<float>(), out.data_ptr<float>(),",
            "                        has_x0 ? x0.data_ptr<float>() : nullptr, has_x0, gv.data_ptr<float>(),",
            "                        glam.data_ptr<float>(), has_x0 ? gx0.data_ptr<float>() : nullptr, M, L, N);",
            "}",
        ]
    )


def _get_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        src = _source()
        cflags, ldflags = _build_flags()
        tag = hashlib.md5("".join((src, *cflags, *ldflags)).encode()).hexdigest()[:10]
        _EXT = load_inline(
            name=f"tsfast_diag_c_{tag}",
            cpp_sources=src,
            functions=["diag_fwd", "diag_bwd"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
    return _EXT


def supports(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> str | None:
    """Reason this backend cannot handle the inputs, or None when it can (see module docstring)."""
    if v.device.type != "cpu":
        return f"input on {v.device.type}, C backend is CPU-only"
    if v.dtype not in _DTYPES:
        return f"dtype {v.dtype} unsupported (need float32 or complex64)"
    if lam.dtype != v.dtype:
        return f"lam dtype {lam.dtype} != v dtype {v.dtype}"
    if x0 is not None and x0.dtype != v.dtype:
        return f"x0 dtype {x0.dtype} != v dtype {v.dtype}"
    if v.dim() < 2:
        return "v must have at least a time and a state axis"
    if lam.shape[-1] != v.shape[-1]:
        return f"state dim mismatch: lam {tuple(lam.shape)} vs v {tuple(v.shape)}"
    if not is_available():
        return "no host C++ toolchain / ninja"
    return None


def _forward(ext, lam_lane, v_flat, x0_lane, meta):
    _os, _bd, m, L, n = meta
    is_complex = v_flat.is_complex()
    out = torch.empty_like(v_flat)
    x0 = x0_lane if x0_lane is not None else v_flat[:0]
    ext.diag_fwd(lam_lane, v_flat, x0, out, x0_lane is not None, is_complex, m, L, n)
    return out


class _CDiagonal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ext, lam, v, x0):
        lam_lane, v_flat, x0_lane, meta = _prep(lam, v, x0)
        out = _forward(ext, lam_lane, v_flat, x0_lane, meta)
        ctx.ext, ctx.meta = ext, meta
        ctx.lam, ctx.v, ctx.x0 = lam, v, x0
        ctx.save_for_backward(lam_lane, out, x0_lane)
        return out.reshape(meta[0])

    @staticmethod
    def backward(ctx, grad_out):
        ext = ctx.ext
        _os, _bd, m, L, n = ctx.meta
        lam_lane, out, x0_lane = ctx.saved_tensors
        has_x0 = x0_lane is not None
        is_complex = out.is_complex()
        g = grad_out.reshape(m, L, n).contiguous()
        gv = torch.empty_like(out)
        glam = torch.empty_like(lam_lane)
        gx0 = torch.empty_like(lam_lane) if has_x0 else lam_lane[:0]
        x0 = x0_lane if has_x0 else out[:0]
        ext.diag_bwd(g, lam_lane, out, x0, gv, glam, gx0, has_x0, is_complex, m, L, n)
        needs = (ctx.needs_input_grad[1], ctx.needs_input_grad[2], ctx.needs_input_grad[3])
        grad_lam, grad_v, grad_x0 = _reduce(gv, glam, gx0, ctx.lam, ctx.v, ctx.x0, ctx.meta, needs)
        return None, grad_lam, grad_v, grad_x0


def run(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    """Run the constant-coefficient diagonal recurrence through the C++ extension (autograd-capable)."""
    ext = _get_ext()
    if not torch.is_grad_enabled() or not any(t is not None and t.requires_grad for t in (lam, v, x0)):
        lam_lane, v_flat, x0_lane, meta = _prep(lam, v, x0)
        return _forward(ext, lam_lane, v_flat, x0_lane, meta).reshape(meta[0])
    return _CDiagonal.apply(ext, lam, v, x0)
