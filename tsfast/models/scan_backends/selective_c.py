"""Generated-C++ backend for the selective (time-varying diagonal) recurrence.

The pure-PyTorch doubling scan materializes O(L log L) intermediates and re-reads the full
``[B, L, N]`` tensors ~log2(L) times. This backend runs the recurrence as a flat loop
compiled once via ``torch.utils.cpp_extension.load_inline``: OpenMP (or ATen's thread pool)
parallelizes over the ``B x N`` lanes, and each lane streams sequentially over ``L`` carrying
a scalar state. It serves the CPU training fallback and the long CPU test-signal inference
(identibench feeds the whole simulation test signal in one call).

Backward is the analytic reverse-time adjoint ``G_t = g_t + a_{t+1} G_{t+1}`` swept once per
lane (time descending), emitting ``grad_v = G``, ``grad_lam = G x_{t-1}`` and
``grad_x0 = a_0 G_0``. Only ``lam`` and the forward output feed the backward, so its memory
is O(L). Real float32 only (the Mamba selective scan), so no conjugation is needed. The
toolchain probe, compile flags, and batch-parallel scaffold are shared with the SSM C backend.
"""

__all__ = [
    "supports",
    "run",
]

import hashlib
import sys

import torch

from ..ssm.backend_c import _BATCH_PARALLEL_ATEN, _BATCH_PARALLEL_GCD, _build_flags, is_available

_EXTENSION = None


def _source() -> str:
    darwin = sys.platform == "darwin"
    return "\n".join(
        [
            "#include <torch/extension.h>",
            "#include <ATen/Parallel.h>",
            "#include <algorithm>",
            "",
            _BATCH_PARALLEL_GCD if darwin else _BATCH_PARALLEL_ATEN,
            "",
            # ------------------------------------------------------------ forward
            "void sel_fwd(torch::Tensor lam, torch::Tensor v, torch::Tensor x0,",
            "             torch::Tensor out, bool has_x0) {",
            "    const int64_t B = lam.size(0), L = lam.size(1), N = lam.size(2);",
            "    const float* lamp = lam.data_ptr<float>();",
            "    const float* vp = v.data_ptr<float>();",
            "    const float* x0p = has_x0 ? x0.data_ptr<float>() : nullptr;",
            "    float* outp = out.data_ptr<float>();",
            "    batch_parallel(B * N, [&](int64_t l0, int64_t l1) {",
            "    for (int64_t l = l0; l < l1; ++l) {",
            "        const int64_t b = l / N, n = l % N;",
            "        float x = has_x0 ? x0p[b * N + n] : 0.0f;",
            "        for (int64_t t = 0; t < L; ++t) {",
            "            const int64_t p = (b * L + t) * N + n;",
            "            x = lamp[p] * x + vp[p];",
            "            outp[p] = x;",
            "        }",
            "    }",
            "    });",
            "}",
            "",
            # ----------------------------------------------------------- backward
            "void sel_bwd(torch::Tensor lam, torch::Tensor out, torch::Tensor g, torch::Tensor x0,",
            "             torch::Tensor glam, torch::Tensor gv, torch::Tensor gx0,",
            "             bool has_x0, bool need_x0) {",
            "    const int64_t B = lam.size(0), L = lam.size(1), N = lam.size(2);",
            "    const float* lamp = lam.data_ptr<float>();",
            "    const float* outp = out.data_ptr<float>();",
            "    const float* gp = g.data_ptr<float>();",
            "    const float* x0p = has_x0 ? x0.data_ptr<float>() : nullptr;",
            "    float* glamp = glam.data_ptr<float>();",
            "    float* gvp = gv.data_ptr<float>();",
            "    float* gx0p = need_x0 ? gx0.data_ptr<float>() : nullptr;",
            "    batch_parallel(B * N, [&](int64_t l0, int64_t l1) {",
            "    for (int64_t l = l0; l < l1; ++l) {",
            "        const int64_t b = l / N, n = l % N;",
            "        float G = 0.0f;",
            "        for (int64_t t = L - 1; t >= 0; --t) {",
            "            const int64_t p = (b * L + t) * N + n;",
            "            const float a_next = (t + 1 < L) ? lamp[p + N] : 0.0f;",
            "            G = gp[p] + a_next * G;",
            "            gvp[p] = G;",
            "            const float xprev = (t > 0) ? outp[p - N] : (has_x0 ? x0p[b * N + n] : 0.0f);",
            "            glamp[p] = G * xprev;",
            "        }",
            "        if (need_x0) gx0p[b * N + n] = lamp[(b * L) * N + n] * G;",
            "    }",
            "    });",
            "}",
        ]
    )


def _get_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load_inline

        src = _source()
        cflags, ldflags = _build_flags()
        tag = hashlib.md5("".join((src, *cflags, *ldflags)).encode()).hexdigest()[:10]
        _EXTENSION = load_inline(
            name=f"tsfast_scan_selective_c_{tag}",
            cpp_sources=src,
            functions=["sel_fwd", "sel_bwd"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
    return _EXTENSION


def _forward(ext, lam, v, x0):
    out = torch.empty_like(lam)
    has_x0 = x0 is not None
    ext.sel_fwd(lam, v, x0 if has_x0 else lam, out, has_x0)
    return out


class _SelectiveC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ext, lam, v, x0):
        out = _forward(ext, lam, v, x0)
        ctx.ext = ext
        ctx.save_for_backward(lam, out, x0)
        ctx.has_x0 = x0 is not None
        return out

    @staticmethod
    def backward(ctx, g):
        ext = ctx.ext
        lam, out, x0 = ctx.saved_tensors
        g = g.contiguous()
        B, L, N = lam.shape
        has_x0 = ctx.has_x0
        _, need_lam, need_v, need_x0 = ctx.needs_input_grad
        need_x0 = bool(need_x0 and has_x0)
        glam = torch.empty_like(lam)
        gv = torch.empty_like(lam)
        gx0 = torch.empty(B, N, dtype=lam.dtype) if need_x0 else lam
        ext.sel_bwd(lam, out, g, x0 if has_x0 else lam, glam, gv, gx0, has_x0, need_x0)
        grad_lam = glam if need_lam else None
        grad_v = gv if need_v else None
        grad_x0 = gx0 if need_x0 else None
        return None, grad_lam, grad_v, grad_x0


def supports(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> str | None:
    """Reason the C++ kernel cannot handle these tensors, or None if it can."""
    if lam.device.type != "cpu" or v.device.type != "cpu":
        return "inputs not on CPU"
    if not is_available():
        return "no host C++ toolchain / ninja"
    if lam.dtype != torch.float32 or v.dtype != torch.float32:
        return f"needs float32, got lam={lam.dtype}, v={v.dtype}"
    if lam.shape != v.shape:
        return f"lam.shape {tuple(lam.shape)} != v.shape {tuple(v.shape)}"
    if lam.dim() < 2:
        return f"needs a leading batch and a time dim, got {lam.dim()} dims"
    if x0 is not None:
        expected = lam.shape[:-2] + lam.shape[-1:]
        if x0.dtype != torch.float32:
            return f"x0 must be float32, got {x0.dtype}"
        if tuple(x0.shape) != tuple(expected):
            return f"x0.shape {tuple(x0.shape)} != expected {tuple(expected)}"
    return None


def run(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    """Selective recurrence ``x_t = lam_t x_{t-1} + v_t`` via the compiled C++ extension."""
    ext = _get_extension()
    lam3 = lam.contiguous().reshape(-1, lam.shape[-2], lam.shape[-1])
    v3 = v.contiguous().reshape(-1, v.shape[-2], v.shape[-1])
    x03 = x0.contiguous().reshape(-1, x0.shape[-1]) if x0 is not None else None
    if not torch.is_grad_enabled() or not (
        lam3.requires_grad or v3.requires_grad or (x03 is not None and x03.requires_grad)
    ):
        out = _forward(ext, lam3, v3, x03)
    else:
        out = _SelectiveC.apply(ext, lam3, v3, x03)
    return out.reshape(lam.shape)
