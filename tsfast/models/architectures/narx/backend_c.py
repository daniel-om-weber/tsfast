"""Generated-C++ execution backend for NarxMLP: fast CPU free-run rollout and BPTT.

The per-step Python dispatch of the naive rollout dominates its runtime on CPU. This backend
generates a C++ recurrence specialized to the layer spec (dims baked as compile-time constants
so the tiny GEMVs fully unroll and vectorize), compiles it once per spec via
``torch.utils.cpp_extension.load_inline``, and parallelizes over the batch. Toolchain probing,
compile flags, activation snippets, and the batch-parallel scaffolding are shared with the SSM
C backend.

The kernel owns only the sequential feedback recurrence: it receives the precomputed
input-window contribution ``hu = conv_u(u)`` per step and returns the prediction sequence.
Backward follows the split-BPTT design: the reverse sweep carries the lag-buffer adjoint
(each fed-back prediction receives gradient from up to ``na`` future steps) and emits the
per-step adjoints; parameter gradients are batched GEMMs over the ``[B*L, .]`` flattened
adjoints (``narx_param_grads``), and the gradient w.r.t. ``hu`` hands the chain back to
autograd for the convolution path. The entry points serve the ``tsfast::narx_rollout*``
custom ops, which deliver contiguous float32 tensors.
"""

__all__ = [
    "supports",
    "forward_infer",
    "forward_train",
    "backward",
    "is_available",
]

import hashlib
import sys

import torch

from ..._core.kernel_c import (
    _ACT_C,
    _BATCH_PARALLEL_ATEN,
    _BATCH_PARALLEL_GCD,
    _FAST_TANH_C,
    _build_flags,
    is_available,
)
from .core import NarxSpec

_EXTENSIONS: dict[NarxSpec, object] = {}


def supports(spec: NarxSpec, hu: torch.Tensor, y_true: torch.Tensor) -> str | None:
    """Reason the C++ kernel cannot handle these inputs, or None when it can."""
    if hu.device.type != "cpu" or y_true.device.type != "cpu":
        return "inputs not on CPU"
    if not is_available():
        return "no host C++ toolchain / ninja"
    if hu.dtype != torch.float32 or y_true.dtype != torch.float32:
        return f"needs float32, got hu={hu.dtype}, y_true={y_true.dtype}"
    if hu.dim() != 3 or hu.shape[-1] != spec.hidden[0]:
        return f"expected hu of shape [B, L, {spec.hidden[0]}], got {tuple(hu.shape)}"
    if y_true.shape != (*hu.shape[:2], spec.n_y):
        return f"expected y_true of shape [{hu.shape[0]}, {hu.shape[1]}, {spec.n_y}], got {tuple(y_true.shape)}"
    # The kernel indexes raw data pointers, so a non-dense layout reads the wrong elements.
    if not hu.is_contiguous() or not y_true.is_contiguous():
        return "inputs must be contiguous"
    return None


def _gen_source(spec: NarxSpec) -> str:
    """Emit the spec-specialized C++ forward/backward recurrence."""
    dims = spec.dims
    ny, nb, k = spec.n_y, spec.n_buf, spec.n_linear
    darwin = sys.platform == "darwin"
    act, dact = _ACT_C[spec.act]
    lines: list[str] = [
        "#include <torch/extension.h>",
        "#include <ATen/Parallel.h>",
        "#include <algorithm>",
        "#include <cmath>",
        "",
        _BATCH_PARALLEL_GCD if darwin else _BATCH_PARALLEL_ATEN,
    ]
    if "fast_tanhf" in act:
        lines.append(_FAST_TANH_C)
    lines += [
        f"constexpr int NY = {ny};",
        f"constexpr int NB = {nb};",
        f"constexpr int H0 = {dims[1]};",
        "",
    ]

    # ---------------------------------------------------------------- forward
    z_args = "".join(f", torch::Tensor z{i}" for i in range(k - 1))
    w_args = "torch::Tensor wy" + "".join(f", torch::Tensor w{i}, torch::Tensor c{i}" for i in range(1, k))
    lines += [
        f"void narx_fwd(torch::Tensor hu, torch::Tensor ytrue, int64_t washout, {w_args}, "
        f"torch::Tensor out{z_args}, bool store_z) {{",
        "    const int64_t B = hu.size(0), L = hu.size(1);",
        "    const float* hup = hu.data_ptr<float>();",
        "    const float* ytp = ytrue.data_ptr<float>();",
        "    float* outp = out.data_ptr<float>();",
        "    const float* wyp = wy.data_ptr<float>();",
    ]
    for i in range(1, k):
        lines.append(f"    const float* w{i}p = w{i}.data_ptr<float>();")
        lines.append(f"    const float* c{i}p = c{i}.data_ptr<float>();")
    for i in range(k - 1):
        lines.append(f"    float* z{i}p = z{i}.data_ptr<float>();")
    lines += [
        "    batch_parallel(B, [&](int64_t b_begin, int64_t b_end) {",
        "    for (int64_t b = b_begin; b < b_end; ++b) {",
        "        float v[NB] = {0.0f};",
        "        for (int64_t t = 0; t < L; ++t) {",
    ]
    # first layer: hu[t] + Wy v, activation
    lines += [
        f"            float h0[{dims[1]}];",
        f"            for (int o = 0; o < {dims[1]}; ++o) {{",
        "                const float* wr = wyp + o * NB;",
        "                float acc = hup[(b * L + t) * H0 + o];",
        "                for (int j = 0; j < NB; ++j) acc += wr[j] * v[j];",
        "                h0[o] = acc;",
        "            }",
        f"            for (int o = 0; o < {dims[1]}; ++o) h0[o] = " + act.format(a="h0[o]") + ";",
        f"            if (store_z) for (int o = 0; o < {dims[1]}; ++o) z0p[(b * L + t) * {dims[1]} + o] = h0[o];",
    ]
    prev = "h0"
    for i in range(1, k):
        n_in, n_out = dims[i], dims[i + 1]
        dst = "yt" if i == k - 1 else f"h{i}"
        lines.append(f"            float {dst}[{n_out}];")
        lines.append(f"            for (int o = 0; o < {n_out}; ++o) {{")
        lines.append(f"                const float* wr = w{i}p + o * {n_in};")
        lines.append(f"                float acc = c{i}p[o];")
        lines.append(f"                for (int j = 0; j < {n_in}; ++j) acc += wr[j] * {prev}[j];")
        lines.append(f"                {dst}[o] = acc;")
        lines.append("            }")
        if i < k - 1:
            lines.append(f"            for (int o = 0; o < {n_out}; ++o) {dst}[o] = " + act.format(a=f"{dst}[o]") + ";")
            lines.append(
                f"            if (store_z) for (int o = 0; o < {n_out}; ++o) "
                f"z{i}p[(b * L + t) * {n_out} + o] = {dst}[o];"
            )
        prev = dst
    lines += [
        "            const float* fed = (t < washout) ? (ytp + (b * L + t) * NY) : yt;",
        "            for (int i = 0; i < NY; ++i) outp[(b * L + t) * NY + i] = yt[i];",
        "            for (int j = 0; j < NB - NY; ++j) v[j] = v[j + NY];",
        "            for (int i = 0; i < NY; ++i) v[NB - NY + i] = fed[i];",
        "        }",
        "    }",
        "    });",
        "}",
        "",
    ]

    # --------------------------------------------------------------- backward
    # Reverse buffer-adjoint sweep. wyt/wt{i} are the transposed weights so per-row
    # reductions read contiguous memory. Emits gy (total per-step prediction adjoint),
    # gfed (adjoint of the fed sample, routed to y_true during washout) and ga{i}
    # (hidden pre-activation adjoints; ga0 doubles as the hu gradient).
    zb_args = "".join(f", torch::Tensor z{i}" for i in range(k - 1))
    wt_args = "torch::Tensor wyt" + "".join(f", torch::Tensor wt{i}" for i in range(1, k))
    ga_args = "".join(f", torch::Tensor ga{i}" for i in range(k - 1))
    lines += [
        f"void narx_bwd(torch::Tensor gout, int64_t washout{zb_args}, {wt_args}, "
        f"torch::Tensor gy, torch::Tensor gfed{ga_args}) {{",
        "    const int64_t B = gout.size(0), L = gout.size(1);",
        "    const float* goutp = gout.data_ptr<float>();",
        "    float* gyp = gy.data_ptr<float>();",
        "    float* gfedp = gfed.data_ptr<float>();",
        "    const float* wytp = wyt.data_ptr<float>();",
    ]
    for i in range(k - 1):
        lines.append(f"    const float* z{i}p = z{i}.data_ptr<float>();")
        lines.append(f"    float* ga{i}p = ga{i}.data_ptr<float>();")
    for i in range(1, k):
        lines.append(f"    const float* wt{i}p = wt{i}.data_ptr<float>();")
    lines += [
        "    batch_parallel(B, [&](int64_t b_begin, int64_t b_end) {",
        "    for (int64_t b = b_begin; b < b_end; ++b) {",
        "        float carry[NB] = {0.0f};",
        "        for (int64_t t = L - 1; t >= 0; --t) {",
        "            float gyv[NY];",
        "            for (int i = 0; i < NY; ++i) {",
        "                const float gf = carry[NB - NY + i];",
        "                gfedp[(b * L + t) * NY + i] = gf;",
        "                gyv[i] = goutp[(b * L + t) * NY + i] + (t >= washout ? gf : 0.0f);",
        "                gyp[(b * L + t) * NY + i] = gyv[i];",
        "            }",
    ]
    prev = "gyv"
    for i in range(k - 1, 0, -1):  # back through linears k-1..1 onto their inputs
        n_in, n_out = dims[i], dims[i + 1]
        lines.append(f"            float g{i - 1}[{n_in}];")
        lines.append(f"            for (int j = 0; j < {n_in}; ++j) {{")
        lines.append(f"                const float* wr = wt{i}p + j * {n_out};")
        lines.append("                float acc = 0.0f;")
        lines.append(f"                for (int o = 0; o < {n_out}; ++o) acc += wr[o] * {prev}[o];")
        lines.append(f"                const float zv = z{i - 1}p[(b * L + t) * {n_in} + j];")
        lines.append(f"                g{i - 1}[j] = acc * " + dact.format(z="zv") + ";")
        lines.append(f"                ga{i - 1}p[(b * L + t) * {n_in} + j] = g{i - 1}[j];")
        lines.append("            }")
        prev = f"g{i - 1}"
    lines += [
        "            for (int j = NB - 1; j >= 0; --j) {",
        f"                const float* wr = wytp + j * {dims[1]};",
        "                float acc = j >= NY ? carry[j - NY] : 0.0f;",
        f"                for (int o = 0; o < {dims[1]}; ++o) acc += wr[o] * {prev}[o];",
        "                carry[j] = acc;",
        "            }",
        "        }",
        "    }",
        "    });",
        "}",
    ]
    return "\n".join(lines)


def _get_extension(spec: NarxSpec):
    ext = _EXTENSIONS.get(spec)
    if ext is None:
        from torch.utils.cpp_extension import load_inline

        src = _gen_source(spec)
        cflags, ldflags = _build_flags()
        tag = hashlib.md5("".join((src, *cflags, *ldflags)).encode()).hexdigest()[:10]
        ext = load_inline(
            name=f"tsfast_narx_c_{tag}",
            cpp_sources=src,
            functions=["narx_fwd", "narx_bwd"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
        _EXTENSIONS[spec] = ext
    return ext


def _run_fwd(spec: NarxSpec, hu, y_true, washout, params, store_z):
    ext = _get_extension(spec)
    B, L = hu.shape[0], hu.shape[1]
    out = torch.empty(B, L, spec.n_y, dtype=torch.float32)
    zs = [torch.empty(B, L, h, dtype=torch.float32) for h in spec.hidden]
    wb = [t.contiguous() for t in params]
    ext.narx_fwd(hu, y_true, washout, *wb, out, *zs, store_z)
    return out, zs


def forward_infer(
    spec: NarxSpec, hu: torch.Tensor, y_true: torch.Tensor, washout: int, params: list[torch.Tensor]
) -> torch.Tensor:
    """Forward rollout without stored intermediates, returning ``out [B, L, n_y]``."""
    out, _ = _run_fwd(spec, hu, y_true, washout, params, store_z=False)
    return out


def forward_train(
    spec: NarxSpec, hu: torch.Tensor, y_true: torch.Tensor, washout: int, params: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward rollout storing the post-activation hidden sequences, returning ``(out, zs)``."""
    return _run_fwd(spec, hu, y_true, washout, params, store_z=True)


def backward(
    spec: NarxSpec, grad_out: torch.Tensor, zs: list[torch.Tensor], weights: list[torch.Tensor], washout: int
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Reverse buffer-adjoint sweep, returning ``(gy, gfed, gas)`` (see ``narx_param_grads``)."""
    ext = _get_extension(spec)
    B, L = grad_out.shape[0], grad_out.shape[1]
    wts = [w.t().contiguous() for w in weights]
    gy = torch.empty(B, L, spec.n_y, dtype=torch.float32)
    gfed = torch.empty(B, L, spec.n_y, dtype=torch.float32)
    gas = [torch.empty_like(z) for z in zs]
    ext.narx_bwd(grad_out, washout, *zs, *wts, gy, gfed, *gas)
    return gy, gfed, gas
