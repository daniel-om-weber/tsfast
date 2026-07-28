"""Generated-C++ execution backend for NeuralStateSpace: fast CPU rollout and BPTT.

The per-step Python dispatch of the naive rollout dominates its runtime on CPU. This backend
generates a C++ rollout specialized to the layer spec (dims baked as compile-time constants so
the tiny GEMVs fully unroll and vectorize), builds it once per spec through ``load_cabi``, and
parallelizes over the batch: with ATen's intra-op thread pool (``at::parallel_for``) where the
toolchain matches torch's threading backend, and with Grand Central Dispatch on macOS, which
ships with the OS and frees the user from providing an OpenMP runtime for Apple clang.

The entry points are plain ``extern "C"`` functions over ``float*`` and the two sizes, called
through ``ctypes``. Keeping ``torch/extension.h`` out of the translation unit is what makes the
per-spec build cheap: that header costs some forty times what a kernel body costs to compile,
and pybind11 buys nothing at a boundary that only passes pointers.

Requires a host C++ toolchain (g++/clang); ``is_available`` verifies this by building a tiny
probe once per process (disk-cached afterwards). The object targets the host CPU
(``-march=native``/``-mcpu=native``), so a kernel cache shared between machines of different
CPU generations must not be reused.

Backward follows the split-BPTT design: the sequential state-adjoint recurrence runs in C++
(reverse sweep re-using the hidden activations stored by the training forward), while the
parameter gradients are batched GEMMs over the ``[B*L, .]`` flattened adjoints
(``mlp_param_grads``, applied inside the ``tsfast::ssm_rollout_bwd`` op). Inputs arrive
from the ``tsfast::ssm_rollout*`` custom ops as contiguous float32 CPU tensors.
"""

__all__ = [
    "supports",
    "forward_train",
    "forward_infer",
    "backward",
    "is_available",
]

import ctypes
import sys

import torch

from ..._core.kernel_c import (
    _ACT_C,
    _ACT_C_DARWIN,
    _BATCH_PARALLEL_ATEN,
    _BATCH_PARALLEL_GCD,
    _FAST_TANH_C,
    is_available,
    load_cabi,
)
from .core import SSMSpec, rollout_unsupported, saved_widths

_EXTENSIONS: dict[SSMSpec, object] = {}

#: Gates this generator emits, screened in ``supports`` via ``rollout_unsupported``.
_GATES = ("none", "leak", "gru", "residual")


def _gen_source(spec: SSMSpec) -> str:
    """Emit the spec-specialized C++ forward/backward rollout."""
    dims = spec.dims
    nx, nu, k = spec.n_state, spec.n_input, spec.n_linear
    gated = spec.gate != "none"
    # Every gated mode scales a stored vector by its gate; they differ in which vector (the
    # candidate offset for the lerp forms, the candidate itself for the unit-carry form), in
    # the eps factor, and in how much of the adjoint bypasses the MLP. "leak" further takes
    # its gate from a parameter rather than the MLP, so its final layer stays n_state wide
    # and it stores no per-step pre-activation.
    leaky = spec.gate == "leak"
    residual = spec.gate == "residual"
    gv = "cv" if residual else "dv"
    gv_init = "xn[i]" if residual else "xn[i] - x[i]"
    scale = "EPS * " if residual else ""
    a_arg = ", const float* ap" if leaky else ""
    gl_arg = ", float* glp" if leaky else ""
    n_out_last = dims[k]
    darwin = sys.platform == "darwin"
    act, dact = (_ACT_C_DARWIN if darwin else _ACT_C)[spec.act]
    lines: list[str] = [
        "#include <ATen/Parallel.h>",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstdint>",
        "",
        _BATCH_PARALLEL_GCD if darwin else _BATCH_PARALLEL_ATEN,
    ]
    if darwin and "fast_tanhf" in act:
        lines.append(_FAST_TANH_C)
    lines += [
        f"constexpr int NX = {nx};",
        f"constexpr int NU = {nu};",
        *([f"constexpr float EPS = {spec.eps!r}f;"] if residual else []),
        "",
    ]

    # ---------------------------------------------------------------- forward
    # The gated forward stores the candidate offset d = c - x and the gate pre-activation s
    # alongside the hidden activations; the reverse sweep needs no other per-step state.
    gate_store = (", float* gdp" + ("" if leaky else ", float* gsp")) if gated else ""
    z_args = "".join(f", float* z{i}p" for i in range(k - 1)) + gate_store
    w_args = "".join(f", const float* w{i}p, const float* c{i}p" for i in range(k))
    lines += [
        f'extern "C" void ssm_fwd(int64_t B, int64_t L, const float* up, const float* x0p{a_arg}{w_args}, '
        f"float* outp{z_args}, bool store_z) {{",
        "    batch_parallel(B, [&](int64_t b_begin, int64_t b_end) {",
        "    for (int64_t b = b_begin; b < b_end; ++b) {",
        "        float x[NX];",
        "        for (int i = 0; i < NX; ++i) x[i] = x0p[b * NX + i];",
        "        for (int64_t t = 0; t < L; ++t) {",
        "            const float* ut = up + (b * L + t) * NU;",
    ]
    prev = None  # local array holding the previous layer's output
    for i in range(k):
        n_in, n_out = dims[i], dims[i + 1]
        dst = "xn" if i == k - 1 else f"h{i}"
        lines.append(f"            float {dst}[{n_out}];")
        lines.append(f"            for (int o = 0; o < {n_out}; ++o) {{")
        lines.append(f"                const float* wr = w{i}p + o * {n_in};")
        lines.append(f"                float acc = c{i}p[o];")
        if i == 0:
            lines.append("                for (int j = 0; j < NX; ++j) acc += wr[j] * x[j];")
            lines.append("                for (int j = 0; j < NU; ++j) acc += wr[NX + j] * ut[j];")
        else:
            lines.append(f"                for (int j = 0; j < {n_in}; ++j) acc += wr[j] * {prev}[j];")
        lines.append(f"                {dst}[o] = acc;")
        lines.append("            }")
        if i < k - 1:
            # separate loop so the activation vectorizes (libmvec via -ffast-math on
            # glibc, the inline fast_tanhf polynomial on macOS)
            lines.append(f"            for (int o = 0; o < {n_out}; ++o) {dst}[o] = " + act.format(a=f"{dst}[o]") + ";")
            lines.append(
                f"            if (store_z) for (int o = 0; o < {n_out}; ++o) "
                f"z{i}p[(b * L + t) * {n_out} + o] = {dst}[o];"
            )
        prev = dst
    if leaky:
        # The gate is the parameter vector ap, so xn is just the candidate; the offset it
        # moves along is still what the reverse sweep needs for dL/da.
        lines += [
            "            float dv[NX];",
            "            for (int i = 0; i < NX; ++i) dv[i] = xn[i] - x[i];",
            "            if (store_z) for (int i = 0; i < NX; ++i) gdp[(b * L + t) * NX + i] = dv[i];",
            "            for (int i = 0; i < NX; ++i) {",
            "                x[i] += ap[i] * dv[i];",
            "                outp[(b * L + t) * NX + i] = x[i];",
            "            }",
        ]
    elif gated:
        # xn holds [candidate | gate pre-activation]. The lerp moves the state a z-fraction of
        # the way to the candidate, keeping a diag(1 - z) path in the Jacobian; "residual"
        # instead adds a gated increment and keeps an exactly unit carry.
        lines += [
            f"            float {gv}[NX], sv[NX];",
            f"            for (int i = 0; i < NX; ++i) {{ {gv}[i] = {gv_init}; sv[i] = xn[NX + i]; }}",
            "            if (store_z) for (int i = 0; i < NX; ++i) {",
            f"                gdp[(b * L + t) * NX + i] = {gv}[i];",
            "                gsp[(b * L + t) * NX + i] = sv[i];",
            "            }",
            "            for (int i = 0; i < NX; ++i) {",
            f"                x[i] += {scale}{gv}[i] / (1.0f + expf(-sv[i]));",
            "                outp[(b * L + t) * NX + i] = x[i];",
            "            }",
        ]
    else:
        lines.append("            for (int i = 0; i < NX; ++i) { x[i] = xn[i]; outp[(b * L + t) * NX + i] = xn[i]; }")
    lines += [
        "        }",
        "    }",
        "    });",
        "}",
        "",
    ]

    # --------------------------------------------------------------- backward
    # Reverse state-adjoint sweep. wt{i} are the transposed weights [n_in, n_out] so the
    # per-row reductions read contiguous memory. Emits gy (total per-step output adjoint),
    # ga{i} (hidden pre-activation adjoints) and gx0 (= final carry) for the GEMM stage.
    gate_read = (", const float* gdp" + ("" if leaky else ", const float* gsp")) if gated else ""
    zb_args = "".join(f", const float* z{i}p" for i in range(k - 1)) + gate_read
    wt_args = "".join(f", const float* wt{i}p" for i in range(k))
    ga_args = "".join(f", float* ga{i}p" for i in range(k - 1))
    lines += [
        f'extern "C" void ssm_bwd(int64_t B, int64_t L, const float* goutp{a_arg}{zb_args}{wt_args}, '
        f"float* gyp{ga_args}, float* gx0p{gl_arg}) {{",
        "    batch_parallel(B, [&](int64_t b_begin, int64_t b_end) {",
        "    for (int64_t b = b_begin; b < b_end; ++b) {",
        "        float carry[NX] = {0.0f};",
        *(["        float glacc[NX] = {0.0f};"] if leaky else []),
        "        for (int64_t t = L - 1; t >= 0; --t) {",
    ]
    if leaky:
        # dL/da sums g * d over batch and time; accumulate the time axis here and let the
        # host reduce the batch, which keeps the sequential sweep free of atomics.
        lines += [
            "            float gyv[NX], gtot[NX], gdir[NX];",
            "            for (int i = 0; i < NX; ++i) {",
            "                gtot[i] = goutp[(b * L + t) * NX + i] + carry[i];",
            "                gyv[i] = ap[i] * gtot[i];",
            "                gdir[i] = (1.0f - ap[i]) * gtot[i];",
            "                glacc[i] += gtot[i] * gdp[(b * L + t) * NX + i];",
            "                gyp[(b * L + t) * NX + i] = gyv[i];",
            "            }",
        ]
    elif gated:
        # gtot is the total adjoint of x_{t+1}; it splits into the candidate and gate columns
        # of the final linear, plus a direct path that bypasses the MLP entirely — (1 - z) of
        # the adjoint for the lerp, all of it for the unit carry of "residual".
        lines += [
            f"            float gyv[{n_out_last}], gtot[NX], gdir[NX];",
            "            for (int i = 0; i < NX; ++i) {",
            "                gtot[i] = goutp[(b * L + t) * NX + i] + carry[i];",
            "                const float z = 1.0f / (1.0f + expf(-gsp[(b * L + t) * NX + i]));",
            f"                gyv[i] = {scale}z * gtot[i];",
            f"                gyv[NX + i] = {scale}z * (1.0f - z) * gtot[i] * gdp[(b * L + t) * NX + i];",
            "                gdir[i] = " + ("gtot[i];" if residual else "(1.0f - z) * gtot[i];"),
            "            }",
            f"            for (int o = 0; o < {n_out_last}; ++o) gyp[(b * L + t) * {n_out_last} + o] = gyv[o];",
        ]
    else:
        lines += [
            "            float gyv[NX];",
            "            for (int i = 0; i < NX; ++i) {",
            "                gyv[i] = goutp[(b * L + t) * NX + i] + carry[i];",
            "                gyp[(b * L + t) * NX + i] = gyv[i];",
            "            }",
        ]
    prev = "gyv"
    for i in range(k - 1, 0, -1):  # back through linears K-1..1 onto their inputs
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
    n_out0 = dims[1]
    lines += [
        "            for (int j = 0; j < NX; ++j) {",
        f"                const float* wr = wt0p + j * {n_out0};",
        "                float acc = 0.0f;",
        f"                for (int o = 0; o < {n_out0}; ++o) acc += wr[o] * {prev}[o];",
        "                carry[j] = acc" + (" + gdir[j];" if gated else ";"),
        "            }",
        "        }",
        "        for (int i = 0; i < NX; ++i) gx0p[b * NX + i] = carry[i];",
        *(["        for (int i = 0; i < NX; ++i) glp[b * NX + i] = glacc[i];"] if leaky else []),
        "    }",
        "    });",
        "}",
    ]
    return "\n".join(lines)


def _get_extension(spec: SSMSpec):
    ext = _EXTENSIONS.get(spec)
    if ext is None:
        ext = load_cabi(_gen_source(spec), "tsfast_ssm_c")
        _EXTENSIONS[spec] = ext
    return ext


def _call(fn, B: int, L: int, tensors: list[torch.Tensor], *tail) -> None:
    """Invoke a generated entry point on the tensors' raw storage.

    The kernels index every argument as contiguous float32, and a C ABI has no way to notice
    otherwise: a narrower dtype makes the kernel read past the end of the allocation. So the
    layout is checked here rather than trusted from the callers — ``supports`` screens the
    rollout inputs but not the parameters, and ``backward`` does not pass through it at all.
    The check is per rollout, not per step.
    """
    for t in tensors:
        if t.dtype is not torch.float32 or t.device.type != "cpu" or not t.is_contiguous():
            raise TypeError(
                "the C backend requires contiguous float32 CPU tensors, got "
                f"{t.dtype} on {t.device}{'' if t.is_contiguous() else ' (non-contiguous)'}"
            )
    fn(ctypes.c_int64(B), ctypes.c_int64(L), *(ctypes.c_void_p(t.data_ptr()) for t in tensors), *tail)


def _run_fwd(ext, spec: SSMSpec, u, x0, params, store_z, leak=None):
    B, L = u.shape[0], u.shape[1]
    out = torch.empty(B, L, spec.n_state, dtype=torch.float32)
    zs = [torch.empty(B, L, w, dtype=torch.float32) for w in saved_widths(spec)]
    wb = [t.detach().contiguous() for t in params]
    a = [leak.detach().contiguous()] if spec.gate == "leak" else []
    _call(ext.ssm_fwd, B, L, [u, x0, *a, *wb, out, *zs], ctypes.c_bool(store_z))
    return out, zs


def supports(spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor) -> str | None:
    """Reason the generated C++ kernels cannot handle these inputs, or None when they can."""
    reason = rollout_unsupported(spec, u, x0, "cpu", _GATES)
    if reason is not None:
        return reason
    if not is_available():
        return "no host C++ toolchain / ninja"
    return None


def forward_train(
    spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, params: list[torch.Tensor], leak: torch.Tensor | None = None
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Rollout that also stores the hidden activations: returns ``(out, zs)``."""
    return _run_fwd(_get_extension(spec), spec, u, x0, params, store_z=True, leak=leak)


def forward_infer(
    spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, params: list[torch.Tensor], leak: torch.Tensor | None = None
) -> torch.Tensor:
    """Rollout without stored intermediates: returns ``out`` of shape ``[B, L, n_state]``."""
    out, _ = _run_fwd(_get_extension(spec), spec, u, x0, params, store_z=False, leak=leak)
    return out


def backward(
    spec: SSMSpec,
    grad_out: torch.Tensor,
    zs: list[torch.Tensor],
    weights: list[torch.Tensor],
    leak: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
    """Reverse state-adjoint sweep: returns ``(gy, gas, gx0, gleak)`` for the shared GEMM stage."""
    ext = _get_extension(spec)
    B, L = grad_out.shape[0], grad_out.shape[1]
    wts = [w.detach().t().contiguous() for w in weights]
    gy = torch.empty(B, L, spec.out_width, dtype=torch.float32)
    gas = [torch.empty_like(z) for z in zs[: len(spec.hidden)]]
    gx0 = torch.empty(B, spec.n_state, dtype=torch.float32)
    leaky = spec.gate == "leak"
    a = [leak.detach().contiguous()] if leaky else []
    # per-batch partials; the batch axis of dL/da reduces on the host
    gl = [torch.empty(B, spec.n_state, dtype=torch.float32)] if leaky else []
    _call(ext.ssm_bwd, B, L, [grad_out, *a, *zs, *wts, gy, *gas, gx0, *gl])
    return gy, gas, gx0, (gl[0].sum(0) if leaky else None)
