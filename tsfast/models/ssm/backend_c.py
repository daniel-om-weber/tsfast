"""Generated-C++ execution backend for NeuralStateSpace: fast CPU rollout and BPTT.

The per-step Python dispatch of the naive rollout dominates its runtime on CPU. This backend
generates a C++ rollout specialized to the layer spec (dims baked as compile-time constants so
the tiny GEMVs fully unroll and vectorize), compiles it once per spec via
``torch.utils.cpp_extension.load_inline``, and parallelizes over the batch with ATen's
intra-op thread pool (``at::parallel_for``), which exists on every platform PyTorch runs on.

Requires a host C++ toolchain (g++/clang) and ninja; ``is_available`` verifies this by
building a tiny probe extension once per process (disk-cached afterwards). The compiled
extension targets the host CPU (``-march=native``/``-mcpu=native``), so a torch-extensions
cache shared between machines of different CPU generations must not be reused.

Backward follows the split-BPTT design: the sequential state-adjoint recurrence runs in C++
(reverse sweep re-using the hidden activations stored by the training forward), while the
parameter gradients are batched GEMMs over the ``[B*L, .]`` flattened adjoints (``mlp_param_grads``).
"""

__all__ = [
    "c_rollout",
    "is_available",
]

import hashlib
import os
import shutil
import sys
import warnings

import torch

from .core import SSMSpec, check_rollout_args, mlp_param_grads

_ACT_C = {
    "tanh": ("tanhf({a})", "(1.0f - {z} * {z})"),
    "sigmoid": ("1.0f / (1.0f + expf(-({a})))", "({z} * (1.0f - {z}))"),
    "relu": ("({a} > 0.0f ? {a} : 0.0f)", "({z} > 0.0f ? 1.0f : 0.0f)"),
}

_EXTENSIONS: dict[SSMSpec, object] = {}
_AVAILABLE: bool | None = None


def _build_flags() -> tuple[list[str], list[str]]:
    """Compile/link flags matched to the host toolchain and torch's intra-op backend.

    ``at::parallel_for`` in an extension is only parallel when the ``AT_PARALLEL_*``
    macro matching the backend torch was built with is defined: with ``AT_PARALLEL_OPENMP``
    its implementation is an inline OpenMP pragma (so the extension itself must be compiled
    as OpenMP), while with ``AT_PARALLEL_NATIVE`` it calls torch's own thread pool and
    needs no extra flags. Without either macro it silently degrades to a serial loop.
    """
    # Apple clang rejects -march=native on arm64; -mcpu=native is its equivalent.
    arch = "-mcpu=native" if sys.platform == "darwin" else "-march=native"
    cflags = ["-O3", arch, "-ffast-math"]
    ldflags: list[str] = []
    if "OpenMP" in torch.__config__.parallel_info():
        cflags.append("-DAT_PARALLEL_OPENMP=1")
        if sys.platform == "darwin":
            # Apple clang has no -fopenmp driver support and ships no omp.h; use the
            # Homebrew libomp headers and link the libomp that torch already loads.
            brew = next((p for p in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp") if os.path.isdir(p)), None)
            if brew is None:
                raise RuntimeError(
                    "torch uses the OpenMP intra-op backend, which on macOS requires "
                    "Homebrew libomp for the C backend: brew install libomp"
                )
            cflags += ["-Xpreprocessor", "-fopenmp", f"-I{brew}/include"]
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            omp_dir = torch_lib if os.path.exists(os.path.join(torch_lib, "libomp.dylib")) else f"{brew}/lib"
            ldflags += [f"-L{omp_dir}", "-lomp"]
        else:
            cflags.append("-fopenmp")
            ldflags.append("-fopenmp")
    else:
        cflags.append("-DAT_PARALLEL_NATIVE=1")
    return cflags, ldflags


def is_available() -> bool:
    """True if the host toolchain can build the generated extension.

    Verified by compiling a minimal probe spec on first call (a few seconds, then
    disk-cached by ``load_inline``); the result is cached for the process.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _probe()
    return _AVAILABLE


def _probe() -> bool:
    if shutil.which("c++") is None and shutil.which("g++") is None:
        return False
    try:
        import torch.utils.cpp_extension as ce

        ce.verify_ninja_availability()
    except (ImportError, RuntimeError):
        return False
    try:
        _get_extension(SSMSpec(1, 1, (), "tanh"))
    except Exception as e:
        warnings.warn(f"C backend disabled, probe compilation failed: {e}")
        return False
    return True


def _gen_source(spec: SSMSpec) -> str:
    """Emit the spec-specialized C++ forward/backward rollout."""
    dims = spec.dims
    nx, nu, k = spec.n_state, spec.n_input, spec.n_linear
    act, dact = _ACT_C[spec.act]
    lines: list[str] = [
        "#include <torch/extension.h>",
        "#include <ATen/Parallel.h>",
        "#include <cmath>",
        "",
        f"constexpr int NX = {nx};",
        f"constexpr int NU = {nu};",
        "",
    ]

    # ---------------------------------------------------------------- forward
    z_args = "".join(f", torch::Tensor z{i}" for i in range(k - 1))
    w_args = "".join(f", torch::Tensor w{i}, torch::Tensor c{i}" for i in range(k))
    lines += [
        f"void ssm_fwd(torch::Tensor u, torch::Tensor x0{w_args}, torch::Tensor out{z_args}, bool store_z) {{",
        "    const int64_t B = u.size(0), L = u.size(1);",
        "    const float* up = u.data_ptr<float>();",
        "    const float* x0p = x0.data_ptr<float>();",
        "    float* outp = out.data_ptr<float>();",
    ]
    for i in range(k):
        lines.append(f"    const float* w{i}p = w{i}.data_ptr<float>();")
        lines.append(f"    const float* c{i}p = c{i}.data_ptr<float>();")
    for i in range(k - 1):
        lines.append(f"    float* z{i}p = z{i}.data_ptr<float>();")
    lines += [
        "    at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {",
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
            # separate loop so the transcendental vectorizes via libmvec (-ffast-math)
            lines.append(f"            for (int o = 0; o < {n_out}; ++o) {dst}[o] = " + act.format(a=f"{dst}[o]") + ";")
            lines.append(
                f"            if (store_z) for (int o = 0; o < {n_out}; ++o) "
                f"z{i}p[(b * L + t) * {n_out} + o] = {dst}[o];"
            )
        prev = dst
    lines += [
        "            for (int i = 0; i < NX; ++i) { x[i] = xn[i]; outp[(b * L + t) * NX + i] = xn[i]; }",
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
    zb_args = "".join(f", torch::Tensor z{i}" for i in range(k - 1))
    wt_args = "".join(f", torch::Tensor wt{i}" for i in range(k))
    ga_args = "".join(f", torch::Tensor ga{i}" for i in range(k - 1))
    lines += [
        f"void ssm_bwd(torch::Tensor gout{zb_args}{wt_args}, torch::Tensor gy{ga_args}, torch::Tensor gx0) {{",
        "    const int64_t B = gout.size(0), L = gout.size(1);",
        "    const float* goutp = gout.data_ptr<float>();",
        "    float* gyp = gy.data_ptr<float>();",
        "    float* gx0p = gx0.data_ptr<float>();",
    ]
    for i in range(k - 1):
        lines.append(f"    const float* z{i}p = z{i}.data_ptr<float>();")
        lines.append(f"    float* ga{i}p = ga{i}.data_ptr<float>();")
    for i in range(k):
        lines.append(f"    const float* wt{i}p = wt{i}.data_ptr<float>();")
    lines += [
        "    at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {",
        "    for (int64_t b = b_begin; b < b_end; ++b) {",
        "        float carry[NX] = {0.0f};",
        "        for (int64_t t = L - 1; t >= 0; --t) {",
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
        "                carry[j] = acc;",
        "            }",
        "        }",
        "        for (int i = 0; i < NX; ++i) gx0p[b * NX + i] = carry[i];",
        "    }",
        "    });",
        "}",
    ]
    return "\n".join(lines)


def _get_extension(spec: SSMSpec):
    ext = _EXTENSIONS.get(spec)
    if ext is None:
        from torch.utils.cpp_extension import load_inline

        src = _gen_source(spec)
        cflags, ldflags = _build_flags()
        tag = hashlib.md5("".join((src, *cflags, *ldflags)).encode()).hexdigest()[:10]
        ext = load_inline(
            name=f"tsfast_ssm_c_{tag}",
            cpp_sources=src,
            functions=["ssm_fwd", "ssm_bwd"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
        _EXTENSIONS[spec] = ext
    return ext


def _run_fwd(ext, spec: SSMSpec, u, x0, params, store_z):
    B, L = u.shape[0], u.shape[1]
    out = torch.empty(B, L, spec.n_state, dtype=torch.float32)
    zs = [torch.empty(B, L, h, dtype=torch.float32) for h in spec.hidden]
    wb = [t.detach().contiguous() for t in params]
    ext.ssm_fwd(u, x0, *wb, out, *zs, store_z)
    return out, zs


class _CSSMRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ext, spec, u, x0, *params):
        u = u.contiguous()
        x0 = x0.contiguous()
        out, zs = _run_fwd(ext, spec, u, x0, params, store_z=True)
        ctx.ext, ctx.spec = ext, spec
        ctx.save_for_backward(u, x0, out, *zs, *params[0::2])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext, spec = ctx.ext, ctx.spec
        k = spec.n_linear
        saved = ctx.saved_tensors
        u, x0, out = saved[0], saved[1], saved[2]
        zs = list(saved[3 : 3 + k - 1])
        weights = list(saved[3 + k - 1 :])
        B, L = u.shape[0], u.shape[1]
        wts = [w.detach().t().contiguous() for w in weights]
        gy = torch.empty(B, L, spec.n_state, dtype=torch.float32)
        gas = [torch.empty_like(z) for z in zs]
        gx0 = torch.empty(B, spec.n_state, dtype=torch.float32)
        ext.ssm_bwd(grad_out.contiguous(), *zs, *wts, gy, *gas, gx0)
        grads, du = mlp_param_grads(spec, x0, u, out, zs, gy, gas, w0=weights[0], need_du=ctx.needs_input_grad[2])
        dx0 = gx0 if ctx.needs_input_grad[3] else None
        return (None, None, du, dx0, *grads)


def c_rollout(spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    """Run the rollout through the generated C++ extension (autograd-capable)."""
    check_rollout_args(spec, u, x0, "cpu")
    ext = _get_extension(spec)
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in [u, x0, *params]):
        out, _ = _run_fwd(ext, spec, u.contiguous(), x0.contiguous(), params, store_z=False)
        return out
    return _CSSMRollout.apply(ext, spec, u, x0, *params)
