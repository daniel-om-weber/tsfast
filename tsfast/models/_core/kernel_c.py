"""Shared primitives for the generated-C++ execution backends.

Activation source macros, the batch-parallel driver templates, the host-toolchain
compile/link flags, and the toolchain availability probe are common to every generated
C++ backend (NeuralStateSpace, NARX, port-Hamiltonian, and the diagonal/selective scan
backends). They live here so those backends depend on a shared kernel toolkit rather
than on any one model's module.

``is_available`` compiles a trivial probe extension once per process (disk-cached by
``load_inline``) to confirm the host can build the generated code; it is intentionally
spec-free, so a model kernel that fails to compile for its own reasons does not make the
toolchain look unavailable to the others.

Names are kept underscore-prefixed and identical to their original definitions so the
backends import them unchanged (only the module path moved).
"""

__all__ = [
    "is_available",
]

import shutil
import sys
import warnings

import torch

_ACT_C = {
    "tanh": ("tanhf({a})", "(1.0f - {z} * {z})"),
    "sigmoid": ("1.0f / (1.0f + expf(-({a})))", "({z} * (1.0f - {z}))"),
    "relu": ("({a} > 0.0f ? {a} : 0.0f)", "({z} > 0.0f ? 1.0f : 0.0f)"),
}

# Rational tanh approximation (Eigen's single-precision coefficients), accurate to a few
# ulp on the clamped range. Plain arithmetic instead of libm so the activation loop stays
# auto-vectorizable on macOS, which ships no vector libm — a scalar tanhf call per element
# otherwise dominates the rollout.
_FAST_TANH_C = """\
static inline float fast_tanhf(float x) {
    x = fminf(7.90531110763549805f, fmaxf(-7.90531110763549805f, x));
    const float x2 = x * x;
    float p = -2.76076847742355e-16f;
    p = p * x2 + 2.00018790482477e-13f;
    p = p * x2 + -8.60467152213735e-11f;
    p = p * x2 + 5.12229709037114e-08f;
    p = p * x2 + 1.48572235717979e-05f;
    p = p * x2 + 6.37261928875436e-04f;
    p = p * x2 + 4.89352455891786e-03f;
    p = p * x;
    float q = 1.19825839466702e-06f;
    q = q * x2 + 1.18534705686654e-04f;
    q = q * x2 + 2.26843463243900e-03f;
    q = q * x2 + 4.89352518554385e-03f;
    return p / q;
}
"""

_ACT_C_DARWIN = {
    **_ACT_C,
    "tanh": ("fast_tanhf({a})", _ACT_C["tanh"][1]),
    "sigmoid": ("(0.5f + 0.5f * fast_tanhf(0.5f * ({a})))", _ACT_C["sigmoid"][1]),
}

# Chunked to at::get_num_threads() tasks so torch.set_num_threads() is honored; GCD picks
# the worker threads itself. dispatch_apply_f (not dispatch_apply) because blocks capturing
# C++ locals are an Apple extension, while a captureless lambda is a plain function pointer.
_BATCH_PARALLEL_GCD = """\
#include <dispatch/dispatch.h>

template <typename F>
static void batch_parallel(int64_t n, const F& f) {
    if (n <= 0) return;
    struct Ctx { const F* f; int64_t n, n_tasks; };
    Ctx ctx{&f, n, std::min<int64_t>(n, at::get_num_threads())};
    dispatch_apply_f(
        (size_t)ctx.n_tasks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), &ctx,
        [](void* p, size_t i) {
            auto* c = static_cast<Ctx*>(p);
            const int64_t chunk = (c->n + c->n_tasks - 1) / c->n_tasks;
            const int64_t b0 = (int64_t)i * chunk, b1 = std::min(c->n, b0 + chunk);
            if (b0 < b1) (*c->f)(b0, b1);
        });
}
"""

_BATCH_PARALLEL_ATEN = """\
template <typename F>
static void batch_parallel(int64_t n, const F& f) {
    at::parallel_for(0, n, 1, f);
}
"""

_AVAILABLE: bool | None = None


def _build_flags() -> tuple[list[str], list[str]]:
    """Compile/link flags matched to the host toolchain and torch's intra-op backend.

    On macOS the generated source parallelizes via Grand Central Dispatch (part of
    libSystem, always available to Apple clang), so no threading flags are needed.
    Elsewhere it uses ``at::parallel_for``, which is only parallel when the ``AT_PARALLEL_*``
    macro matching the backend torch was built with is defined: with ``AT_PARALLEL_OPENMP``
    its implementation is an inline OpenMP pragma (so the extension itself must be compiled
    as OpenMP), while with ``AT_PARALLEL_NATIVE`` it calls torch's own thread pool and
    needs no extra flags. Without either macro it silently degrades to a serial loop.
    """
    if sys.platform == "darwin":
        # Apple clang rejects -march=native on arm64; -mcpu=native is its equivalent.
        return ["-O3", "-mcpu=native", "-ffast-math"], []
    cflags = ["-O3", "-march=native", "-ffast-math"]
    ldflags: list[str] = []
    if "OpenMP" in torch.__config__.parallel_info():
        cflags += ["-DAT_PARALLEL_OPENMP=1", "-fopenmp"]
        ldflags.append("-fopenmp")
    else:
        cflags.append("-DAT_PARALLEL_NATIVE=1")
    return cflags, ldflags


def is_available() -> bool:
    """True if the host toolchain can build a generated extension.

    Verified by compiling a trivial spec-free probe on first call (a few seconds, then
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
        cflags, ldflags = _build_flags()
        ce.load_inline(
            name="tsfast_kernel_c_probe",
            cpp_sources="int64_t tsfast_kernel_probe() { return 0; }",
            functions=["tsfast_kernel_probe"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
    except Exception as e:
        warnings.warn(f"C backend disabled, probe compilation failed: {e}")
        return False
    return True
