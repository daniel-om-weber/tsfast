"""Shared primitives for the generated-C++ execution backends.

Activation source macros, the batch-parallel driver templates, the host-toolchain
compile/link flags, and the toolchain availability probe are common to every generated
C++ backend (NeuralStateSpace, NARX, port-Hamiltonian, and the diagonal/selective scan
backends). They live here so those backends depend on a shared kernel toolkit rather
than on any one model's module.

``is_available`` compiles a trivial probe once per process (disk-cached) to confirm the host
can build the generated code; it is intentionally spec-free, so a model kernel that fails to
compile for its own reasons does not make the toolchain look unavailable to the others. The
probe also reports the threading macros it was built with, which is what turns a wrong
``-fopenmp`` decision into a warning rather than a silently single-threaded kernel.

``load_cabi`` builds a generated translation unit into a shared object and loads it with
``ctypes``. Kernels built through it must not include ``torch/extension.h``: that header
carries the whole torch C++ frontend and pybind11, and parsing it costs some forty times what
a kernel body costs to compile. The kernels exchange nothing but float pointers and sizes, so
they need none of it — ``ATen/Parallel.h`` alone keeps ``at::parallel_for``, and with it
``torch.set_num_threads``, governing the batch split.

Names stay underscore-prefixed: they are a shared internal toolkit, not public API.
"""

__all__ = [
    "is_available",
    "load_cabi",
]

import ctypes
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from pathlib import Path

import torch

# Rational tanh approximation (Eigen's single-precision coefficients), accurate to 2.4e-07
# absolute over the clamped range. Plain arithmetic instead of libm keeps the activation loop
# auto-vectorizable on every host: macOS ships no vector libm at all, and where glibc's libmvec
# does supply one the polynomial still measures faster than it (1.15-1.26x on the whole rollout).
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

#: Activation expression and its derivative in terms of the stored post-activation ``z``.
#: A generator that emits an entry naming ``fast_tanhf`` must also emit ``_FAST_TANH_C``.
_ACT_C = {
    "tanh": ("fast_tanhf({a})", "(1.0f - {z} * {z})"),
    "sigmoid": ("(0.5f + 0.5f * fast_tanhf(0.5f * ({a})))", "({z} * (1.0f - {z}))"),
    "relu": ("({a} > 0.0f ? {a} : 0.0f)", "({z} > 0.0f ? 1.0f : 0.0f)"),
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

# Apple clang rejects -march=native on arm64; -mcpu=native is its equivalent.
_ARCH_FLAG = "-mcpu=native" if sys.platform == "darwin" else "-march=native"


def _aten_parallel_backend() -> str:
    """Which ``at::parallel_for`` implementation the generated kernels compile against.

    Read from ``ATen/Config.h`` — the header the kernels themselves include — rather than from
    ``torch.__config__.parallel_info()``, whose OpenMP banner reports only that torch was built
    with OpenMP available and appears even when ATen's intra-op backend is the native thread
    pool. Defining ``AT_PARALLEL_*`` on the command line cannot change the answer either:
    Config.h defines both unconditionally and is included last, so the header always wins.

    Returns:
        ``"openmp"``, ``"native"``, or ``"unknown"`` when no readable Config.h was found.
    """
    from torch.utils.cpp_extension import include_paths

    for p in include_paths():
        cfg = Path(p) / "ATen" / "Config.h"
        if not cfg.is_file():
            continue
        m = re.search(r"^\s*#define\s+AT_PARALLEL_OPENMP\s+(\d+)", cfg.read_text(), re.M)
        if m is not None:
            return "native" if m.group(1) == "0" else "openmp"
    return "unknown"


def _build_flags() -> tuple[list[str], list[str]]:
    """Compile/link flags matched to the host toolchain and ATen's intra-op backend.

    On macOS the generated source parallelizes via Grand Central Dispatch (part of libSystem,
    always available to Apple clang), so no threading flags are needed. Elsewhere it uses
    ``at::parallel_for``, whose OpenMP form is an inline pragma in ``ATen/Parallel.h``: unless
    the extension is itself compiled with ``-fopenmp`` that pragma is ignored and the kernel
    runs on one thread with no diagnostic. Against a native-thread-pool ATen the flag is inert,
    so an undetermined backend takes it as well; ``is_available``'s probe reports the pairing
    that was actually built.
    """
    cflags = ["-O3", _ARCH_FLAG, "-ffast-math"]
    if sys.platform == "darwin" or _aten_parallel_backend() == "native":
        return cflags, []
    return [*cflags, "-fopenmp"], ["-fopenmp"]


def _compiler() -> str | None:
    return shutil.which("c++") or shutil.which("g++")


def _torch_flags() -> tuple[list[str], list[str]]:
    """Include and link flags for the ATen threading header the generated kernels use.

    ``at::parallel_for`` resolves to real symbols in ``libtorch_cpu`` even where its OpenMP
    form is an inline pragma, so the object has to link and carry an rpath: nothing guarantees
    torch's own libraries were loaded with their symbols made globally visible.
    """
    from torch.utils.cpp_extension import include_paths

    libdir = str(Path(torch.__file__).parent / "lib")
    return (
        [f"-I{p}" for p in include_paths()],
        [f"-L{libdir}", "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{libdir}"],
    )


_LIBS: dict[str, ctypes.CDLL] = {}


def _cache_dir() -> Path:
    d = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "tsfast" / "kernel_c"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _target_id(cxx: str | None) -> str:
    """Digest of the preprocessor macros the host-targeting flag resolves to.

    ``-E -dM`` on empty input is understood by both gcc and clang and reports the ISA feature
    macros ``-march=native``/``-mcpu=native`` selected, along with the compiler's own version
    macros. Hashing that instead of the literal flag — which reads identically on every host —
    is what keeps two CPU generations sharing a cache directory from loading each other's
    objects, whose SIGILL no ``except`` can catch. Falls back to the version string.
    """
    if cxx is None:
        return "no-compiler"
    try:
        r = subprocess.run([cxx, _ARCH_FLAG, "-E", "-dM", "-x", "c++", "-"], input="", capture_output=True, text=True)
        if r.returncode == 0:
            return hashlib.sha256(r.stdout.encode()).hexdigest()[:16]
        return subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        return str(cxx)


def _toolchain_id() -> str:
    """Everything outside the source that changes the emitted object."""
    global _TOOLCHAIN_ID
    if _TOOLCHAIN_ID is None:
        cxx = _compiler()
        tgt = _target_id(cxx)
        _TOOLCHAIN_ID = f"{cxx}|{tgt}|{torch.__version__}|{platform.machine()}|{sys.platform}"
    return _TOOLCHAIN_ID


_TOOLCHAIN_ID: str | None = None


def load_cabi(source: str, prefix: str) -> ctypes.CDLL:
    """Build a generated translation unit and load it with ``ctypes``.

    The object is keyed by a hash of the source, the flags it was built with, and the
    toolchain identity — which resolves the host-targeting flag to the ISA it selected, so a
    cache directory shared between machines of different CPU generations yields one object per
    microarchitecture. Cached on disk under ``XDG_CACHE_HOME`` and memoised per process.

    Args:
        source: complete translation unit; entry points must be declared ``extern "C"``.
        prefix: readable stem for the cached object, with the source hash appended.
    """
    cflags, ldflags = _build_flags()
    inc, torch_ld = _torch_flags()
    key = "".join((source, *cflags, *ldflags, *inc, _toolchain_id()))
    name = f"{prefix}_{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:10]}"
    lib = _LIBS.get(name)
    if lib is None:
        path = _cache_dir() / f"{name}.so"
        if not path.exists():
            _build(source, path, [*cflags, *inc], [*ldflags, *torch_ld])
        try:
            lib = ctypes.CDLL(str(path))
        except OSError:
            # A truncated object (interrupted write, partially copied cache) would otherwise
            # fail for every future process; drop it and build once more.
            path.unlink(missing_ok=True)
            _build(source, path, [*cflags, *inc], [*ldflags, *torch_ld])
            lib = ctypes.CDLL(str(path))
        _LIBS[name] = lib
    return lib


def _build(source: str, path: Path, cflags: list[str], ldflags: list[str]) -> None:
    cxx = _compiler()
    if cxx is None:
        raise RuntimeError("no host C++ compiler (c++/g++) on PATH")
    # Built into a scratch directory and moved into place, so concurrent builders racing on
    # the same spec cannot expose a half-written object to a third process's CDLL.
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        src, o, so = Path(tmp) / "kernel.cpp", Path(tmp) / "kernel.o", Path(tmp) / "kernel.so"
        src.write_text(source)
        # Compile and link are separate steps because ``-ffast-math`` must not reach the link
        # line: there it pulls in crtfastmath.o, whose ELF constructor sets FTZ/DAZ in MXCSR
        # when the object is dlopened. That would silently flush denormals to zero for every
        # subsequent torch operation in the process, not just for this kernel.
        for cmd in (
            [cxx, *cflags, "-std=c++17", "-fPIC", "-c", str(src), "-o", str(o)],
            [cxx, "-shared", str(o), "-o", str(so), *ldflags],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"kernel build failed:\n{' '.join(cmd)}\n{r.stderr}")
        with open(so, "rb") as f:
            os.fsync(f.fileno())  # the rename is atomic; the contents behind it must be durable
        os.replace(so, path)


# Reports what the toolchain actually produced rather than what was requested, so a threading
# pairing the flags got wrong surfaces as a warning instead of as a kernel that silently runs
# on one thread. at::get_num_threads keeps a real libtorch_cpu symbol in the probe, so a
# broken rpath still fails here rather than in a model's first rollout.
_PROBE_SRC = """\
#include <ATen/Parallel.h>
#include <cstdint>
extern "C" int64_t tsfast_kernel_probe() {
    int64_t f = at::get_num_threads() > 0 ? 4 : 0;
#if AT_PARALLEL_OPENMP
    f |= 1;
#endif
#ifdef _OPENMP
    f |= 2;
#endif
    return f;
}
"""


def is_available() -> bool:
    """True if the host toolchain can build a generated kernel.

    Verified by building a trivial spec-free probe on first call (disk-cached afterwards);
    the result is cached for the process.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _probe()
    return _AVAILABLE


def _load_inline_toolchain() -> bool:
    """Whether the heavier ``load_inline`` path could build.

    This gate covers more than ``load_cabi`` needs, because every generated-C++ backend shares
    ``is_available`` and the ones built through ``load_inline`` need ninja and a pybind11
    translation unit — which means the Python development headers. Probing that by compiling
    would cost the seconds ``load_cabi`` exists to avoid, so the headers are checked by
    presence instead.
    """
    try:
        import torch.utils.cpp_extension as ce

        ce.verify_ninja_availability()
    except (ImportError, RuntimeError):
        return False
    return (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file()


def _probe() -> bool:
    if _compiler() is None or not _load_inline_toolchain():
        return False
    try:
        fn = load_cabi(_PROBE_SRC, "tsfast_kernel_probe").tsfast_kernel_probe
        fn.argtypes, fn.restype = [], ctypes.c_int64
        flags = fn()
    except Exception as e:
        warnings.warn(f"C backend disabled, probe build failed: {e}")
        return False
    # Only the hosts served by _BATCH_PARALLEL_ATEN reach at::parallel_for; on macOS every
    # kernel splices the GCD driver instead, so the missing -fopenmp there costs nothing.
    if sys.platform != "darwin" and flags & 1 and not flags & 2:
        warnings.warn(
            "ATen's intra-op backend is OpenMP but the generated kernels built without -fopenmp: "
            "at::parallel_for will run them on a single thread"
        )
    return True
