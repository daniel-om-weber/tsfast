"""Generated-C++ execution backend for the REN rollout: fast CPU rollout and BPTT.

The REN step is sequential twice over — along the sequence and along the ``n_nl`` neurons
of the equilibrium layer — so the naive rollout issues ``L * n_nl`` tiny kernels and is
dominated entirely by per-op dispatch. This backend compiles ONE generic scalar-templated
kernel (``float`` and ``double``, activation as a template parameter) with the dimensions
passed at runtime, and parallelizes over the batch through ATen's intra-op thread pool.
Being fp64-capable it is the ``torch.autograd.gradcheck`` vehicle as well as the fast CPU
path.

The forward stores the per-step input states and equilibrium activations (``B×L×nx`` and
``B×L×nv``); the backward runs only the sequential adjoint recurrence of ``MATH_REN.md``
§2.1 and hands its two outputs to the shared batched-GEMM stage in :mod:`.core`. Inputs
arrive from the ``tsfast::ren_rollout*`` custom ops as contiguous CPU tensors with the
explicit realization flattened in ``ExplicitREN.tensors`` order.
"""

__all__ = [
    "supports",
    "fits",
    "forward_infer",
    "forward_train",
    "backward",
    "is_available",
]

import hashlib
import sys

import torch

from ..._core.kernel_c import (  # toolchain probe, flags and batch driver shared
    _BATCH_PARALLEL_ATEN,
    _BATCH_PARALLEL_GCD,
    _build_flags,
    is_available,
)
from .common import RENSpec
from .core import rollout_unsupported

_EXTENSION = None

_ACT_ID = {"tanh": 0, "relu": 1, "sigmoid": 2}

#: Stack-array caps of the generated kernel, mirrored by :func:`fits`.
MAX_NX = 128
MAX_NV = 512
MAX_NU = 128
MAX_NY = 128

_SRC_HEAD = r"""
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <vector>
#include <cmath>
"""

_SRC_BODY = r"""
constexpr int MAXNX = 128;
constexpr int MAXNV = 512;
constexpr int MAXNU = 128;
constexpr int MAXNY = 128;

// ACT: 0 tanh, 1 relu, 2 sigmoid. The derivative is taken at the post-activation value,
// which every admissible activation permits (MATH_REN.md §1) and which is all the
// backward stores.
template <typename S, int ACT> static inline S act_f(S v) {
  if (ACT == 0) return std::tanh(v);
  if (ACT == 1) return v > (S)0 ? v : (S)0;
  return (S)1 / ((S)1 + std::exp(-v));
}

template <typename S, int ACT> static inline S act_d(S w) {
  if (ACT == 0) return (S)1 - w * w;
  if (ACT == 1) return w > (S)0 ? (S)1 : (S)0;
  return w * ((S)1 - w);
}

template <typename S> static inline S dot(const S* a, const S* b, int n) {
  S acc = (S)0;
  for (int j = 0; j < n; ++j) acc += a[j] * b[j];
  return acc;
}

template <typename S, int ACT>
static void fwd_impl(const S* u, const S* x0, const S* const* P, S* y, S* xlast, S* xs, S* ws,
                     bool store, int64_t B, int64_t L, int nx, int nu, int ny, int nv) {
  const S *A = P[0], *B1 = P[1], *B2 = P[2], *C1 = P[3], *D11 = P[4], *D12 = P[5];
  const S *C2 = P[6], *D21 = P[7], *D22 = P[8], *bx = P[9], *bv = P[10], *by = P[11];
  batch_parallel(B, [&](int64_t b_begin, int64_t b_end) {
    for (int64_t b = b_begin; b < b_end; ++b) {
      S x[MAXNX], w[MAXNV], xn[MAXNX];
      for (int i = 0; i < nx; ++i) x[i] = x0[b * nx + i];
      for (int64_t t = 0; t < L; ++t) {
        const S* ut = u + (b * L + t) * nu;
        // forward substitution over the neurons: D11 is strictly lower triangular, so
        // neuron i only ever reads 0..i-1 and the equilibrium resolves in one sweep
        for (int i = 0; i < nv; ++i) {
          S acc = bv[i] + dot(C1 + (size_t)i * nx, x, nx) + dot(D12 + (size_t)i * nu, ut, nu);
          acc += dot(D11 + (size_t)i * nv, w, i);
          w[i] = act_f<S, ACT>(acc);
        }
        S* yt = y + (b * L + t) * ny;
        for (int o = 0; o < ny; ++o)
          yt[o] = by[o] + dot(C2 + (size_t)o * nx, x, nx) + dot(D21 + (size_t)o * nv, w, nv)
                        + dot(D22 + (size_t)o * nu, ut, nu);
        if (store) {
          for (int i = 0; i < nx; ++i) xs[(b * L + t) * nx + i] = x[i];
          for (int i = 0; i < nv; ++i) ws[(b * L + t) * nv + i] = w[i];
        }
        for (int k = 0; k < nx; ++k)
          xn[k] = bx[k] + dot(A + (size_t)k * nx, x, nx) + dot(B1 + (size_t)k * nv, w, nv)
                        + dot(B2 + (size_t)k * nu, ut, nu);
        for (int k = 0; k < nx; ++k) x[k] = xn[k];
      }
      for (int i = 0; i < nx; ++i) xlast[b * nx + i] = x[i];
    }
  });
}

template <typename S, int ACT>
static void bwd_impl(const S* gy, const S* gxl, const S* ws, const S* const* T,
                     S* lam, S* gv, S* gx0, int64_t B, int64_t L, int nx, int ny, int nv) {
  // Transposed copies so every reduction below reads a contiguous row.
  const S *At = T[0], *B1t = T[1], *C1t = T[2], *C2t = T[3], *D21t = T[4], *D11t = T[5];
  batch_parallel(B, [&](int64_t b_begin, int64_t b_end) {
    for (int64_t b = b_begin; b < b_end; ++b) {
      S carry[MAXNX], gvv[MAXNV], gwd[MAXNV], nl[MAXNX];
      for (int i = 0; i < nx; ++i) carry[i] = gxl[b * nx + i];
      for (int64_t t = L - 1; t >= 0; --t) {
        const S* gyt = gy + (b * L + t) * ny;
        const S* wt = ws + (b * L + t) * nv;
        for (int i = 0; i < nv; ++i)
          gwd[i] = dot(D21t + (size_t)i * ny, gyt, ny) + dot(B1t + (size_t)i * nx, carry, nx);
        // reverse sweep: w_i feeds every v_k with k > i, so its adjoint collects from the
        // neurons already processed -- column i of D11, read as row i of the transpose
        for (int i = nv - 1; i >= 0; --i) {
          const S* dt = D11t + (size_t)i * nv;
          S acc = gwd[i];
          for (int k = i + 1; k < nv; ++k) acc += dt[k] * gvv[k];
          gvv[i] = acc * act_d<S, ACT>(wt[i]);
        }
        for (int i = 0; i < nx; ++i) lam[(b * L + t) * nx + i] = carry[i];
        for (int i = 0; i < nv; ++i) gv[(b * L + t) * nv + i] = gvv[i];
        for (int j = 0; j < nx; ++j)
          nl[j] = dot(At + (size_t)j * nx, carry, nx) + dot(C2t + (size_t)j * ny, gyt, ny)
                + dot(C1t + (size_t)j * nv, gvv, nv);
        for (int j = 0; j < nx; ++j) carry[j] = nl[j];
      }
      for (int i = 0; i < nx; ++i) gx0[b * nx + i] = carry[i];
    }
  });
}

#define DISPATCH_ACT(actid, BODY)                                     \
  switch (actid) {                                                    \
    case 0: { constexpr int ACT = 0; BODY; break; }                   \
    case 1: { constexpr int ACT = 1; BODY; break; }                   \
    default: { constexpr int ACT = 2; BODY; break; }                  \
  }

template <typename S> static std::vector<const S*> cptrs(const std::vector<torch::Tensor>& v) {
  std::vector<const S*> p;
  for (auto& t : v) p.push_back(t.data_ptr<S>());
  return p;
}

void ren_fwd(torch::Tensor u, torch::Tensor x0, std::vector<torch::Tensor> params,
             torch::Tensor y, torch::Tensor xlast, torch::Tensor xs, torch::Tensor ws,
             int64_t store, int64_t act) {
  const int64_t B = u.size(0), L = u.size(1);
  const int nx = (int)x0.size(1), nu = (int)u.size(2), ny = (int)y.size(2), nv = (int)ws.size(2);
  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "ren_fwd", [&] {
    auto P = cptrs<scalar_t>(params);
    DISPATCH_ACT(act, (fwd_impl<scalar_t, ACT>(
        u.data_ptr<scalar_t>(), x0.data_ptr<scalar_t>(), P.data(), y.data_ptr<scalar_t>(),
        xlast.data_ptr<scalar_t>(), xs.data_ptr<scalar_t>(), ws.data_ptr<scalar_t>(),
        store != 0, B, L, nx, nu, ny, nv)))
  });
}

void ren_bwd(torch::Tensor gy, torch::Tensor gxl, torch::Tensor ws,
             std::vector<torch::Tensor> tparams, torch::Tensor lam, torch::Tensor gv,
             torch::Tensor gx0, int64_t act) {
  const int64_t B = gy.size(0), L = gy.size(1);
  const int nx = (int)lam.size(2), ny = (int)gy.size(2), nv = (int)gv.size(2);
  AT_DISPATCH_FLOATING_TYPES(gy.scalar_type(), "ren_bwd", [&] {
    auto T = cptrs<scalar_t>(tparams);
    DISPATCH_ACT(act, (bwd_impl<scalar_t, ACT>(
        gy.data_ptr<scalar_t>(), gxl.data_ptr<scalar_t>(), ws.data_ptr<scalar_t>(), T.data(),
        lam.data_ptr<scalar_t>(), gv.data_ptr<scalar_t>(), gx0.data_ptr<scalar_t>(),
        B, L, nx, ny, nv)))
  });
}
"""


def _gen_source() -> str:
    """The kernel source with the platform's batch-parallel driver spliced in."""
    driver = _BATCH_PARALLEL_GCD if sys.platform == "darwin" else _BATCH_PARALLEL_ATEN
    return _SRC_HEAD + driver + _SRC_BODY


def _get_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load_inline

        src = _gen_source()
        cflags, ldflags = _build_flags()
        tag = hashlib.md5("".join((src, *cflags, *ldflags)).encode(), usedforsecurity=False).hexdigest()[:10]
        _EXTENSION = load_inline(
            name=f"tsfast_ren_c_{tag}",
            cpp_sources=src,
            functions=["ren_fwd", "ren_bwd"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
    return _EXTENSION


def fits(spec: RENSpec) -> bool:
    """Whether the generated kernel's stack arrays hold this spec."""
    return spec.n_state <= MAX_NX and spec.n_nl <= MAX_NV and spec.n_input <= MAX_NU and spec.n_output <= MAX_NY


def supports(spec: RENSpec, u: torch.Tensor, x0: torch.Tensor) -> str | None:
    """Reason the generated C++ kernels cannot handle these inputs, or None when they can."""
    reason = rollout_unsupported(spec, u, x0, "cpu", (torch.float32, torch.float64))
    if reason is not None:
        return reason
    if not fits(spec):
        return (
            f"spec exceeds the kernel caps (n_state<={MAX_NX}, n_nl<={MAX_NV}, n_input<={MAX_NU}, n_output<={MAX_NY})"
        )
    # The kernel indexes raw data pointers, so a non-dense layout reads the wrong elements.
    if not u.is_contiguous() or not x0.is_contiguous():
        return "inputs must be contiguous"
    if not is_available():
        return "no host C++ toolchain / ninja"
    return None


def _run_fwd(spec: RENSpec, u, x0, params, store: bool):
    b, ln = u.shape[0], u.shape[1]
    y = torch.empty(b, ln, spec.n_output, dtype=u.dtype)
    xlast = torch.empty(b, spec.n_state, dtype=u.dtype)
    xs = torch.empty(b, ln if store else 0, spec.n_state, dtype=u.dtype)
    ws = torch.empty(b, ln if store else 0, spec.n_nl, dtype=u.dtype)
    _get_extension().ren_fwd(u, x0, list(params), y, xlast, xs, ws, int(store), _ACT_ID[spec.act])
    return y, xlast, xs, ws


def forward_infer(spec: RENSpec, u, x0, params) -> tuple[torch.Tensor, torch.Tensor]:
    """Rollout without stored intermediates: returns ``(y, x_L)``."""
    y, xlast, _, _ = _run_fwd(spec, u, x0, params, store=False)
    return y, xlast


def forward_train(spec: RENSpec, u, x0, params) -> tuple[torch.Tensor, ...]:
    """Rollout that also stores the BPTT tape: returns ``(y, x_L, xs, ws)``."""
    return _run_fwd(spec, u, x0, params, store=True)


def backward(spec: RENSpec, gy, gxl, xs, ws, params) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential adjoint recurrence: returns ``(lam, gv, gx0)`` for the shared GEMM stage."""
    b, ln = gy.shape[0], gy.shape[1]
    a, b1, c1, c2, d11, d21 = (params[i] for i in (0, 1, 3, 6, 4, 7))
    tparams = [t.t().contiguous() for t in (a, b1, c1, c2, d21, d11)]
    lam = torch.empty(b, ln, spec.n_state, dtype=gy.dtype)
    gv = torch.empty(b, ln, spec.n_nl, dtype=gy.dtype)
    gx0 = torch.empty(b, spec.n_state, dtype=gy.dtype)
    _get_extension().ren_bwd(gy, gxl, ws, tparams, lam, gv, gx0, _ACT_ID[spec.act])
    return lam, gv, gx0
