"""Generated-C++ execution backend for the constant-coefficient diagonal scan (LRU/S5 core).

The doubling scan of ``scan.diagonal_recurrence`` sweeps the sequence ``ceil(log2(L))`` times;
this backend streams it once. Work is split into channel blocks of up to ``KMAX`` lanes: a task
owns one (row, block) pair and sweeps time once, with the recurrence ``x_t = lam * x_{t-1} + v_t``
(and its adjoint ``G_t = g_t + conj(lam) * G_{t+1}``, reversed) running in an auto-vectorizable
inner loop over the block's separate re/im accumulators — a per-lane scalar walk is
latency-bound and reads 8 bytes per touched cache line, while a full-row block streams
contiguously. When there are too few (row, block) tasks to fill the threads (small batch, e.g.
single-sequence CPU inference), time is additionally split into chunks reconciled by a carry
pass: zero-start chunk scans record their end states, the true chunk-start states compose
sequentially through ``lam^len`` (one tiny vector op per chunk), and a second sweep re-scans
each chunk from its true start. Complex tensors are processed as interleaved re/im float
buffers; the conjugates ``conj(lam)`` (gradient scan) and ``conj(x_{t-1})`` (``grad_lam``
reduction) are open-coded. Toolchain probing, compile flags, and the batch-parallel scaffolding
are shared with the SSM/NARX C backends.

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

from ..kernel_c import (
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


# The kernel proper. Structure: fwd_chunk/bwd_chunk are the vectorizable time sweeps over one
# channel block; fwd_t/bwd_t split the work into (row, block[, time-chunk]) tasks and reconcile
# time chunks through the lam^len carry composition; diag_fwd/diag_bwd dispatch on the dtype.
_KERNEL = r"""
using cf = c10::complex<float>;

// KMAX: channel-block width in lanes. A block covers the whole row when N <= KMAX, so its time
// sweep streams contiguous memory; wider rows split into KMAX-wide blocks. MIN_CHUNK: smallest
// number of timesteps worth a chunk of its own when time is split to occupy idle threads (the
// zero-start pass of the chunked mode re-reads the chunk's input once).
static constexpr int KMAX = 64;
static constexpr int64_t MIN_CHUNK = 2048;

static int64_t pick_chunks(int64_t base_tasks, int64_t L) {
    int64_t nt = at::get_num_threads();
    if (nt < 1) nt = 1;
    if (base_tasks >= 2 * nt || L < 2 * MIN_CHUNK) return 1;
    const int64_t want = (2 * nt + base_tasks - 1) / base_tasks;
    return std::max<int64_t>(1, std::min<int64_t>(want, L / MIN_CHUNK));
}

struct Block {
    int64_t m, cj, c0, t0, t1;
    int nc;
};

static Block block_of(int64_t i, int64_t nblk, int64_t NC, int64_t LCH, int64_t L, int64_t N) {
    const int64_t m = i / (nblk * NC), r = i % (nblk * NC);
    const int64_t cj = r % NC, c0 = (r / NC) * KMAX, t0 = cj * LCH;
    return {m, cj, c0, t0, std::min(L, t0 + LCH), (int)std::min<int64_t>(KMAX, N - c0)};
}

// gather one block of a [M, N] lane vector (lam or x0) into re/im arrays
template <bool C, bool CONJ>
static void load_lanes(const float* buf, int64_t m, int64_t c0, int nc, int64_t N,
                       float* re, float* im) {
    for (int k = 0; k < nc; ++k) {
        if constexpr (C) {
            re[k] = buf[2 * (m * N + c0 + k)];
            im[k] = (CONJ ? -1.f : 1.f) * buf[2 * (m * N + c0 + k) + 1];
        } else {
            re[k] = buf[m * N + c0 + k];
            im[k] = 0.f;
        }
    }
}

// a^len per lane by repeated squaring, flushing underflow to zero: powers of |a| < 1 decay
// into denormals, which multiply ~100x slower — a shared library gets no FTZ/DAZ from
// -ffast-math (crtfastmath.o is linked into executables only). Real lanes ride along with a
// zero imaginary part.
static void pow_clamped(const float* a_re, const float* a_im, int64_t len, int nc,
                        float* p_re, float* p_im) {
    float br[KMAX], bi[KMAX];
    for (int k = 0; k < nc; ++k) {
        p_re[k] = 1.f; p_im[k] = 0.f;
        br[k] = a_re[k]; bi[k] = a_im[k];
    }
    for (uint64_t e = (uint64_t)len; e;) {
        if (e & 1)
            for (int k = 0; k < nc; ++k) {
                const float nr = p_re[k] * br[k] - p_im[k] * bi[k];
                p_im[k] = p_re[k] * bi[k] + p_im[k] * br[k];
                p_re[k] = nr;
                if (p_re[k] * p_re[k] + p_im[k] * p_im[k] < 1e-30f) { p_re[k] = 0.f; p_im[k] = 0.f; }
            }
        e >>= 1;
        if (e)
            for (int k = 0; k < nc; ++k) {
                const float nr = br[k] * br[k] - bi[k] * bi[k];
                bi[k] = 2.f * br[k] * bi[k];
                br[k] = nr;
                if (br[k] * br[k] + bi[k] * bi[k] < 1e-30f) { br[k] = 0.f; bi[k] = 0.f; }
            }
    }
}

// x = a*x + v over [t0, t1) for one channel block; separate re/im accumulators and __restrict
// streams keep the inner loop auto-vectorizable. STORE=false only carries the state (the
// zero-start pass of the chunked mode). KB > 0 fixes the lane count at compile time — the
// vectorizer handles the full-block case much better with a constant trip count; KB == 0
// takes the lane count from `nc` (ragged tail blocks).
template <bool C, bool STORE, int KB>
static void fwd_chunk_impl(const float* __restrict ar, const float* __restrict ai,
                           float* __restrict xr, float* __restrict xi,
                           const float* __restrict v, float* __restrict out,
                           int64_t t0, int64_t t1, int64_t N, int nc) {
    constexpr int64_t S = C ? 2 : 1;
    const int kb = KB ? KB : nc;
    for (int64_t t = t0; t < t1; ++t) {
        const float* __restrict vp = v + S * t * N;
        float* __restrict op = nullptr;
        if constexpr (STORE) op = out + S * t * N;
        for (int k = 0; k < kb; ++k) {
            if constexpr (C) {
                const float nr = ar[k] * xr[k] - ai[k] * xi[k] + vp[2 * k];
                xi[k] = ar[k] * xi[k] + ai[k] * xr[k] + vp[2 * k + 1];
                xr[k] = nr;
                if constexpr (STORE) { op[2 * k] = xr[k]; op[2 * k + 1] = xi[k]; }
            } else {
                xr[k] = ar[k] * xr[k] + vp[k];
                if constexpr (STORE) op[k] = xr[k];
            }
        }
    }
}

template <bool C, bool STORE>
static void fwd_chunk(const float* __restrict ar, const float* __restrict ai,
                      float* __restrict xr, float* __restrict xi,
                      const float* __restrict v, float* __restrict out,
                      int64_t t0, int64_t t1, int64_t N, int nc) {
    if (nc == KMAX)
        fwd_chunk_impl<C, STORE, KMAX>(ar, ai, xr, xi, v, out, t0, t1, N, nc);
    else
        fwd_chunk_impl<C, STORE, 0>(ar, ai, xr, xi, v, out, t0, t1, N, nc);
}

// G = g + a*G swept backward over [t0, t1); `a` arrives pre-conjugated. FULL also writes
// gv = G and accumulates glam += G * conj(x_{t-1}), reading x_{t-1} from the forward output
// (the x0 block at t = 0, peeled off so the hot loop stays branch-free).
template <bool C, bool FULL, int KB>
static void bwd_chunk_impl(const float* __restrict ar, const float* __restrict ai,
                           float* __restrict Gr, float* __restrict Gi,
                           const float* __restrict g, const float* __restrict out,
                           const float* __restrict x0r, const float* __restrict x0i,
                           float* __restrict gv, float* __restrict glr, float* __restrict gli,
                           int64_t t0, int64_t t1, int64_t N, int nc) {
    constexpr int64_t S = C ? 2 : 1;
    const int kb = KB ? KB : nc;
    for (int64_t t = t1 - 1; t >= std::max<int64_t>(t0, 1); --t) {
        const float* __restrict gp = g + S * t * N;
        float* __restrict gvp = nullptr;
        const float* __restrict xp = nullptr;
        if constexpr (FULL) { gvp = gv + S * t * N; xp = out + S * (t - 1) * N; }
        for (int k = 0; k < kb; ++k) {
            if constexpr (C) {
                const float nr = gp[2 * k] + ar[k] * Gr[k] - ai[k] * Gi[k];
                Gi[k] = gp[2 * k + 1] + ar[k] * Gi[k] + ai[k] * Gr[k];
                Gr[k] = nr;
                if constexpr (FULL) {
                    gvp[2 * k] = Gr[k]; gvp[2 * k + 1] = Gi[k];
                    glr[k] += Gr[k] * xp[2 * k] + Gi[k] * xp[2 * k + 1];
                    gli[k] += Gi[k] * xp[2 * k] - Gr[k] * xp[2 * k + 1];
                }
            } else {
                Gr[k] = gp[k] + ar[k] * Gr[k];
                if constexpr (FULL) { gvp[k] = Gr[k]; glr[k] += Gr[k] * xp[k]; }
            }
        }
    }
    if (t0 == 0) {  // t = 0: x_{t-1} is the initial state (zeros when absent)
        for (int k = 0; k < kb; ++k) {
            if constexpr (C) {
                const float nr = g[2 * k] + ar[k] * Gr[k] - ai[k] * Gi[k];
                Gi[k] = g[2 * k + 1] + ar[k] * Gi[k] + ai[k] * Gr[k];
                Gr[k] = nr;
                if constexpr (FULL) {
                    gv[2 * k] = Gr[k]; gv[2 * k + 1] = Gi[k];
                    glr[k] += Gr[k] * x0r[k] + Gi[k] * x0i[k];
                    gli[k] += Gi[k] * x0r[k] - Gr[k] * x0i[k];
                }
            } else {
                Gr[k] = g[k] + ar[k] * Gr[k];
                if constexpr (FULL) { gv[k] = Gr[k]; glr[k] += Gr[k] * x0r[k]; }
            }
        }
    }
}

template <bool C, bool FULL>
static void bwd_chunk(const float* __restrict ar, const float* __restrict ai,
                      float* __restrict Gr, float* __restrict Gi,
                      const float* __restrict g, const float* __restrict out,
                      const float* __restrict x0r, const float* __restrict x0i,
                      float* __restrict gv, float* __restrict glr, float* __restrict gli,
                      int64_t t0, int64_t t1, int64_t N, int nc) {
    if (nc == KMAX)
        bwd_chunk_impl<C, FULL, KMAX>(ar, ai, Gr, Gi, g, out, x0r, x0i, gv, glr, gli, t0, t1, N, nc);
    else
        bwd_chunk_impl<C, FULL, 0>(ar, ai, Gr, Gi, g, out, x0r, x0i, gv, glr, gli, t0, t1, N, nc);
}

template <bool C>
static void fwd_t(const float* lam, const float* v, const float* x0, float* out,
                  bool has_x0, int64_t M, int64_t L, int64_t N) {
    constexpr int64_t S = C ? 2 : 1;
    const int64_t nblk = (N + KMAX - 1) / KMAX;
    const int64_t NC = pick_chunks(M * nblk, L);
    const int64_t LCH = (L + NC - 1) / NC;
    const int64_t tasks = M * nblk * NC;

    // true start state of every chunk task, planar [tasks][re KMAX | im KMAX]
    std::vector<float> starts;
    if (NC > 1) {
        // zero-start scan per chunk records its end state (the last chunk's is never consumed)
        std::vector<float> ends((size_t)tasks * 2 * KMAX);
        batch_parallel(tasks, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const Block b = block_of(i, nblk, NC, LCH, L, N);
                if (b.cj == NC - 1) continue;
                float ar[KMAX], ai[KMAX], xr[KMAX] = {}, xi[KMAX] = {};
                load_lanes<C, false>(lam, b.m, b.c0, b.nc, N, ar, ai);
                fwd_chunk<C, false>(ar, ai, xr, xi, v + S * (b.m * L * N + b.c0), nullptr,
                                    b.t0, b.t1, N, b.nc);
                float* e = ends.data() + (size_t)i * 2 * KMAX;
                for (int k = 0; k < b.nc; ++k) { e[k] = xr[k]; e[KMAX + k] = xi[k]; }
            }
        });
        // per (row, block): compose the true chunk starts, carry -> end_j + a^{len_j} * carry
        starts.resize((size_t)tasks * 2 * KMAX);
        batch_parallel(M * nblk, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t m = i / nblk, c0 = (i % nblk) * KMAX;
                const int nc = (int)std::min<int64_t>(KMAX, N - c0);
                float ar[KMAX], ai[KMAX], cr[KMAX] = {}, ci[KMAX] = {}, pr[KMAX], pi[KMAX];
                load_lanes<C, false>(lam, m, c0, nc, N, ar, ai);
                if (has_x0) load_lanes<C, false>(x0, m, c0, nc, N, cr, ci);
                for (int64_t cj = 0; cj < NC; ++cj) {
                    float* s = starts.data() + ((size_t)i * NC + cj) * 2 * KMAX;
                    for (int k = 0; k < nc; ++k) { s[k] = cr[k]; s[KMAX + k] = ci[k]; }
                    if (cj == NC - 1) break;
                    const int64_t t0 = cj * LCH;
                    pow_clamped(ar, ai, std::min(L, t0 + LCH) - t0, nc, pr, pi);
                    const float* e = ends.data() + ((size_t)i * NC + cj) * 2 * KMAX;
                    for (int k = 0; k < nc; ++k) {
                        const float nr = pr[k] * cr[k] - pi[k] * ci[k] + e[k];
                        ci[k] = pr[k] * ci[k] + pi[k] * cr[k] + e[KMAX + k];
                        cr[k] = nr;
                    }
                }
            }
        });
    }
    batch_parallel(tasks, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            const Block b = block_of(i, nblk, NC, LCH, L, N);
            float ar[KMAX], ai[KMAX], xr[KMAX] = {}, xi[KMAX] = {};
            load_lanes<C, false>(lam, b.m, b.c0, b.nc, N, ar, ai);
            if (NC > 1) {
                const float* s = starts.data() + (size_t)i * 2 * KMAX;
                for (int k = 0; k < b.nc; ++k) { xr[k] = s[k]; xi[k] = s[KMAX + k]; }
            } else if (has_x0) {
                load_lanes<C, false>(x0, b.m, b.c0, b.nc, N, xr, xi);
            }
            fwd_chunk<C, true>(ar, ai, xr, xi, v + S * (b.m * L * N + b.c0),
                               out + S * (b.m * L * N + b.c0), b.t0, b.t1, N, b.nc);
        }
    });
}

template <bool C>
static void bwd_t(const float* g, const float* lam, const float* out, const float* x0,
                  bool has_x0, float* gv, float* glam, float* gx0,
                  int64_t M, int64_t L, int64_t N) {
    constexpr int64_t S = C ? 2 : 1;
    const int64_t nblk = (N + KMAX - 1) / KMAX;
    const int64_t NC = pick_chunks(M * nblk, L);
    const int64_t LCH = (L + NC - 1) / NC;
    const int64_t tasks = M * nblk * NC;

    // right-edge boundary G of every chunk task (G_L = 0 behind the last chunk)
    std::vector<float> bounds;
    if (NC > 1) {
        std::vector<float> locals((size_t)tasks * 2 * KMAX);
        batch_parallel(tasks, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const Block b = block_of(i, nblk, NC, LCH, L, N);
                if (b.cj == 0) continue;  // chunk 0's zero-start local is never consumed
                float ar[KMAX], ai[KMAX], Gr[KMAX] = {}, Gi[KMAX] = {};
                load_lanes<C, true>(lam, b.m, b.c0, b.nc, N, ar, ai);
                bwd_chunk<C, false>(ar, ai, Gr, Gi, g + S * (b.m * L * N + b.c0), nullptr,
                                    nullptr, nullptr, nullptr, nullptr, nullptr,
                                    b.t0, b.t1, N, b.nc);
                float* lo = locals.data() + (size_t)i * 2 * KMAX;
                for (int k = 0; k < b.nc; ++k) { lo[k] = Gr[k]; lo[KMAX + k] = Gi[k]; }
            }
        });
        // per (row, block): compose right-edge boundaries, B_{j-1} = local_j + conj(a)^{len_j} * B_j
        bounds.resize((size_t)tasks * 2 * KMAX);
        batch_parallel(M * nblk, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t m = i / nblk, c0 = (i % nblk) * KMAX;
                const int nc = (int)std::min<int64_t>(KMAX, N - c0);
                float ar[KMAX], ai[KMAX], Br[KMAX] = {}, Bi[KMAX] = {}, pr[KMAX], pi[KMAX];
                load_lanes<C, true>(lam, m, c0, nc, N, ar, ai);
                for (int64_t cj = NC - 1;; --cj) {
                    float* s = bounds.data() + ((size_t)i * NC + cj) * 2 * KMAX;
                    for (int k = 0; k < nc; ++k) { s[k] = Br[k]; s[KMAX + k] = Bi[k]; }
                    if (cj == 0) break;
                    const int64_t t0 = cj * LCH;
                    pow_clamped(ar, ai, std::min(L, t0 + LCH) - t0, nc, pr, pi);
                    const float* lo = locals.data() + ((size_t)i * NC + cj) * 2 * KMAX;
                    for (int k = 0; k < nc; ++k) {
                        const float nr = pr[k] * Br[k] - pi[k] * Bi[k] + lo[k];
                        Bi[k] = pr[k] * Bi[k] + pi[k] * Br[k] + lo[KMAX + k];
                        Br[k] = nr;
                    }
                }
            }
        });
    }
    // main reverse sweep: writes gv, accumulates per-task glam partials, and records each
    // (row, block)'s G at t = 0 for gx0
    std::vector<float> gl_part((size_t)tasks * 2 * KMAX);
    std::vector<float> finals(has_x0 ? (size_t)(M * nblk) * 2 * KMAX : 0);
    batch_parallel(tasks, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            const Block b = block_of(i, nblk, NC, LCH, L, N);
            float ar[KMAX], ai[KMAX], Gr[KMAX] = {}, Gi[KMAX] = {};
            float x0r[KMAX] = {}, x0i[KMAX] = {}, glr[KMAX] = {}, gli[KMAX] = {};
            load_lanes<C, true>(lam, b.m, b.c0, b.nc, N, ar, ai);
            if (NC > 1) {
                const float* s = bounds.data() + (size_t)i * 2 * KMAX;
                for (int k = 0; k < b.nc; ++k) { Gr[k] = s[k]; Gi[k] = s[KMAX + k]; }
            }
            if (has_x0 && b.t0 == 0) load_lanes<C, false>(x0, b.m, b.c0, b.nc, N, x0r, x0i);
            bwd_chunk<C, true>(ar, ai, Gr, Gi, g + S * (b.m * L * N + b.c0),
                               out + S * (b.m * L * N + b.c0), x0r, x0i,
                               gv + S * (b.m * L * N + b.c0), glr, gli, b.t0, b.t1, N, b.nc);
            float* p = gl_part.data() + (size_t)i * 2 * KMAX;
            for (int k = 0; k < b.nc; ++k) { p[k] = glr[k]; p[KMAX + k] = gli[k]; }
            if (has_x0 && b.cj == 0) {
                float* f = finals.data() + (size_t)(b.m * nblk + b.c0 / KMAX) * 2 * KMAX;
                for (int k = 0; k < b.nc; ++k) { f[k] = Gr[k]; f[KMAX + k] = Gi[k]; }
            }
        }
    });
    // reduce glam over the chunks; gx0 = conj(lam) * G_0
    batch_parallel(M * nblk, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            const int64_t m = i / nblk, c0 = (i % nblk) * KMAX;
            const int nc = (int)std::min<int64_t>(KMAX, N - c0);
            float sr[KMAX] = {}, si[KMAX] = {};
            for (int64_t cj = 0; cj < NC; ++cj) {
                const float* p = gl_part.data() + ((size_t)i * NC + cj) * 2 * KMAX;
                for (int k = 0; k < nc; ++k) { sr[k] += p[k]; si[k] += p[KMAX + k]; }
            }
            for (int k = 0; k < nc; ++k) {
                if constexpr (C) {
                    glam[2 * (m * N + c0 + k)] = sr[k];
                    glam[2 * (m * N + c0 + k) + 1] = si[k];
                } else {
                    glam[m * N + c0 + k] = sr[k];
                }
            }
            if (has_x0) {
                float ar[KMAX], ai[KMAX];
                load_lanes<C, true>(lam, m, c0, nc, N, ar, ai);
                const float* f = finals.data() + (size_t)i * 2 * KMAX;
                for (int k = 0; k < nc; ++k) {
                    if constexpr (C) {
                        gx0[2 * (m * N + c0 + k)] = ar[k] * f[k] - ai[k] * f[KMAX + k];
                        gx0[2 * (m * N + c0 + k) + 1] = ar[k] * f[KMAX + k] + ai[k] * f[k];
                    } else {
                        gx0[m * N + c0 + k] = ar[k] * f[k];
                    }
                }
            }
        }
    });
}

void diag_fwd(torch::Tensor lam, torch::Tensor v, torch::Tensor x0, torch::Tensor out,
              bool has_x0, bool is_complex, int64_t M, int64_t L, int64_t N) {
    if (is_complex)
        fwd_t<true>(reinterpret_cast<const float*>(lam.data_ptr<cf>()),
                    reinterpret_cast<const float*>(v.data_ptr<cf>()),
                    has_x0 ? reinterpret_cast<const float*>(x0.data_ptr<cf>()) : nullptr,
                    reinterpret_cast<float*>(out.data_ptr<cf>()), has_x0, M, L, N);
    else
        fwd_t<false>(lam.data_ptr<float>(), v.data_ptr<float>(),
                     has_x0 ? x0.data_ptr<float>() : nullptr, out.data_ptr<float>(),
                     has_x0, M, L, N);
}

void diag_bwd(torch::Tensor g, torch::Tensor lam, torch::Tensor out, torch::Tensor x0,
              torch::Tensor gv, torch::Tensor glam, torch::Tensor gx0, bool has_x0,
              bool is_complex, int64_t M, int64_t L, int64_t N) {
    if (is_complex)
        bwd_t<true>(reinterpret_cast<const float*>(g.data_ptr<cf>()),
                    reinterpret_cast<const float*>(lam.data_ptr<cf>()),
                    reinterpret_cast<const float*>(out.data_ptr<cf>()),
                    has_x0 ? reinterpret_cast<const float*>(x0.data_ptr<cf>()) : nullptr, has_x0,
                    reinterpret_cast<float*>(gv.data_ptr<cf>()),
                    reinterpret_cast<float*>(glam.data_ptr<cf>()),
                    has_x0 ? reinterpret_cast<float*>(gx0.data_ptr<cf>()) : nullptr, M, L, N);
    else
        bwd_t<false>(g.data_ptr<float>(), lam.data_ptr<float>(), out.data_ptr<float>(),
                     has_x0 ? x0.data_ptr<float>() : nullptr, has_x0, gv.data_ptr<float>(),
                     glam.data_ptr<float>(),
                     has_x0 ? gx0.data_ptr<float>() : nullptr, M, L, N);
}
"""


def _source() -> str:
    darwin = sys.platform == "darwin"
    return "\n".join(
        [
            "#include <torch/extension.h>",
            "#include <ATen/Parallel.h>",
            "#include <c10/util/complex.h>",
            "#include <algorithm>",
            "#include <cstdint>",
            "#include <vector>",
            "",
            _BATCH_PARALLEL_GCD if darwin else _BATCH_PARALLEL_ATEN,
            _KERNEL,
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
