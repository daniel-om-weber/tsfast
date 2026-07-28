"""Fused generated-C++ backend for the Mamba selective SSM section on CPU.

The generic ``selective_recurrence`` path materializes ``lam = exp(delta A)``,
``v = (delta u) B_t`` and the states ``h`` as ``[B, L, d_inner*d_state]`` tensors and makes
roughly forty passes over them per training step, which is what the section costs on CPU —
not the scan arithmetic, which is a single fused multiply-add per state element. This backend
fuses the whole section — softplus discretization, ``exp(delta A)``, input injection,
recurrence, output contraction ``y_t = h_t . C_t``, skip ``D u`` and the SiLU gate — into one
kernel over the *unexpanded* inputs, so nothing ``[B, L, D, N]``-sized is written at all in
inference and only sparse state checkpoints are written in training.

Each task owns a ``(batch, channel-block)`` slice and walks time carrying ``[BLOCK_D, N]``
states in registers; ``N`` is a compile-time constant so the inner state loop unrolls and
vectorizes, and the block over channels lets one load of ``B_t``/``C_t`` serve ``BLOCK_D``
channels. Parallelism spans ``batch x channel-blocks`` through the shared batch-parallel
driver. The backward walks the checkpoints in reverse, recomputes each chunk's states with
the forward recurrence into an L1-resident buffer, then sweeps the chunk backwards running
the adjoint ``G_t = gy_t C_t + lam_{t+1} G_{t+1}``. ``lam_t h_{t-1}`` is obtained as
``h_t - v_t``, so states before the chunk are never needed. ``grad_Bt``/``grad_Ct`` are
reduced over the channel blocks (and ``grad_A``/``grad_D`` over the batch) via partial
buffers summed in PyTorch, keeping the kernel free of atomics (deterministic).

Real float32 only, matching the regime of the reference CUDA kernel and of the Triton
backend whose op contract this mirrors.
"""

__all__ = [
    "supports",
    "run",
]

import ctypes
import sys

import torch

from ..._core.kernel_c import _BATCH_PARALLEL_ATEN, _BATCH_PARALLEL_GCD, is_available, load_cabi

#: State-checkpoint interval, and with it the backward's recompute chunk length. The reverse
#: sweep holds ``CHK x BLOCK_D x N`` recomputed states, which must stay in L1; within that
#: bound larger is better, since the checkpoint buffer shrinks as 1/CHK while the recompute
#: cost (exactly one extra forward pass) does not depend on it.
_CHK = 16

#: Channels per task. Bounds the same L1 buffer as ``_CHK`` and sets the task count
#: ``B * ceil(d_inner / BLOCK_D)``, which has to cover the thread pool for small batches.
_BLOCK_D = 8

_EXTENSIONS: dict[int, ctypes.CDLL] = {}

# exp on [-88, 88] by range reduction to 2^k and a degree-6 Taylor series of the remainder,
# accurate to ~1e-7 relative. Plain arithmetic instead of libm keeps the state loop — which
# evaluates one exp per state element, the kernel's densest operation — auto-vectorizable on
# every host, macOS shipping no vector libm at all.
_FAST_EXP_C = """\
static inline float fast_expf(float x) {
    x = fminf(88.0f, fmaxf(-88.0f, x));
    const float t = x * 1.44269504088896341f;
    const int32_t k = (int32_t)(t + (t >= 0.0f ? 0.5f : -0.5f));
    const float kf = (float)k;
    // ln2 split so that kf * hi is exact in float and the remainder cannot cancel
    float r = x - kf * 0.693359375f;
    r = r + kf * 2.12194440e-4f;
    float p = 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 5.0e-1f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    union { float f; int32_t i; } s;
    s.i = (k + 127) << 23;
    return p * s.f;
}
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + fast_expf(-x)); }
static inline float softplusf_(float x) { return x > 20.0f ? x : log1pf(fast_expf(x)); }
"""


def _gen_source(n_state: int) -> str:
    """Emit the forward and backward kernels specialized to the state dimension."""
    return "\n".join(
        [
            "#include <ATen/Parallel.h>",
            "#include <algorithm>",
            "#include <cmath>",
            "#include <cstdint>",
            "",
            _BATCH_PARALLEL_GCD if sys.platform == "darwin" else _BATCH_PARALLEL_ATEN,
            _FAST_EXP_C,
            f"constexpr int N = {n_state};",
            f"constexpr int BD = {_BLOCK_D};",
            f"constexpr int CHK = {_CHK};",
            "",
            _FWD_C,
            "",
            _BWD_C,
        ]
    )


# h_t = exp(delta_t A) h_{t-1} + delta_t u_t B_t, out_t = (h_t . C_t + Dp u_t) silu(z_t).
# Checkpoint slot k holds the state at t = (k + 1) * CHK - 1, which is what chunk k + 1
# of the backward recomputes from.
_FWD_C = """\
extern "C" void mamba_fwd(int64_t B, int64_t L, int64_t D, const float* draw, const float* A,
        const float* Bt, const float* Ct, const float* u, const float* z, const float* Dp,
        const float* h0, float* out, float* chk, float* hlast, bool has_h0, bool store_chk) {
    const int64_t ndb = (D + BD - 1) / BD, nc = (L + CHK - 1) / CHK;
    batch_parallel(B * ndb, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
        const int64_t b = i / ndb, d0 = (i % ndb) * BD;
        const int nd = (int)std::min<int64_t>(BD, D - d0);
        float h[BD][N];
        for (int j = 0; j < nd; ++j)
            for (int n = 0; n < N; ++n) h[j][n] = has_h0 ? h0[(b * D + d0 + j) * N + n] : 0.0f;
        for (int64_t t = 0; t < L; ++t) {
            const float* Btt = Bt + (b * L + t) * N;
            const float* Ctt = Ct + (b * L + t) * N;
            const int64_t pd = (b * L + t) * D + d0;
            for (int j = 0; j < nd; ++j) {
                const float* Ad = A + (d0 + j) * N;
                const float delta = softplusf_(draw[pd + j]);
                const float uu = u[pd + j], du = delta * uu;
                float acc = 0.0f;
                for (int n = 0; n < N; ++n) {
                    const float hv = fast_expf(delta * Ad[n]) * h[j][n] + du * Btt[n];
                    h[j][n] = hv;
                    acc += hv * Ctt[n];
                }
                const float zz = z[pd + j];
                out[pd + j] = (acc + Dp[d0 + j] * uu) * zz * sigmoidf_(zz);
            }
            if (store_chk && (t + 1) % CHK == 0) {
                const int64_t slot = (t + 1) / CHK - 1;
                for (int j = 0; j < nd; ++j)
                    for (int n = 0; n < N; ++n) chk[((b * nc + slot) * D + d0 + j) * N + n] = h[j][n];
            }
        }
        for (int j = 0; j < nd; ++j)
            for (int n = 0; n < N; ++n) hlast[(b * D + d0 + j) * N + n] = h[j][n];
    }
    });
}"""


# The adjoint coefficient at step t is lam_{t+1}; at t = L-1 it is the identity, which is what
# passes an incoming state gradient into G_{L-1} unchanged.
_BWD_C = """\
extern "C" void mamba_bwd(int64_t B, int64_t L, int64_t D, const float* draw, const float* A,
        const float* Bt, const float* Ct, const float* u, const float* z, const float* Dp,
        const float* h0, const float* chk, const float* gout, const float* ghlast, float* gdraw,
        float* gA_part, float* gBt_part, float* gCt_part, float* gu, float* gz, float* gDp_part,
        float* gh0, bool has_h0, bool has_glast, bool need_gh0) {
    const int64_t ndb = (D + BD - 1) / BD, nc = (L + CHK - 1) / CHK;
    batch_parallel(B * ndb, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
        const int64_t b = i / ndb, blk = i % ndb, d0 = blk * BD;
        const int nd = (int)std::min<int64_t>(BD, D - d0);
        float G[BD][N], gA[BD][N], gDp_acc[BD], hs[BD][N], hbuf[CHK][BD][N], dbuf[CHK][BD];
        for (int j = 0; j < nd; ++j) {
            gDp_acc[j] = 0.0f;
            for (int n = 0; n < N; ++n) {
                G[j][n] = has_glast ? ghlast[(b * D + d0 + j) * N + n] : 0.0f;
                gA[j][n] = 0.0f;
            }
        }
        // this task owns the (block, batch) slice of the B_t/C_t partials outright
        float* gBp = gBt_part + (blk * B + b) * L * N;
        float* gCp = gCt_part + (blk * B + b) * L * N;
        for (int64_t p = 0; p < L * N; ++p) { gBp[p] = 0.0f; gCp[p] = 0.0f; }

        for (int64_t c = nc - 1; c >= 0; --c) {
            const int64_t t0 = c * CHK;
            const int len = (int)(std::min<int64_t>(L, t0 + CHK) - t0);
            for (int j = 0; j < nd; ++j)
                for (int n = 0; n < N; ++n)
                    hs[j][n] = c > 0 ? chk[((b * nc + c - 1) * D + d0 + j) * N + n]
                                     : (has_h0 ? h0[(b * D + d0 + j) * N + n] : 0.0f);
            for (int ii = 0; ii < len; ++ii) {
                const float* Btt = Bt + (b * L + t0 + ii) * N;
                const int64_t pd = (b * L + t0 + ii) * D + d0;
                for (int j = 0; j < nd; ++j) {
                    const float* Ad = A + (d0 + j) * N;
                    const float delta = softplusf_(draw[pd + j]);
                    const float du = delta * u[pd + j];
                    dbuf[ii][j] = delta;
                    for (int n = 0; n < N; ++n) {
                        hs[j][n] = fast_expf(delta * Ad[n]) * hs[j][n] + du * Btt[n];
                        hbuf[ii][j][n] = hs[j][n];
                    }
                }
            }
            for (int ii = len - 1; ii >= 0; --ii) {
                const int64_t t = t0 + ii;
                const float* Btt = Bt + (b * L + t) * N;
                const float* Ctt = Ct + (b * L + t) * N;
                const int64_t pd = (b * L + t) * D + d0;
                float* gBr = gBp + t * N;
                float* gCr = gCp + t * N;
                for (int j = 0; j < nd; ++j) {
                    const float* Ad = A + (d0 + j) * N;
                    const float* hv = hbuf[ii][j];
                    const float delta = dbuf[ii][j];
                    const float uu = u[pd + j], zz = z[pd + j], du = delta * uu;
                    float y = Dp[d0 + j] * uu;
                    for (int n = 0; n < N; ++n) y += hv[n] * Ctt[n];
                    const float sig = sigmoidf_(zz), go = gout[pd + j];
                    const float gy = go * zz * sig;
                    gz[pd + j] = go * y * sig * (1.0f + zz * (1.0f - sig));
                    gDp_acc[j] += gy * uu;
                    if (t + 1 < L) {
                        const float dn = softplusf_(draw[pd + j + D]);
                        for (int n = 0; n < N; ++n) G[j][n] = gy * Ctt[n] + fast_expf(dn * Ad[n]) * G[j][n];
                    } else {
                        for (int n = 0; n < N; ++n) G[j][n] = gy * Ctt[n] + G[j][n];
                    }
                    float gdelta = 0.0f, gus = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        const float lam_hprev = hv[n] - du * Btt[n], g = G[j][n];
                        gCr[n] += gy * hv[n];
                        gBr[n] += g * du;
                        gA[j][n] += g * lam_hprev * delta;
                        gdelta += g * (Ad[n] * lam_hprev + uu * Btt[n]);
                        gus += g * Btt[n];
                    }
                    gdraw[pd + j] = gdelta * sigmoidf_(draw[pd + j]);
                    gu[pd + j] = delta * gus + gy * Dp[d0 + j];
                }
            }
        }
        for (int j = 0; j < nd; ++j) {
            gDp_part[b * D + d0 + j] = gDp_acc[j];
            for (int n = 0; n < N; ++n) gA_part[(b * D + d0 + j) * N + n] = gA[j][n];
        }
        if (need_gh0) {
            for (int j = 0; j < nd; ++j) {
                const float* Ad = A + (d0 + j) * N;
                const float d_0 = softplusf_(draw[b * L * D + d0 + j]);
                for (int n = 0; n < N; ++n) gh0[(b * D + d0 + j) * N + n] = fast_expf(d_0 * Ad[n]) * G[j][n];
            }
        }
    }
    });
}"""


def _get_extension(n_state: int) -> ctypes.CDLL:
    ext = _EXTENSIONS.get(n_state)
    if ext is None:
        ext = load_cabi(_gen_source(n_state), f"tsfast_mamba_c_n{n_state}")
        _EXTENSIONS[n_state] = ext
    return ext


def _call(ext, name: str, sizes: tuple[int, int, int], tensors: list[torch.Tensor], flags: list[bool]) -> None:
    """Invoke a generated entry point on the tensors' raw storage.

    The kernels index every argument as contiguous float32 of a size the shapes fix, and a C
    ABI has no way to notice otherwise. ``supports`` screens the layer inputs, but the buffers
    allocated here and the gradients arriving from autograd do not pass through it, so dtype,
    device and layout are checked once per call; pinning ``argtypes`` covers a marshalling list
    that has drifted from the generated signature.
    """
    for t in tensors:
        if t.dtype is not torch.float32 or t.device.type != "cpu" or not t.is_contiguous():
            raise TypeError(
                "the C backend requires contiguous float32 CPU tensors, got "
                f"{t.dtype} on {t.device}{'' if t.is_contiguous() else ' (non-contiguous)'}"
            )
    fn = getattr(ext, name)
    if fn.argtypes is None:
        fn.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * len(tensors) + [ctypes.c_bool] * len(flags)
        fn.restype = None
    fn(*sizes, *(ctypes.c_void_p(t.data_ptr()) for t in tensors), *flags)


def _forward(draw, A, Bt, Ct, u, z, Dp, h0, store_chk: bool):
    B, L, D = draw.shape
    N = A.shape[-1]
    nc = -(-L // _CHK)
    out = torch.empty_like(draw)
    h_last = torch.empty(B, D, N)
    chk = torch.empty(B, nc, D, N) if store_chk else torch.empty(0)
    has_h0 = h0 is not None
    _call(
        _get_extension(N),
        "mamba_fwd",
        (B, L, D),
        [draw, A, Bt, Ct, u, z, Dp, h0 if has_h0 else h_last, out, chk if store_chk else out, h_last],
        [has_h0, store_chk],
    )
    return out, h_last, (chk if store_chk else None)


def _contig(*ts: torch.Tensor | None) -> list[torch.Tensor | None]:
    return [t.contiguous() if t is not None else None for t in ts]


@torch.library.custom_op("tsfast::mamba_scan_c", mutates_args=())
def _scan_op(
    draw: torch.Tensor,
    A: torch.Tensor,
    Bt: torch.Tensor,
    Ct: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    Dp: torch.Tensor,
    h0: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    draw, A, Bt, Ct, u, z, Dp, h0 = _contig(draw, A, Bt, Ct, u, z, Dp, h0)
    out, h_last, _ = _forward(draw, A, Bt, Ct, u, z, Dp, h0, store_chk=False)
    return out, h_last


@torch.library.custom_op("tsfast::mamba_scan_train_c", mutates_args=())
def _scan_train_op(
    draw: torch.Tensor,
    A: torch.Tensor,
    Bt: torch.Tensor,
    Ct: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    Dp: torch.Tensor,
    h0: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    draw, A, Bt, Ct, u, z, Dp, h0 = _contig(draw, A, Bt, Ct, u, z, Dp, h0)
    return _forward(draw, A, Bt, Ct, u, z, Dp, h0, store_chk=True)


def _fake_outs(draw, A, with_chk: bool):
    B, L, D = draw.shape
    N = A.shape[-1]
    out = torch.empty_like(draw)
    h_last = draw.new_empty(B, D, N)
    if not with_chk:
        return out, h_last
    return out, h_last, draw.new_empty(B, -(-L // _CHK), D, N)


@_scan_op.register_fake
def _(draw, A, Bt, Ct, u, z, Dp, h0):
    return _fake_outs(draw, A, with_chk=False)


@_scan_train_op.register_fake
def _(draw, A, Bt, Ct, u, z, Dp, h0):
    return _fake_outs(draw, A, with_chk=True)


@torch.library.custom_op("tsfast::mamba_scan_bwd_c", mutates_args=())
def _scan_bwd_op(
    gout: torch.Tensor,
    ghlast: torch.Tensor | None,
    chk: torch.Tensor,
    draw: torch.Tensor,
    A: torch.Tensor,
    Bt: torch.Tensor,
    Ct: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    Dp: torch.Tensor,
    h0: torch.Tensor | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    gout, ghlast, chk, draw, A, Bt, Ct, u, z, Dp, h0 = _contig(gout, ghlast, chk, draw, A, Bt, Ct, u, z, Dp, h0)
    B, L, D = draw.shape
    N = A.shape[-1]
    ndb = -(-D // _BLOCK_D)
    has_h0 = h0 is not None
    # a zero upstream state-gradient arrives as None from autograd
    has_glast = ghlast is not None
    gdraw, gu, gz = (torch.empty_like(draw) for _ in range(3))
    gA_part = torch.empty(B, D, N)
    gBt_part = torch.empty(ndb, B, L, N)
    gCt_part = torch.empty(ndb, B, L, N)
    gDp_part = torch.empty(B, D)
    gh0 = torch.empty(B, D, N) if has_h0 else torch.empty(0)
    _call(
        _get_extension(N),
        "mamba_bwd",
        (B, L, D),
        [
            draw,
            A,
            Bt,
            Ct,
            u,
            z,
            Dp,
            h0 if has_h0 else gA_part,
            chk,
            gout,
            ghlast if has_glast else gA_part,
            gdraw,
            gA_part,
            gBt_part,
            gCt_part,
            gu,
            gz,
            gDp_part,
            gh0 if has_h0 else gA_part,
        ],
        [has_h0, has_glast, has_h0],
    )
    return (
        gdraw,
        gA_part.sum(dim=0),
        gBt_part.sum(dim=0),
        gCt_part.sum(dim=0),
        gu,
        gz,
        gDp_part.sum(dim=0),
        (gh0 if has_h0 else gout.new_empty(0)),
    )


@_scan_bwd_op.register_fake
def _(gout, ghlast, chk, draw, A, Bt, Ct, u, z, Dp, h0):
    B, L, D = draw.shape
    N = A.shape[-1]
    return (
        torch.empty_like(draw),
        draw.new_empty(D, N),
        draw.new_empty(B, L, N),
        draw.new_empty(B, L, N),
        torch.empty_like(draw),
        torch.empty_like(draw),
        draw.new_empty(D),
        (draw.new_empty(B, D, N) if h0 is not None else draw.new_empty(0)),
    )


def _scan_setup(ctx, inputs, output):
    draw, A, Bt, Ct, u, z, Dp, h0 = inputs
    ctx.save_for_backward(draw, A, Bt, Ct, u, z, Dp, h0, output[2])


def _scan_backward(ctx, gout, ghlast, gchk):
    draw, A, Bt, Ct, u, z, Dp, h0, chk = ctx.saved_tensors
    gdraw, gA, gBt, gCt, gu, gz, gDp, gh0 = _scan_bwd_op(gout, ghlast, chk, draw, A, Bt, Ct, u, z, Dp, h0)
    return gdraw, gA, gBt, gCt, gu, gz, gDp, (gh0 if h0 is not None else None)


_scan_train_op.register_autograd(_scan_backward, setup_context=_scan_setup)


def supports(draw, A, Bt, Ct, u, z, Dp, h0) -> str | None:
    """Reason the fused C++ kernel cannot handle these tensors, or None if it can."""
    tensors = {"draw": draw, "A": A, "Bt": Bt, "Ct": Ct, "u": u, "z": z, "Dp": Dp}
    if h0 is not None:
        tensors["h0"] = h0
    for name, t in tensors.items():
        if t.device.type != "cpu":
            return f"{name} not on CPU"
        if t.dtype != torch.float32:
            return f"needs float32, got {name}={t.dtype}"
    if not is_available():
        return "no host C++ toolchain / ninja"
    if draw.dim() != 3:
        return f"draw must be [B, L, D], got {draw.dim()} dims"
    B, L, D = draw.shape
    N = A.shape[-1]
    if tuple(A.shape) != (D, N):
        return f"A.shape {tuple(A.shape)} incompatible with draw {tuple(draw.shape)}"
    if tuple(u.shape) != (B, L, D) or tuple(z.shape) != (B, L, D):
        return f"u/z shapes {tuple(u.shape)}/{tuple(z.shape)} != {(B, L, D)}"
    if tuple(Bt.shape) != (B, L, N) or tuple(Ct.shape) != (B, L, N):
        return f"Bt/Ct shapes {tuple(Bt.shape)}/{tuple(Ct.shape)} != {(B, L, N)}"
    if tuple(Dp.shape) != (D,):
        return f"Dp.shape {tuple(Dp.shape)} != {(D,)}"
    if h0 is not None and tuple(h0.shape) != (B, D, N):
        return f"h0.shape {tuple(h0.shape)} != {(B, D, N)}"
    return None


def run(draw, A, Bt, Ct, u, z, Dp, h0):
    """Fused Mamba SSM: ``h_t = exp(softplus(draw_t) A) h_{t-1} + softplus(draw_t) u_t B_t``,
    output ``(h_t . C_t + Dp u_t) * silu(z_t)``.

    Returns ``(out [B, L, D], h_last [B, D, N])``; differentiable in all inputs (via the
    ``tsfast::mamba_scan_c`` ops; the training op additionally stores state checkpoints).
    """
    inputs = (draw, A, Bt, Ct, u, z, Dp) + ((h0,) if h0 is not None else ())
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in inputs):
        return _scan_op(draw, A, Bt, Ct, u, z, Dp, h0)
    out, h_last, _chk = _scan_train_op(draw, A, Bt, Ct, u, z, Dp, h0)
    return out, h_last
