"""Time-tiled Triton execution backend for the constant-coefficient diagonal scan (LRU/S5 core).

The doubling scan of ``scan.diagonal_recurrence`` re-reads the whole sequence ``ceil(log2(L))``
times; this backend streams it once. Like the fused Mamba backend (``mamba_triton``), each
program owns a block of the ``batch x n`` independent lanes and walks time in tiles: tile-sized
loads amortize DRAM latency (a scalar step-by-step walk is latency-bound at ~one memory
round-trip per step) and ``tl.associative_scan`` runs the recurrence ``x_t = lam * x_{t-1} + v_t``
across the tile in registers, so the sequential dependency only crosses tile boundaries. The
analytic adjoint ``G_t = g_t + conj(lam) * G_{t+1}`` runs the same way as a ``reverse=True``
tile scan. Tile length, lane-block size, and warps are autotuned per shape.

Triton has no complex dtype, so complex64 tensors are passed through ``torch.view_as_real``
(zero-copy, interleaved re/im) and the recurrence is carried as explicit re/im pairs — the
affine maps compose through a 4-component combine function; the conjugates ``conj(lam)``
(gradient scan) and ``conj(x_{t-1})`` (``grad_lam`` reduction) are open-coded. A ``STR``
constexpr of 2 (complex) or 1 (real) selects the interleaved vs plain element stride so one
kernel source serves both float32 and complex64.

Only the coefficients and the forward output are saved for backward (matching the doubling
implementation's O(L) memory contract); unlike the Mamba backend no checkpointing or state
recomputation is needed, since the saved output *is* the state and ``x_{t-1}`` is re-read from
it with a one-step shift. Each lane reduces its own ``grad_lam`` over time locally; the
cross-batch / broadcast reduction of ``grad_lam`` (and of ``grad_v`` / ``grad_x0``) is finished
on the torch side by ``_reduce``, shared with the C backend.
"""

__all__ = [
    "supports",
    "run",
]

import torch
import triton
import triton.language as tl

from ..kernel_triton import is_available
from .diagonal_c import _prep, _reduce

_DTYPES = (torch.float32, torch.complex64)


def _configs():
    # BLOCK (lanes/program) trades load width against grid size — small lane counts need small
    # blocks to oversubscribe the SMs; BLOCK_T is the tile length of the in-register scan.
    return [
        triton.Config({"BLOCK": bl, "BLOCK_T": bt}, num_warps=w)
        for bl in (8, 16, 32)
        for bt in (32, 64)
        for w in (2, 4, 8)
    ]


@triton.jit
def _combine_r(a1, b1, a2, b2):
    # composition of x -> a x + b maps: second map applied after the first
    return a1 * a2, a2 * b1 + b2


@triton.jit
def _combine_c(a1r, a1i, b1r, b1i, a2r, a2i, b2r, b2i):
    # complex affine composition, re/im carried explicitly
    ar = a2r * a1r - a2i * a1i
    ai = a2r * a1i + a2i * a1r
    br = a2r * b1r - a2i * b1i + b2r
    bi = a2r * b1i + a2i * b1r + b2i
    return ar, ai, br, bi


@triton.autotune(configs=_configs(), key=["M", "L", "N", "STR", "HAS_X0"])
@triton.jit
def _diag_fwd_kernel(
    lam_ptr,
    v_ptr,
    x0_ptr,
    out_ptr,
    M,
    L,
    N,
    BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STR: tl.constexpr,
    HAS_X0: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    mask = lane < M * N
    m = lane // N
    c = lane % N
    base = (m * L * N + c) * STR  # element offset of (m, t=0, c) in the re/im-interleaved buffer

    lr = tl.load(lam_ptr + lane * STR, mask=mask)
    li = tl.load(lam_ptr + lane * STR + 1, mask=mask) if STR == 2 else 0.0
    if HAS_X0:
        xr = tl.load(x0_ptr + lane * STR, mask=mask)
        xi = tl.load(x0_ptr + lane * STR + 1, mask=mask) if STR == 2 else 0.0
    else:
        xr = tl.zeros([BLOCK], dtype=tl.float32)
        xi = tl.zeros([BLOCK], dtype=tl.float32)

    rows = tl.arange(0, BLOCK_T)
    last = rows[:, None] == BLOCK_T - 1
    for tc in tl.range(0, tl.cdiv(L, BLOCK_T)):
        offs_t = tc * BLOCK_T + rows
        m2 = (offs_t < L)[:, None] & mask[None, :]
        off = base[None, :] + offs_t[:, None] * (N * STR)
        vr = tl.load(v_ptr + off, mask=m2, other=0.0)
        # masked rows compose as exactly the identity map (a=1, b=0), so the carry extracted
        # from the last row is x_{L-1} even in a partial final tile
        ar = tl.where(m2, lr[None, :], 1.0)
        if STR == 2:
            vi = tl.load(v_ptr + off + 1, mask=m2, other=0.0)
            ai = tl.where(m2, li[None, :], 0.0)
            car, cai, cbr, cbi = tl.associative_scan((ar, ai, vr, vi), 0, _combine_c)
            xr_c = cbr + car * xr[None, :] - cai * xi[None, :]
            xi_c = cbi + car * xi[None, :] + cai * xr[None, :]
            tl.store(out_ptr + off, xr_c, mask=m2)
            tl.store(out_ptr + off + 1, xi_c, mask=m2)
            xi = tl.sum(tl.where(last, xi_c, 0.0), axis=0)
        else:
            car, cbr = tl.associative_scan((ar, vr), 0, _combine_r)
            xr_c = cbr + car * xr[None, :]
            tl.store(out_ptr + off, xr_c, mask=m2)
        xr = tl.sum(tl.where(last, xr_c, 0.0), axis=0)


@triton.autotune(configs=_configs(), key=["M", "L", "N", "STR", "HAS_X0"])
@triton.jit
def _diag_bwd_kernel(
    g_ptr,
    lam_ptr,
    out_ptr,
    x0_ptr,
    gv_ptr,
    glam_ptr,
    gx0_ptr,
    M,
    L,
    N,
    BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STR: tl.constexpr,
    HAS_X0: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    mask = lane < M * N
    m = lane // N
    c = lane % N
    base = (m * L * N + c) * STR

    lr = tl.load(lam_ptr + lane * STR, mask=mask)
    li = tl.load(lam_ptr + lane * STR + 1, mask=mask) if STR == 2 else 0.0
    # adjoint coefficient conj(lam) = (lr, -li)
    if HAS_X0:
        x0r = tl.load(x0_ptr + lane * STR, mask=mask)
        x0i = tl.load(x0_ptr + lane * STR + 1, mask=mask) if STR == 2 else 0.0
    else:
        x0r = tl.zeros([BLOCK], dtype=tl.float32)
        x0i = tl.zeros([BLOCK], dtype=tl.float32)

    Gr = tl.zeros([BLOCK], dtype=tl.float32)
    Gi = tl.zeros([BLOCK], dtype=tl.float32)
    glr = tl.zeros([BLOCK], dtype=tl.float32)
    gli = tl.zeros([BLOCK], dtype=tl.float32)

    rows = tl.arange(0, BLOCK_T)
    first = rows[:, None] == 0
    for ci in tl.range(0, tl.cdiv(L, BLOCK_T)):
        # tiles run backwards through time (the adjoint carry G flows that way); inside a tile
        # the reverse=True scan carries G from the last row down to the first.
        tc = tl.cdiv(L, BLOCK_T) - 1 - ci
        offs_t = tc * BLOCK_T + rows
        mask_t = offs_t < L
        m2 = mask_t[:, None] & mask[None, :]
        off = base[None, :] + offs_t[:, None] * (N * STR)
        gr = tl.load(g_ptr + off, mask=m2, other=0.0)
        ar = tl.where(m2, lr[None, :], 1.0)
        # x_{t-1}: previous output for t>0, else x0 (or 0); zeroed on masked rows so the
        # grad_lam accumulation only picks up valid steps
        poff = base[None, :] + (offs_t - 1)[:, None] * (N * STR)
        m_prev = (mask_t & (offs_t > 0))[:, None] & mask[None, :]
        xpr = tl.load(out_ptr + poff, mask=m_prev, other=0.0)
        if HAS_X0:
            at0 = (offs_t == 0)[:, None] & mask[None, :]
            xpr = tl.where(at0, x0r[None, :], xpr)
        if STR == 2:
            gi = tl.load(g_ptr + off + 1, mask=m2, other=0.0)
            ai = tl.where(m2, -li[None, :], 0.0)
            xpi = tl.load(out_ptr + poff + 1, mask=m_prev, other=0.0)
            if HAS_X0:
                xpi = tl.where(at0, x0i[None, :], xpi)
            car, cai, cbr, cbi = tl.associative_scan((ar, ai, gr, gi), 0, _combine_c, reverse=True)
            # G_t = g_t + conj(lam) G_{t+1}: compose the tile's reverse scan with the carry
            Gr_c = cbr + car * Gr[None, :] - cai * Gi[None, :]
            Gi_c = cbi + car * Gi[None, :] + cai * Gr[None, :]
            tl.store(gv_ptr + off, Gr_c, mask=m2)
            tl.store(gv_ptr + off + 1, Gi_c, mask=m2)
            # grad_lam += G_t * conj(x_{t-1})
            glr += tl.sum(Gr_c * xpr + Gi_c * xpi, axis=0)
            gli += tl.sum(Gi_c * xpr - Gr_c * xpi, axis=0)
            Gi = tl.sum(tl.where(first, Gi_c, 0.0), axis=0)
        else:
            car, cbr = tl.associative_scan((ar, gr), 0, _combine_r, reverse=True)
            Gr_c = cbr + car * Gr[None, :]
            tl.store(gv_ptr + off, Gr_c, mask=m2)
            glr += tl.sum(Gr_c * xpr, axis=0)
        Gr = tl.sum(tl.where(first, Gr_c, 0.0), axis=0)

    tl.store(glam_ptr + lane * STR, glr, mask=mask)
    if STR == 2:
        tl.store(glam_ptr + lane * STR + 1, gli, mask=mask)
    if HAS_X0:
        # grad_x0 = conj(lam) * G_0 (G holds G at t=0 after the reverse sweep)
        tl.store(gx0_ptr + lane * STR, lr * Gr + li * Gi, mask=mask)
        if STR == 2:
            tl.store(gx0_ptr + lane * STR + 1, lr * Gi - li * Gr, mask=mask)


def _as_real(t: torch.Tensor) -> torch.Tensor:
    """Flat real view for the kernel: view_as_real for complex, the tensor itself for float32."""
    return torch.view_as_real(t) if t.is_complex() else t


def supports(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> str | None:
    """Reason this backend cannot handle the inputs, or None when it can (see module docstring)."""
    if v.device.type != "cuda":
        return f"input on {v.device.type}, triton backend is CUDA-only"
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
        return "no CUDA / triton"
    return None


def _forward(lam_lane, v_flat, x0_lane, meta):
    _os, _bd, m, L, n = meta
    str_ = 2 if v_flat.is_complex() else 1
    out = torch.empty_like(v_flat)
    has_x0 = x0_lane is not None
    x0v = _as_real(x0_lane) if has_x0 else out.new_empty(0, dtype=torch.float32)
    lanes = m * n
    grid = lambda META: (triton.cdiv(lanes, META["BLOCK"]),)  # noqa: E731
    _diag_fwd_kernel[grid](
        _as_real(lam_lane),
        _as_real(v_flat),
        x0v,
        _as_real(out),
        m,
        L,
        n,
        STR=str_,
        HAS_X0=has_x0,
    )
    return out


class _TritonDiagonal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, v, x0):
        lam_lane, v_flat, x0_lane, meta = _prep(lam, v, x0)
        out = _forward(lam_lane, v_flat, x0_lane, meta)
        ctx.meta = meta
        ctx.lam, ctx.v, ctx.x0 = lam, v, x0
        ctx.save_for_backward(lam_lane, out, x0_lane)
        return out.reshape(meta[0])

    @staticmethod
    def backward(ctx, grad_out):
        _os, _bd, m, L, n = ctx.meta
        lam_lane, out, x0_lane = ctx.saved_tensors
        has_x0 = x0_lane is not None
        str_ = 2 if out.is_complex() else 1
        g = grad_out.reshape(m, L, n).contiguous()
        gv = torch.empty_like(out)
        glam = torch.empty_like(lam_lane)
        dummy = out.new_empty(0, dtype=torch.float32)
        gx0 = torch.empty_like(lam_lane) if has_x0 else None
        x0v = _as_real(x0_lane) if has_x0 else dummy
        grid = lambda META: (triton.cdiv(m * n, META["BLOCK"]),)  # noqa: E731
        _diag_bwd_kernel[grid](
            _as_real(g),
            _as_real(lam_lane),
            _as_real(out),
            x0v,
            _as_real(gv),
            _as_real(glam),
            _as_real(gx0) if has_x0 else dummy,
            m,
            L,
            n,
            STR=str_,
            HAS_X0=has_x0,
        )
        needs = (ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        return _reduce(gv, glam, gx0, ctx.lam, ctx.v, ctx.x0, ctx.meta, needs)


def run(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    """Run the constant-coefficient diagonal recurrence through the Triton kernels (autograd-capable)."""
    if not torch.is_grad_enabled() or not any(t is not None and t.requires_grad for t in (lam, v, x0)):
        lam_lane, v_flat, x0_lane, meta = _prep(lam, v, x0)
        return _forward(lam_lane, v_flat, x0_lane, meta).reshape(meta[0])
    return _TritonDiagonal.apply(lam, v, x0)
