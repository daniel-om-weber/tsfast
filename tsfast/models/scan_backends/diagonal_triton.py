"""Persistent-Triton execution backend for the constant-coefficient diagonal scan (LRU/S5 core).

The doubling scan of ``scan.diagonal_recurrence`` re-reads the whole sequence ``ceil(log2(L))``
times; this backend instead streams it once. The recurrence ``x_t = lam * x_{t-1} + v_t`` and its
analytic adjoint (a reverse-time scan with coefficient ``conj(lam)``) each run as a single
persistent kernel: the grid is over the ``batch x n`` independent lanes flattened into blocks,
every lane loads its constant coefficient once into registers and sweeps its column of the
sequence in one pass. Triton has no complex dtype, so complex64 tensors are passed through
``torch.view_as_real`` (zero-copy, interleaved re/im) and the real/imaginary parts are carried
explicitly through the kernel; the conjugates ``conj(lam)`` (gradient scan) and ``conj(x_{t-1})``
(``grad_lam`` reduction) are open-coded. A ``STR`` constexpr of 2 (complex) or 1 (real) selects
the interleaved vs plain element stride so one kernel serves both float32 and complex64.

Only the coefficients and the forward output are saved for backward (matching the doubling
implementation's O(L) memory contract). Each lane reduces its own ``grad_lam`` over time locally;
the cross-batch / broadcast reduction of ``grad_lam`` (and of ``grad_v`` / ``grad_x0``) is
finished on the torch side by ``_reduce``, shared with the C backend.
"""

__all__ = [
    "supports",
    "run",
]

import torch
import triton
import triton.language as tl

from ..ssm.backend_triton import is_available
from .diagonal_c import _prep, _reduce

_DTYPES = (torch.float32, torch.complex64)
# The recurrence is sequential in time, so a lane's throughput is memory-latency bound; with only
# B*n independent lanes at model-realistic batch sizes there are too few warps to hide that latency
# by occupancy alone. Software-pipelining the loop (num_stages) prefetches the independent per-step
# input loads several iterations ahead, which is what recovers most of the speedup at small batch.
_BLOCK = 32
_NUM_STAGES = 8


@triton.jit
def _diag_fwd_kernel(
    lam_ptr, v_ptr, x0_ptr, out_ptr, M, L, N,
    BLOCK: tl.constexpr, STR: tl.constexpr, HAS_X0: tl.constexpr, NUM_STAGES: tl.constexpr,
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

    for t in tl.range(0, L, num_stages=NUM_STAGES):
        off = base + t * N * STR
        vr = tl.load(v_ptr + off, mask=mask)
        if STR == 2:
            vi = tl.load(v_ptr + off + 1, mask=mask)
            nxr = lr * xr - li * xi + vr
            xi = lr * xi + li * xr + vi
            xr = nxr
            tl.store(out_ptr + off, xr, mask=mask)
            tl.store(out_ptr + off + 1, xi, mask=mask)
        else:
            xr = lr * xr + vr
            tl.store(out_ptr + off, xr, mask=mask)


@triton.jit
def _diag_bwd_kernel(
    g_ptr, lam_ptr, out_ptr, x0_ptr, gv_ptr, glam_ptr, gx0_ptr, M, L, N,
    BLOCK: tl.constexpr, STR: tl.constexpr, HAS_X0: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    mask = lane < M * N
    m = lane // N
    c = lane % N
    base = (m * L * N + c) * STR

    lr = tl.load(lam_ptr + lane * STR, mask=mask)
    li = tl.load(lam_ptr + lane * STR + 1, mask=mask) if STR == 2 else 0.0
    # conj(lam) = (lr, -li)
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

    for ti in tl.range(0, L, num_stages=NUM_STAGES):
        t = L - 1 - ti
        off = base + t * N * STR
        gr = tl.load(g_ptr + off, mask=mask)
        # x_{t-1}: previous output for t>0, else x0 (or 0)
        poff = base + tl.maximum(t - 1, 0) * N * STR
        opr = tl.load(out_ptr + poff, mask=mask)
        xpr = tl.where(t > 0, opr, x0r)
        if STR == 2:
            gi = tl.load(g_ptr + off + 1, mask=mask)
            opi = tl.load(out_ptr + poff + 1, mask=mask)
            xpi = tl.where(t > 0, opi, x0i)
            # G = g + conj(lam) * G
            nGr = gr + lr * Gr + li * Gi
            Gi = gi + lr * Gi - li * Gr
            Gr = nGr
            tl.store(gv_ptr + off, Gr, mask=mask)
            tl.store(gv_ptr + off + 1, Gi, mask=mask)
            # grad_lam += G * conj(x_prev)
            glr += Gr * xpr + Gi * xpi
            gli += Gi * xpr - Gr * xpi
        else:
            Gr = gr + lr * Gr
            tl.store(gv_ptr + off, Gr, mask=mask)
            glr += Gr * xpr

    tl.store(glam_ptr + lane * STR, glr, mask=mask)
    if STR == 2:
        tl.store(glam_ptr + lane * STR + 1, gli, mask=mask)
    if HAS_X0:
        # grad_x0 = conj(lam) * G_1 (G holds G at t=0 after the reverse sweep)
        tl.store(gx0_ptr + lane * STR, lr * Gr + li * Gi, mask=mask)
        if STR == 2:
            tl.store(gx0_ptr + lane * STR + 1, lr * Gi - li * Gr, mask=mask)


def _num_warps(block: int) -> int:
    return max(1, min(8, block // 32))


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
    grid = (triton.cdiv(lanes, _BLOCK),)
    _diag_fwd_kernel[grid](
        _as_real(lam_lane), _as_real(v_flat), x0v, _as_real(out), m, L, n,
        BLOCK=_BLOCK, STR=str_, HAS_X0=has_x0, NUM_STAGES=_NUM_STAGES, num_warps=_num_warps(_BLOCK),
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
        grid = (triton.cdiv(m * n, _BLOCK),)
        _diag_bwd_kernel[grid](
            _as_real(g), _as_real(lam_lane), _as_real(out), x0v,
            _as_real(gv), _as_real(glam), _as_real(gx0) if has_x0 else dummy, m, L, n,
            BLOCK=_BLOCK, STR=str_, HAS_X0=has_x0, NUM_STAGES=_NUM_STAGES, num_warps=_num_warps(_BLOCK),
        )
        needs = (ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        return _reduce(gv, glam, gx0, ctx.lam, ctx.v, ctx.x0, ctx.meta, needs)


def run(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    """Run the constant-coefficient diagonal recurrence through the Triton kernels (autograd-capable)."""
    if not torch.is_grad_enabled() or not any(t is not None and t.requires_grad for t in (lam, v, x0)):
        lam_lane, v_flat, x0_lane, meta = _prep(lam, v, x0)
        return _forward(lam_lane, v_flat, x0_lane, meta).reshape(meta[0])
    return _TritonDiagonal.apply(lam, v, x0)
