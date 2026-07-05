"""Persistent-Triton backend for the selective (time-varying diagonal) recurrence.

The doubling scan in ``tsfast.models._core.scan`` re-reads the full ``[B, L, N]`` coefficient
and input tensors ~log2(L) times. This backend instead runs the recurrence as a single
sequential-in-time persistent kernel: the grid covers ``(batch x N-chunks)`` and each
program streams over ``L`` once, loading ``a_t`` and ``v_t`` and writing ``x_t`` per step
while carrying the state in registers. One read, one write, no O(L log L) intermediates.

Backward is the analytic reverse-time adjoint ``G_t = g_t + a_{t+1} G_{t+1}`` run as a
second persistent kernel (time descending), emitting ``grad_v = G``, ``grad_lam = G x_{t-1}``
and ``grad_x0 = a_0 G_0`` in one sweep. Only ``lam`` and the forward output are saved for
backward, so the backward memory stays O(L). Real float32 only (the Mamba selective scan),
so no conjugation is needed.
"""

__all__ = [
    "supports",
    "run",
]

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    _HAVE_TRITON = False


def _configs():
    return [
        triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=s) for bn in (64, 128) for w in (1, 2, 4) for s in (1, 2)
    ]


if _HAVE_TRITON:

    @triton.autotune(configs=_configs(), key=["L", "N"])
    @triton.jit
    def _sel_fwd(lam_ptr, v_ptr, x0_ptr, out_ptr, B, L, N, HAS_X0: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        n_chunks = tl.cdiv(N, BLOCK_N)
        b = pid // n_chunks
        offs = (pid - b * n_chunks) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        row = b * L * N + offs
        if HAS_X0:
            x = tl.load(x0_ptr + b * N + offs, mask=mask, other=0.0)
        else:
            x = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in tl.range(0, L):
            p = row + t * N
            a = tl.load(lam_ptr + p, mask=mask, other=0.0)
            vt = tl.load(v_ptr + p, mask=mask, other=0.0)
            x = a * x + vt
            tl.store(out_ptr + p, x, mask=mask)

    @triton.autotune(configs=_configs(), key=["L", "N"])
    @triton.jit
    def _sel_bwd(
        lam_ptr,
        out_ptr,
        g_ptr,
        x0_ptr,
        glam_ptr,
        gv_ptr,
        gx0_ptr,
        B,
        L,
        N,
        HAS_X0: tl.constexpr,
        NEED_X0: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_chunks = tl.cdiv(N, BLOCK_N)
        b = pid // n_chunks
        offs = (pid - b * n_chunks) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        row = b * L * N + offs
        G = tl.zeros([BLOCK_N], dtype=tl.float32)
        gx0 = tl.zeros([BLOCK_N], dtype=tl.float32)
        for ti in tl.range(0, L):
            t = L - 1 - ti
            p = row + t * N
            g = tl.load(g_ptr + p, mask=mask, other=0.0)
            a_next = tl.load(lam_ptr + p + N, mask=mask & (t + 1 < L), other=0.0)
            G = g + a_next * G
            tl.store(gv_ptr + p, G, mask=mask)
            a = tl.load(lam_ptr + p, mask=mask, other=0.0)
            xprev = tl.load(out_ptr + p - N, mask=mask & (t > 0), other=0.0)
            if HAS_X0:
                x0v = tl.load(x0_ptr + b * N + offs, mask=mask, other=0.0)
                xprev = tl.where(t > 0, xprev, x0v)
            tl.store(glam_ptr + p, G * xprev, mask=mask)
            if NEED_X0:
                gx0 = tl.where(t == 0, a * G, gx0)
        if NEED_X0:
            tl.store(gx0_ptr + b * N + offs, gx0, mask=mask)


def _grid(B, N):
    return lambda meta: (B * triton.cdiv(N, meta["BLOCK_N"]),)


def _forward(lam, v, x0):
    B, L, N = lam.shape
    out = torch.empty_like(lam)
    has_x0 = x0 is not None
    x0_arg = x0 if has_x0 else lam  # unused pointer when HAS_X0 is False
    _sel_fwd[_grid(B, N)](lam, v, x0_arg, out, B, L, N, HAS_X0=has_x0)
    return out


class _SelectiveTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, v, x0):
        out = _forward(lam, v, x0)
        ctx.save_for_backward(lam, out, x0)
        ctx.has_x0 = x0 is not None
        return out

    @staticmethod
    def backward(ctx, g):
        lam, out, x0 = ctx.saved_tensors
        g = g.contiguous()
        B, L, N = lam.shape
        has_x0 = ctx.has_x0
        need_lam, need_v, need_x0 = ctx.needs_input_grad
        need_x0 = bool(need_x0 and has_x0)
        glam = torch.empty_like(lam)
        gv = torch.empty_like(lam)
        gx0 = torch.empty(B, N, device=lam.device, dtype=lam.dtype) if need_x0 else lam
        x0_arg = x0 if has_x0 else lam
        _sel_bwd[_grid(B, N)](lam, out, g, x0_arg, glam, gv, gx0, B, L, N, HAS_X0=has_x0, NEED_X0=need_x0)
        grad_lam = glam if need_lam else None
        grad_v = gv if need_v else None
        grad_x0 = gx0 if need_x0 else None
        return grad_lam, grad_v, grad_x0


def supports(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> str | None:
    """Reason the persistent Triton kernel cannot handle these tensors, or None if it can."""
    if not _HAVE_TRITON:
        return "triton not importable"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if lam.device.type != "cuda" or v.device.type != "cuda":
        return "inputs not on CUDA"
    if lam.dtype != torch.float32 or v.dtype != torch.float32:
        return f"needs float32, got lam={lam.dtype}, v={v.dtype}"
    if lam.shape != v.shape:
        return f"lam.shape {tuple(lam.shape)} != v.shape {tuple(v.shape)}"
    if lam.dim() < 2:
        return f"needs a leading batch and a time dim, got {lam.dim()} dims"
    if x0 is not None:
        expected = lam.shape[:-2] + lam.shape[-1:]
        if x0.dtype != torch.float32:
            return f"x0 must be float32, got {x0.dtype}"
        if tuple(x0.shape) != tuple(expected):
            return f"x0.shape {tuple(x0.shape)} != expected {tuple(expected)}"
    return None


def run(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    """Selective recurrence ``x_t = lam_t x_{t-1} + v_t`` via the persistent Triton kernel."""
    lam3 = lam.contiguous().reshape(-1, lam.shape[-2], lam.shape[-1])
    v3 = v.contiguous().reshape(-1, v.shape[-2], v.shape[-1])
    x03 = x0.contiguous().reshape(-1, x0.shape[-1]) if x0 is not None else None
    if not torch.is_grad_enabled() or not (
        lam3.requires_grad or v3.requires_grad or (x03 is not None and x03.requires_grad)
    ):
        out = _forward(lam3, v3, x03)
    else:
        out = _SelectiveTriton.apply(lam3, v3, x03)
    return out.reshape(lam.shape)
