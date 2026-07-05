"""Fused causal depthwise convolution + SiLU Triton backend for the Mamba conv section.

The eager path around ``MambaLayer``'s convolution costs four extra passes over the
``[B, L, d_inner]`` activation: the ``cat`` gluing the carried tail onto the sequence,
a contiguity copy of the strided ``in_proj`` chunk, cuDNN's conv1d, and a separate SiLU
(each again in the backward). The convolved signal itself must stay materialized —
``x_proj`` consumes it as a GEMM input — so this backend fuses everything *around* it
into one kernel: it reads the strided chunk view directly, folds the carried tail in as
the causal left-context (no ``cat``), and writes the SiLU-activated result once, already
in the channel-last layout the downstream GEMM and scan kernel want.

There is no recurrence here, so programs tile ``(batch, time, channels)`` freely; each
tap is a shifted tile load that hits L2 after the first pass. The backward recomputes
the pre-activation (cheaper than saving it), accumulates the input gradient from the
``K`` shifted output gradients, and reduces the weight/bias gradients over ``(batch,
time-block)`` via partial buffers summed in PyTorch — no atomics (deterministic). The
tiny gradient to the carried tail (``K - 1`` leading steps) is handled in PyTorch by the
autograd wrapper, not here. Real float32 only, matching the scan kernel's regime.
"""

__all__ = [
    "supports",
    "run",
]

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    _HAVE_TRITON = False

# The backward's time-tile length is fixed (not autotuned) because the weight/bias
# partial buffers are sized by the number of time blocks before launch.
_BWD_BLOCK_T = 64


def _fwd_configs():
    return [
        triton.Config({"BLOCK_T": bt, "BLOCK_D": bd}, num_warps=w)
        for bt in (64, 256)
        for bd in (32, 64)
        for w in (2, 4)
    ]


def _bwd_configs():
    return [triton.Config({"BLOCK_D": bd}, num_warps=w) for bd in (32, 64) for w in (2, 4, 8)]


if _HAVE_TRITON:

    @triton.jit
    def _load_xbuf(x_ptr, tail_ptr, b, rows, offs_d, m_d, L, D, sxb, sxl, sxd, K: tl.constexpr):
        # the virtual buffer cat(tail, x): rows < 0 come from the carried tail
        m_x = (rows >= 0) & (rows < L)
        xv = tl.load(
            x_ptr + b * sxb + rows[:, None] * sxl + offs_d[None, :] * sxd,
            mask=m_x[:, None] & m_d[None, :],
            other=0.0,
        )
        m_t = rows < 0
        tv = tl.load(
            tail_ptr + (b * D + offs_d[None, :]) * (K - 1) + (rows[:, None] + K - 1),
            mask=m_t[:, None] & m_d[None, :],
            other=0.0,
        )
        return xv + tv

    @triton.autotune(configs=_fwd_configs(), key=["L", "D", "K"])
    @triton.jit
    def _conv_fwd(
        x_ptr,  # [B, L, D] strided (chunk view of in_proj)
        tail_ptr,  # [B, D, K-1] carried causal left-context
        w_ptr,  # [D, K] depthwise taps
        b_ptr,  # [D] or unused
        out_ptr,  # [B, L, D] contiguous, SiLU applied
        B,
        L,
        D,
        sxb,
        sxl,
        sxd,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t_chunks = tl.cdiv(L, BLOCK_T)
        d_chunks = tl.cdiv(D, BLOCK_D)
        b = pid // (t_chunks * d_chunks)
        tb = pid % (t_chunks * d_chunks) // d_chunks
        db = pid % d_chunks
        offs_t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
        m_d = offs_d < D

        if HAS_BIAS:
            p = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32) + tl.load(b_ptr + offs_d, mask=m_d, other=0.0)[None, :]
        else:
            p = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        for k in tl.static_range(K):
            w_k = tl.load(w_ptr + offs_d * K + k, mask=m_d, other=0.0)
            p += w_k[None, :] * _load_xbuf(
                x_ptr, tail_ptr, b, offs_t - (K - 1) + k, offs_d, m_d, L, D, sxb, sxl, sxd, K
            )
        out = p * tl.sigmoid(p)
        m_td = (offs_t < L)[:, None] & m_d[None, :]
        tl.store(out_ptr + b * L * D + offs_t[:, None] * D + offs_d[None, :], out, mask=m_td)

    @triton.autotune(configs=_bwd_configs(), key=["L", "D", "K"])
    @triton.jit
    def _conv_bwd(
        x_ptr,  # [B, L, D] strided
        tail_ptr,  # [B, D, K-1]
        w_ptr,  # [D, K]
        b_ptr,  # [D] or unused
        gy_ptr,  # [B, L, D] upstream gradient of the activated output
        gx_ptr,  # [B, L, D] contiguous
        gw_ptr,  # [NBT, D, K] partials, summed over NBT outside
        gb_ptr,  # [NBT, D] partials or unused
        B,
        L,
        D,
        sxb,
        sxl,
        sxd,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        NEED_GB: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t_chunks = tl.cdiv(L, BLOCK_T)
        d_chunks = tl.cdiv(D, BLOCK_D)
        b = pid // (t_chunks * d_chunks)
        tb = pid % (t_chunks * d_chunks) // d_chunks
        db = pid % d_chunks
        offs_t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
        m_d = offs_d < D

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_d, mask=m_d, other=0.0)
        else:
            bias = tl.zeros([BLOCK_D], dtype=tl.float32)

        # gx_t = sum_k w_k gpre_{t+K-1-k}, with gpre_r = gy_r silu'(p_r); the j-th pass
        # rebuilds p at rows offs_t + j (shifted loads, L2-resident after the first
        # pass). Rows beyond L carry gy = 0, so their gpre vanishes regardless of p.
        gx = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        for j in tl.static_range(K):
            rows = offs_t + j
            p = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32) + bias[None, :]
            for k in tl.static_range(K):
                w_k = tl.load(w_ptr + offs_d * K + k, mask=m_d, other=0.0)
                p += w_k[None, :] * _load_xbuf(
                    x_ptr, tail_ptr, b, rows - (K - 1) + k, offs_d, m_d, L, D, sxb, sxl, sxd, K
                )
            m_r = (rows < L)[:, None] & m_d[None, :]
            gy_j = tl.load(gy_ptr + b * L * D + rows[:, None] * D + offs_d[None, :], mask=m_r, other=0.0)
            sig = tl.sigmoid(p)
            gpre = gy_j * sig * (1.0 + p * (1.0 - sig))
            w_j = tl.load(w_ptr + offs_d * K + (K - 1 - j), mask=m_d, other=0.0)
            gx += w_j[None, :] * gpre
            if j == 0:
                # weight/bias gradients only need gpre at the unshifted rows
                for k in tl.static_range(K):
                    xk = _load_xbuf(x_ptr, tail_ptr, b, offs_t - (K - 1) + k, offs_d, m_d, L, D, sxb, sxl, sxd, K)
                    tl.store(
                        gw_ptr + ((b * t_chunks + tb) * D + offs_d) * K + k,
                        tl.sum(gpre * xk, axis=0),
                        mask=m_d,
                    )
                if NEED_GB:
                    tl.store(gb_ptr + (b * t_chunks + tb) * D + offs_d, tl.sum(gpre, axis=0), mask=m_d)
        m_td = (offs_t < L)[:, None] & m_d[None, :]
        tl.store(gx_ptr + b * L * D + offs_t[:, None] * D + offs_d[None, :], gx, mask=m_td)


def _fwd_grid(B, L, D):
    return lambda meta: (B * triton.cdiv(L, meta["BLOCK_T"]) * triton.cdiv(D, meta["BLOCK_D"]),)


def _forward(x, tail, weight, bias):
    B, L, D = x.shape
    K = weight.shape[-1]
    out = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
    _conv_fwd[_fwd_grid(B, L, D)](
        x,
        tail,
        weight,
        bias if bias is not None else weight,  # unused pointer when HAS_BIAS is False
        out,
        B,
        L,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        K=K,
        HAS_BIAS=bias is not None,
    )
    return out


class _FusedConvSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tail, weight, bias):
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, tail, weight, *((bias,) if bias is not None else ()))
        return _forward(x, tail, weight, bias)

    @staticmethod
    def backward(ctx, gy):
        x, tail, weight, *rest = ctx.saved_tensors
        bias = rest[0] if ctx.has_bias else None
        B, L, D = x.shape
        K = weight.shape[-1]
        gy = gy.contiguous()
        need_gb = bool(ctx.has_bias and ctx.needs_input_grad[3])
        NBT = B * triton.cdiv(L, _BWD_BLOCK_T)
        gx = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
        gw_part = torch.empty(NBT, D, K, device=x.device, dtype=x.dtype)
        gb_part = torch.empty(NBT, D, device=x.device, dtype=x.dtype) if need_gb else gx
        _conv_bwd[_fwd_grid(B, L, D)](
            x,
            tail,
            weight,
            bias if ctx.has_bias else weight,
            gy,
            gx,
            gw_part,
            gb_part,
            B,
            L,
            D,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            K=K,
            HAS_BIAS=ctx.has_bias,
            NEED_GB=need_gb,
            BLOCK_T=_BWD_BLOCK_T,
        )
        g_tail = None
        if ctx.needs_input_grad[1]:
            # only the first K-1 outputs see the tail; recompute their pre-activation
            # and correlate with the taps in eager PyTorch (a handful of tiny ops)
            Lh = min(K - 1, L)
            xbuf_head = torch.cat((tail, x[:, :Lh].mT), dim=-1)
            p_head = F.conv1d(xbuf_head, weight.unsqueeze(1), bias, groups=D)
            sig = torch.sigmoid(p_head)
            gpre_head = gy[:, :Lh].mT * sig * (1.0 + p_head * (1.0 - sig))
            g_tail = torch.zeros_like(tail)
            for j in range(K - 1):
                for t in range(min(j, Lh - 1) + 1):
                    g_tail[:, :, j] += gpre_head[:, :, t] * weight[:, j - t]
        return (
            gx if ctx.needs_input_grad[0] else None,
            g_tail,
            gw_part.sum(dim=0) if ctx.needs_input_grad[2] else None,
            gb_part.sum(dim=0) if need_gb else None,
        )


def supports(x, tail, weight, bias) -> str | None:
    """Reason the fused conv kernel cannot handle these tensors, or None if it can."""
    if not _HAVE_TRITON:
        return "triton not importable"
    if not torch.cuda.is_available():
        return "CUDA not available"
    tensors = {"x": x, "tail": tail, "weight": weight}
    if bias is not None:
        tensors["bias"] = bias
    for name, tt in tensors.items():
        if tt.device.type != "cuda":
            return f"{name} not on CUDA"
        if tt.dtype != torch.float32:
            return f"needs float32, got {name}={tt.dtype}"
    if x.dim() != 3:
        return f"x must be [B, L, D], got {x.dim()} dims"
    B, L, D = x.shape
    if weight.dim() != 3 or weight.shape[0] != D or weight.shape[1] != 1:
        return f"weight.shape {tuple(weight.shape)} is not depthwise [D, 1, K] for D={D}"
    K = weight.shape[-1]
    if not 2 <= K <= 8:
        return f"kernel width {K} outside the supported range [2, 8]"
    if tuple(tail.shape) != (B, D, K - 1):
        return f"tail.shape {tuple(tail.shape)} != {(B, D, K - 1)}"
    if bias is not None and tuple(bias.shape) != (D,):
        return f"bias.shape {tuple(bias.shape)} != {(D,)}"
    return None


def run(x, tail, weight, bias):
    """Fused causal depthwise conv + SiLU: ``silu(conv1d(cat(tail, x), weight, bias))``.

    ``x`` is channel-last ``[B, L, D]`` (any strides); ``tail`` is the carried causal
    left-context ``[B, D, K-1]``. Returns the activated signal ``[B, L, D]``,
    contiguous; differentiable in all inputs.
    """
    tail = tail.contiguous()
    weight = weight.contiguous().squeeze(1)
    bias = bias.contiguous() if bias is not None else None
    inputs = (x, tail, weight) + ((bias,) if bias is not None else ())
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in inputs):
        return _forward(x, tail, weight, bias)
    return _FusedConvSilu.apply(x, tail, weight, bias)
