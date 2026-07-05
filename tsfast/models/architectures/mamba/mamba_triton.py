"""Fused time-tiled Triton backend for the Mamba selective SSM section.

The generic ``selective_recurrence`` path materializes ``lam = exp(delta A)``,
``v = (delta u) B_t`` and the states ``h`` as ``[B, L, d_inner*d_state]`` tensors, so the
scan section moves ~20x the global-memory traffic of the reference CUDA kernel
(state-spaces/mamba). This backend fuses the whole section — softplus discretization,
``exp(delta A)``, input injection, recurrence, output contraction ``y_t = h_t . C_t``,
skip ``D u`` and the SiLU gate — into one kernel over the *unexpanded* inputs. The
``[B, L, D, N]`` states live only in registers: the forward writes the gated output and,
in training, one state checkpoint every ``_CHK`` steps (~3% of a full state store); under
``torch.no_grad`` nothing DN-sized touches memory at all.

Each program owns a ``(batch, channel-block)`` slice and walks time in tiles:
tile-sized loads amortize DRAM latency (a scalar step-by-step walk is latency-bound at
~one memory round-trip per step) and ``tl.associative_scan`` runs the recurrence across
the tile in registers. The backward walks the checkpoints in reverse, recomputes the
chunk's states with the forward tile scan, and runs the adjoint
``G_t = gy_t C_t + lam_{t+1} G_{t+1}`` as a ``reverse=True`` tile scan in the same pass.
``lam_t h_{t-1}`` is obtained as ``h_t - v_t``, so previous states are never needed.
``grad_Bt``/``grad_Ct`` are reduced over the d_inner blocks (and ``grad_A``/``grad_D``
over the batch) via partial buffers summed in PyTorch, keeping the kernel free of
atomics (deterministic). Real float32 only, matching the reference kernel's regime.
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

# _CHK is the state-checkpoint interval and the backward's tile length; fixed (not
# autotuned) because the checkpoint buffer is allocated before launch. Larger values
# raise register pressure in the backward faster than they save checkpoint traffic.
_CHK = 8


def _block_d(B: int, D: int) -> int:
    """Channels per program: the largest tile that still oversubscribes the SMs.

    Chosen at launch (not autotuned) so the backward's partial gradient buffers can be
    allocated to the exact number of channel blocks; forward and backward must agree.
    """
    for bd in (16, 8):
        if B * -(-D // bd) >= 192:
            return bd
    return 4


def _fwd_configs():
    return [triton.Config({"BLOCK_T": bt}, num_warps=w) for bt in (16, 32, 64) for w in (2, 4, 8)]


def _bwd_configs():
    return [triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)]


if _HAVE_TRITON:

    @triton.jit
    def _affine_combine(a1, b1, a2, b2):
        # composition of x -> a x + b maps: second map applied after the first
        return a1 * a2, a2 * b1 + b2

    @triton.jit
    def _softplus(x):
        # matches F.softplus's overflow threshold of 20
        return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))

    # L is intentionally NOT an autotune key: the launch grid is B*cdiv(D, BLOCK_D)
    # (L is only the inner tile-loop trip count), so the optimal (BLOCK_T, num_warps)
    # is L-invariant. Keying on L would re-search on every horizon-schedule length.
    @triton.autotune(configs=_fwd_configs(), key=["D", "N", "STORE_CHK", "BLOCK_D"])
    @triton.jit
    def _mamba_fwd(
        draw_ptr,  # [B, L, D] dt_proj output, pre-softplus
        A_ptr,  # [D, N]
        Bt_ptr,  # [B, L, N]
        Ct_ptr,  # [B, L, N]
        u_ptr,  # [B, L, D]
        z_ptr,  # [B, L, D] gate input
        Dp_ptr,  # [D] skip coefficients
        h0_ptr,  # [B, D, N] or unused
        out_ptr,  # [B, L, D] gated output
        chk_ptr,  # [B, NC, D, N] state checkpoints or unused
        hlast_ptr,  # [B, D, N]
        B,
        L,
        D,
        N,
        NC,
        HAS_H0: tl.constexpr,
        STORE_CHK: tl.constexpr,
        CHK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        d_chunks = tl.cdiv(D, BLOCK_D)
        b = pid // d_chunks
        offs_d = (pid - b * d_chunks) * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        mask_d = offs_d < D
        mask_n = offs_n < N
        mask_dn = mask_d[:, None] & mask_n[None, :]
        dn_off = offs_d[:, None] * N + offs_n[None, :]

        A = tl.load(A_ptr + dn_off, mask=mask_dn, other=0.0)
        Dp = tl.load(Dp_ptr + offs_d, mask=mask_d, other=0.0)
        if HAS_H0:
            h = tl.load(h0_ptr + b * D * N + dn_off, mask=mask_dn, other=0.0)
        else:
            h = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)

        rows = tl.arange(0, BLOCK_T)
        for c in tl.range(0, tl.cdiv(L, BLOCK_T)):
            offs_t = c * BLOCK_T + rows
            mask_t = offs_t < L
            ptr_td = b * L * D + offs_t[:, None] * D + offs_d[None, :]
            m_td = mask_t[:, None] & mask_d[None, :]
            ptr_tn = b * L * N + offs_t[:, None] * N + offs_n[None, :]
            m_tn = mask_t[:, None] & mask_n[None, :]
            draw_c = tl.load(draw_ptr + ptr_td, mask=m_td, other=0.0)
            u_c = tl.load(u_ptr + ptr_td, mask=m_td, other=0.0)
            z_c = tl.load(z_ptr + ptr_td, mask=m_td, other=0.0)
            B_c = tl.load(Bt_ptr + ptr_tn, mask=m_tn, other=0.0)
            C_c = tl.load(Ct_ptr + ptr_tn, mask=m_tn, other=0.0)

            # masked rows compose as exactly the identity map (delta=0 -> lam=1, v=0), so
            # the carry extracted from the last row is h_{L-1} even in a partial final tile
            delta_c = tl.where(m_td, _softplus(draw_c), 0.0)
            lam_c = tl.exp(delta_c[:, :, None] * A[None, :, :])
            v_c = (delta_c * u_c)[:, :, None] * B_c[:, None, :]
            acum, s = tl.associative_scan((lam_c, v_c), 0, _affine_combine)
            h_c = s + acum * h[None, :, :]
            y_c = tl.sum(h_c * C_c[:, None, :], axis=2) + Dp[None, :] * u_c
            out_c = y_c * z_c * tl.sigmoid(z_c)
            tl.store(out_ptr + ptr_td, out_c, mask=m_td)
            if STORE_CHK:
                # h at t = k*CHK - 1 lands in checkpoint slot k-1, read back by chunk k
                m_chk = mask_t & ((offs_t + 1) % CHK == 0)
                slot = (offs_t + 1) // CHK - 1
                offs_chk = (b * NC + slot)[:, None, None] * D * N + dn_off[None, :, :]
                tl.store(chk_ptr + offs_chk, h_c, mask=m_chk[:, None, None] & mask_dn[None, :, :])
            h = tl.sum(tl.where(rows[:, None, None] == BLOCK_T - 1, h_c, 0.0), axis=0)
        tl.store(hlast_ptr + b * D * N + dn_off, h, mask=mask_dn)

    @triton.autotune(configs=_bwd_configs(), key=["D", "N", "BLOCK_D"])  # L-invariant, see fwd
    @triton.jit
    def _mamba_bwd(
        draw_ptr,  # [B, L, D]
        A_ptr,  # [D, N]
        Bt_ptr,  # [B, L, N]
        Ct_ptr,  # [B, L, N]
        u_ptr,  # [B, L, D]
        z_ptr,  # [B, L, D]
        Dp_ptr,  # [D]
        h0_ptr,  # [B, D, N] or unused
        chk_ptr,  # [B, NC, D, N] checkpoints from the forward
        gout_ptr,  # [B, L, D] upstream gradient of the gated output
        ghlast_ptr,  # [B, D, N] or unused
        gdraw_ptr,  # [B, L, D]
        gA_ptr,  # [B, D, N] partials, summed over B outside
        gBt_ptr,  # [GD, B, L, N] partials, summed over GD outside
        gCt_ptr,  # [GD, B, L, N] partials, summed over GD outside
        gu_ptr,  # [B, L, D]
        gz_ptr,  # [B, L, D]
        gDp_ptr,  # [B, D] partials, summed over B outside
        gh0_ptr,  # [B, D, N] or unused
        B,
        L,
        D,
        N,
        NC,
        HAS_H0: tl.constexpr,
        HAS_GLAST: tl.constexpr,
        NEED_GH0: tl.constexpr,
        CHK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        d_chunks = tl.cdiv(D, BLOCK_D)
        b = pid // d_chunks
        pid_d = pid - b * d_chunks
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        mask_d = offs_d < D
        mask_n = offs_n < N
        mask_dn = mask_d[:, None] & mask_n[None, :]
        dn_off = offs_d[:, None] * N + offs_n[None, :]

        A = tl.load(A_ptr + dn_off, mask=mask_dn, other=0.0)
        Dp = tl.load(Dp_ptr + offs_d, mask=mask_d, other=0.0)
        if HAS_GLAST:
            G = tl.load(ghlast_ptr + b * D * N + dn_off, mask=mask_dn, other=0.0)
        else:
            G = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
        if HAS_H0:
            h0v = tl.load(h0_ptr + b * D * N + dn_off, mask=mask_dn, other=0.0)
        else:
            h0v = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
        gA_acc = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
        gDp_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        rows = tl.arange(0, CHK)
        for ci in tl.range(0, tl.cdiv(L, CHK)):
            # chunks run backwards through time (the adjoint carry G flows that way);
            # rows inside a chunk run forwards, so the states recompute with the same
            # tile scan as the forward while G uses a reverse=True scan.
            c = tl.cdiv(L, CHK) - 1 - ci
            offs_t = c * CHK + rows
            mask_t = offs_t < L
            ptr_td = b * L * D + offs_t[:, None] * D + offs_d[None, :]
            m_td = mask_t[:, None] & mask_d[None, :]
            ptr_tn = b * L * N + offs_t[:, None] * N + offs_n[None, :]
            m_tn = mask_t[:, None] & mask_n[None, :]
            draw_c = tl.load(draw_ptr + ptr_td, mask=m_td, other=0.0)
            # adjoint coefficient is lam at t+1; the masked position t = L-1 becomes
            # delta=0, i.e. lam=1, which forwards ghlast into G_{L-1} unchanged
            m_td1 = (mask_t & (offs_t + 1 < L))[:, None] & mask_d[None, :]
            draw_n = tl.load(draw_ptr + ptr_td + D, mask=m_td1, other=0.0)
            u_c = tl.load(u_ptr + ptr_td, mask=m_td, other=0.0)
            z_c = tl.load(z_ptr + ptr_td, mask=m_td, other=0.0)
            gout_c = tl.load(gout_ptr + ptr_td, mask=m_td, other=0.0)
            B_c = tl.load(Bt_ptr + ptr_tn, mask=m_tn, other=0.0)
            C_c = tl.load(Ct_ptr + ptr_tn, mask=m_tn, other=0.0)
            if HAS_H0:
                hs = tl.load(chk_ptr + (b * NC + c - 1) * D * N + dn_off, mask=mask_dn & (c > 0), other=0.0)
                h_start = tl.where(c > 0, hs, h0v)
            else:
                h_start = tl.load(chk_ptr + (b * NC + c - 1) * D * N + dn_off, mask=mask_dn & (c > 0), other=0.0)

            delta_c = tl.where(m_td, _softplus(draw_c), 0.0)
            a_c = tl.exp(tl.where(m_td1, _softplus(draw_n), 0.0)[:, :, None] * A[None, :, :])
            lam_c = tl.exp(delta_c[:, :, None] * A[None, :, :])
            v_c = (delta_c * u_c)[:, :, None] * B_c[:, None, :]
            acum_h, s_h = tl.associative_scan((lam_c, v_c), 0, _affine_combine)
            h_c = s_h + acum_h * h_start[None, :, :]

            # gate: out = y * silu(z) with y = h . C + Dp u
            y_c = tl.sum(h_c * C_c[:, None, :], axis=2) + Dp[None, :] * u_c
            sig = tl.sigmoid(z_c)
            gy_c = gout_c * z_c * sig
            gz_c = gout_c * y_c * sig * (1.0 + z_c * (1.0 - sig))
            gDp_acc += tl.sum(gy_c * u_c, axis=0)

            q_c = gy_c[:, :, None] * C_c[:, None, :]
            acum_g, s_g = tl.associative_scan((a_c, q_c), 0, _affine_combine, reverse=True)
            G_c = s_g + acum_g * G[None, :, :]

            lam_hprev = h_c - v_c  # = lam_t h_{t-1}, so previous states are never needed
            gC_c = tl.sum(gy_c[:, :, None] * h_c, axis=1)
            gdelta_c = tl.sum(G_c * (A[None, :, :] * lam_hprev + u_c[:, :, None] * B_c[:, None, :]), axis=2)
            gdraw_c = gdelta_c * tl.sigmoid(draw_c)
            gu_c = delta_c * tl.sum(G_c * B_c[:, None, :], axis=2) + gy_c * Dp[None, :]
            gB_c = tl.sum(G_c * (delta_c * u_c)[:, :, None], axis=1)
            gA_acc += tl.sum(G_c * lam_hprev * delta_c[:, :, None], axis=0)

            tl.store(gdraw_ptr + ptr_td, gdraw_c, mask=m_td)
            tl.store(gu_ptr + ptr_td, gu_c, mask=m_td)
            tl.store(gz_ptr + ptr_td, gz_c, mask=m_td)
            row_gp = (pid_d * B + b) * L * N + offs_t[:, None] * N + offs_n[None, :]
            tl.store(gBt_ptr + row_gp, gB_c, mask=m_tn)
            tl.store(gCt_ptr + row_gp, gC_c, mask=m_tn)

            G = tl.sum(tl.where(rows[:, None, None] == 0, G_c, 0.0), axis=0)
        tl.store(gA_ptr + b * D * N + dn_off, gA_acc, mask=mask_dn)
        tl.store(gDp_ptr + b * D + offs_d, gDp_acc, mask=mask_d)
        if NEED_GH0:
            draw_0 = tl.load(draw_ptr + b * L * D + offs_d, mask=mask_d, other=-30.0)
            lam_0 = tl.exp(_softplus(draw_0)[:, None] * A)
            tl.store(gh0_ptr + b * D * N + dn_off, lam_0 * G, mask=mask_dn)


def _grid(B, D):
    return (B * triton.cdiv(D, _block_d(B, D)),)


def _forward(draw, A, Bt, Ct, u, z, Dp, h0, store_chk: bool):
    B, L, D = draw.shape
    N = A.shape[-1]
    NC = triton.cdiv(L, _CHK)
    out = torch.empty_like(draw)
    h_last = torch.empty(B, D, N, device=draw.device, dtype=draw.dtype)
    chk = torch.empty(B, NC, D, N, device=draw.device, dtype=draw.dtype) if store_chk else out
    has_h0 = h0 is not None
    h0_arg = h0 if has_h0 else out  # unused pointer when HAS_H0 is False
    _mamba_fwd[_grid(B, D)](
        draw,
        A,
        Bt,
        Ct,
        u,
        z,
        Dp,
        h0_arg,
        out,
        chk,
        h_last,
        B,
        L,
        D,
        N,
        NC,
        HAS_H0=has_h0,
        STORE_CHK=store_chk,
        CHK=_CHK,
        BLOCK_D=_block_d(B, D),
        BLOCK_N=triton.next_power_of_2(N),
    )
    return out, h_last, (chk if store_chk else None)


class _FusedMambaScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, draw, A, Bt, Ct, u, z, Dp, h0):
        out, h_last, chk = _forward(draw, A, Bt, Ct, u, z, Dp, h0, store_chk=True)
        ctx.save_for_backward(draw, A, Bt, Ct, u, z, Dp, h0, chk)
        return out, h_last

    @staticmethod
    def backward(ctx, gout, ghlast):
        draw, A, Bt, Ct, u, z, Dp, h0, chk = ctx.saved_tensors
        B, L, D = draw.shape
        N = A.shape[-1]
        NC = triton.cdiv(L, _CHK)
        gout = gout.contiguous()
        has_h0 = h0 is not None
        need_gh0 = bool(ctx.needs_input_grad[7] and has_h0)
        # a zero upstream state-gradient arrives as None from autograd
        has_glast = ghlast is not None
        ghlast_arg = ghlast.contiguous() if has_glast else gout
        GD = triton.cdiv(D, _block_d(B, D))
        gdraw = torch.empty_like(draw)
        gu = torch.empty_like(draw)
        gz = torch.empty_like(draw)
        gA_part = torch.empty(B, D, N, device=draw.device, dtype=draw.dtype)
        gBt_part = torch.empty(GD, B, L, N, device=draw.device, dtype=draw.dtype)
        gCt_part = torch.empty(GD, B, L, N, device=draw.device, dtype=draw.dtype)
        gDp_part = torch.empty(B, D, device=draw.device, dtype=draw.dtype)
        gh0 = torch.empty(B, D, N, device=draw.device, dtype=draw.dtype) if need_gh0 else gout
        h0_arg = h0 if has_h0 else gout
        _mamba_bwd[_grid(B, D)](
            draw,
            A,
            Bt,
            Ct,
            u,
            z,
            Dp,
            h0_arg,
            chk,
            gout,
            ghlast_arg,
            gdraw,
            gA_part,
            gBt_part,
            gCt_part,
            gu,
            gz,
            gDp_part,
            gh0,
            B,
            L,
            D,
            N,
            NC,
            HAS_H0=has_h0,
            HAS_GLAST=has_glast,
            NEED_GH0=need_gh0,
            CHK=_CHK,
            BLOCK_D=_block_d(B, D),
            BLOCK_N=triton.next_power_of_2(N),
        )
        return (
            gdraw if ctx.needs_input_grad[0] else None,
            gA_part.sum(dim=0) if ctx.needs_input_grad[1] else None,
            gBt_part.sum(dim=0) if ctx.needs_input_grad[2] else None,
            gCt_part.sum(dim=0) if ctx.needs_input_grad[3] else None,
            gu if ctx.needs_input_grad[4] else None,
            gz if ctx.needs_input_grad[5] else None,
            gDp_part.sum(dim=0) if ctx.needs_input_grad[6] else None,
            gh0 if need_gh0 else None,
        )


def supports(draw, A, Bt, Ct, u, z, Dp, h0) -> str | None:
    """Reason the fused Triton kernel cannot handle these tensors, or None if it can."""
    if not _HAVE_TRITON:
        return "triton not importable"
    if not torch.cuda.is_available():
        return "CUDA not available"
    tensors = {"draw": draw, "A": A, "Bt": Bt, "Ct": Ct, "u": u, "z": z, "Dp": Dp}
    if h0 is not None:
        tensors["h0"] = h0
    for name, tt in tensors.items():
        if tt.device.type != "cuda":
            return f"{name} not on CUDA"
        if tt.dtype != torch.float32:
            return f"needs float32, got {name}={tt.dtype}"
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

    Returns ``(out [B, L, D], h_last [B, D, N])``; differentiable in all inputs.
    """
    draw = draw.contiguous()
    A = A.contiguous()
    Bt = Bt.contiguous()
    Ct = Ct.contiguous()
    u = u.contiguous()
    z = z.contiguous()
    Dp = Dp.contiguous()
    h0 = h0.contiguous() if h0 is not None else None
    inputs = (draw, A, Bt, Ct, u, z, Dp) + ((h0,) if h0 is not None else ())
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in inputs):
        out, h_last, _ = _forward(draw, A, Bt, Ct, u, z, Dp, h0, store_chk=False)
        return out, h_last
    return _FusedMambaScan.apply(draw, A, Bt, Ct, u, z, Dp, h0)
