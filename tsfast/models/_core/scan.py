"""Parallel scans for diagonal linear recurrences, the compute core of LRU/S5/Mamba-style layers.

Both public functions are registered ``torch.library`` custom ops (``tsfast::scan_diagonal``,
``tsfast::scan_selective``) with analytic-adjoint backward ops, so they compose with
``torch.compile`` (no graph breaks), fake/meta tracing, and export. Inside the op, dispatch
picks the fastest usable kernel backend for the input's device from ``_BACKENDS`` —
honoring the process preference set via ``tsfast.models.set_backend``/``use_backend`` —
and falls back to the pure-PyTorch log-doubling (Hillis-Steele) scan below.

The gradient of a linear recurrence ``x_t = a_t x_{t-1} + v_t`` is itself a reverse-time
linear recurrence ``G_t = g_t + conj(a_{t+1}) G_{t+1}`` with ``dL/dv_t = G_t``,
``dL/da_t = G_t conj(x_{t-1})`` and ``dL/dx_0 = conj(a_1) G_1``. Only the coefficients and
the forward output are saved, so the backward memory is O(L) instead of the O(L log L)
intermediates plain autograd would retain across the doubling levels — the difference
between fitting and OOM for long sequences at large batch sizes.
"""

__all__ = [
    "diagonal_recurrence",
    "selective_recurrence",
    "complex_in_proj",
    "real_out_proj",
]

import torch

from .dispatch import resolve

# Kernel backends per op, resolved through dispatch.resolve: each module exposes
# supports(lam, v, x0) -> str | None and forward/backward entry points on the
# flattened lane layout the ops below establish.
_BACKENDS = {
    "diagonal": {
        "triton": "tsfast.models._core.scan_backends.diagonal_triton",
        "c": "tsfast.models._core.scan_backends.diagonal_c",
    },
    "selective": {
        "triton": "tsfast.models._core.scan_backends.selective_triton",
        "c": "tsfast.models._core.scan_backends.selective_c",
    },
}
_AUTO_ORDER = {"cuda": ("triton",), "cpu": ("c",)}


def _resolve(op: str, lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None):
    return resolve(f"scan.{op}", _BACKENDS[op], _AUTO_ORDER.get(v.device.type, ()), (lam, v, x0))


# --------------------------------------------------------------- reference implementation


def _scan_diagonal_(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """In-place log-doubling scan ``x_t += lam^s * x_{t-s}`` for constant diagonal ``lam``.

    ``x`` must own its memory (no autograd tracking); each level's update materializes
    its right-hand side before the in-place add, so the overlapping views are safe.
    """
    L, s, lam_p = x.shape[-2], 1, lam.unsqueeze(-2)
    while s < L:
        x[..., s:, :] += lam_p * x[..., :-s, :]
        lam_p = lam_p * lam_p
        s *= 2
    return x


def _scan_selective_(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """In-place log-doubling scan for time-varying diagonal coefficients ``a``.

    Composes the affine maps ``(a_t, x_t)`` with the rule
    ``(a2, b2) . (a1, b1) = (a1*a2, a2*b1 + b2)``; both buffers are mutated. Each level
    materializes its updates from the pre-level values before writing.
    """
    L, s = x.shape[-2], 1
    while s < L:
        upd_x = a[..., s:, :] * x[..., :-s, :]
        upd_a = a[..., s:, :] * a[..., :-s, :]
        x[..., s:, :] += upd_x
        a[..., s:, :] = upd_a
        s *= 2
    return x


def _x_prev(x: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    """States ``x_0 .. x_{L-1}`` aligned with steps ``1 .. L`` (zeros for a cold start)."""
    first = torch.zeros_like(x[..., :1, :]) if x0 is None else x0.unsqueeze(-2).expand_as(x[..., :1, :])
    return torch.cat((first, x[..., :-1, :]), dim=-2)


# ------------------------------------------------------------------------- custom ops
#
# Both recurrences are exposed as forward/backward custom-op pairs operating on a
# flattened lane layout (all leading batch dims collapsed, everything contiguous,
# broadcasting already materialized by the public wrappers): diagonal takes
# lam [M, N] constant over time with v [M, L, N]; selective takes lam [B, L, N]
# matching v. Keeping the broadcast/reshape outside the ops means plain autograd
# reduces the lane gradients back to the callers' (possibly broadcast) shapes.
# The backward is its own registered op so compiled autograd also sees no graph break.


@torch.library.custom_op("tsfast::scan_diagonal", mutates_args=())
def _scan_diagonal_op(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    lam, v = lam.contiguous(), v.contiguous()
    x0 = x0.contiguous() if x0 is not None else None
    mod = _resolve("diagonal", lam, v, x0)
    if mod is not None:
        return mod.forward(lam, v, x0)
    x = v.clone()
    if x0 is not None:
        x[..., 0, :] += lam * x0
    return _scan_diagonal_(x, lam)


@_scan_diagonal_op.register_fake
def _(lam, v, x0):
    return torch.empty_like(v)


@torch.library.custom_op("tsfast::scan_diagonal_bwd", mutates_args=())
def _scan_diagonal_bwd_op(
    g: torch.Tensor, lam: torch.Tensor, out: torch.Tensor, x0: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # lam/x0 come from setup_context and may still be broadcast views; the kernels
    # index raw data pointers, so materialize everything (no-ops when already dense).
    g, lam, out = g.contiguous(), lam.contiguous(), out.contiguous()
    x0 = x0.contiguous() if x0 is not None else None
    mod = _resolve("diagonal", lam, g, x0)
    if mod is not None:
        return mod.backward(g, lam, out, x0)
    lam_c = lam.conj()
    # G_t = g_t + conj(lam) G_{t+1}: the same constant-coefficient scan, time-reversed.
    G = _scan_diagonal_(g.flip(-2).clone(), lam_c).flip(-2)
    glam = (G * _x_prev(out, x0).conj()).sum(-2)
    gx0 = lam_c * G[..., 0, :] if x0 is not None else lam.new_empty(0)
    return glam, G, gx0


@_scan_diagonal_bwd_op.register_fake
def _(g, lam, out, x0):
    gx0 = torch.empty_like(lam) if x0 is not None else lam.new_empty(0)
    return torch.empty_like(lam), torch.empty_like(g), gx0


def _diag_setup(ctx, inputs, output):
    lam, v, x0 = inputs
    ctx.save_for_backward(lam, output, x0)


def _diag_backward(ctx, g):
    lam, out, x0 = ctx.saved_tensors
    glam, gv, gx0 = _scan_diagonal_bwd_op(g, lam, out, x0)
    return glam, gv, (gx0 if x0 is not None else None)


_scan_diagonal_op.register_autograd(_diag_backward, setup_context=_diag_setup)


@torch.library.custom_op("tsfast::scan_selective", mutates_args=())
def _scan_selective_op(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None) -> torch.Tensor:
    lam, v = lam.contiguous(), v.contiguous()
    x0 = x0.contiguous() if x0 is not None else None
    mod = _resolve("selective", lam, v, x0)
    if mod is not None:
        return mod.forward(lam, v, x0)
    x = v.clone()
    a = lam.clone()
    if x0 is not None:
        x[..., 0, :] += a[..., 0, :] * x0
    return _scan_selective_(x, a)


@_scan_selective_op.register_fake
def _(lam, v, x0):
    return torch.empty_like(v)


@torch.library.custom_op("tsfast::scan_selective_bwd", mutates_args=())
def _scan_selective_bwd_op(
    g: torch.Tensor, lam: torch.Tensor, out: torch.Tensor, x0: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # lam/x0 come from setup_context and may still be broadcast views; the kernels
    # index raw data pointers, so materialize everything (no-ops when already dense).
    g, lam, out = g.contiguous(), lam.contiguous(), out.contiguous()
    x0 = x0.contiguous() if x0 is not None else None
    mod = _resolve("selective", lam, g, x0)
    if mod is not None:
        return mod.backward(g, lam, out, x0)
    # G_t = g_t + conj(a_{t+1}) G_{t+1}: time-reverse, then the flipped coefficient at
    # step s is conj(a) flipped and shifted right by one (the first slot multiplies the
    # zero initial state of the reverse scan, so its value never matters).
    a_f = lam.conj().flip(-2)
    c = torch.cat((torch.zeros_like(a_f[..., :1, :]), a_f[..., :-1, :]), dim=-2)
    G = _scan_selective_(g.flip(-2).clone(), c).flip(-2)
    glam = G * _x_prev(out, x0).conj()
    gx0 = lam[..., 0, :].conj() * G[..., 0, :] if x0 is not None else lam.new_empty(0)
    return glam, G, gx0


@_scan_selective_bwd_op.register_fake
def _(g, lam, out, x0):
    gx0 = torch.empty_like(lam[..., 0, :]) if x0 is not None else lam.new_empty(0)
    return torch.empty_like(lam), torch.empty_like(g), gx0


def _sel_setup(ctx, inputs, output):
    lam, v, x0 = inputs
    ctx.save_for_backward(lam, output, x0)


def _sel_backward(ctx, g):
    lam, out, x0 = ctx.saved_tensors
    glam, gv, gx0 = _scan_selective_bwd_op(g, lam, out, x0)
    return glam, gv, (gx0 if x0 is not None else None)


_scan_selective_op.register_autograd(_sel_backward, setup_context=_sel_setup)


# ------------------------------------------------------------------------- public API


def diagonal_recurrence(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x_t = lam * x_{t-1} + v_t`` with constant diagonal ``lam`` via a log-doubling scan.

    The constant-coefficient specialization of ``selective_recurrence``: because ``lam`` does
    not vary along the sequence, each doubling step extends the summation window with one
    elementwise multiply by the carried power ``lam**s``, so the sequential depth is
    ``ceil(log2(L))``. Exact for any spectral radius, real or complex. Gradients come from
    the analytic adjoint (a reverse-time scan), so backward memory is O(L).

    Args:
        lam: diagonal coefficients ``[..., n]``, broadcast against the leading dims of ``v``.
        v: input sequence ``[..., L, n]``.
        x0: initial state ``[..., n]``; zeros if None.

    Returns:
        States ``x_1 .. x_L`` as ``[..., L, n]``.
    """
    out_shape = torch.broadcast_shapes(lam.unsqueeze(-2).shape, v.shape)
    bdims, L, n = out_shape[:-2], out_shape[-2], out_shape[-1]
    lam_lane = lam.broadcast_to(bdims + (n,)).reshape(-1, n)
    v_flat = v.broadcast_to(out_shape).reshape(-1, L, n)
    x0_lane = None if x0 is None else x0.broadcast_to(bdims + (n,)).reshape(-1, n)
    return _scan_diagonal_op(lam_lane, v_flat, x0_lane).reshape(out_shape)


def selective_recurrence(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x_t = lam_t * x_{t-1} + v_t`` with time-varying diagonal ``lam_t`` via a parallel scan.

    Hillis-Steele scan over the affine maps ``(lam_t, v_t)`` with the composition rule
    ``(a2, b2) . (a1, b1) = (a1*a2, a2*b1 + b2)``: each doubling step composes every prefix
    with the prefix ``s`` steps earlier, so the sequential depth is ``ceil(log2(L))`` at
    ``O(L log L)`` elementwise work. Real or complex. Gradients come from the analytic
    adjoint (a reverse-time scan), so backward memory is O(L). This is the recurrence form
    of Mamba-style selective state-space layers.

    Args:
        lam: diagonal coefficients per step ``[..., L, n]``, broadcast against ``v``.
        v: input sequence ``[..., L, n]``.
        x0: initial state ``[..., n]``; zeros if None.

    Returns:
        States ``x_1 .. x_L`` as ``[..., L, n]``.
    """
    out_shape = torch.broadcast_shapes(lam.shape, v.shape)
    bdims, L, n = out_shape[:-2], out_shape[-2], out_shape[-1]
    lam3 = lam.broadcast_to(out_shape).reshape(-1, L, n)
    v3 = v.broadcast_to(out_shape).reshape(-1, L, n)
    x03 = None if x0 is None else x0.broadcast_to(bdims + (n,)).reshape(-1, n)
    return _scan_selective_op(lam3, v3, x03).reshape(out_shape)


def complex_in_proj(u: torch.Tensor, W_re: torch.Tensor, W_im: torch.Tensor) -> torch.Tensor:
    """Project a real sequence into complex state space, ``u @ (W_re + i W_im).mT``, as one real GEMM.

    Interleaving the real and imaginary rows of ``W`` into a real ``[d, 2n]`` weight makes the
    GEMM output contiguous in exactly ``torch.view_as_real``'s layout, so the complex result is
    a free view. Compared to ``torch.complex(u @ W_re.mT, u @ W_im.mT)`` this saves a GEMM call
    and the sequence-sized complex packing kernel (plus its unpacking in backward) — for the
    diagonal-recurrence layers these boundary projections otherwise rival the scan itself in
    memory traffic.

    Args:
        u: real input sequence ``[..., d]``.
        W_re: real part of the projection ``[n, d]``.
        W_im: imaginary part of the projection ``[n, d]``.

    Returns:
        Complex sequence ``[..., n]``.
    """
    n, d = W_re.shape
    W = torch.stack((W_re, W_im), dim=-1).permute(1, 0, 2).reshape(d, 2 * n)
    return torch.view_as_complex((u @ W).unflatten(-1, (n, 2)))


def real_out_proj(x: torch.Tensor, W_re: torch.Tensor, W_im: torch.Tensor) -> torch.Tensor:
    """Read a real sequence out of complex state space, ``Re(x @ (W_re + i W_im).mT)``, as one real GEMM.

    The counterpart of ``complex_in_proj``: ``view_as_real`` of the (contiguous) scan output is
    a free view whose last axis folds into an interleaved real ``[out, 2n]`` weight. Compared to
    ``x.real @ W_re.mT - x.imag @ W_im.mT`` this saves a GEMM call, the strided ``.real`` /
    ``.imag`` extraction copies, the subtraction, and the complex gradient repacking in backward.

    Args:
        x: complex sequence ``[..., n]``.
        W_re: real part of the readout ``[out, n]``.
        W_im: imaginary part of the readout ``[out, n]``.

    Returns:
        Real sequence ``[..., out]``.
    """
    out, n = W_re.shape
    W = torch.stack((W_re, -W_im), dim=-1).reshape(out, 2 * n)
    return torch.view_as_real(x).flatten(-2) @ W.mT


def _diagonal_recurrence_sequential(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Reference implementation of both recurrences as a per-timestep Python loop.

    ``lam`` may be constant ``[..., n]`` or time-varying ``[..., L, n]``.
    """
    time_varying = lam.dim() == v.dim()
    x = torch.zeros_like(v[..., 0, :]) if x0 is None else x0
    outs = []
    for t in range(v.shape[-2]):
        lam_t = lam[..., t, :] if time_varying else lam
        x = lam_t * x + v[..., t, :]
        outs.append(x)
    return torch.stack(outs, dim=-2)
