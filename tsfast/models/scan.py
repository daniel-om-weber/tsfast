"""Parallel scans for diagonal linear recurrences, the compute core of LRU/S5/Mamba-style layers.

Both public functions run a log-doubling (Hillis-Steele) scan inside a custom
``autograd.Function`` with the analytic adjoint: the gradient of a linear recurrence
``x_t = a_t x_{t-1} + v_t`` is itself a reverse-time linear recurrence
``G_t = g_t + conj(a_{t+1}) G_{t+1}`` with ``dL/dv_t = G_t``, ``dL/da_t = G_t conj(x_{t-1})``
and ``dL/dx_0 = conj(a_1) G_1``. Only the coefficients and the forward output are saved,
so the backward memory is O(L) instead of the O(L log L) intermediates plain autograd
would retain across the doubling levels — the difference between fitting and OOM for
long sequences at large batch sizes.
"""

__all__ = [
    "diagonal_recurrence",
    "selective_recurrence",
]

import torch
from torch.autograd.function import once_differentiable


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


class _DiagonalRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, v, x0):
        x = v.broadcast_to(torch.broadcast_shapes(lam.unsqueeze(-2).shape, v.shape)).clone()
        if x0 is not None:
            x[..., 0, :] += lam * x0
        _scan_diagonal_(x, lam)
        ctx.save_for_backward(lam, x, x0)
        ctx.v_shape = v.shape
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        lam, x, x0 = ctx.saved_tensors
        lam_c = lam.conj()
        # G_t = g_t + conj(lam) G_{t+1}: the same constant-coefficient scan, time-reversed.
        G = _scan_diagonal_(g.flip(-2).clone(), lam_c).flip(-2)
        grad_lam = grad_v = grad_x0 = None
        if ctx.needs_input_grad[0]:
            grad_lam = (G * _x_prev(x, x0).conj()).sum_to_size(lam.unsqueeze(-2).shape).squeeze(-2)
        if ctx.needs_input_grad[1]:
            grad_v = G.sum_to_size(ctx.v_shape)
        if x0 is not None and ctx.needs_input_grad[2]:
            grad_x0 = (lam_c * G[..., 0, :]).sum_to_size(x0.shape)
        return grad_lam, grad_v, grad_x0


class _SelectiveRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, v, x0):
        shape = torch.broadcast_shapes(lam.shape, v.shape)
        x = v.broadcast_to(shape).clone()
        a = lam.broadcast_to(shape).clone()
        if x0 is not None:
            x[..., 0, :] += a[..., 0, :] * x0
        _scan_selective_(x, a)
        ctx.save_for_backward(lam, x, x0)
        ctx.v_shape = v.shape
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        lam, x, x0 = ctx.saved_tensors
        shape = x.shape
        # G_t = g_t + conj(a_{t+1}) G_{t+1}: time-reverse, then the flipped coefficient at
        # step s is conj(a) flipped and shifted right by one (the first slot multiplies the
        # zero initial state of the reverse scan, so its value never matters).
        a_f = lam.conj().broadcast_to(shape).flip(-2)
        c = torch.cat((torch.zeros_like(a_f[..., :1, :]), a_f[..., :-1, :]), dim=-2)
        G = _scan_selective_(g.flip(-2).clone(), c).flip(-2)
        grad_lam = grad_v = grad_x0 = None
        if ctx.needs_input_grad[0]:
            grad_lam = (G * _x_prev(x, x0).conj()).sum_to_size(lam.shape)
        if ctx.needs_input_grad[1]:
            grad_v = G.sum_to_size(ctx.v_shape)
        if x0 is not None and ctx.needs_input_grad[2]:
            grad_x0 = (lam.broadcast_to(shape)[..., 0, :].conj() * G[..., 0, :]).sum_to_size(x0.shape)
        return grad_lam, grad_v, grad_x0


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
    return _DiagonalRecurrence.apply(lam, v, x0)


def selective_recurrence(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x_t = lam_t * x_{t-1} + v_t`` with time-varying diagonal ``lam_t`` via a parallel scan.

    Hillis-Steele scan over the affine maps ``(lam_t, v_t)`` with the composition rule
    ``(a2, b2) . (a1, b1) = (a1*a2, a2*b1 + b2)``: each doubling step composes every prefix
    with the prefix ``s`` steps earlier, so the sequential depth is ``ceil(log2(L))`` at
    ``O(L log L)`` elementwise work. Real or complex. Gradients come from the analytic
    adjoint (a reverse-time scan), so backward memory is O(L). This is the recurrence form
    of Mamba-style selective state-space layers.

    Args:
        lam: diagonal coefficients per step ``[..., L, n]``.
        v: input sequence ``[..., L, n]``.
        x0: initial state ``[..., n]``; zeros if None.

    Returns:
        States ``x_1 .. x_L`` as ``[..., L, n]``.
    """
    return _SelectiveRecurrence.apply(lam, v, x0)


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
