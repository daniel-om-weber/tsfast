"""Parallel scans for diagonal linear recurrences, the compute core of LRU/S5/Mamba-style layers."""

__all__ = [
    "diagonal_recurrence",
    "selective_recurrence",
]

import torch


def diagonal_recurrence(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x_t = lam * x_{t-1} + v_t`` with constant diagonal ``lam`` via a log-doubling scan.

    The constant-coefficient specialization of ``selective_recurrence``: because ``lam`` does
    not vary along the sequence, each doubling step extends the summation window with one
    elementwise multiply by the carried power ``lam**s``, so the sequential depth is
    ``ceil(log2(L))``. Exact for any spectral radius, real or complex, and differentiable by
    plain autograd on any device.

    Args:
        lam: diagonal coefficients ``[..., n]``, broadcast against the leading dims of ``v``.
        v: input sequence ``[..., L, n]``.
        x0: initial state ``[..., n]``; zeros if None.

    Returns:
        States ``x_1 .. x_L`` as ``[..., L, n]``.
    """
    L = v.shape[-2]
    if x0 is not None:
        v = torch.cat((v[..., :1, :] + (lam * x0).unsqueeze(-2), v[..., 1:, :]), dim=-2)
    x, lam_p, s = v, lam.unsqueeze(-2), 1
    while s < L:
        x_sh = torch.cat((torch.zeros_like(x[..., :s, :]), x[..., :-s, :]), dim=-2)
        x = x + lam_p * x_sh
        lam_p = lam_p * lam_p
        s *= 2
    return x


def selective_recurrence(lam: torch.Tensor, v: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x_t = lam_t * x_{t-1} + v_t`` with time-varying diagonal ``lam_t`` via a parallel scan.

    Hillis-Steele scan over the affine maps ``(lam_t, v_t)`` with the composition rule
    ``(a2, b2) . (a1, b1) = (a1*a2, a2*b1 + b2)``: each doubling step composes every prefix
    with the prefix ``s`` steps earlier, so the sequential depth is ``ceil(log2(L))`` at
    ``O(L log L)`` elementwise work. Real or complex, differentiable by plain autograd on any
    device. This is the recurrence form of Mamba-style selective state-space layers.

    Args:
        lam: diagonal coefficients per step ``[..., L, n]``.
        v: input sequence ``[..., L, n]``.
        x0: initial state ``[..., n]``; zeros if None.

    Returns:
        States ``x_1 .. x_L`` as ``[..., L, n]``.
    """
    L = v.shape[-2]
    if x0 is not None:
        v = torch.cat((v[..., :1, :] + lam[..., :1, :] * x0.unsqueeze(-2), v[..., 1:, :]), dim=-2)
    a, x, s = lam, v, 1
    while s < L:
        a_sh = torch.cat((torch.ones_like(a[..., :s, :]), a[..., :-s, :]), dim=-2)
        x_sh = torch.cat((torch.zeros_like(x[..., :s, :]), x[..., :-s, :]), dim=-2)
        x = x + a * x_sh
        a = a * a_sh
        s *= 2
    return x


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
