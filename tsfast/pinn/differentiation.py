"""Finite-difference operators for PINN physics losses."""

__all__ = [
    "diff1_forward",
    "diff1_forward_double",
    "diff1_central",
    "diff1_central4_double",
    "diff2_forward",
    "diff2_central",
    "diff2_central_double",
    "diff3_forward",
    "diff3_central",
]

import torch


def diff1_forward(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    First-order forward difference.
    f'(x) ≈ (f(x+h) - f(x)) / h
    Accuracy: O(dt)
    """
    interior = (signal[:, 1:] - signal[:, :-1]) / dt
    last = interior[:, -1:]
    return torch.cat(
        [interior, last],
        dim=1,
    )


def diff1_forward_double(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    First-order forward difference (float64).
    f'(x) ≈ (f(x+h) - f(x)) / h
    Accuracy: O(dt)
    """
    signal_double = signal.type(torch.float64)
    interior = (signal_double[:, 1:] - signal_double[:, :-1]) / dt
    last = interior[:, -1:]
    return torch.cat(
        [interior, last],
        dim=1,
    ).type(signal.dtype)


def diff1_central(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    First-order central difference.
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    Accuracy: O(dt²)
    """
    dt2 = 2.0 * dt
    interior = (signal[:, 2:] - signal[:, :-2]) / dt2
    first = (signal[:, 1:2] - signal[:, 0:1]) / dt
    last = (signal[:, -1:] - signal[:, -2:-1]) / dt
    return torch.cat([first, interior, last], dim=1)


def diff1_central4_double(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    4th-order central difference:
    f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    Accuracy: O(dt⁴)
    """
    signal_double = signal.type(torch.float64)

    # 4th order central for interior
    interior = (
        -signal_double[:, 4:] + 8 * signal_double[:, 3:-1] - 8 * signal_double[:, 1:-3] + signal_double[:, :-4]
    ) / (12 * dt)

    # Handle boundaries with lower-order approximations
    # Second-order central for points 1 and -2
    point1 = (signal_double[:, 2:3] - signal_double[:, 0:1]) / (2 * dt)
    point_m2 = (signal_double[:, -1:] - signal_double[:, -3:-2]) / (2 * dt)

    # Forward/backward difference for edges
    first = (signal_double[:, 1:2] - signal_double[:, 0:1]) / dt
    last = (signal_double[:, -1:] - signal_double[:, -2:-1]) / dt

    return torch.cat([first, point1, interior, point_m2, last], dim=1).type(signal.dtype)


def diff2_forward(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Second-order forward difference.
    f''(x) ≈ (f(x+2h) - 2f(x+h) + f(x)) / h²
    Accuracy: O(dt)
    """
    dt_sq = dt * dt
    interior = (signal[:, 2:] - 2.0 * signal[:, 1:-1] + signal[:, :-2]) / dt_sq
    first = (signal[:, 2:3] - 2.0 * signal[:, 1:2] + signal[:, 0:1]) / dt_sq
    last = (signal[:, -1:] - 2.0 * signal[:, -2:-1] + signal[:, -3:-2]) / dt_sq
    return torch.cat([first, interior, last], dim=1)


def diff2_central(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Second-order central difference.
    f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    Accuracy: O(dt²)
    """
    dt_sq = dt * dt
    interior = (signal[:, 2:] - 2.0 * signal[:, 1:-1] + signal[:, :-2]) / dt_sq
    first = (signal[:, 2:3] - 2.0 * signal[:, 1:2] + signal[:, 0:1]) / dt_sq
    last = (signal[:, -1:] - 2.0 * signal[:, -2:-1] + signal[:, -3:-2]) / dt_sq
    return torch.cat([first, interior, last], dim=1)


def diff2_central_double(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Second-order central difference (float64).
    f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    Accuracy: O(dt²)
    """
    dt_sq = dt * dt
    signal_double = signal.type(torch.float64)
    interior = (signal_double[:, 2:] - 2.0 * signal_double[:, 1:-1] + signal_double[:, :-2]) / dt_sq
    first = (signal_double[:, 2:3] - 2.0 * signal_double[:, 1:2] + signal_double[:, 0:1]) / dt_sq
    last = (signal_double[:, -1:] - 2.0 * signal_double[:, -2:-1] + signal_double[:, -3:-2]) / dt_sq
    return torch.cat(
        [first, interior, last],
        dim=1,
    ).type(signal.dtype)


def diff3_forward(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Third-order forward difference.
    f'''(x) ≈ (f(x+3h) - 3f(x+2h) + 3f(x+h) - f(x)) / h³
    Accuracy: O(dt)
    """
    dt_cb = dt * dt * dt
    interior = (signal[:, 3:] - 3.0 * signal[:, 2:-1] + 3.0 * signal[:, 1:-2] - signal[:, :-3]) / dt_cb
    first = (signal[:, 3:4] - 3.0 * signal[:, 2:3] + 3.0 * signal[:, 1:2] - signal[:, 0:1]) / dt_cb
    second = first
    last = (signal[:, -1:] - 3.0 * signal[:, -2:-1] + 3.0 * signal[:, -3:-2] - signal[:, -4:-3]) / dt_cb
    return torch.cat([first, second, interior, last], dim=1)


def diff3_central(signal: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Third-order central difference.
    f'''(x) ≈ (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h³)
    Accuracy: O(dt²)
    """
    dt_cb = 2.0 * dt * dt * dt
    interior = (signal[:, 4:] - 2.0 * signal[:, 3:-1] + 2.0 * signal[:, 1:-3] - signal[:, :-4]) / dt_cb

    # Boundary using forward formula
    dt_cb_fwd = dt * dt * dt
    d1 = (signal[:, 3:4] - 3.0 * signal[:, 2:3] + 3.0 * signal[:, 1:2] - signal[:, 0:1]) / dt_cb_fwd
    d2 = (signal[:, 4:5] - 2.0 * signal[:, 3:4] + 2.0 * signal[:, 1:2] - signal[:, 0:1]) / dt_cb
    d_n1 = (signal[:, -1:] - 2.0 * signal[:, -2:-1] + 2.0 * signal[:, -4:-3] - signal[:, -5:-4]) / dt_cb
    d_n = (signal[:, -1:] - 3.0 * signal[:, -2:-1] + 3.0 * signal[:, -3:-2] - signal[:, -4:-3]) / dt_cb_fwd

    return torch.cat([d1, d2, interior, d_n1, d_n], dim=1)
