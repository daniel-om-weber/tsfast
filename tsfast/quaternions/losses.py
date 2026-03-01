"""Quaternion loss functions and evaluation metrics."""

__all__ = [
    "inclination_loss",
    "inclination_loss_abs",
    "inclination_loss_squared",
    "inclination_loss_smooth",
    "angle_loss",
    "angle_loss_opt",
    "abs_inclination",
    "ms_inclination",
    "rms_inclination",
    "smooth_inclination",
    "rms_inclination_deg",
    "rms_pitch_deg",
    "rms_roll_deg",
    "mean_inclination_deg",
    "ms_rel_angle",
    "abs_rel_angle",
    "rms_rel_angle_deg",
    "mean_rel_angle_deg",
    "deg_rmse",
]

import torch
import torch.nn.functional as F

from .ops import (
    conjQuat,
    diffQuat,
    inclinationAngle,
    norm_quaternion,
    pitchAngle,
    rad2deg,
    relativeAngle,
    rollAngle,
)


# --- Loss functions ---


def inclination_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Root mean squared inclination loss."""
    q = diffQuat(q1, q2)
    q_abs = (q[..., 3] ** 2 + q[..., 0] ** 2).sqrt() - 1
    return (q_abs**2).mean().sqrt()


def inclination_loss_abs(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute inclination loss."""
    q = diffQuat(q1, q2)
    q_abs = (q[..., 3] ** 2 + q[..., 0] ** 2).sqrt() - 1
    return q_abs.abs().mean()


def inclination_loss_squared(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean squared inclination loss."""
    q = diffQuat(q1, q2)
    q_abs = (q[..., 3] ** 2 + q[..., 0] ** 2).sqrt() - 1
    return (q_abs**2).mean()


def inclination_loss_smooth(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Smooth L1 inclination loss."""
    q = diffQuat(q1, q2)
    q_abs = (q[..., 3] ** 2 + q[..., 0] ** 2).sqrt() - 1
    return F.smooth_l1_loss(q_abs, torch.zeros_like(q_abs))


def angle_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute angle loss based on the w component of the difference quaternion."""
    q = diffQuat(q1, q2)
    return (q[..., 0] - 1).abs().mean()


def angle_loss_opt(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Optimized angle loss avoiding full quaternion multiplication."""
    q1 = norm_quaternion(q1)
    q2 = norm_quaternion(q2)

    q2 = conjQuat(q2)
    q = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    return (q - 1).abs().mean()


# --- Metrics ---


def abs_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute inclination angle metric."""
    inclination = inclinationAngle(q1, q2)
    return inclination.abs().mean()


def ms_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean squared inclination angle metric."""
    inclination = inclinationAngle(q1, q2)
    return (inclination**2).mean()


def rms_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Root mean squared inclination angle metric."""
    inclination = inclinationAngle(q1, q2)
    return (inclination**2).mean().sqrt()


def smooth_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Smooth L1 inclination angle metric."""
    inclination = inclinationAngle(q1, q2)
    return F.smooth_l1_loss(inclination, torch.zeros_like(inclination))


def rms_inclination_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Root mean squared inclination angle in degrees."""
    inclination = inclinationAngle(q1, q2)
    return rad2deg((inclination**2).mean().sqrt())


def rms_pitch_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Root mean squared pitch angle in degrees."""
    inclination = pitchAngle(q1, q2)
    return rad2deg((inclination**2).mean().sqrt())


def rms_roll_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Root mean squared roll angle in degrees."""
    inclination = rollAngle(q1, q2)
    return rad2deg((inclination**2).mean().sqrt())


def mean_inclination_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean inclination angle in degrees."""
    inclination = inclinationAngle(q1, q2)
    return rad2deg(inclination.mean())


def ms_rel_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean squared relative angle metric."""
    rel_angle = relativeAngle(q1, q2)
    return (rel_angle**2).mean()


def abs_rel_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute relative angle metric."""
    rel_angle = relativeAngle(q1, q2)
    return rel_angle.abs().mean()


def rms_rel_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Root mean squared relative angle in degrees."""
    rel_angle = relativeAngle(q1, q2)
    return rad2deg((rel_angle**2).mean().sqrt())


def mean_rel_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean relative angle in degrees."""
    rel_angle = relativeAngle(q1, q2)
    return rad2deg(rel_angle.mean())


def deg_rmse(inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """RMSE metric converted to degrees."""
    from ..training import fun_rmse

    return rad2deg(fun_rmse(inp, targ))
