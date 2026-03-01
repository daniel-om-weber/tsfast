"""Quaternion loss functions and evaluation metrics."""

__all__ = [
    "inclination_error",
    "inclination_loss",
    "inclination_loss_abs",
    "inclination_loss_squared",
    "inclination_loss_smooth",
    "angle_loss",
    "angle_loss_opt",
    "inclination_angle",
    "abs_inclination",
    "ms_inclination",
    "rms_inclination",
    "smooth_inclination",
    "rms_inclination_deg",
    "pitch_angle",
    "rms_pitch_deg",
    "roll_angle",
    "rms_roll_deg",
    "mean_inclination_deg",
    "rel_angle",
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


# --- Reduction helpers ---


def _smooth_l1(x: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(x, torch.zeros_like(x))


def _deg_rms(x: torch.Tensor) -> torch.Tensor:
    return rad2deg(x.pow(2).mean().sqrt())


# --- Per-element loss functions ---


def inclination_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element inclination error from difference quaternion."""
    q = diffQuat(q1, q2)
    return (q[..., 3] ** 2 + q[..., 0] ** 2).sqrt() - 1


def angle_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element absolute angle error from difference quaternion w component."""
    q = diffQuat(q1, q2)
    return (q[..., 0] - 1).abs()


def angle_loss_opt(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element absolute angle error (optimized, no full quaternion multiply)."""
    q1 = norm_quaternion(q1)
    q2 = norm_quaternion(q2)
    q2 = conjQuat(q2)
    q = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    return (q - 1).abs()


def inclination_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element inclination angle."""
    return inclinationAngle(q1, q2)


def pitch_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element pitch angle."""
    return pitchAngle(q1, q2)


def roll_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element roll angle."""
    return rollAngle(q1, q2)


def rel_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element relative angle."""
    return relativeAngle(q1, q2)


# --- Reduced losses (from inclination_error) ---


def inclination_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """RMS inclination error."""
    return inclination_error(q1, q2).pow(2).mean().sqrt()


def inclination_loss_abs(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute inclination error."""
    return inclination_error(q1, q2).abs().mean()


def inclination_loss_squared(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean squared inclination error."""
    return inclination_error(q1, q2).pow(2).mean()


def inclination_loss_smooth(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Smooth L1 inclination error."""
    return _smooth_l1(inclination_error(q1, q2))


# --- Reduced metrics (from inclination_angle) ---


def abs_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute inclination angle."""
    return inclination_angle(q1, q2).abs().mean()


def ms_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean squared inclination angle."""
    return inclination_angle(q1, q2).pow(2).mean()


def rms_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """RMS inclination angle."""
    return inclination_angle(q1, q2).pow(2).mean().sqrt()


def smooth_inclination(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Smooth L1 inclination angle."""
    return _smooth_l1(inclination_angle(q1, q2))


def rms_inclination_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """RMS inclination angle in degrees."""
    return rad2deg(inclination_angle(q1, q2)).pow(2).mean().sqrt()


def mean_inclination_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean inclination angle in degrees."""
    return rad2deg(inclination_angle(q1, q2)).mean()


# --- Reduced metrics (from pitch_angle / roll_angle) ---


def rms_pitch_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """RMS pitch angle in degrees."""
    return _deg_rms(pitch_angle(q1, q2))


def rms_roll_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """RMS roll angle in degrees."""
    return _deg_rms(roll_angle(q1, q2))


# --- Reduced metrics (from rel_angle) ---


def ms_rel_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean squared relative angle."""
    return rel_angle(q1, q2).pow(2).mean()


def abs_rel_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean absolute relative angle."""
    return rel_angle(q1, q2).abs().mean()


def rms_rel_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """RMS relative angle in degrees."""
    return _deg_rms(rel_angle(q1, q2))


def mean_rel_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Mean relative angle in degrees."""
    return rad2deg(rel_angle(q1, q2).mean())


def deg_rmse(inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """RMSE metric converted to degrees."""
    from ..training import fun_rmse

    return rad2deg(fun_rmse(inp, targ))
