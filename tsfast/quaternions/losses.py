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

from ..training.losses import loss_fn
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


# --- Loss functions ---


@loss_fn
def inclination_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element inclination error from difference quaternion."""
    q = diffQuat(q1, q2)
    return (q[..., 3] ** 2 + q[..., 0] ** 2).sqrt() - 1


inclination_loss = inclination_error.reduce("rms")
inclination_loss_abs = inclination_error.reduce(lambda x: x.abs().mean())
inclination_loss_squared = inclination_error.reduce(lambda x: x.pow(2).mean())
inclination_loss_smooth = inclination_error.reduce(_smooth_l1)


@loss_fn
def angle_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element absolute angle error from difference quaternion w component."""
    q = diffQuat(q1, q2)
    return (q[..., 0] - 1).abs()


@loss_fn
def angle_loss_opt(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element absolute angle error (optimized, no full quaternion multiply)."""
    q1 = norm_quaternion(q1)
    q2 = norm_quaternion(q2)
    q2 = conjQuat(q2)
    q = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    return (q - 1).abs()


# --- Metrics ---


@loss_fn
def inclination_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element inclination angle."""
    return inclinationAngle(q1, q2)


abs_inclination = inclination_angle.reduce(lambda x: x.abs().mean())
ms_inclination = inclination_angle.reduce(lambda x: x.pow(2).mean())
rms_inclination = inclination_angle.reduce("rms")
smooth_inclination = inclination_angle.reduce(_smooth_l1)
rms_inclination_deg = inclination_angle.reduce(lambda x: rad2deg(x).mean().sqrt())
mean_inclination_deg = inclination_angle.reduce(lambda x: rad2deg(x).mean())


@loss_fn
def pitch_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element pitch angle."""
    return pitchAngle(q1, q2)


rms_pitch_deg = pitch_angle.reduce(_deg_rms)


@loss_fn
def roll_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element roll angle."""
    return rollAngle(q1, q2)


rms_roll_deg = roll_angle.reduce(_deg_rms)


@loss_fn
def rel_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Per-element relative angle."""
    return relativeAngle(q1, q2)


ms_rel_angle = rel_angle.reduce(lambda x: x.pow(2).mean())
abs_rel_angle = rel_angle.reduce(lambda x: x.abs().mean())
rms_rel_angle_deg = rel_angle.reduce(_deg_rms)
mean_rel_angle_deg = rel_angle.reduce(lambda x: rad2deg(x.mean()))


def deg_rmse(inp: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """RMSE metric converted to degrees."""
    from ..training import fun_rmse

    return rad2deg(fun_rmse(inp, targ))
