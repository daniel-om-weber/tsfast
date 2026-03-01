"""Core quaternion math, angle computations, and generation utilities."""

__all__ = [
    "rad2deg",
    "multiplyQuat",
    "norm_quaternion",
    "conjQuat",
    "diffQuat",
    "relativeQuat",
    "safe_acos",
    "safe_acos_double",
    "inclinationAngle",
    "relativeAngle",
    "rollAngle",
    "pitchAngle",
    "inclinationAngleAbs",
    "rand_quat",
    "rot_vec",
    "quatFromAngleAxis",
    "quatInterp",
]

import warnings

import numpy as np
import torch
import torch.nn.functional as F


_pi = torch.Tensor([3.14159265358979323846])


def rad2deg(t: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees."""
    return 180.0 * t / _pi.to(t.device).type(t.dtype)


def multiplyQuat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions element-wise."""
    o1 = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    o2 = q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2]
    o3 = q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1]
    o4 = q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]
    return torch.stack([o1, o2, o3, o4], dim=-1)


def norm_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit norm."""
    return q / q.norm(p=2, dim=-1)[..., None]


_conjugate_quaternion = torch.tensor([1, -1, -1, -1])


def conjQuat(q: torch.Tensor) -> torch.Tensor:
    """Compute the conjugate of a quaternion."""
    return q * _conjugate_quaternion.to(q.device).type(q.dtype)


def diffQuat(q1: torch.Tensor, q2: torch.Tensor, norm: bool = True) -> torch.Tensor:
    """Compute the difference quaternion between q1 and q2.

    Args:
        q1: first quaternion tensor.
        q2: second quaternion tensor.
        norm: whether to normalize inputs before computing the difference.
    """
    if norm:
        nq1 = norm_quaternion(q1)
        nq2 = norm_quaternion(q2)
    else:
        nq1 = q1
        nq2 = q2

    return relativeQuat(nq1, nq2)


def relativeQuat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the relative quaternion as quat1*inv(quat2)."""

    o1 = q1[..., 0] * q2[..., 0] + q1[..., 1] * q2[..., 1] + q1[..., 2] * q2[..., 2] + q1[..., 3] * q2[..., 3]
    o2 = -q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] - q1[..., 2] * q2[..., 3] + q1[..., 3] * q2[..., 2]
    o3 = -q1[..., 0] * q2[..., 2] + q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] - q1[..., 3] * q2[..., 1]
    o4 = -q1[..., 0] * q2[..., 3] - q1[..., 1] * q2[..., 2] + q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]

    return torch.stack([o1, o2, o3, o4], dim=-1)


def safe_acos(t: torch.Tensor, eps: float = 4e-8) -> torch.Tensor:
    """Numerically stable variant of arccosine."""
    return t.clamp(-1.0 + eps, 1.0 - eps).acos()


def safe_acos_double(t: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """Numerically stable arccosine using float64 internally for accuracy."""
    try:
        return t.type(torch.float64).clamp(-1.0 + eps, 1.0 - eps).acos().type(t.dtype)
    except TypeError as e:
        warnings.warn(
            f"Float64 precision not supported on {t.device} device. Falling back to float32. This may reduce numerical accuracy of quaternion operations. Error: {e}"
        )
        return t.clamp(-1.0 + 1e-6, 1.0 - 1e-6).acos()


# --- Angle computations ---


def inclinationAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Inclination (tilt) angle between two quaternions.

    Uses ``atan2`` instead of ``acos`` for numerical stability.
    """
    q = diffQuat(q1, q2)
    return 2 * torch.atan2(
        (q[..., 1] ** 2 + q[..., 2] ** 2).sqrt(),
        (q[..., 0] ** 2 + q[..., 3] ** 2).sqrt(),
    )


def relativeAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Full rotation angle between two quaternions.

    Uses ``atan2`` instead of ``acos`` for numerical stability.
    """
    q = diffQuat(q1, q2)
    return 2 * torch.atan2(q[..., 1:].norm(dim=-1), q[..., 0].abs())


def rollAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the roll angle between two quaternions."""
    q = diffQuat(q1, q2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    return torch.atan2(t0, t1)


def pitchAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Euler pitch angle of the difference quaternion.

    Uses ``atan2(sin, cos)`` instead of ``asin`` for numerical stability
    near gimbal lock (``+/-pi/2``).
    """
    q = diffQuat(q1, q2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sin_p = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    cos_p = (1.0 - sin_p**2).sqrt()
    return torch.atan2(sin_p, cos_p)


_unit_quaternion = torch.tensor([1.0, 0, 0, 0])


def inclinationAngleAbs(q: torch.Tensor) -> torch.Tensor:
    """Absolute inclination angle relative to the identity quaternion.

    Uses ``atan2`` instead of ``acos`` for numerical stability.
    """
    q = diffQuat(q, _unit_quaternion[None, :].to(q.device))
    return 2 * torch.atan2(
        (q[..., 1] ** 2 + q[..., 2] ** 2).sqrt(),
        (q[..., 0] ** 2 + q[..., 3] ** 2).sqrt(),
    )


# --- Generation / manipulation ---


def rand_quat() -> torch.Tensor:
    """Generate a random unit quaternion."""
    q = torch.rand((4)) * 2 - 1
    q /= q.norm()
    return q


def rot_vec(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate a 3D vector by a quaternion."""
    v = F.pad(v, (1, 0), "constant", 0)
    return multiplyQuat(conjQuat(q), multiplyQuat(v, q))[..., 1:]


def quatFromAngleAxis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Create quaternions from angle-axis representation.

    Args:
        angle: rotation angles, shape (N,) or (1,).
        axis: rotation axes, shape (3,) or (N, 3) or (1, 3).
    """
    if len(axis.shape) == 2:
        N = max(angle.shape[0], axis.shape[0])
        assert angle.shape in ((1,), (N,))
        assert axis.shape == (N, 3) or axis.shape == (1, 3)
    else:
        N = angle.shape[0]
        assert angle.shape == (N,)
        assert axis.shape == (3,)
        axis = axis[None, :]

    axis = axis / torch.norm(axis, dim=1)[:, None]
    quat = torch.cat([torch.cos(angle / 2)[:, None], axis * torch.sin(angle / 2)[:, None]], dim=-1)
    return quat


def quatInterp(quat: torch.Tensor, ind: torch.Tensor, extend: bool = False) -> torch.Tensor:
    """Interpolate quaternions at non-integer indices using Slerp.

    Sampling indices are in the range 0..N-1. For values outside this range,
    depending on ``extend``, the first/last element or NaN is returned.

    Args:
        quat: input quaternions, shape (N(xB)x4).
        ind: sampling indices, shape (M,).
        extend: if true, extend input by repeating first/last value.

    Returns:
        Interpolated quaternions, shape (Mx4).
    """
    N = quat.shape[0]
    M = ind.shape[0]
    assert quat.shape[-1] == 4
    assert ind.shape == (M,)

    ind = ind.to(quat.device)
    ind0 = torch.clamp(torch.floor(ind).type(torch.long), 0, N - 1)
    ind1 = torch.clamp(torch.ceil(ind).type(torch.long), 0, N - 1)

    q0 = quat[ind0].type(torch.float64)
    q1 = quat[ind1].type(torch.float64)
    q_1_0 = diffQuat(q0, q1)

    # normalize the quaternion for positive w component to ensure
    # that the angle will be [0, 180deg]
    invert_sign_ind = q_1_0[..., 0] < 0
    q_1_0[invert_sign_ind] = -q_1_0[invert_sign_ind]

    angle = 2 * torch.atan2(q_1_0[..., 1:].norm(dim=-1), q_1_0[..., 0])
    axis = q_1_0[..., 1:]

    # copy over (almost) direct hits
    direct_ind = angle < 1e-06
    quat_out = torch.empty_like(q0)
    quat_out[direct_ind] = q0[direct_ind]

    interp_ind = ~direct_ind
    t01 = ind - ind0
    if len(quat.shape) == 3:
        t01 = t01[:, None]  # extend shape if batches are part of the tensor
    q_t_0 = quatFromAngleAxis((t01 * angle)[interp_ind], axis[interp_ind])
    quat_out[interp_ind] = multiplyQuat(q0[interp_ind], q_t_0)

    if not extend:
        quat_out[ind < 0] = np.nan
        quat_out[ind > N - 1] = np.nan

    return quat_out.type_as(quat)
