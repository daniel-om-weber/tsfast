"""Quaternion math, loss functions, and augmentations."""

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
    "inclination_loss",
    "inclination_loss_abs",
    "inclination_loss_squared",
    "inclination_loss_smooth",
    "abs_inclination",
    "ms_inclination",
    "rms_inclination",
    "smooth_inclination",
    "rms_inclination_deg",
    "rms_pitch_deg",
    "rms_roll_deg",
    "mean_inclination_deg",
    "angle_loss",
    "angle_loss_opt",
    "ms_rel_angle",
    "abs_rel_angle",
    "rms_rel_angle_deg",
    "mean_rel_angle_deg",
    "deg_rmse",
    "QuaternionRegularizer",
    "augmentation_groups",
    "QuaternionAugmentation",
    "plot_scalar_inclination",
    "plot_quaternion_inclination",
    "plot_quaternion_rel_angle",
]

import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .training import fun_rmse


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

    return relativeQuat(nq1, nq2)  # somehow relativeQuat does not work for backpropagation


#     return multiplyQuat(nq1, conjQuat(nq2))


def relativeQuat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the relative quaternion as quat1*inv(quat2)."""

    o1 = q1[..., 0] * q2[..., 0] + q1[..., 1] * q2[..., 1] + q1[..., 2] * q2[..., 2] + q1[..., 3] * q2[..., 3]
    o2 = -q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] - q1[..., 2] * q2[..., 3] + q1[..., 3] * q2[..., 2]
    o3 = -q1[..., 0] * q2[..., 2] + q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] - q1[..., 3] * q2[..., 1]
    o4 = -q1[..., 0] * q2[..., 3] - q1[..., 1] * q2[..., 2] + q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]

    return torch.stack([o1, o2, o3, o4], dim=-1)


def safe_acos(t: torch.Tensor, eps: float = 4e-8) -> torch.Tensor:
    """Numerically stable variant of arccosine."""
    #     eps = 4e-8 #minimum value for acos(1) != 0
    return t.clamp(-1.0 + eps, 1.0 - eps).acos()


def safe_acos_double(t: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """Numerically stable arccosine using float64 internally for accuracy."""
    try:
        # Try to use float64 for higher precision
        return t.type(torch.float64).clamp(-1.0 + eps, 1.0 - eps).acos().type(t.dtype)
    except TypeError as e:
        # If float64 is not supported on this device, warn the user and fall back to float32
        warnings.warn(
            f"Float64 precision not supported on {t.device} device. Falling back to float32. This may reduce numerical accuracy of quaternion operations. Error: {e}"
        )
        return t.clamp(-1.0 + 1e-6, 1.0 - 1e-6).acos()


def inclinationAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the inclination angle between two quaternions."""
    q = diffQuat(q1, q2)
    return 2 * safe_acos_double((q[..., 3] ** 2 + q[..., 0] ** 2).sqrt())


def relativeAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the relative rotation angle between two quaternions."""
    q = diffQuat(q1, q2)
    return 2 * safe_acos_double((q[..., 0]).abs())


def rollAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the roll angle between two quaternions."""
    q = diffQuat(q1, q2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    return torch.atan2(t0, t1)


def pitchAngle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the pitch angle between two quaternions."""
    q = diffQuat(q1, q2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t2 = +2.0 * (w * y - z * x)
    t2 = t2.clamp(-1.0, 1.0)
    return torch.asin(t2)


_unit_quaternion = torch.tensor([1.0, 0, 0, 0])


def inclinationAngleAbs(q: torch.Tensor) -> torch.Tensor:
    """Compute the absolute inclination angle relative to the identity quaternion."""
    q = diffQuat(q, _unit_quaternion[None, :].to(q.device))
    return 2 * ((q[..., 3] ** 2 + q[..., 0] ** 2).sqrt()).acos()


def rand_quat() -> torch.Tensor:
    """Generate a random unit quaternion."""
    q = torch.rand((4)) * 2 - 1
    q /= q.norm()
    return q


def rot_vec(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate a 3D vector by a quaternion."""
    v = F.pad(v, (1, 0), "constant", 0)
    return multiplyQuat(conjQuat(q), multiplyQuat(v, q))[..., 1:]


#     return multiplyQuat(q,multiplyQuat(v,conjQuat(q)))[...,1:]


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

    #     angle = 2 * torch.acos(torch.clamp(q_1_0[:, 0], -1, 1))
    angle = 2 * safe_acos_double((q_1_0[..., 0]))  # .type_as(quat)
    axis = q_1_0[..., 1:]

    # copy over (almost) direct hits
    #     with np.errstate(invalid='ignore'):
    direct_ind = angle < 1e-06
    quat_out = torch.empty_like(q0)
    # print(quat_out.shape, direct_ind.shape, q0.shape)
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
    #     import pdb;pdb.set_trace()

    return quat_out.type_as(quat)


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
    return rad2deg(fun_rmse(inp, targ))


class QuaternionRegularizer:
    """Regularization loss that penalizes non-unit quaternion outputs.

    Args:
        modules: list of nn.Module instances whose outputs are captured via hooks.
        reg_unit: weight for the unit-norm regularization term.
    """

    def __init__(self, modules: list, reg_unit: float = 0.0):
        self.modules = modules
        self.reg_unit = reg_unit
        self._hooks: list = []
        self._captured: torch.Tensor | None = None

    def _hook_fn(self, module, input, output):
        if type(output) is torch.Tensor:
            self._captured = output
        else:
            self._captured = output[0]

    def setup(self, trainer):
        """Register forward hooks on the target modules."""
        for m in self.modules:
            self._hooks.append(m.register_forward_hook(self._hook_fn))

    def teardown(self, trainer):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Compute unit-norm regularization loss from captured hook output."""
        if self._captured is None or self.reg_unit == 0.0:
            return torch.tensor(0.0, device=pred.device)

        h = self._captured.float()
        l_a = float(self.reg_unit) * ((1 - h.norm(dim=-1)) ** 2).mean()
        return l_a


def augmentation_groups(u_groups: list[int]) -> list[list[int]]:
    """Convert channel group sizes into start/end index pairs.

    Args:
        u_groups: list of group sizes (number of channels per group).
    """
    u_groups = np.cumsum([0] + u_groups)
    return [[u_groups[i], u_groups[i + 1] - 1] for i in range(len(u_groups) - 1)]


class QuaternionAugmentation:
    """Apply random quaternion rotation to grouped signals during training.

    Each call samples a new random quaternion and applies it to all specified
    signal groups.  Groups of size 4 are rotated as quaternions, groups of
    size 3 are rotated as vectors.

    Args:
        inp_groups: list of [start, end] index pairs defining signal groups
            (groups of size 4 are rotated as quaternions, size 3 as vectors).
    """

    def __init__(self, inp_groups: list[list[int]]):
        self.inp_groups = inp_groups
        for g in inp_groups:
            group_len = g[1] - g[0] + 1
            if group_len != 4 and group_len != 3:
                raise AttributeError

    def __call__(self, xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random quaternion rotation augmentation.

        Returns:
            Tuple of (augmented xb, augmented yb).
        """
        r_quat = rand_quat()

        # Augment input groups
        for g in self.inp_groups:
            tmp = xb[..., g[0] : g[1] + 1]
            if tmp.shape[-1] == 3:
                xb[..., g[0] : g[1] + 1] = rot_vec(tmp, r_quat)
            else:
                xb[..., g[0] : g[1] + 1] = multiplyQuat(tmp, r_quat)

        # Augment target (quaternion rotation)
        yb = multiplyQuat(yb, r_quat)

        return xb, yb


def relativeQuat_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute the relative quaternion as inv(quat1)*quat2 (numpy)."""
    if isinstance(q1, np.ndarray) and q1.shape == (4,):
        q1 = q1[np.newaxis]  # convert to 1x4 matrix
        shape = q2.shape
    elif isinstance(q1, np.ndarray) and q1.shape == (1, 4):
        shape = q2.shape
    elif isinstance(q2, np.ndarray) and q2.shape == (4,):
        q2 = q2[np.newaxis]  # convert to 1x4 matrix
        shape = q1.shape
    elif isinstance(q2, np.ndarray) and q2.shape == (1, 4):
        shape = q1.shape
    else:
        assert q1.shape == q2.shape
        shape = q1.shape
    output = np.zeros(shape=shape)
    output[:, 0] = q1[:, 0] * q2[:, 0] + q1[:, 1] * q2[:, 1] + q1[:, 2] * q2[:, 2] + q1[:, 3] * q2[:, 3]
    output[:, 1] = q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0] - q1[:, 2] * q2[:, 3] + q1[:, 3] * q2[:, 2]
    output[:, 2] = q1[:, 0] * q2[:, 2] + q1[:, 1] * q2[:, 3] - q1[:, 2] * q2[:, 0] - q1[:, 3] * q2[:, 1]
    output[:, 3] = q1[:, 0] * q2[:, 3] - q1[:, 1] * q2[:, 2] + q1[:, 2] * q2[:, 1] - q1[:, 3] * q2[:, 0]
    return output


def quatFromAngleAxis_np(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Create quaternions from angle-axis representation (numpy).

    If angle is 0, the output will be an identity quaternion. If axis is a
    zero vector, a ValueError will be raised unless the corresponding angle
    is 0.

    Args:
        angle: scalar or N angles in radians.
        axis: rotation axes, shape (3,) or (Nx3) or (1x3).

    Returns:
        Quaternion array, shape (Nx4) or (1x4).
    """

    angle = np.asarray(angle, np.float64)
    axis = np.asarray(axis, np.float64)

    is1D = (angle.shape == tuple() or angle.shape == (1,)) and axis.shape == (3,)

    if angle.shape == tuple():
        angle = angle.reshape(1)  # equivalent to np.atleast_1d
    if axis.shape == (3,):
        axis = axis.reshape((1, 3))

    N = max(angle.shape[0], axis.shape[0])

    # for (1x1) case
    if angle.shape == (1, 1):
        angle = angle.ravel()

    assert angle.shape == (N,) or angle.shape == (1,), f"invalid angle shape: {angle.shape}"
    assert axis.shape == (N, 3) or axis.shape == (1, 3), f"invalid axis shape: {axis.shape}"

    angle_brodcasted = np.broadcast_to(angle, (N,))
    axis_brodcasted = np.broadcast_to(axis, (N, 3))

    norm = np.linalg.norm(axis_brodcasted, axis=1)

    identity = norm < np.finfo(np.float64).eps

    q = np.zeros((N, 4), np.float64)
    q[identity] = np.array([1, 0, 0, 0])
    q[~identity] = np.concatenate(
        (
            np.cos(angle_brodcasted[~identity][:, np.newaxis] / 2),
            axis_brodcasted[~identity]
            * np.array(np.sin(angle_brodcasted[~identity] / 2.0) / norm[~identity])[:, np.newaxis],
        ),
        axis=1,
    )

    if is1D:
        q = q.reshape((4,))

    return q


def multiplyQuat_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions element-wise (numpy)."""
    if isinstance(q1, np.ndarray) and q1.shape == (4,):
        q1 = q1[np.newaxis]  # convert to 1x4 matrix
        shape = q2.shape
    elif isinstance(q1, np.ndarray) and q1.shape == (1, 4):
        shape = q2.shape
    elif isinstance(q2, np.ndarray) and q2.shape == (4,):
        q2 = q2[np.newaxis]  # convert to 1x4 matrix
        shape = q1.shape
    elif isinstance(q2, np.ndarray) and q2.shape == (1, 4):
        shape = q1.shape
    else:
        assert q1.shape == q2.shape
        shape = q1.shape
    output = np.zeros(shape=shape)
    output[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    output[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    output[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    output[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    return output


def quatInterp_np(quat: np.ndarray, ind: np.ndarray, extend: bool = True) -> np.ndarray:
    """Interpolate quaternions at non-integer indices using Slerp (numpy).

    Sampling indices are in the range 0..N-1. For values outside this range,
    depending on ``extend``, the first/last element or NaN is returned.

    Args:
        quat: input quaternions, shape (Nx4).
        ind: sampling indices, shape (M,).
        extend: if true, extend input by repeating first/last value.

    Returns:
        Interpolated quaternions, shape (Mx4).
    """
    ind = np.atleast_1d(ind)
    N = quat.shape[0]
    M = ind.shape[0]
    assert quat.shape == (N, 4)
    assert ind.shape == (M,)

    ind0 = np.clip(np.floor(ind).astype(int), 0, N - 1)
    ind1 = np.clip(np.ceil(ind).astype(int), 0, N - 1)

    q0 = quat[ind0]
    q1 = quat[ind1]
    q_1_0 = relativeQuat_np(q0, q1)

    # normalize the quaternion for positive w component to ensure
    # that the angle will be [0, 180deg]
    invert_sign_ind = q_1_0[:, 0] < 0
    q_1_0[invert_sign_ind] = -q_1_0[invert_sign_ind]

    angle = 2 * np.arccos(np.clip(q_1_0[:, 0], -1, 1))
    axis = q_1_0[:, 1:]

    # copy over (almost) direct hits
    with np.errstate(invalid="ignore"):
        direct_ind = angle < 1e-06
    quat_out = np.empty_like(q0)
    # print(quat_out.shape, direct_ind.shape, q0.shape)
    quat_out[direct_ind] = q0[direct_ind]

    interp_ind = ~direct_ind
    t01 = ind - ind0
    q_t_0 = quatFromAngleAxis_np((t01 * angle)[interp_ind], axis[interp_ind])
    quat_out[interp_ind] = multiplyQuat_np(q0[interp_ind], q_t_0)

    if not extend:
        quat_out[ind < 0] = np.nan
        quat_out[ind > N - 1] = np.nan

    return quat_out


def plot_scalar_inclination(
    axs: list, in_sig: torch.Tensor, targ_sig: torch.Tensor, out_sig: torch.Tensor | None = None, **kwargs
):
    """Plot scalar inclination target, prediction, and error.

    Args:
        axs: list of matplotlib axes to plot on.
        in_sig: input signal tensor.
        targ_sig: target inclination tensor.
        out_sig: predicted inclination tensor, or None for batch display.
    """
    axs[0].plot(rad2deg(targ_sig).detach().cpu().numpy())
    axs[0].label_outer()
    axs[0].set_ylabel("inclination[deg]")

    if out_sig is not None:
        axs[0].plot(rad2deg(out_sig).detach().cpu().numpy())
        axs[0].legend(["y", "y-hat"])
        axs[1].plot(rad2deg(targ_sig - out_sig).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[deg]")

    axs[-1].plot(in_sig)


def plot_quaternion_inclination(
    axs: list, in_sig: torch.Tensor, targ_sig: torch.Tensor, out_sig: torch.Tensor | None = None, **kwargs
):
    """Plot quaternion inclination target, prediction, and error.

    Args:
        axs: list of matplotlib axes to plot on.
        in_sig: input signal tensor.
        targ_sig: target quaternion tensor.
        out_sig: predicted quaternion tensor, or None for batch display.
    """
    axs[0].plot(rad2deg(inclinationAngleAbs(targ_sig)).detach().cpu().numpy())
    axs[0].label_outer()
    axs[0].legend(["y"])
    axs[0].set_ylabel("inclination[deg]")

    if out_sig is not None:
        axs[0].plot(rad2deg(inclinationAngleAbs(out_sig)).detach().cpu().numpy())
        axs[0].legend(["y", "y-hat"])
        axs[1].plot(rad2deg(inclinationAngle(out_sig, targ_sig)).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[deg]")
        if "ref" in kwargs:
            #             axs[0].plot(rad2deg(inclinationAngleAbs(kwargs['ref'])))
            #             axs[0].legend(['y','y-hat','y_ref'])
            axs[1].plot(rad2deg(inclinationAngle(targ_sig, kwargs["ref"])).detach().cpu().numpy())
            axs[1].legend(["y-hat", "y_ref"])

    axs[-1].plot(in_sig)


def plot_quaternion_rel_angle(
    axs: list, in_sig: torch.Tensor, targ_sig: torch.Tensor, out_sig: torch.Tensor | None = None, **kwargs
):
    """Plot relative quaternion angle target, prediction, and error.

    Args:
        axs: list of matplotlib axes to plot on.
        in_sig: input signal tensor.
        targ_sig: target quaternion tensor.
        out_sig: predicted quaternion tensor, or None for batch display.
    """
    first_targ = targ_sig[0].repeat(targ_sig.shape[0], 1)
    axs[0].plot(rad2deg(relativeAngle(first_targ, targ_sig)).detach().cpu().numpy())
    axs[0].label_outer()
    axs[0].legend(["y"])
    axs[0].set_ylabel("angle[deg]")

    if out_sig is not None:
        axs[0].plot(rad2deg(relativeAngle(first_targ, out_sig)).detach().cpu().numpy())
        axs[0].legend(["y", "y-hat"])
        axs[1].plot(rad2deg(relativeAngle(out_sig, targ_sig)).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[deg]")

    axs[-1].plot(in_sig)
