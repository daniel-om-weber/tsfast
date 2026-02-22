"""Quaternion math, loss functions, augmentations, and data blocks."""

__all__ = [
    "TensorQuaternionInclination",
    "TensorQuaternionAngle",
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
    "Quaternion_ResamplingModel",
    "HDF2Quaternion",
    "QuaternionBlock",
    "TensorInclination",
    "HDF2Inclination",
    "InclinationBlock",
    "plot_scalar_inclination",
    "plot_quaternion_inclination",
    "plot_quaternion_rel_angle",
    "show_results",
    "show_batch",
]

import math
import warnings

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.callback.core import TrainEvalCallback
from fastai.callback.hook import HookCallback
from fastai.data.block import TransformBlock
from fastai.data.transforms import Normalize
from fastai.torch_basics import Transform, tensor
from fastcore.meta import delegates
from plum import dispatch
from scipy.signal import resample

from .data.block import pad_sequence
from .data.core import (
    HDF2Sequence,
    TensorSequences,
    TensorSequencesOutput,
    plot_seqs_multi_figures,
    plot_seqs_single_figure,
)
from .learner.losses import fun_rmse


class TensorQuaternionInclination(TensorSequences):
    pass


class TensorQuaternionAngle(TensorSequences):
    pass


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


_conjugate_quaternion = tensor([1, -1, -1, -1])


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


_unit_quaternion = tensor([1.0, 0, 0, 0])


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
    # that the angle will be [0, 180°]
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


@delegates()
class QuaternionRegularizer(HookCallback):
    """Regularize quaternion output toward unit norm.

    Args:
        reg_unit: weight for the unit-norm regularization term.
        detach: whether to detach the hook output from the computation graph.
    """

    run_before = TrainEvalCallback

    def __init__(self, reg_unit: float = 0.0, detach: bool = False, **kwargs):
        super().__init__(detach=detach, cpu=False, **kwargs)
        self.reg_unit = reg_unit

    def hook(self, m, i, o):
        if type(o) is torch.Tensor:
            self.out = o
        else:
            self.out = o[0]

    def after_loss(self):
        if not self.training:
            return
        h = self.out.float()

        if self.reg_unit != 0.0:
            l_a = float(self.reg_unit) * ((1 - h.norm(dim=-1)) ** 2).mean()
            #             import pdb; pdb.set_trace()
            self.learn.loss_grad += l_a


def augmentation_groups(u_groups: list[int]) -> list[list[int]]:
    """Convert channel group sizes into start/end index pairs.

    Args:
        u_groups: list of group sizes (number of channels per group).
    """
    u_groups = np.cumsum([0] + u_groups)
    return [[u_groups[i], u_groups[i + 1] - 1] for i in range(len(u_groups) - 1)]


class QuaternionAugmentation(Transform):
    """Apply random quaternion rotation to grouped signals.

    Only applied to training data. Each call samples a new random quaternion
    and applies it to all specified signal groups.

    Args:
        inp_groups: list of [start, end] index pairs defining signal groups
            (groups of size 4 are rotated as quaternions, size 3 as vectors).
    """

    split_idx = 0

    def __init__(self, inp_groups: list[list[int]], **kwargs):
        super().__init__(**kwargs)
        self.inp_groups = inp_groups
        self.r_quat = None
        for g in inp_groups:
            group_len = g[1] - g[0] + 1
            if group_len != 4 and group_len != 3:
                raise AttributeError

    def __call__(self, b, split_idx=None, **kwargs):
        # import pdb; pdb.set_trace()
        self.r_quat = rand_quat()
        return super().__call__(b, split_idx=split_idx, **kwargs)

    def encodes(self, x: (TensorSequences)):
        # import pdb; pdb.set_trace()
        for g in self.inp_groups:
            tmp = x[..., g[0] : g[1] + 1]
            if tmp.shape[-1] == 3:
                x[..., g[0] : g[1] + 1] = rot_vec(tmp, self.r_quat)
            else:
                x[..., g[0] : g[1] + 1] = multiplyQuat(tmp, self.r_quat)
        return x

    def encodes(self, x: TensorQuaternionInclination):
        return multiplyQuat(x, self.r_quat)

    def encodes(self, x: TensorQuaternionAngle):
        return multiplyQuat(x, self.r_quat)


class Quaternion_ResamplingModel(nn.Module):
    """Resample signals before and after model prediction.

    Useful for applying models to datasets with different sampling rates.

    Args:
        model: wrapped prediction model.
        fs_targ: target sampling frequency for resampling.
        fs_mean: mean used for denormalizing the sampling frequency input.
        fs_std: std used for denormalizing the sampling frequency input.
        quaternion_sampling: use quaternion Slerp interpolation for output
            resampling instead of linear interpolation.
    """

    def __init__(
        self, model: nn.Module, fs_targ: float, fs_mean: float = 0, fs_std: float = 1, quaternion_sampling: bool = True
    ):
        super().__init__()
        self.model = model
        self.fs_targ = fs_targ
        self.register_buffer("fs_mean", tensor(fs_mean))
        self.register_buffer("fs_std", tensor(fs_std))
        self.quaternion_sampling = quaternion_sampling

    def forward(self, x):
        dt = (x[0, 0, -1] * self.fs_std) + self.fs_mean
        fs_src = 1 / dt
        x_len = x.shape[1]
        res_len = int(x.shape[1] * self.fs_targ / (fs_src + 10))
        x_raw = x[..., :-1]
        if x_len == res_len:
            res = self.model(x_raw)
        else:
            #             x_new = nn.functional.interpolate(x_raw.transpose(1,2), size=res_len, mode='linear',align_corners=False).transpose(1,2)
            #             import pdb;pdb.set_trace()
            x_new = tensor(resample(x_raw.detach().cpu().numpy(), res_len, axis=1)).to(x_raw.device)
            res = self.model(x_new)
            if self.quaternion_sampling:
                res = quatInterp(res.transpose(0, 1), torch.linspace(0, res.shape[1] - 1, x_len)).transpose(0, 1)
            else:
                res = nn.functional.interpolate(
                    res.transpose(1, 2), size=x_len, mode="linear", align_corners=False
                ).transpose(1, 2)

        return res


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
    # that the angle will be [0, 180°]
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


class HDF2Quaternion(HDF2Sequence):
    """Extract quaternion sequences from HDF5 files with Slerp resampling."""

    def _hdf_extract_sequence(
        self,
        hdf_path,
        dataset=None,
        l_slc=None,
        r_slc=None,
        resampling_factor=None,
        fs_idx=None,
        dt_idx=False,
        fast_resample=True,
    ):

        if resampling_factor is not None:
            seq_len = (
                r_slc - l_slc if l_slc is not None and r_slc is not None else None
            )  # calculate seq_len for later slicing, necesary because of rounding errors in resampling
            if l_slc is not None:
                l_slc = math.floor(l_slc / resampling_factor)
            if r_slc is not None:
                r_slc = math.ceil(r_slc / resampling_factor)

        with h5py.File(hdf_path, "r") as f:
            ds = f if dataset is None else f[dataset]
            l_array = [(ds[n][l_slc:r_slc]) for n in self.clm_names]
            seq = np.stack(l_array, axis=-1)

        if resampling_factor is not None:
            #             res_seq = resample_interp(seq,resampling_factor)
            res_seq = quatInterp_np(seq, np.linspace(0, seq.shape[0] - 1, int(seq.shape[0] * resampling_factor)))
            if fs_idx is not None:
                res_seq[:, fs_idx] = seq[0, fs_idx] * resampling_factor
            if dt_idx is not None:
                res_seq[:, dt_idx] = seq[0, dt_idx] / resampling_factor
            seq = res_seq

            if seq_len is not None:
                seq = seq[
                    :seq_len
                ]  # cut the part of the sequence that is too long because of resampling rounding errors

        return seq


class QuaternionBlock(TransformBlock):
    """TransformBlock for quaternion sequence data with normalization.

    Args:
        seq_extract: transform that extracts the quaternion sequence.
        padding: whether to pad sequences of different lengths.
    """

    def __init__(self, seq_extract: Transform, padding: bool = False):
        return super().__init__(
            type_tfms=[seq_extract],
            batch_tfms=[Normalize(axes=[0, 1])],
            dls_kwargs={} if not padding else {"before_batch": pad_sequence},
        )

    @classmethod
    @delegates(HDF2Quaternion, keep=True)
    def from_hdf(
        cls, clm_names: list[str], seq_cls: type = TensorQuaternionInclination, padding: bool = False, **kwargs
    ):
        """Create a QuaternionBlock from HDF5 files.

        Args:
            clm_names: column/dataset names to extract.
            seq_cls: tensor class for the extracted quaternion sequences.
            padding: whether to pad sequences of different lengths.
        """
        return cls(HDF2Quaternion(clm_names, to_cls=seq_cls, **kwargs), padding)


class TensorInclination(TensorSequences):
    pass


class HDF2Inclination(HDF2Sequence):
    """Extract inclination angle sequences from HDF5 quaternion data."""

    def _hdf_extract_sequence(self, hdf_path, dataset=None, l_slc=None, r_slc=None, down_s=None):
        with h5py.File(hdf_path, "r") as f:
            ds = f if dataset is None else f[dataset]
            l_array = [ds[n][l_slc:r_slc] for n in self.clm_names]
            seq = np.vstack(l_array).T
            seq = np.array(inclinationAngleAbs(tensor(seq))[:, None])
            return seq


class InclinationBlock(TransformBlock):
    """TransformBlock for inclination angle sequence data.

    Args:
        seq_extract: transform that extracts the inclination sequence.
        padding: whether to pad sequences of different lengths.
    """

    def __init__(self, seq_extract: Transform, padding: bool = False):
        return super().__init__(
            type_tfms=[seq_extract],
            batch_tfms=[Normalize(axes=[0, 1])],
            dls_kwargs={} if not padding else {"before_batch": pad_sequence},
        )

    @classmethod
    @delegates(HDF2Inclination, keep=True)
    def from_hdf(cls, clm_names: list[str], seq_cls: type = TensorInclination, padding: bool = False, **kwargs):
        """Create an InclinationBlock from HDF5 files.

        Args:
            clm_names: column/dataset names to extract.
            seq_cls: tensor class for the extracted inclination sequences.
            padding: whether to pad sequences of different lengths.
        """
        return cls(HDF2Inclination(clm_names, to_cls=seq_cls, **kwargs), padding)


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
    axs[0].set_ylabel("inclination[°]")

    if out_sig is not None:
        axs[0].plot(rad2deg(out_sig).detach().cpu().numpy())
        axs[0].legend(["y", "ŷ"])
        axs[1].plot(rad2deg(targ_sig - out_sig).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[°]")

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
    axs[0].set_ylabel("inclination[°]")

    if out_sig is not None:
        axs[0].plot(rad2deg(inclinationAngleAbs(out_sig)).detach().cpu().numpy())
        axs[0].legend(["y", "ŷ"])
        axs[1].plot(rad2deg(inclinationAngle(out_sig, targ_sig)).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[°]")
        if "ref" in kwargs:
            #             axs[0].plot(rad2deg(inclinationAngleAbs(kwargs['ref'])))
            #             axs[0].legend(['y','ŷ','y_ref'])
            axs[1].plot(rad2deg(inclinationAngle(targ_sig, kwargs["ref"])).detach().cpu().numpy())
            axs[1].legend(["ŷ", "y_ref"])

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
    axs[0].set_ylabel("angle[°]")

    if out_sig is not None:
        axs[0].plot(rad2deg(relativeAngle(first_targ, out_sig)).detach().cpu().numpy())
        axs[0].legend(["y", "ŷ"])
        axs[1].plot(rad2deg(relativeAngle(out_sig, targ_sig)).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[°]")

    axs[-1].plot(in_sig)


@dispatch
def show_results(x: TensorSequences, y: TensorInclination, samples, outs, ctxs=None, max_n=2, **kwargs):
    """Show prediction results for scalar inclination targets."""
    n_samples = min(len(samples), max_n)
    n_targ = 2
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_scalar_inclination, outs, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_scalar_inclination, outs, **kwargs)
    return ctxs


@dispatch
def show_batch(x: TensorSequences, y: TensorInclination, samples, ctxs=None, max_n=6, **kwargs):
    """Show a batch of scalar inclination samples."""
    n_samples = min(len(samples), max_n)
    n_targ = 1
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_scalar_inclination, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_scalar_inclination, **kwargs)
    return ctxs


@dispatch
def show_results(x: TensorSequences, y: TensorQuaternionInclination, samples, outs, ctxs=None, max_n=2, **kwargs):
    """Show prediction results for quaternion inclination targets."""
    if "quat" in kwargs:
        return show_results(x, TensorSequencesOutput(y), samples, outs, ctxs, max_n, **kwargs)
    n_samples = min(len(samples), max_n)
    n_targ = 2
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_quaternion_inclination, outs, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_quaternion_inclination, outs, **kwargs)
    return ctxs


@dispatch
def show_batch(x: TensorSequences, y: TensorQuaternionInclination, samples, ctxs=None, max_n=6, **kwargs):
    """Show a batch of quaternion inclination samples."""
    n_samples = min(len(samples), max_n)
    n_targ = 1
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_quaternion_inclination)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_quaternion_inclination)
    return ctxs


@dispatch
def show_results(x: TensorSequences, y: TensorQuaternionAngle, samples, outs, ctxs=None, max_n=2, **kwargs):
    """Show prediction results for quaternion angle targets."""
    n_samples = min(len(samples), max_n)
    n_targ = 2
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_quaternion_rel_angle, outs, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_quaternion_rel_angle, outs, **kwargs)
    return ctxs


@dispatch
def show_batch(x: TensorSequences, y: TensorQuaternionAngle, samples, ctxs=None, max_n=6, **kwargs):
    """Show a batch of quaternion angle samples."""
    n_samples = min(len(samples), max_n)
    n_targ = 1
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_quaternion_rel_angle, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_quaternion_rel_angle, **kwargs)
    return ctxs
