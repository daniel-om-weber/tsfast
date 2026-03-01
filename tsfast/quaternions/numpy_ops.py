"""NumPy implementations of quaternion operations."""

__all__ = [
    "relativeQuat_np",
    "quatFromAngleAxis_np",
    "multiplyQuat_np",
    "quatInterp_np",
]

import numpy as np


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
    quat_out[direct_ind] = q0[direct_ind]

    interp_ind = ~direct_ind
    t01 = ind - ind0
    q_t_0 = quatFromAngleAxis_np((t01 * angle)[interp_ind], axis[interp_ind])
    quat_out[interp_ind] = multiplyQuat_np(q0[interp_ind], q_t_0)

    if not extend:
        quat_out[ind < 0] = np.nan
        quat_out[ind > N - 1] = np.nan

    return quat_out
