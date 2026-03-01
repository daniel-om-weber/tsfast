"""Quaternion math, loss functions, and augmentations."""

from .core import (
    conjQuat,
    diffQuat,
    inclinationAngle,
    inclinationAngleAbs,
    multiplyQuat,
    norm_quaternion,
    pitchAngle,
    quatFromAngleAxis,
    quatInterp,
    rad2deg,
    rand_quat,
    relativeAngle,
    relativeQuat,
    rollAngle,
    rot_vec,
    safe_acos,
    safe_acos_double,
)
from .losses import (
    abs_inclination,
    abs_rel_angle,
    angle_loss,
    angle_loss_opt,
    deg_rmse,
    inclination_loss,
    inclination_loss_abs,
    inclination_loss_smooth,
    inclination_loss_squared,
    mean_inclination_deg,
    mean_rel_angle_deg,
    ms_inclination,
    ms_rel_angle,
    rms_inclination,
    rms_inclination_deg,
    rms_pitch_deg,
    rms_rel_angle_deg,
    rms_roll_deg,
    smooth_inclination,
)
from .aux_losses import QuaternionRegularizer
from .transforms import QuaternionAugmentation, augmentation_groups
from .numpy_ops import (
    multiplyQuat_np,
    quatFromAngleAxis_np,
    quatInterp_np,
    relativeQuat_np,
)
from .viz import (
    plot_quaternion_inclination,
    plot_quaternion_rel_angle,
    plot_scalar_inclination,
)

__all__ = [
    # core math
    "rad2deg",
    "multiplyQuat",
    "norm_quaternion",
    "conjQuat",
    "diffQuat",
    "relativeQuat",
    "safe_acos",
    "safe_acos_double",
    # angles
    "inclinationAngle",
    "relativeAngle",
    "rollAngle",
    "pitchAngle",
    "inclinationAngleAbs",
    # generation / manipulation
    "rand_quat",
    "rot_vec",
    "quatFromAngleAxis",
    "quatInterp",
    # losses & metrics
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
    # aux losses
    "QuaternionRegularizer",
    # transforms
    "augmentation_groups",
    "QuaternionAugmentation",
    # numpy
    "relativeQuat_np",
    "quatFromAngleAxis_np",
    "multiplyQuat_np",
    "quatInterp_np",
    # viz
    "plot_scalar_inclination",
    "plot_quaternion_inclination",
    "plot_quaternion_rel_angle",
]
