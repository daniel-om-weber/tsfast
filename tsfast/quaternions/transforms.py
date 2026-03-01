"""Quaternion data augmentation transforms."""

__all__ = ["augmentation_groups", "QuaternionAugmentation"]

import numpy as np
import torch

from .core import multiplyQuat, rand_quat, rot_vec


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
