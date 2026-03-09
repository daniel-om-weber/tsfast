"""Transforms and augmentations following the ``__call__(xb, yb) -> (xb, yb)`` protocol."""

__all__ = [
    "prediction_concat",
    "noise",
    "noise_varying",
    "noise_grouped",
    "bias",
    "vary_seq_len",
    "truncate_sequence",
]

import random
from collections.abc import Callable

import torch
from torch import Tensor

from .schedulers import sched_ramp


# ──────────────────────────────────────────────────────────────────────────────
#  Transforms (applied train + valid)
# ──────────────────────────────────────────────────────────────────────────────


class prediction_concat:
    """Concatenate y onto x for autoregressive prediction, shortening both by t_offset.

    Args:
        t_offset: number of steps the output is shifted in the past
    """

    def __init__(self, t_offset: int = 1):
        self.t_offset = t_offset

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        x = xb
        y = yb

        if self.t_offset != 0:
            x = x[:, self.t_offset :, :]
            y = y[:, : -self.t_offset, :]
            yb = yb[:, self.t_offset :, :]

        return torch.cat((x, y), dim=-1), yb


# ──────────────────────────────────────────────────────────────────────────────
#  Augmentations (train only)
# ──────────────────────────────────────────────────────────────────────────────


class noise:
    """Add normal-distributed noise with per-signal mean and std to the input.

    Args:
        std: standard deviation of the noise per signal (scalar or vector)
        mean: mean of the noise per signal (scalar or vector)
        p: probability of applying the augmentation
    """

    def __init__(self, std: float | Tensor = 0.1, mean: float | Tensor = 0.0, p: float = 1.0):
        self.std = torch.as_tensor(std, dtype=torch.float)
        self.mean = torch.as_tensor(mean, dtype=torch.float)
        self.p = p

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        if self.p < 1.0 and random.random() > self.p:
            return xb, yb
        std = self.std.to(xb.device)
        mean = self.mean.to(xb.device)
        xb = xb + torch.normal(mean=mean.expand_as(xb), std=std.expand_as(xb))
        return xb, yb


class noise_varying:
    """Add noise with a randomly sampled standard deviation per application.

    Args:
        std_std: standard deviation of the noise std distribution
        p: probability of applying the augmentation
    """

    def __init__(self, std_std: float = 0.1, p: float = 1.0):
        self.std_std = torch.as_tensor(std_std, dtype=torch.float)
        self.p = p

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        if self.p < 1.0 and random.random() > self.p:
            return xb, yb
        std_std = self.std_std.to(xb.device)
        std = torch.normal(mean=0, std=std_std).abs()
        xb = xb + torch.normal(mean=0, std=std.expand_as(xb))
        return xb, yb


class noise_grouped:
    """Add noise with per-group randomly sampled standard deviations.

    Args:
        std_std: standard deviation of the noise std distribution per group
        std_idx: index mapping each signal to its noise group
        p: probability of applying the augmentation
    """

    def __init__(self, std_std, std_idx, p: float = 1.0):
        self.std_std = torch.as_tensor(std_std, dtype=torch.float)
        self.std_idx = torch.as_tensor(std_idx, dtype=torch.long)
        self.p = p

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        if self.p < 1.0 and random.random() > self.p:
            return xb, yb
        std_std = self.std_std.to(xb.device)
        std_idx = self.std_idx.to(xb.device)
        std = torch.normal(mean=0, std=std_std).abs()[std_idx]
        xb = xb + torch.normal(mean=0, std=std.expand_as(xb))
        return xb, yb


class bias:
    """Add a constant normal-distributed offset per signal per sample to the input.

    Args:
        std: standard deviation of the bias per signal (scalar or vector)
        mean: mean of the bias per signal (scalar or vector)
        p: probability of applying the augmentation
    """

    def __init__(self, std: float | Tensor = 0.1, mean: float | Tensor = 0.0, p: float = 1.0):
        self.std = torch.as_tensor(std, dtype=torch.float)
        self.mean = torch.as_tensor(mean, dtype=torch.float)
        self.p = p

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        if self.p < 1.0 and random.random() > self.p:
            return xb, yb
        mean = self.mean.to(xb.device)
        std = self.std.to(xb.device)
        # Constant offset per sample: shape [batch, 1, features]
        mean = mean.repeat((xb.shape[0], 1, 1)).expand((xb.shape[0], 1, xb.shape[2]))
        std = std.repeat((xb.shape[0], 1, 1)).expand((xb.shape[0], 1, xb.shape[2]))
        n = torch.normal(mean=mean, std=std).expand_as(xb)
        xb = xb + n
        return xb, yb


class vary_seq_len:
    """Randomly vary sequence length of every minibatch.

    Args:
        min_len: minimum sequence length to keep
    """

    def __init__(self, min_len: int = 50):
        self.min_len = min_len

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        seq_len_x = xb.shape[1]
        ly = yb.shape[1]
        lim = random.randint(self.min_len, ly)
        if ly < seq_len_x:
            xb = xb[:, : -(ly - lim)]
        else:
            xb = xb[:, :lim]
        yb = yb[:, :lim]
        return xb, yb


class truncate_sequence:
    """Progressively truncate sequence length during training using a scheduler.

    Stateful: call ``setup(trainer)`` before training to access ``trainer.pct_train``.

    Args:
        truncate_length: maximum number of time steps to truncate
        scheduler: scheduling function controlling truncation over training
    """

    def __init__(self, truncate_length: int = 50, scheduler: Callable = sched_ramp):
        self._truncate_length = truncate_length
        self._scheduler = scheduler
        self._trainer = None

    def setup(self, trainer):
        self._trainer = trainer

    def teardown(self, trainer):
        self._trainer = None

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        pct = self._trainer.pct_train if self._trainer is not None else 0.0
        ly = yb.shape[1]
        lim = int(self._scheduler(ly - self._truncate_length, 0, pct))
        if lim > 0:
            xb = xb[:, :-lim]
            yb = yb[:, :-lim]
        return xb, yb
