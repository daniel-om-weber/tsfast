"""Transforms and augmentations following the ``__call__(xb, yb) -> (xb, yb)`` protocol."""

__all__ = [
    "prediction_concat",
    "ar_init",
    "noise",
    "noise_varying",
    "noise_grouped",
    "bias",
    "vary_seq_len",
    "truncate_sequence",
    "variable_init_sz",
    "alternating_encoder",
]

import random
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from .losses import sched_ramp


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
        y = yb.as_subclass(type(x))

        if self.t_offset != 0:
            x = x[:, self.t_offset :, :]
            y = y[:, : -self.t_offset, :]
            yb = yb[:, self.t_offset :, :]

        return torch.cat((x, y), dim=-1), yb


class ar_init:
    """Concatenate the target variable to the input for autoregression."""

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        y = yb.as_subclass(type(xb))
        return torch.cat((xb, y), dim=-1), yb


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


class variable_init_sz:
    """Randomizes ``model.init_sz`` during training, restores during validation.

    Transform (applied train + valid): during training, sets a random init_sz
    per batch; during validation, restores the original init_sz.

    Args:
        init_sz_min: minimum initialization window size
        init_sz_max: maximum initialization window size (inclusive)
        model: explicit model reference (auto-detected via unwrap_model if None)
    """

    def __init__(self, init_sz_min: int, init_sz_max: int, model: torch.nn.Module | None = None):
        self.init_sz_min = init_sz_min
        self.init_sz_max = init_sz_max
        self.inner_model = model
        self._init_sz_valid: int | None = None
        self._training: bool = False

    def setup(self, trainer):
        if self.inner_model is None:
            from ..models.layers import unwrap_model

            self.inner_model = unwrap_model(trainer.model)
        self._trainer = trainer

    def teardown(self, trainer):
        self._trainer = None

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        model = self.inner_model
        if model is None or not hasattr(model, "init_sz"):
            return xb, yb

        training = self._trainer.model.training if self._trainer is not None else False

        if self._init_sz_valid is None:
            self._init_sz_valid = model.init_sz

        if training:
            model.init_sz = np.random.randint(self.init_sz_min, self.init_sz_max + 1)
        else:
            model.init_sz = self._init_sz_valid

        return xb, yb


class alternating_encoder:
    """Randomly alternates between sequence and state encoder per training batch.

    Augmentation (train only): randomly switches ``model.default_encoder_mode``.
    Resets to ``'sequence'`` on teardown.

    Args:
        p_state: probability of using state encoder per batch
        model: explicit model reference (auto-detected via unwrap_model if None)
    """

    def __init__(self, p_state: float = 0.3, model: torch.nn.Module | None = None):
        self.p_state = p_state
        self.inner_model = model

    def setup(self, trainer):
        if self.inner_model is None:
            from ..models.layers import unwrap_model

            self.inner_model = unwrap_model(trainer.model)

    def teardown(self, trainer):
        if self.inner_model is not None and hasattr(self.inner_model, "default_encoder_mode"):
            self.inner_model.default_encoder_mode = "sequence"

    def __call__(self, xb: Tensor, yb: Tensor) -> tuple[Tensor, Tensor]:
        if self.inner_model is not None and hasattr(self.inner_model, "default_encoder_mode"):
            self.inner_model.default_encoder_mode = "state" if np.random.rand() < self.p_state else "sequence"
        return xb, yb
