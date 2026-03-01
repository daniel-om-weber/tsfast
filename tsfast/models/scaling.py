"""Feature scalers and normalized model wrappers for time series models."""

__all__ = [
    "Scaler",
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "ScaledModel",
    "unwrap_model",
]

import numpy as np
import torch
from torch import nn


class Scaler(nn.Module):
    "Base class for feature scaling on [batch, seq, features] tensors."

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.denormalize(x)

    @classmethod
    def from_stats(cls, stats) -> "Scaler":
        "Create a Scaler from a NormPair. Override in subclasses."
        raise NotImplementedError(f"{cls.__name__} must implement from_stats")


def _ensure_tensor(arr):
    "Convert numpy array or tensor to shape [1,1,features] float tensor."
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr[None, None, :]).float()
    if isinstance(arr, torch.Tensor):
        if arr.ndim == 1:
            return arr[None, None, :].float()
        return arr.float()
    raise TypeError(f"Expected ndarray or Tensor, got {type(arr)}")


class StandardScaler(Scaler):
    """Normalize by ``(x - mean) / std``.

    Args:
        mean: per-feature mean, as ndarray or tensor
        std: per-feature standard deviation, as ndarray or tensor
    """

    _epsilon = 1e-16

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", _ensure_tensor(mean))
        self.register_buffer("std", _ensure_tensor(std) + self._epsilon)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    @classmethod
    def from_stats(cls, stats) -> "StandardScaler":
        return cls(stats.mean, stats.std)


class MinMaxScaler(Scaler):
    """Normalize by ``(x - min) / (max - min)`` to ``[0, 1]``.

    Args:
        min_val: per-feature minimum, as ndarray or tensor
        max_val: per-feature maximum, as ndarray or tensor
    """

    _epsilon = 1e-16

    def __init__(self, min_val, max_val):
        super().__init__()
        self.register_buffer("min_val", _ensure_tensor(min_val))
        self.register_buffer("range_val", _ensure_tensor(max_val) - _ensure_tensor(min_val) + self._epsilon)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / self.range_val

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.range_val + self.min_val

    @classmethod
    def from_stats(cls, stats) -> "MinMaxScaler":
        return cls(stats.min, stats.max)


class MaxAbsScaler(Scaler):
    """Normalize by ``x / max(|min|, |max|)``.

    Args:
        min_val: per-feature minimum, as ndarray or tensor
        max_val: per-feature maximum, as ndarray or tensor
    """

    _epsilon = 1e-16

    def __init__(self, min_val, max_val):
        super().__init__()
        self.register_buffer(
            "max_abs", torch.max(torch.abs(_ensure_tensor(min_val)), torch.abs(_ensure_tensor(max_val))) + self._epsilon
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.max_abs

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.max_abs

    @classmethod
    def from_stats(cls, stats) -> "MaxAbsScaler":
        return cls(stats.min, stats.max)


class ScaledModel(nn.Module):
    """Wraps a model with input normalization and optional output denormalization.

    Args:
        model: inner model to wrap
        input_norm: scaler applied to inputs before the model
        output_norm: scaler applied to outputs after the model
    """

    def __init__(self, model: nn.Module, input_norm: Scaler, output_norm: Scaler | None = None):
        super().__init__()
        self.model = model
        self.input_norm = input_norm
        self.output_norm = output_norm

    @classmethod
    def from_stats(
        cls, model: nn.Module, input_stats, output_stats=None, scaler_cls: type | None = None
    ) -> "ScaledModel":
        "Create from NormPair stats with the given Scaler class."
        if scaler_cls is None:
            scaler_cls = StandardScaler
        input_norm = scaler_cls.from_stats(input_stats)
        output_norm = scaler_cls.from_stats(output_stats) if output_stats is not None else None
        return cls(model, input_norm, output_norm)

    @classmethod
    def from_dls(
        cls,
        model: nn.Module,
        dls,
        input_norm: type[Scaler] | None = StandardScaler,
        output_norm: type[Scaler] | None = None,
        *,
        autoregressive: bool = False,
    ) -> nn.Module:
        """Create from DataLoaders norm_stats, or return *model* unchanged if *input_norm* is None.

        Args:
            model: inner model to wrap
            dls: DataLoaders with ``norm_stats`` attribute (populated automatically if missing)
            input_norm: scaler class for input normalization, or None to skip wrapping
            output_norm: scaler class for output denormalization, or None to skip
            autoregressive: if True, input stats are ``norm_u + norm_y`` and output
                stats use ``input_norm`` (AR models use the same scaler for both)
        """
        if input_norm is None:
            return model
        norm_u, norm_y = dls.norm_stats
        if autoregressive:
            in_scaler = input_norm.from_stats(norm_u + norm_y)
            out_scaler = input_norm.from_stats(norm_y)
        else:
            in_scaler = input_norm.from_stats(norm_u)
            out_scaler = output_norm.from_stats(norm_y) if output_norm is not None else None
        return cls(model, in_scaler, out_scaler)

    def forward(self, xb: torch.Tensor, **kwargs) -> torch.Tensor:
        xb = self.input_norm.normalize(xb)
        result = self.model(xb, **kwargs)
        if isinstance(result, tuple):
            out, state = result
            if self.output_norm is not None:
                out = self.output_norm.denormalize(out)
            return out, state
        if self.output_norm is not None:
            result = self.output_norm.denormalize(result)
        return result


def _unwrap_ddp(model):
    "Unwrap DistributedDataParallel/DataParallel wrappers."
    while hasattr(model, "module"):
        model = model.module
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    "Get the inner model, unwrapping DDP/DP and ScaledModel if present."
    model = _unwrap_ddp(model)
    return model.model if isinstance(model, ScaledModel) else model
