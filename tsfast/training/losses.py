"""Loss functions and metrics for training."""

__all__ = [
    "mse",
    "mse_nan",
    "ignore_nan",
    "float64_func",
    "skip_n_loss",
    "cut_loss",
    "norm_loss",
    "weighted_mae",
    "rand_seq_len_loss",
    "fun_rmse",
    "cos_sim_loss",
    "cos_sim_loss_pow",
    "nrmse",
    "nrmse_std",
    "mean_vaf",
    "zero_loss",
]

import functools
import warnings
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
#  Pure loss functions and metrics
# ──────────────────────────────────────────────────────────────────────────────


def mse(inp: Tensor, targ: Tensor) -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(inp, targ)


def ignore_nan(func: Callable) -> Callable:
    """Decorator that removes NaN samples from (inp, targ) before computing a loss.

    A sample is removed if any feature in the target is NaN.
    Reduces tensors to a flat array.

    Args:
        func: loss function with signature (inp, targ) -> Tensor
    """

    @functools.wraps(func)
    def wrapper(inp: Tensor, targ: Tensor) -> Tensor:
        mask = ~torch.isnan(targ).any(dim=-1)
        return func(inp[mask], targ[mask])

    return wrapper


mse_nan = ignore_nan(mse)


def float64_func(func: Callable) -> Callable:
    """Decorator that computes a function in float64 and converts the result back.

    Args:
        func: function to wrap with float64 promotion
    """

    @functools.wraps(func)
    def float64_func_decorator(*args, **kwargs):
        typ = args[0].dtype
        try:
            args = tuple([x.double() if isinstance(x, Tensor) else x for x in args])
            return func(*args, **kwargs).type(typ)
        except TypeError as e:
            if "doesn't support float64" in str(e):
                warnings.warn(
                    f"Float64 precision not supported on {args[0].device} device. Using original precision. This may reduce numerical accuracy. Error: {e}"
                )
                return func(*args, **kwargs)
            else:
                raise

    return float64_func_decorator


def skip_n_loss(fn: Callable, n_skip: int = 0) -> Callable:
    """Loss-function modifier that skips the first n time steps of sequential data.

    Args:
        fn: base loss function to wrap
        n_skip: number of initial time steps to discard
    """

    @functools.wraps(fn)
    def _inner(input, target):
        return fn(input[:, n_skip:].contiguous(), target[:, n_skip:].contiguous())

    return _inner


def cut_loss(fn: Callable, l_cut: int = 0, r_cut: int | None = None) -> Callable:
    """Loss-function modifier that slices the sequence from l_cut to r_cut.

    Args:
        fn: base loss function to wrap
        l_cut: left index to start the slice
        r_cut: right index to end the slice (None keeps the rest)
    """

    @functools.wraps(fn)
    def _inner(input, target):
        return fn(input[:, l_cut:r_cut], target[:, l_cut:r_cut])

    return _inner


def norm_loss(fn: Callable, norm_stats, scaler_cls: type | None = None) -> Callable:
    """Loss wrapper that normalizes predictions and targets before computing loss.

    Args:
        fn: base loss function to wrap
        norm_stats: normalization statistics used to build the scaler
        scaler_cls: scaler class to use (defaults to StandardScaler1D)
    """
    from ..models.layers import StandardScaler1D

    if scaler_cls is None:
        scaler_cls = StandardScaler1D
    scaler = scaler_cls.from_stats(norm_stats)

    @functools.wraps(fn)
    def _inner(input, target):
        scaler.to(input.device)
        return fn(scaler.normalize(input), scaler.normalize(target))

    return _inner


def weighted_mae(input: Tensor, target: Tensor) -> Tensor:
    """Weighted MAE with log-spaced weights decaying along the sequence axis."""
    max_weight = 1.0
    min_weight = 0.1
    seq_len = input.shape[1]

    device = input.device
    compute_device = device
    if device.type == "mps":
        compute_device = "cpu"
        warnings.warn(
            f"torch.logspace not supported on {device} device. Using cpu. This may reduce numerical performance"
        )
    weights = torch.logspace(
        start=torch.log10(torch.tensor(max_weight)),
        end=torch.log10(torch.tensor(min_weight)),
        steps=seq_len,
        device=compute_device,
    ).to(device)

    weights = (weights / weights.sum())[None, :, None]

    return ((input - target).abs() * weights).sum(dim=1).mean()


def rand_seq_len_loss(
    fn: Callable, min_idx: int = 1, max_idx: int | None = None, mid_idx: int | None = None
) -> Callable:
    """Loss-function modifier that randomly truncates each sequence in the minibatch individually.

    Uses a triangular distribution. Slow for very large batch sizes.

    Args:
        fn: base loss function to wrap
        min_idx: minimum sequence length
        max_idx: maximum sequence length (defaults to full sequence)
        mid_idx: mode of the triangular distribution (defaults to min_idx)
    """

    @functools.wraps(fn)
    def _inner(input, target):
        bs, seq_len, _ = input.shape
        _max = max_idx if max_idx is not None else seq_len
        _mid = mid_idx if mid_idx is not None else min_idx
        len_list = np.random.triangular(min_idx, _mid, _max, (bs,)).astype(int)
        return torch.stack([fn(input[i, : len_list[i]], target[i, : len_list[i]]) for i in range(bs)]).mean()

    return _inner


def fun_rmse(inp: Tensor, targ: Tensor) -> Tensor:
    """RMSE loss function defined as a plain function."""
    return torch.sqrt(F.mse_loss(inp, targ))


def cos_sim_loss(inp: Tensor, targ: Tensor) -> Tensor:
    """Cosine similarity loss (1 - cosine similarity), averaged over the batch."""
    return (1 - F.cosine_similarity(inp, targ, dim=-1)).mean()


def cos_sim_loss_pow(inp: Tensor, targ: Tensor) -> Tensor:
    """Squared cosine similarity loss, averaged over the batch."""
    return (1 - F.cosine_similarity(inp, targ, dim=-1)).pow(2).mean()


def nrmse(inp: Tensor, targ: Tensor) -> Tensor:
    """RMSE loss normalized by variance of each target variable."""
    mse_val = (inp - targ).pow(2).mean(dim=[0, 1])
    var = targ.var(dim=[0, 1])
    return (mse_val / var).sqrt().mean()


def nrmse_std(inp: Tensor, targ: Tensor) -> Tensor:
    """RMSE loss normalized by standard deviation of each target variable."""
    mse_val = (inp - targ).pow(2).mean(dim=[0, 1])
    std = targ.std(dim=[0, 1])
    return (mse_val / std).sqrt().mean()


def mean_vaf(inp: Tensor, targ: Tensor) -> Tensor:
    """Variance accounted for (VAF) metric, returned as a percentage."""
    return (1 - ((targ - inp).var() / targ.var())) * 100


def zero_loss(pred: Tensor, targ: Tensor) -> Tensor:
    """Always-zero loss that preserves the computation graph."""
    return (pred * 0).sum()
