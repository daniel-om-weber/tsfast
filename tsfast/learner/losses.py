"""Loss functions and loss-function modifiers for time series training."""

__all__ = [
    "mse_nan",
    "ignore_nan",
    "float64_func",
    "SkipNLoss",
    "CutLoss",
    "NormLoss",
    "weighted_mae",
    "RandSeqLenLoss",
    "fun_rmse",
    "cos_sim_loss",
    "cos_sim_loss_pow",
    "nrmse",
    "nrmse_std",
    "mean_vaf",
    "zero_loss",
]

from ..data import *
from fastai.basics import *
import warnings

import functools
from collections.abc import Callable


def ignore_nan(func: Callable) -> Callable:
    """Decorator that removes NaN values from tensors before function execution.

    Reduces tensors to a flat array. Apply to functions such as mse.

    Args:
        func: loss function to wrap
    """

    @functools.wraps(func)
    def ignore_nan_decorator(*args, **kwargs):
        #         mask = ~torch.isnan(args[-1]) #nan mask of target tensor
        #         args = tuple([x[mask] for x in args]) #remove nan values
        mask = ~torch.isnan(args[-1][..., -1])  # nan mask of target tensor
        args = tuple([x[mask, :] for x in args])  # remove nan values
        return func(*args, **kwargs)

    return ignore_nan_decorator


mse_nan = ignore_nan(mse)

import functools
import warnings


def float64_func(func: Callable) -> Callable:
    """Decorator that computes a function in float64 and converts the result back.

    Args:
        func: function to wrap with float64 promotion
    """

    @functools.wraps(func)
    def float64_func_decorator(*args, **kwargs):
        typ = args[0].dtype
        try:
            # Try to use float64 for higher precision
            args = tuple([x.double() if issubclass(type(x), Tensor) else x for x in args])
            return func(*args, **kwargs).type(typ)
        except TypeError as e:
            # If float64 is not supported on this device, warn the user and fall back to float32
            if "doesn't support float64" in str(e):
                warnings.warn(
                    f"Float64 precision not supported on {args[0].device} device. Using original precision. This may reduce numerical accuracy. Error: {e}"
                )
                return func(*args, **kwargs)
            else:
                raise  # Re-raise if it's some other error

    return float64_func_decorator


def SkipNLoss(fn: Callable, n_skip: int = 0) -> Callable:
    """Loss-function modifier that skips the first n time steps of sequential data.

    Args:
        fn: base loss function to wrap
        n_skip: number of initial time steps to discard
    """

    @functools.wraps(fn)
    def _inner(input, target):
        return fn(input[:, n_skip:].contiguous(), target[:, n_skip:].contiguous())

    return _inner


def CutLoss(fn: Callable, l_cut: int = 0, r_cut: int | None = None) -> Callable:
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


def NormLoss(fn: Callable, norm_stats, scaler_cls: type | None = None) -> Callable:
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
    if device.type == "mps":
        # Compute on CPU because MPS does not support logspace yet
        weights = torch.logspace(
            start=torch.log10(torch.tensor(max_weight)),
            end=torch.log10(torch.tensor(min_weight)),
            steps=seq_len,
            device="cpu",
        ).to(device)
        warnings.warn(
            f"torch.logspace not supported on {device} device. Using cpu. This may reduce numerical performance"
        )
    else:
        # Compute directly on the target device
        weights = torch.logspace(
            start=torch.log10(torch.tensor(max_weight)),
            end=torch.log10(torch.tensor(min_weight)),
            steps=seq_len,
            device=device,
        )

    weights = (weights / weights.sum())[None, :, None]

    return ((input - target).abs() * weights).sum(dim=1).mean()


def RandSeqLenLoss(fn: Callable, min_idx: int = 1, max_idx: int | None = None, mid_idx: int | None = None) -> Callable:
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
        if "max_idx" not in locals():
            max_idx = seq_len
        if "mid_idx" not in locals():
            mid_idx = min_idx  # +(max_idx-min_idx)//4
        # len_list = torch.randint(min_idx,max_idx,(bs,))
        len_list = np.random.triangular(min_idx, mid_idx, max_idx, (bs,)).astype(int)
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
    mse = (inp - targ).pow(2).mean(dim=[0, 1])
    var = targ.var(dim=[0, 1])
    return (mse / var).sqrt().mean()


def nrmse_std(inp: Tensor, targ: Tensor) -> Tensor:
    """RMSE loss normalized by standard deviation of each target variable."""
    mse = (inp - targ).pow(2).mean(dim=[0, 1])
    var = targ.std(dim=[0, 1])
    return (mse / var).sqrt().mean()


def mean_vaf(inp: Tensor, targ: Tensor) -> Tensor:
    """Variance accounted for (VAF) metric, returned as a percentage."""
    return (1 - ((targ - inp).var() / targ.var())) * 100


def zero_loss(pred: Tensor, targ: Tensor) -> Tensor:
    """Always-zero loss that preserves the computation graph."""
    return (pred * 0).sum()
