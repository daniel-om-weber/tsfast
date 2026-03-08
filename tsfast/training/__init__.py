"""Lightweight pure-PyTorch training framework."""

from .learner import CudaGraphTbpttLearner, Learner, Recorder, SimpleCudaGraphLearner, TbpttLearner
from .losses import (
    cos_sim_loss,
    cos_sim_loss_pow,
    cut_loss,
    float64_func,
    fun_rmse,
    ignore_nan,
    mean_vaf,
    mse,
    mse_nan,
    nan_mean,
    norm_loss,
    nrmse,
    nrmse_std,
    rand_seq_len_loss,
    weighted_mae,
    zero_loss,
)
from .aux_losses import (
    ActivationRegularizer,
    AuxiliaryLoss,
    FranSysRegularizer,
    TemporalActivationRegularizer,
)
from .schedulers import sched_lin_p, sched_ramp
from .transforms import (
    bias,
    noise,
    noise_grouped,
    noise_varying,
    prediction_concat,
    truncate_sequence,
    vary_seq_len,
)
from .profiling import DataProfiler, benchmark_dataloaders
from .viz import grad_norm, layout_samples, plot_grad_flow, plot_sequence

__all__ = [
    # core
    "Learner",
    "TbpttLearner",
    "CudaGraphTbpttLearner",
    "SimpleCudaGraphLearner",
    "Recorder",
    # losses & metrics
    "nan_mean",
    "mse",
    "mse_nan",
    "ignore_nan",
    "float64_func",
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
    # schedulers
    "sched_lin_p",
    "sched_ramp",
    # aux losses
    "AuxiliaryLoss",
    "ActivationRegularizer",
    "TemporalActivationRegularizer",
    "FranSysRegularizer",
    # transforms
    "prediction_concat",
    "noise",
    "noise_varying",
    "noise_grouped",
    "bias",
    "vary_seq_len",
    "truncate_sequence",
    # profiling
    "DataProfiler",
    "benchmark_dataloaders",
    # viz
    "plot_sequence",
    "plot_grad_flow",
    "grad_norm",
    "layout_samples",
]
