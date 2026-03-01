"""Lightweight pure-PyTorch training framework."""

from .core import Learner, Recorder, TbpttLearner
from .losses import (
    CutLoss,
    FranSysRegularizer,
    NormLoss,
    RandSeqLenLoss,
    SkipNLoss,
    TimeSeriesRegularizerLoss,
    add_loss,
    consistency_loss,
    cos_sim_loss,
    cos_sim_loss_pow,
    float64_func,
    fun_rmse,
    ignore_nan,
    mean_vaf,
    mse,
    mse_nan,
    nrmse,
    nrmse_std,
    physics_loss,
    sched_lin_p,
    sched_ramp,
    transition_smoothness,
    weighted_mae,
    zero_loss,
)
from .transforms import (
    ar_init,
    bias,
    noise,
    noise_grouped,
    noise_varying,
    prediction_concat,
    truncate_sequence,
    vary_seq_len,
)
from .viz import grad_norm, layout_samples, plot_grad_flow, plot_sequence

__all__ = [
    # core
    "Learner",
    "TbpttLearner",
    "Recorder",
    # losses & metrics
    "mse",
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
    # schedulers
    "sched_lin_p",
    "sched_ramp",
    # aux losses
    "add_loss",
    "physics_loss",
    "transition_smoothness",
    "TimeSeriesRegularizerLoss",
    "FranSysRegularizer",
    "consistency_loss",
    # transforms
    "prediction_concat",
    "ar_init",
    "noise",
    "noise_varying",
    "noise_grouped",
    "bias",
    "vary_seq_len",
    "truncate_sequence",
    # viz
    "plot_sequence",
    "plot_grad_flow",
    "grad_norm",
    "layout_samples",
]
