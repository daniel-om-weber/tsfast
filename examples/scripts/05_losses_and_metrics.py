# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 05: Loss Functions and Metrics
#
# The choice of loss function and evaluation metrics significantly affects
# training behavior and how you assess model quality. TSFast provides
# specialized losses and metrics for time series system identification:
#
# - **Loss modifiers** that wrap any base loss to skip transients, slice
#   windows, or normalize scales
# - **Evaluation metrics** standard in the system identification literature
#
# This example walks through each one and explains when to use it.

# %% [markdown]
# ## Prerequisites
#
# This example builds on concepts from Examples 00-02. Make sure you have
# completed those first.

# %% [markdown]
# ## Setup

# %%
from torch import nn

from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.training import (
    RNNLearner,
    fun_rmse, nrmse, nrmse_std, mean_vaf,
    weighted_mae, norm_loss, cut_loss,
)

# %% [markdown]
# ## Load Dataset
#
# We use the Silverbox benchmark throughout this example. Each model trains for
# only 5 epochs to keep things fast -- the focus here is on the loss and metric
# behavior, not on achieving the best possible fit.

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## The Default: MAE Loss
#
# TSFast uses `nn.L1Loss()` (Mean Absolute Error) as the default training loss.
# MAE is more robust to outliers and measurement spikes than MSE (Mean Squared
# Error). MSE heavily penalizes large errors, which can cause the model to
# overfit to noisy data points. For system identification, where measurement
# noise and occasional spikes are common, MAE provides a more stable training
# signal.

# %%
lrn_mae = RNNLearner(dls, rnn_type='lstm', loss_func=nn.L1Loss(), metrics=[fun_rmse])
lrn_mae.show_batch(max_n=2)

# %%
lrn_mae.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## Evaluation Metrics
#
# While you train with a loss function, you evaluate with metrics. TSFast
# provides several standard metrics for system identification:
#
# - **`fun_rmse`** -- Root Mean Square Error. The standard reporting metric in
#   many fields. Penalizes large errors more than MAE because of the squaring.
#
# - **`nrmse`** -- RMSE normalized by the variance of each target variable.
#   This allows fair comparison across outputs with different scales. A value
#   of 0 means perfect prediction; a value of 1 means the model is no better
#   than predicting the mean.
#
# - **`nrmse_std`** -- RMSE normalized by the standard deviation of each target
#   variable. Similar to `nrmse` but uses std instead of variance for the
#   denominator.
#
# - **`mean_vaf`** -- Variance Accounted For, expressed as a percentage. Measures
#   what fraction of the target signal's variance is explained by the model.
#   100% means perfect prediction. This metric is widely used in the system
#   identification literature.

# %%
lrn = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse, nrmse, nrmse_std, mean_vaf])
lrn.fit_flat_cos(n_epoch=5, lr=3e-3)

# %%
lrn.show_results(max_n=2)

# %% [markdown]
# ## n_skip: Ignoring Transient Warmup
#
# RNNs start from a zero hidden state. During the first few timesteps, the
# hidden state is "warming up" and predictions are unreliable. Setting
# `n_skip` on the Learner discards the first N timesteps from both loss
# and metric computation. This prevents the optimizer from wasting effort on
# the unavoidable warmup transient.
#
# - **`n_skip=50`** -- skip the first 50 timesteps when computing the loss

# %%
lrn_skip = RNNLearner(dls, rnn_type='lstm', n_skip=50, metrics=[fun_rmse])
lrn_skip.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## cut_loss: Evaluating a Window
#
# `cut_loss` slices the sequence to a specific range before computing the loss.
# This is useful when you only care about predictions in a particular part of
# the sequence.
#
# - **`l_cut=50`** -- trim 50 timesteps from the left (start of the sequence)
# - **`r_cut=450`** -- keep up to timestep 450 (trim from the right)
#
# This evaluates only timesteps 50 through 450 of each 500-step window.

# %%
my_cut_loss = cut_loss(nn.L1Loss(), l_cut=50, r_cut=450)
lrn_cut = RNNLearner(dls, rnn_type='lstm', loss_func=my_cut_loss, metrics=[fun_rmse])
lrn_cut.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## norm_loss: Scale-Invariant Training
#
# When your system has multiple outputs with very different magnitudes (e.g.,
# position in meters and velocity in m/s), the loss is dominated by the
# largest-scale output. `norm_loss` normalizes both predictions and targets
# before computing the loss, so all outputs contribute equally regardless of
# their physical scale.
#
# `norm_loss` takes the output normalization statistics from the DataLoaders
# (`dls.norm_stats.y`) and uses them to normalize both prediction and target
# tensors before passing them to the base loss function.

# %%
my_norm_loss = norm_loss(nn.L1Loss(), dls.norm_stats.y)
lrn_norm = RNNLearner(dls, rnn_type='lstm', loss_func=my_norm_loss, metrics=[fun_rmse])
lrn_norm.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## Weighted MAE
#
# `weighted_mae` applies log-spaced weights along the time axis, giving higher
# weight to earlier timesteps and lower weight to later ones. This is useful
# when early dynamics matter more than steady-state behavior, for example when
# modeling transient responses or step responses where the initial trajectory is
# most informative.

# %%
lrn_wmae = RNNLearner(dls, rnn_type='lstm', loss_func=weighted_mae, metrics=[fun_rmse])
lrn_wmae.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## Custom Learning Rate Schedules
#
# All examples so far use `fit_flat_cos`, which keeps the learning rate flat
# then cosine-decays it to zero. For more control, call `fit()` directly with
# a `scheduler_fn`. TSFast provides several schedule functions:
#
# - **`sched_flat_cos`** -- flat then cosine decay (used by `fit_flat_cos`)
# - **`sched_ramp`** -- constant at `start`, linear ramp to `end`, then constant
# - **`sched_lin_p`** -- linear decay from `start` to `end`, reaching `end` by
#   position `p`
#
# `scheduler_fn` is a factory `(optimizer, total_steps) -> scheduler` that
# creates a PyTorch LR scheduler. The schedule functions return a multiplier
# for a given position in [0, 1].

# %%
from torch.optim.lr_scheduler import LambdaLR
from tsfast.training import sched_ramp

lrn_sched = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_sched.fit(
    n_epoch=5,
    lr=3e-3,
    scheduler_fn=lambda opt, steps: LambdaLR(
        opt, lambda s: sched_ramp(start=1.0, end=0.01, pos=s / steps)
    ),
)

# %% [markdown]
# ### Using PyTorch Built-in Schedulers
#
# Any `torch.optim.lr_scheduler` works as a `scheduler_fn`. For example,
# `OneCycleLR` implements the 1cycle policy (warmup then cosine decay) and
# controls the learning rate entirely via `max_lr` -- the `lr` passed to
# `fit()` sets the optimizer baseline but `OneCycleLR` overrides it
# immediately.

# %%
from torch.optim.lr_scheduler import OneCycleLR

lrn_onecycle = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_onecycle.fit(
    n_epoch=5,
    lr=3e-3,
    scheduler_fn=lambda opt, steps: OneCycleLR(opt, max_lr=3e-3, total_steps=steps),
)

# %% [markdown]
# ## Key Takeaways
#
# - **MAE (default)** is robust to outliers -- a good default for system
#   identification where measurement noise and spikes are common.
# - **`fun_rmse`**, **`nrmse`**, and **`mean_vaf`** are standard evaluation
#   metrics. `nrmse` enables fair comparison across different-scale outputs,
#   and `mean_vaf` reports the percentage of variance explained.
# - **`n_skip`** excludes the RNN warmup transient from the loss, preventing
#   the optimizer from fitting the unavoidable zero-state startup.
# - **`cut_loss`** restricts the loss to a specific time window, useful when
#   only part of the sequence matters.
# - **`norm_loss`** enables scale-invariant training for multi-output systems by
#   computing the loss in normalized space.
# - **`weighted_mae`** emphasizes early timesteps, useful for transient-response
#   modeling.
# - **`fit()` with `scheduler_fn`** gives full control over the learning rate
#   schedule. Use `sched_ramp` or `sched_lin_p` for custom warmup/decay
#   profiles, or pass any PyTorch built-in scheduler like `OneCycleLR`.
# - Choose metrics that match your evaluation requirements -- different
#   applications call for different measures of model quality.
