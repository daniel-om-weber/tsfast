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
# # Example 09: Autoregressive Models
#
# Autoregressive (AR) models feed their own predictions back as input, enabling
# free-running multi-step-ahead simulation. During training, they use "teacher
# forcing" (feeding true outputs), then switch to their own predictions at
# inference time.

# %% [markdown]
# ## Prerequisites
#
# This notebook builds on concepts from Examples 00-04. Make sure you are
# familiar with simulation (Example 02), prediction mode (Example 03), and
# model architectures (Example 04) before proceeding.

# %% [markdown]
# ## Setup

# %%
from tsfast.datasets.benchmark import create_dls_silverbox
from tsfast.models.rnn import AR_RNNLearner, RNNLearner
from tsfast.models.cnn import AR_TCNLearner
from tsfast.learner.losses import fun_rmse

# %% [markdown]
# ## What is Autoregressive Prediction?
#
# In standard simulation (Example 02), the model maps input u(t) to output y(t)
# in a single forward pass. In autoregressive mode:
#
# - **Training (teacher forcing)**: the model receives `[u(t), y_true(t-1)]` as
#   input. The `ARInitCB` callback concatenates the true target to the input
#   automatically.
# - **Inference (free-running)**: the model uses its own prediction
#   `[u(t), y_pred(t-1)]`. This tests whether the model is stable -- errors can
#   accumulate and cause divergence.
#
# AR models are more powerful for long-horizon prediction but require stronger
# regularization to stay stable.

# %% [markdown]
# ## Load the Dataset

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
dls.show_batch(max_n=2)

# %% [markdown]
# ## Standard Simulation Baseline
#
# Train a standard RNN for comparison. This model sees only the input u(t) and
# must predict y(t) without any output feedback. It serves as a baseline to
# highlight the difference autoregressive models make.

# %%
lrn_std = RNNLearner(dls, rnn_type='lstm', hidden_size=40, n_skip=50, metrics=[fun_rmse])
lrn_std.fit_flat_cos(n_epoch=10, lr=3e-3)
lrn_std.show_results(max_n=2)
print(f"Standard RNN: {lrn_std.validate()}")

# %% [markdown]
# ## Autoregressive RNN
#
# `AR_RNNLearner` wraps the model with autoregressive behavior and adds
# `TimeSeriesRegularizer` automatically. `alpha` and `beta` control activation
# and temporal regularization respectively -- AR models need these for stability.
#
# Key parameters:
#
# - **`rnn_type='lstm'`**: use LSTM cells for the recurrent layer.
# - **`hidden_size=40`**: 40 hidden units in the LSTM.
# - **`alpha=1.0`**: penalty weight for large activations (AR regularization).
# - **`beta=1.0`**: penalty weight for abrupt activation changes between
#   timesteps (TAR regularization).

# %%
lrn_ar = AR_RNNLearner(
    dls, rnn_type='lstm', hidden_size=40,
    alpha=1.0, beta=1.0, metrics=[fun_rmse]
)
lrn_ar.fit_flat_cos(n_epoch=10, lr=3e-3)
lrn_ar.show_results(max_n=2)
print(f"AR-RNN: {lrn_ar.validate()}")

# %% [markdown]
# ## Autoregressive TCN
#
# AR mode also works with temporal convolutional networks. `AR_TCNLearner`
# combines causal convolutions with autoregressive output feedback. The
# `hl_depth` parameter controls the number of TCN blocks (and therefore the
# receptive field, which is `2**hl_depth` timesteps).

# %%
lrn_ar_tcn = AR_TCNLearner(dls, hl_depth=4, metrics=[fun_rmse])
lrn_ar_tcn.fit_flat_cos(n_epoch=10, lr=3e-3)
lrn_ar_tcn.show_results(max_n=2)
print(f"AR-TCN: {lrn_ar_tcn.validate()}")

# %% [markdown]
# ## Stability and Regularization
#
# AR models can diverge during free-running inference if prediction errors
# accumulate. Regularization helps:
#
# - **`alpha`** penalizes large activations, keeping the model in a
#   well-behaved region.
# - **`beta`** penalizes abrupt changes in predictions, encouraging smoothness.
# - Higher `alpha` and `beta` improve stability but may reduce accuracy on
#   easy regions.
#
# Train with stronger regularization to demonstrate the trade-off:

# %%
lrn_ar_strong = AR_RNNLearner(
    dls, rnn_type='lstm', hidden_size=40,
    alpha=3.0, beta=3.0, metrics=[fun_rmse]
)
lrn_ar_strong.fit_flat_cos(n_epoch=10, lr=3e-3)
lrn_ar_strong.show_results(max_n=2)

# %% [markdown]
# ## Key Takeaways
#
# - AR models feed their own predictions back as input for multi-step-ahead
#   simulation.
# - Teacher forcing during training provides stable gradients; free-running at
#   inference tests stability.
# - `AR_RNNLearner` and `AR_TCNLearner` handle the autoregressive logic
#   automatically.
# - Regularization (`alpha`, `beta`) is essential to prevent error accumulation
#   and divergence.
# - Trade-off: stronger regularization leads to more stable but potentially
#   less accurate predictions.
