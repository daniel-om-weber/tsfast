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
# # Example 02: Simulation -- Training on Multiple Datasets
#
# Simulation is the most common mode in system identification: the model predicts
# output y(t) from input u(t) alone, with no access to past measured outputs. This
# example trains simulation models on benchmark datasets and introduces
# `InferenceWrapper` for numpy-based inference.

# %% [markdown]
# ## Prerequisites
#
# This notebook builds on concepts from Examples 00 and 01. Make sure you are
# familiar with creating DataLoaders and training a basic model before proceeding.

# %% [markdown]
# ## Setup

# %%
from tsfast.tsdata.benchmark import create_dls_silverbox, create_dls_wh
from tsfast.inference import InferenceWrapper
from tsfast.training import RNNLearner, fun_rmse

# %% [markdown]
# ## What is Simulation?
#
# In simulation mode, the model sees **only the input signal** u(t) and must predict
# the output y(t). The model has no access to measured outputs -- it must simulate
# the system's behavior purely from the input.
#
# This is the simplest and most common mode for system identification. Think of it
# as a black-box model that takes a control signal and predicts what the system will
# do, without ever "peeking" at the real measurements during inference.

# %% [markdown]
# ## Load the Silverbox Dataset
#
# The Silverbox is a standard benchmark in system identification. It is an
# electronic circuit that mimics a nonlinear mass-spring-damper system.
#
# - `bs=16`: batch size of 16 windows per training step
# - `win_sz=500`: each training window is 500 timesteps long
# - `stp_sz=10`: consecutive windows are offset by 10 timesteps (overlapping windows)

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Train an LSTM with n_skip
#
# RNNs start with a zero hidden state, so the first N predictions are unreliable
# because the network hasn't "warmed up" yet. The `n_skip` parameter excludes the
# first N timesteps from the loss computation, so the model isn't penalized for the
# transient warmup period.
#
# Key parameters:
#
# - `rnn_type='lstm'`: use an LSTM cell (alternatives: `'gru'`, `'rnn'`)
# - `n_skip=50`: exclude the first 50 timesteps from the loss
# - `hidden_size=40`: 40 hidden units in the LSTM layer
# - `metrics=[fun_rmse]`: track root mean squared error during training

# %%
lrn = RNNLearner(dls, rnn_type='lstm', n_skip=50, hidden_size=40, metrics=[fun_rmse])
lrn.show_batch(max_n=4)

# %%
lrn.fit_flat_cos(n_epoch=10, lr=3e-3)

# %% [markdown]
# ## Visualize Results
#
# `show_results` overlays the model's predictions against the true output on
# validation windows. The model has never seen these windows during training.

# %%
lrn.show_results(max_n=3)

# %% [markdown]
# ## Evaluating on the Validation Set
#
# `validate()` runs the model on the validation set and returns a tuple of
# `(loss, {metric_name: value})`. You can pass a different DataLoader via
# `dl=` to evaluate on other splits (e.g., `lrn.validate(dl=dls.test)`).

# %%
val_loss, val_metrics = lrn.validate()
print(f"Validation loss: {val_loss}")
print(f"Validation metrics: {val_metrics}")

# %% [markdown]
# ## Getting Predictions
#
# `get_preds` returns a tuple of `(predictions, targets)` as tensors. This is
# useful for custom analysis, plotting, or computing metrics that aren't built
# into tsfast.

# %%
preds, targs = lrn.get_preds()
print(f"Predictions shape: {preds.shape}")
print(f"Targets shape: {targs.shape}")

# %% [markdown]
# ## Training on a Different Dataset
#
# The same workflow applies to any benchmark dataset. Here we train on the
# Wiener-Hammerstein benchmark, which models a different nonlinear dynamic system.
# The only change is the DataLoader factory function -- the model architecture
# and training loop are identical.

# %%
dls_wh = create_dls_wh()
lrn_wh = RNNLearner(dls_wh, rnn_type='lstm', n_skip=50, hidden_size=40, metrics=[fun_rmse])
lrn_wh.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_wh.show_results(max_n=3)

# %% [markdown]
# ## Using Your Model: InferenceWrapper
#
# After training, you often want to run inference with numpy arrays -- for example,
# in a deployment pipeline or when integrating with scipy/control toolboxes.
#
# `InferenceWrapper` handles the full pipeline automatically:
#
# 1. Converts numpy input to a PyTorch tensor
# 2. Applies the same input normalization used during training
# 3. Runs the model forward pass
# 4. Converts the output back to a numpy array

# %%
wrapper = InferenceWrapper(lrn)

xb, yb = dls.valid.one_batch()
np_input = xb.cpu().numpy()

y_pred = wrapper.inference(np_input)
print(f"Input shape:  {np_input.shape}")
print(f"Output shape: {y_pred.shape}")

# %% [markdown]
# ## Key Takeaways
#
# - **Simulation models predict output from input alone** (no output feedback).
#   The model must learn the full system dynamics from the excitation signal u(t).
# - **`n_skip` handles the RNN warmup transient** by excluding early timesteps from
#   the loss, so the model isn't penalized while its hidden state initializes.
# - **Pass `dl=` to evaluate on a specific split**: e.g. `dl=lrn.dls.test`
#   for the test set (defaults to validation).
# - **`InferenceWrapper` provides numpy-in / numpy-out inference** with automatic
#   normalization, making it easy to use trained models outside of the training loop.
