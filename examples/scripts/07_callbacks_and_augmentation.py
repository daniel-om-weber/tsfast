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
# # Example 07: Augmentations, Regularizers, and Training Options
#
# TSFast provides composable building blocks that customize training without
# modifying model code:
#
# - **Augmentations** modify training data on-the-fly (noise, bias, sequence
#   length variation)
# - **Auxiliary losses** add regularization terms to the main loss
#   (activation smoothing, gradient penalties)
# - **Training options** control optimizer behavior (gradient clipping)
#
# This example demonstrates each category and shows how to combine them.

# %% [markdown]
# ## Prerequisites
#
# - [Example 00: Your First Model](00_your_first_model.ipynb)
# - [Example 01: Understanding the Data Pipeline](01_data_pipeline.ipynb)
# - [Example 02: Simulation](02_simulation.ipynb)
# - [Benchmark RNN](../../benchmarks/benchmark_rnn.py)

# %% [markdown]
# ## Setup

# %%
from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.models.scaling import unwrap_model
from tsfast.training import (
    RNNLearner,
    fun_rmse,
    ActivationRegularizer,
    TemporalActivationRegularizer,
    noise, bias, vary_seq_len, truncate_sequence,
    plot_grad_flow,
)

# %% [markdown]
# ## Load the Dataset

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Data Augmentations
#
# Augmentations modify training data on-the-fly. They only apply during
# training (not validation or test), so your evaluation metrics stay
# comparable. Pass them as `augmentations=[...]` when creating the Learner.

# %% [markdown]
# ### noise
#
# Adds Gaussian noise to input signals. `std` controls the noise magnitude.

# %%
lrn_noisy = RNNLearner(
    dls, rnn_type='lstm', metrics=[fun_rmse],
    augmentations=[noise(std=0.05)],
)

# %% [markdown]
# ### bias
#
# Adds a constant offset per signal per sample. This simulates sensor drift or
# calibration errors, making the model more robust to such shifts.

# %%
lrn_bias = RNNLearner(
    dls, rnn_type='lstm', metrics=[fun_rmse],
    augmentations=[bias(std=0.05)],
)

# %% [markdown]
# ### Training with Augmentation
#
# Train two models -- one with augmentation, one without -- to see the effect
# on validation performance.

# %%
lrn_base = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_base.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"Without augmentation: {lrn_base.validate()}")

# %%
lrn_noisy.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"With noise augmentation: {lrn_noisy.validate()}")

# %% [markdown]
# ## Activation Regularization
#
# Two regularizers can be added to the loss:
#
# - **`ActivationRegularizer`**: L2 penalty on RNN activations -- prevents
#   activations from growing too large.
# - **`TemporalActivationRegularizer`**: L2 penalty on temporal differences of
#   activations -- encourages smooth predictions over time.
#
# `modules` specifies which model components to regularize (typically the RNN
# layers). Pass them as `aux_losses=[...]` when creating the Learner.

# %%
lrn_reg = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_reg.aux_losses.append(
    ActivationRegularizer(modules=[unwrap_model(lrn_reg.model).rnn], alpha=2.0)
)
lrn_reg.aux_losses.append(
    TemporalActivationRegularizer(modules=[unwrap_model(lrn_reg.model).rnn], beta=1.0)
)
lrn_reg.fit_flat_cos(n_epoch=5, lr=3e-3)
lrn_reg.show_results(max_n=2)

# %% [markdown]
# ## Gradient Clipping
#
# Clips the gradient norm during backpropagation. This prevents exploding
# gradients, which are common with RNNs on long sequences. Pass `grad_clip=`
# when creating the Learner.

# %%
lrn_clip = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse], grad_clip=10)
lrn_clip.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## Diagnosing Gradient Issues
#
# When training is unstable or the model isn't learning, visualizing gradients
# helps identify the problem. `plot_grad_flow` shows the gradient magnitude
# at each layer -- vanishing gradients appear as near-zero bars, exploding
# gradients as very tall bars.

# %%
lrn_clip.train_one_epoch()
plot_grad_flow(lrn_clip.model.named_parameters())

# %% [markdown]
# ## vary_seq_len
#
# Randomly truncates sequences to different lengths each batch. This acts as
# data augmentation by preventing the model from overfitting to a fixed window
# size. `min_len` sets the minimum allowed length.

# %%
lrn_vary = RNNLearner(
    dls, rnn_type='lstm', metrics=[fun_rmse],
    augmentations=[vary_seq_len(min_len=100)],
)
lrn_vary.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## truncate_sequence
#
# Progressively increases sequence length during training. Starts with short
# sequences (easier for the model) and gradually increases to full length.
# This is a form of curriculum learning that helps the model learn short-term
# dynamics first before tackling longer dependencies.

# %%
lrn_trunc = RNNLearner(
    dls, rnn_type='lstm', metrics=[fun_rmse],
    augmentations=[truncate_sequence(truncate_length=100)],
)
lrn_trunc.fit_flat_cos(n_epoch=10, lr=3e-3)

# %% [markdown]
# ## Combining Options
#
# Augmentations, auxiliary losses, and gradient clipping can be combined freely.
# Pass them all at Learner creation time.

# %%
lrn_combined = RNNLearner(
    dls, rnn_type='lstm', metrics=[fun_rmse],
    grad_clip=10,
)
lrn_combined.aux_losses.append(
    ActivationRegularizer(modules=[unwrap_model(lrn_combined.model).rnn], alpha=2.0)
)
lrn_combined.aux_losses.append(
    TemporalActivationRegularizer(modules=[unwrap_model(lrn_combined.model).rnn], beta=1.0)
)
lrn_combined.fit_flat_cos(n_epoch=10, lr=3e-3)
lrn_combined.show_results(max_n=2)

# %% [markdown]
# ## Key Takeaways
#
# - **`noise`** and **`bias`** augment training data for better
#   generalization. Pass them as `augmentations=[...]` on the Learner.
# - **`ActivationRegularizer`** and **`TemporalActivationRegularizer`** smooth
#   predictions with activation and temporal penalties. Pass them as
#   `aux_losses=[...]` or via `lrn.aux_losses.append(...)`.
# - **`grad_clip`** prevents exploding gradients on long sequences.
# - **`plot_grad_flow`** visualizes gradient magnitudes per layer -- use it to
#   diagnose vanishing or exploding gradients.
# - **`vary_seq_len`** acts as augmentation by varying sequence length each
#   batch.
# - **`truncate_sequence`** implements curriculum learning with progressive
#   sequence length.
# - All options compose -- combine augmentations, auxiliary losses, and
#   gradient clipping for best results.
