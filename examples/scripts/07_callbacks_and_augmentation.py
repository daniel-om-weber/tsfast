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
# # Example 07: Training Callbacks and Data Augmentation
#
# Callbacks let you customize every aspect of the training loop without
# modifying model code. Data augmentation transforms add noise or bias to
# training data, improving generalization. This example demonstrates both.

# %% [markdown]
# ## Prerequisites
#
# - [Example 00: Your First Model](00_your_first_model.ipynb)
# - [Example 01: Understanding the Data Pipeline](01_data_pipeline.ipynb)
# - [Example 02: Simulation](02_simulation.ipynb)
# - [Example 04: Benchmark RNN](04_benchmark_rnn.py)

# %% [markdown]
# ## Setup

# %%
from tsfast.datasets.benchmark import create_dls_silverbox
from tsfast.models.rnn import RNNLearner
from tsfast.learner.callbacks import (
    TimeSeriesRegularizer, GradientClipping, VarySeqLen,
    BatchLossFilter, CB_TruncateSequence,
)
from tsfast.data.transforms import SeqNoiseInjection, SeqBiasInjection
from tsfast.learner.losses import fun_rmse

# %% [markdown]
# ## Load the Dataset

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
dls.show_batch(max_n=2)

# %% [markdown]
# ## Data Augmentation Transforms
#
# Transforms modify training data on-the-fly. They only apply during training
# (not validation or test), so your evaluation metrics stay comparable.

# %% [markdown]
# ### SeqNoiseInjection
#
# Adds Gaussian noise to input signals. `std` controls the noise magnitude and
# `p` is the probability of applying the transform per batch.

# %%
dls_noisy = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
dls_noisy.train.after_batch.add(SeqNoiseInjection(std=0.05, p=1.0))
dls_noisy.show_batch(max_n=2)

# %% [markdown]
# Compare this to the clean batch above -- you should see slight noise on the
# input signal.

# %% [markdown]
# ### SeqBiasInjection
#
# Adds a constant offset per signal per sample. This simulates sensor drift or
# calibration errors, making the model more robust to such shifts.

# %%
dls_biased = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
dls_biased.train.after_batch.add(SeqBiasInjection(std=0.1, p=1.0))

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
lrn_aug = RNNLearner(dls_noisy, rnn_type='lstm', metrics=[fun_rmse])
lrn_aug.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"With noise augmentation: {lrn_aug.validate()}")

# %% [markdown]
# ## TimeSeriesRegularizer
#
# Adds two regularization terms to the loss:
#
# - **`alpha`**: L2 penalty on RNN activations -- prevents activations from
#   growing too large.
# - **`beta`**: L2 penalty on temporal differences of activations -- encourages
#   smooth predictions over time.
#
# `modules` specifies which model components to regularize (typically the RNN
# layers).

# %%
lrn_reg = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_reg.fit_flat_cos(n_epoch=5, lr=3e-3, cbs=[
    TimeSeriesRegularizer(alpha=2.0, beta=1.0)
])
lrn_reg.show_results(max_n=2)

# %% [markdown]
# ## GradientClipping
#
# Clips the gradient norm during backpropagation. This prevents exploding
# gradients, which are common with RNNs on long sequences. `clip_val` is the
# maximum allowed gradient norm.

# %%
lrn_clip = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_clip.fit_flat_cos(n_epoch=5, lr=3e-3, cbs=[GradientClipping(clip_val=10)])

# %% [markdown]
# ## VarySeqLen
#
# Randomly truncates sequences to different lengths each batch. This acts as
# data augmentation by preventing the model from overfitting to a fixed window
# size. `min_len` sets the minimum allowed length.

# %%
lrn_vary = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_vary.fit_flat_cos(n_epoch=5, lr=3e-3, cbs=[VarySeqLen(min_len=100)])

# %% [markdown]
# ## BatchLossFilter
#
# Keeps only the hardest batches (those with the highest loss) for gradient
# updates. `loss_perc=0.5` means only the top 50% of samples by loss
# contribute to learning -- a form of curriculum learning that focuses on the
# most informative examples.

# %%
lrn_filter = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_filter.fit_flat_cos(n_epoch=5, lr=3e-3, cbs=[BatchLossFilter(loss_perc=0.5)])

# %% [markdown]
# ## CB_TruncateSequence
#
# Progressively increases sequence length during training. Starts with short
# sequences (easier for the model) and gradually increases to full length.
# This is a form of curriculum learning that helps the model learn short-term
# dynamics first before tackling longer dependencies.

# %%
lrn_trunc = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_trunc.fit_flat_cos(n_epoch=10, lr=3e-3, cbs=[CB_TruncateSequence(truncate_length=100)])

# %% [markdown]
# ## Combining Callbacks
#
# Multiple callbacks can be combined. They execute in order during each
# training step, so you can layer regularization, gradient control, and
# curriculum strategies together.

# %%
lrn_combined = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_combined.fit_flat_cos(n_epoch=10, lr=3e-3, cbs=[
    TimeSeriesRegularizer(alpha=2.0, beta=1.0),
    GradientClipping(clip_val=10),
])
lrn_combined.show_results(max_n=2)

# %% [markdown]
# ## Key Takeaways
#
# - **`SeqNoiseInjection`** and **`SeqBiasInjection`** augment training data
#   for better generalization.
# - **`TimeSeriesRegularizer`** smooths predictions with activation and
#   temporal penalties.
# - **`GradientClipping`** prevents exploding gradients on long sequences.
# - **`VarySeqLen`** acts as augmentation by varying sequence length each
#   batch.
# - **`BatchLossFilter`** focuses learning on the hardest examples.
# - **`CB_TruncateSequence`** implements curriculum learning with progressive
#   sequence length.
# - Callbacks compose -- combine multiple for best results.
