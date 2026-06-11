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
# # Example 17: Stateful Models and TBPTT
#
# Standard backpropagation through time (BPTT) requires storing intermediate
# activations for every timestep in a sequence. For very long sequences this
# quickly exhausts GPU memory. Truncated Backpropagation Through Time (TBPTT)
# solves this by splitting long sequences into manageable sub-windows while
# carrying the RNN hidden state across them, enabling training on arbitrarily
# long sequences with bounded memory.

# %% [markdown]
# ## Prerequisites
#
# This example builds on concepts from Examples 00-04. Make sure you have
# completed those first.

# %% [markdown]
# ## Setup

# %%
from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.training import RNNLearner, fun_rmse

# %% [markdown]
# ## The Memory Problem
#
# When training RNNs, backpropagation stores intermediate activations for every
# timestep. A sequence of length 10,000 requires roughly 100x more memory than
# a sequence of length 100. For very long sequences this exceeds GPU memory.
#
# TBPTT solves this by:
#
# 1. **Splitting** the long sequence into sub-windows (e.g., 100 timesteps each)
# 2. **Running** the forward pass on one sub-window at a time
# 3. **Computing gradients** only within each sub-window (truncated)
# 4. **Carrying the hidden state** from the end of one sub-window to the start
#    of the next
#
# This means the model still "sees" the full sequence through its hidden state,
# but memory usage is bounded by `sub_seq_len` rather than the full sequence
# length.

# %% [markdown]
# ## Standard Training (Baseline)
#
# First, train normally with a moderately large window to establish a reference
# point. The full 500-step sequence is backpropagated through in one pass.

# %%
dls_standard = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

lrn_standard = RNNLearner(dls_standard, rnn_type='lstm', hidden_size=40, metrics=[fun_rmse])
lrn_standard.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_standard.show_results(max_n=2)

# %%
print(f"Standard: {lrn_standard.validate()}")

# %% [markdown]
# ## TBPTT Training
#
# The DataLoaders for TBPTT are identical to the baseline -- they yield full
# 500-step windows. The splitting into sub-windows happens inside the learner,
# so `sub_seq_len` is a learner parameter, not a DataLoader parameter.

# %%
dls_tbptt = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Stateful Model
#
# Create a stateful RNN that maintains hidden state across sub-windows.
#
# - **`sub_seq_len=100`** -- enables TBPTT: each 500-step window is split into
#   5 sub-windows of 100 timesteps. The RNN does **not** reset its hidden
#   state between sub-windows. Instead, the state from the previous sub-window
#   initializes the next one, allowing information to flow across sub-window
#   boundaries. The window size `win_sz` must be divisible by `sub_seq_len`.
#
# When `sub_seq_len` is set, `RNNLearner` automatically uses `TbpttLearner`
# and a stateful RNN. The hidden state is carried across the sub-windows of
# one window and reset between batches, so the state from one training sample
# never bleeds into the next, unrelated sample.

# %%
lrn_tbptt = RNNLearner(
    dls_tbptt, rnn_type='lstm', hidden_size=40,
    sub_seq_len=100, metrics=[fun_rmse],
)

# %% [markdown]
# ## Train with TBPTT
#
# Training proceeds exactly like standard training. Under the hood, the
# learner splits each batch into sub-windows, runs forward/backward on each
# sub-window in turn, and carries the detached hidden state from one
# sub-window to the next.

# %%
lrn_tbptt.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_tbptt.show_results(max_n=2)

# %%
print(f"TBPTT: {lrn_tbptt.validate()}")

# %% [markdown]
# ## Comparison
#
# Compare the final validation metrics for both approaches.

# %%
print(f"Standard (full BPTT):  {lrn_standard.validate()}")
print(f"Stateful (TBPTT):      {lrn_tbptt.validate()}")

# %% [markdown]
# TBPTT may have slightly different loss because gradients are truncated at
# sub-window boundaries. However, performance should be comparable. The key
# advantage is **memory efficiency** -- TBPTT can handle sequences that would
# cause out-of-memory errors with standard training.

# %% [markdown]
# ## When to Use TBPTT
#
# TBPTT is most useful when:
#
# - **Sequences are very long** (thousands of timesteps or more) and
#   full backpropagation would exhaust GPU memory.
# - **GPU memory is limited** and you need to keep memory usage bounded.
# - **The system has long-range dependencies** that benefit from a large
#   `win_sz`, but you cannot afford to backpropagate through the entire window.
#
# For short sequences (under ~1000 timesteps), standard training is simpler and
# usually sufficient. The overhead of managing sub-windows and stateful hidden
# state is not worth the complexity for sequences that already fit in memory.

# %% [markdown]
# ## Key Takeaways
#
# - **TBPTT splits long windows into sub-windows** inside the learner; the
#   DataLoaders are the same as for standard training.
# - **`sub_seq_len`** in `RNNLearner` enables TBPTT and makes the RNN carry its
#   hidden state across sub-windows instead of resetting to zero each time.
# - **Hidden state is reset between batches** so different training samples do
#   not bleed into each other.
# - **Gradients are truncated** to `sub_seq_len` timesteps, bounding memory
#   usage regardless of the full sequence length.
# - **Hidden state spans the full sequence**, preserving long-range information
#   even though gradients are truncated.
# - **Use TBPTT when sequences are too long** for standard backpropagation to
#   fit in GPU memory.
