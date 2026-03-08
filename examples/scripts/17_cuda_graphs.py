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
# # Example 17: CUDA Graphs
#
# CUDA graphs record a sequence of GPU operations once, then replay
# them — eliminating CPU-side kernel launch overhead on every call.
# The speedup is largest for small, fast kernels where launch overhead
# dominates (typical for RNNs and lightweight CNNs).
#
# **Key constraint:** all input tensors must have the **same shape** on every
# call. Dynamic batch sizes or sequence lengths are not supported.
#
# This example shows how to apply CUDA graphs to:
#
# 1. **Stateless models** (TCN) — use `torch.cuda.make_graphed_callables` directly
# 2. **Stateful models** (SimpleRNN) — use `GraphedStatefulModel` which handles
#    state flattening/unflattening automatically

# %% [markdown]
# ## Prerequisites
#
# This example requires a CUDA GPU and builds on concepts from Examples 00 and 16.

# %% [markdown]
# ## Setup

# %%
import time

import torch

from tsfast.models.cnn import TCN
from tsfast.models.cudagraph import GraphedStatefulModel
from tsfast.models.rnn import SimpleRNN

B, T, N_IN, N_OUT = 32, 200, 3, 1


def bench(fn, n=200, warmup=50):
    """Time a function in milliseconds (GPU-synchronized)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

# %% [markdown]
# ## Stateless Model — TCN
#
# Stateless models take a single input tensor and return a single output tensor.
# `torch.cuda.make_graphed_callables` works directly — just provide the model
# and a sample input with the exact shape that will be used at runtime.

# %%
tcn = TCN(N_IN, N_OUT, hl_depth=4, hl_width=32).cuda()
sample_x = torch.randn(B, T, N_IN, device="cuda")

graphed_tcn = torch.cuda.make_graphed_callables(tcn, (sample_x,), num_warmup_iters=3)

# %%
# Verify outputs match
with torch.no_grad():
    y_eager = tcn(sample_x)
    y_graph = graphed_tcn(sample_x)

print(f"Max difference: {(y_eager - y_graph).abs().max().item():.2e}")
assert torch.allclose(y_eager, y_graph, atol=1e-5)

# %%
# Benchmark: eager vs CUDA-graphed inference
with torch.no_grad():
    t_eager = bench(lambda: tcn(sample_x))
    t_graph = bench(lambda: graphed_tcn(sample_x))

print(f"TCN eager:   {t_eager:.2f} ms")
print(f"TCN graphed: {t_graph:.2f} ms")
print(f"Speedup:     {t_eager / t_graph:.1f}x")

# %% [markdown]
# ## Stateful Model — SimpleRNN
#
# Stateful models return `(output, state)` where state is a nested structure of
# tensors (e.g. a list of hidden-state tensors for each RNN layer).
# `torch.cuda.make_graphed_callables` only accepts flat Tensor arguments, so it
# cannot be applied directly.
#
# `GraphedStatefulModel` solves this by:
#
# 1. Probing the model to discover its state structure (shapes, nesting)
# 2. Flattening nested state into a single `[B, D]` tensor before graph capture
# 3. Unflattening back to the original structure on output
#
# From the outside, the wrapped model has the same `forward(x, state=None)`
# interface as the original.

# %%
rnn = SimpleRNN(N_IN, N_OUT, num_layers=2, hidden_size=64, return_state=True).cuda()
graphed_rnn = GraphedStatefulModel(rnn)

# First call triggers graph capture
x = torch.randn(B, T, N_IN, device="cuda")
pred, state = graphed_rnn(x)
print(f"Output shape: {pred.shape}")
print(f"State: {len(state)} layers, each {state[0].shape}")

# State can be passed back in for continuation
pred2, state2 = graphed_rnn(x, state=state)

# %%
# Benchmark: eager vs CUDA-graphed inference
rnn_eager = SimpleRNN(N_IN, N_OUT, num_layers=2, hidden_size=64, return_state=True).cuda()

with torch.no_grad():
    t_eager = bench(lambda: rnn_eager(x, state=None))
    t_graph = bench(lambda: graphed_rnn(x, state=None))

print(f"RNN eager:   {t_eager:.2f} ms")
print(f"RNN graphed: {t_graph:.2f} ms")
print(f"Speedup:     {t_eager / t_graph:.1f}x")

# %% [markdown]
# ## Training Integration
#
# For training, `RNNLearner` handles CUDA graph wrapping automatically when
# you pass `cuda_graph=True` together with `sub_seq_len`:
#
# ```python
# from tsfast.training import RNNLearner
#
# lrn = RNNLearner(dls, sub_seq_len=100, cuda_graph=True)
# lrn.fit_flat_cos(n_epoch=10, lr=3e-3)
# ```
#
# Under the hood this wraps the model in `GraphedStatefulModel` before passing
# it to `TbpttLearner`. See Example 16 for TBPTT details and
# `examples/04_benchmark_rnn.py` for comprehensive training benchmarks.

# %% [markdown]
# ## Key Takeaways
#
# - **Stateless models** (TCN, CNN): use
#   `torch.cuda.make_graphed_callables(model, (sample_x,))` directly
# - **Stateful models** (RNN, LSTM, GRU): use `GraphedStatefulModel(model)`
#   which handles state flattening automatically
# - **Training**: `RNNLearner(..., cuda_graph=True)` does this for you
# - **Fixed shapes required**: all inputs must have identical shapes across calls
# - **Best for**: models where CPU kernel-launch overhead dominates GPU compute
