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
# # Example 04: Choosing Your Model Architecture
#
# TSFast supports three model families for time series system identification:
#
# - **RNNs** (LSTM, GRU) -- sequential memory, processing one timestep at a time
# - **TCNs** -- parallel causal convolutions with exponentially growing receptive field
# - **CRNNs** -- hybrid combining CNN feature extraction with RNN sequential modeling
#
# This example trains each architecture on the same dataset and compares their
# performance and trade-offs.

# %% [markdown]
# ## Prerequisites
#
# This example builds on concepts from Examples 00-02. Make sure you have
# completed those first.

# %% [markdown]
# ## Setup

# %%
from tsfast.datasets.benchmark import create_dls_silverbox
from tsfast.models.rnn import RNNLearner
from tsfast.models.cnn import TCNLearner, CRNNLearner
from tsfast.learner.losses import fun_rmse

# %% [markdown]
# ## Load the Dataset
#
# We use the Silverbox benchmark for a fair comparison. All four models train on
# exactly the same data and are evaluated on the same validation set.

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
dls.show_batch(max_n=2)

# %% [markdown]
# ## LSTM
#
# Long Short-Term Memory: processes the sequence step-by-step, maintaining a
# cell state and hidden state at each timestep. The gating mechanism (input,
# forget, output gates) allows it to selectively remember or discard information,
# making it effective at capturing long-range dependencies.
#
# Key parameter:
#
# - **`hidden_size=40`** -- dimension of the hidden state vector. Larger values
#   give the model more capacity to represent complex dynamics, but increase
#   memory usage and training time.

# %%
lrn_lstm = RNNLearner(dls, rnn_type='lstm', hidden_size=40, metrics=[fun_rmse])
lrn_lstm.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_lstm.show_results(max_n=2)

# %% [markdown]
# ## GRU
#
# Gated Recurrent Unit: similar to LSTM but with a simpler gating mechanism
# (2 gates instead of 3). The update gate and reset gate combine the roles of
# LSTM's input, forget, and output gates. GRUs often train faster than LSTMs
# with comparable performance on many tasks.

# %%
lrn_gru = RNNLearner(dls, rnn_type='gru', hidden_size=40, metrics=[fun_rmse])
lrn_gru.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_gru.show_results(max_n=2)

# %% [markdown]
# ## TCN (Temporal Convolutional Network)
#
# TCNs use 1D causal convolutions with exponentially increasing dilation factors.
# Each layer doubles the dilation (1, 2, 4, 8, ...), so the receptive field grows
# as 2^depth. Unlike RNNs, TCNs process the entire sequence in parallel, making
# them significantly faster to train on GPUs.
#
# Key parameters:
#
# - **`num_layers=4`** -- number of TCN blocks. Controls the receptive field
#   (2^4 = 16 timesteps). Deeper networks see further back in time.
# - **`hidden_size=40`** -- number of channels (feature maps) per layer.
#   Controls the width of the network.

# %%
lrn_tcn = TCNLearner(dls, num_layers=4, hidden_size=40, metrics=[fun_rmse])
lrn_tcn.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_tcn.show_results(max_n=2)

# %% [markdown]
# ## CRNN (Convolutional Recurrent Neural Network)
#
# The CRNN combines a TCN front-end with an RNN back-end. The convolutional
# layers extract local temporal features (patterns, transients) efficiently in
# parallel, then the RNN captures long-range dynamics and sequential
# dependencies. This hybrid approach can outperform either architecture alone.
#
# Key parameters:
#
# - **`num_cnn_layers=4`** -- depth of the TCN feature extractor
# - **`num_rnn_layers=1`** -- depth of the RNN sequential modeler

# %%
lrn_crnn = CRNNLearner(dls, num_cnn_layers=4, num_rnn_layers=1, metrics=[fun_rmse])
lrn_crnn.fit_flat_cos(n_epoch=10, lr=3e-3)

# %%
lrn_crnn.show_results(max_n=2)

# %% [markdown]
# ## Comparison
#
# Let's compare the final validation loss and RMSE across all four models.

# %%
results = {
    'LSTM': lrn_lstm.validate(),
    'GRU': lrn_gru.validate(),
    'TCN': lrn_tcn.validate(),
    'CRNN': lrn_crnn.validate(),
}
for name, val in results.items():
    print(f"{name:6s}: loss={val[0]:.4f}, RMSE={val[1]:.4f}")

# %% [markdown]
# ## Trade-offs
#
# **TCN**
#
# - Parallel computation makes training fast, especially on GPUs.
# - Fixed receptive field (2^depth) may miss very long-range dependencies unless
#   you add enough layers.
# - Good default choice for many problems -- fast to train with strong
#   performance.
#
# **RNN (LSTM / GRU)**
#
# - Sequential processing is inherently slower than parallel convolutions.
# - Flexible memory allows learning arbitrarily long dependencies in principle.
# - GRU is simpler than LSTM (fewer parameters), often sufficient for moderate
#   complexity systems.
# - LSTM's explicit cell state can help when the system has very long memory.
#
# **CRNN**
#
# - Best of both worlds: the CNN extracts local features efficiently, the RNN
#   models long-range dynamics.
# - More hyperparameters to tune (CNN depth, RNN depth, hidden sizes for each
#   stage).
# - Can be the strongest choice when you have compute budget for tuning.

# %% [markdown]
# ## Key Takeaways
#
# - **LSTM and GRU** process sequences step-by-step. GRU is simpler and often
#   trains faster; LSTM has more capacity for complex dynamics.
# - **TCN** processes sequences in parallel via causal convolutions. Fast to
#   train, with receptive field controlled by network depth.
# - **CRNN** combines CNN feature extraction with RNN sequential modeling,
#   offering a flexible hybrid architecture.
# - All architectures are accessed through the same simple API:
#   `RNNLearner`, `TCNLearner`, `CRNNLearner`.
# - Start with TCN or GRU as a baseline, then try CRNN if you need more
#   capacity.
