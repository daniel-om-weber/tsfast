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
# # Example 08: FranSys -- Diagnosis/Prognosis Architecture
#
# FranSys (Framework for Nonlinear System identification) uses a two-phase
# approach: a **diagnosis** RNN estimates the system's hidden state from an
# initialization window, then a **prognosis** RNN predicts forward from that
# state. This architecture handles variable initial conditions gracefully and
# generalizes to unseen starting points.

# %% [markdown]
# ## Prerequisites
#
# This notebook builds on concepts from Examples 00-03. Make sure you understand
# simulation mode (Example 02) and prediction mode (Example 03) before
# proceeding.

# %% [markdown]
# ## Setup

# %%
from tsfast.datasets.benchmark import create_dls_cascaded_tanks
from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
from tsfast.learner.callbacks import TimeSeriesRegularizer
from tsfast.learner.losses import fun_rmse

# %% [markdown]
# ## The Diagnosis/Prognosis Concept
#
# Traditional RNNs start from a zero hidden state, which doesn't match
# reality -- real systems are rarely at rest. FranSys addresses this with a
# two-phase architecture:
#
# - **Diagnosis phase** (first `init_sz` timesteps): The diagnosis RNN processes
#   the initialization window of measured input+output data to estimate the
#   system's internal state. The final hidden state captures where the system
#   "is" at the end of the window.
#
# - **Prognosis phase** (remaining timesteps): The prognosis RNN receives the
#   diagnosis hidden state and predicts forward. It processes incoming input
#   (and optionally output feedback) to generate predictions.
#
# The two RNNs share the same hidden dimension but are trained jointly
# end-to-end.

# %% [markdown]
# ## The Cascaded Tanks Benchmark
#
# The Cascaded Tanks system is a benchmark where water flows between two tanks
# in series. It exhibits strong nonlinear behavior due to the square-root
# relationship between water level and flow rate. The system has one input
# (pump voltage) and one output (water level in the second tank).

# %%
dls = create_dls_cascaded_tanks()
dls.show_batch(max_n=4)

# %% [markdown]
# ## Training a Basic FranSys Model
#
# Key parameters:
#
# - **`init_sz=50`**: use the first 50 timesteps for diagnosis (state
#   estimation). Predictions are only evaluated after this window.
# - **`attach_output=True`**: activates PredictionCallback -- the model receives
#   past measured outputs as additional input. This is standard for
#   prediction-mode FranSys.
# - **`hidden_size=40`**: dimension of the RNN hidden state for both diagnosis
#   and prognosis.

# %%
lrn = FranSysLearner(
    dls, init_sz=50, attach_output=True,
    hidden_size=40, metrics=[fun_rmse]
)
lrn.fit_flat_cos(n_epoch=10, lr=3e-3)

# %% [markdown]
# ## Visualize Results
#
# `ds_idx=-1` shows the last validation/test set. The first 50 timesteps
# (diagnosis window) are zero-padded because the model uses that region for
# state estimation rather than prediction.

# %%
lrn.show_results(ds_idx=-1, max_n=2)

# %% [markdown]
# ## Adding TimeSeriesRegularizer
#
# FranSys models benefit significantly from activation regularization, which
# encourages smoother predictions. `alpha` penalizes large activations, `beta`
# penalizes abrupt changes between timesteps. We need to extract the prognosis
# RNN module from the model so the regularizer knows which layer to hook into.

# %%
from tsfast.models.layers import unwrap_model

lrn_reg = FranSysLearner(
    dls, init_sz=50, attach_output=True,
    hidden_size=40, metrics=[fun_rmse]
)
model_reg = unwrap_model(lrn_reg.model)
lrn_reg.fit_flat_cos(n_epoch=10, lr=3e-3, cbs=[
    TimeSeriesRegularizer(alpha=6.0, beta=6.0, modules=[model_reg.rnn_prognosis])
])
lrn_reg.show_results(ds_idx=-1, max_n=2)

# %% [markdown]
# ## FranSys Callback for State Synchronization
#
# `FranSysCallback` adds an auxiliary loss that encourages the prognosis RNN to
# maintain state consistency: the hidden state at any point in the prognosis
# should be similar to what the diagnosis RNN would produce from the same data
# window. This improves long-horizon stability.
#
# The callback requires the diagnosis and prognosis modules to be passed
# explicitly so it can hook into both and compare their hidden states.

# %%
lrn_sync = FranSysLearner(
    dls, init_sz=50, attach_output=True,
    hidden_size=40, metrics=[fun_rmse]
)
model_sync = unwrap_model(lrn_sync.model)
lrn_sync.fit_flat_cos(n_epoch=10, lr=3e-3, cbs=[
    TimeSeriesRegularizer(alpha=6.0, beta=6.0, modules=[model_sync.rnn_prognosis]),
    FranSysCallback(
        modules=[model_sync.rnn_diagnosis, model_sync.rnn_prognosis],
        model=model_sync,
    ),
])
lrn_sync.show_results(ds_idx=-1, max_n=2)

# %% [markdown]
# ## Key Takeaways
#
# - FranSys separates state estimation (diagnosis) from forward prediction
#   (prognosis).
# - `init_sz` controls how many timesteps are used to initialize the hidden
#   state from measured data.
# - `attach_output=True` enables prediction mode (output feedback).
# - `TimeSeriesRegularizer` is especially important for FranSys -- it
#   encourages smooth, stable predictions.
# - `FranSysCallback` adds state synchronization regularization for improved
#   long-horizon stability. It requires the diagnosis and prognosis modules
#   to be passed so it can compare their hidden states.
# - The architecture naturally handles variable initial conditions.
