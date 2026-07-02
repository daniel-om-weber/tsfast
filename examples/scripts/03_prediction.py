# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 03: Prediction -- Using Output Feedback
#
# In prediction mode, the model receives measured outputs in addition to the
# input signal -- but only within an initialization window at the start of each
# sequence. It uses those measurements to estimate the system's current state,
# then predicts forward from that state driven by the input alone. Starting
# from a good state estimate typically gives more accurate predictions than
# simulating from scratch, but requires measured outputs at inference time.

# %% [markdown]
# ## Prerequisites
#
# This notebook builds on Examples 00-02. In particular, make sure you understand
# the simulation concept from Example 02 before reading on, because prediction
# mode is defined by how it differs from simulation.

# %% [markdown]
# ## Setup

# %%
from tsfast.tsdata.benchmark import create_dls_silverbox_prediction
from tsfast.prediction.fransys import FranSysLearner
from tsfast.training import fun_rmse

# %% [markdown]
# ## Simulation vs. Prediction
#
# This is **the** core conceptual distinction in tsfast. Every modeling choice
# flows from this decision:
#
# ### Simulation
#
# ```
# Model input:  [ u(t) ]
# Model output: y(t)
# ```
#
# The model must simulate the system purely from the excitation signal. It never
# sees the real output. This is **harder** but more general -- at deployment time,
# you only need the input signal.
#
# ### Prediction
#
# ```
# Model input:  [ u(t),  y(t) ]   (y is consumed only in the first init_sz steps)
# Model output: y(t) for the timesteps after the initialization window
# ```
#
# The model additionally receives the measured outputs, but it only consumes
# them inside an initialization window at the start of each sequence: it uses
# those measurements to estimate the system's current internal state, then
# rolls out from that state driven by u alone. This is typically more accurate
# than simulation, because the model starts from a known operating point instead
# of a zero state. The tradeoff: you need a sensor measuring y at deployment
# time to fill the initialization window.
#
# ### How tsfast implements prediction
#
# Prediction DataLoaders (e.g., `create_dls_silverbox_prediction`) provide the
# same tensors as simulation ones: u as input and y as target. The output
# feedback is added by the learner: `FranSysLearner(attach_output=True)` inserts
# a `prediction_concat` transform that concatenates the measured output y(t)
# channel-wise onto u(t) at the same timestep (no time shift). All input
# channels -- including the attached y -- are z-score normalized by the model's
# input scaler, using combined statistics for u and y.

# %% [markdown]
# ## Load a Prediction Dataset
#
# `create_dls_silverbox_prediction` creates DataLoaders for the Silverbox
# prediction benchmark. Compared to the simulation variant, it sizes each window
# to the benchmark's initialization window plus prediction horizon; the batches
# themselves are still plain `(u, y)` pairs -- the output feedback is attached
# later by the learner.

# %%
dls = create_dls_silverbox_prediction()

# %% [markdown]
# ## The FranSys Architecture
#
# For prediction tasks, tsfast provides the **FranSys** (Framework for Analysis
# of Systems) architecture. It separates the model into two networks:
#
# 1. **Diagnosis** (first `init_sz` timesteps): a dedicated RNN reads the
#    initialization window -- both input u(t) and measured output y(t) -- and
#    estimates the prognosis network's hidden state from measurement data.
#
# 2. **Prognosis** (remaining timesteps): a second RNN starts from the estimated
#    hidden state and rolls out over the rest of the sequence driven only by the
#    input u(t). The measured outputs are not used beyond the initialization
#    window.
#
# The key insight is that `init_sz` controls how many timesteps of real
# measurements are used to estimate the hidden state. Predictions are only
# evaluated **after** this initialization window, so `init_sz` also acts like
# `n_skip` -- the first `init_sz` timesteps are excluded from the loss.

# %% [markdown]
# ## Train a FranSys Model
#
# Key parameters:
#
# - `init_sz=50`: use the first 50 timesteps for state estimation (diagnosis);
#   predictions are evaluated on the remainder (prognosis)
# - `attach_output=True`: concatenate the measured output y onto the input
#   channels (via the `prediction_concat` transform) -- this is what enables the
#   output feedback that prediction mode is about
# - `hidden_size=40`: 40 hidden units in the RNN layers
# - `metrics=[fun_rmse]`: track root mean squared error

# %%
lrn = FranSysLearner(dls, init_sz=50, hidden_size=40, metrics=[fun_rmse], attach_output=True)
lrn.show_batch(max_n=4)

# %% [markdown]
# Notice that the input has **more channels** than in the simulation example.
# The extra channels are the measured outputs y(t) that the `prediction_concat`
# transform concatenates onto u(t).

# %%
lrn.fit_flat_cos(n_epoch=10, lr=3e-3)

# %% [markdown]
# ## Visualize Results
#
# `show_results` overlays predictions against targets on validation windows.
# Note the initialization region (first 50 timesteps): the model uses this
# window to estimate the system state, so predictions in that region are
# zero-padded and excluded from evaluation.

# %%
lrn.show_results(max_n=3)

# %% [markdown]
# ## Key Takeaways
#
# - **Prediction mode feeds measured outputs into the model**, but FranSys
#   consumes them only inside the initialization window -- the evaluated
#   predictions are rolled out from the estimated state using the input alone.
# - **`attach_output=True` enables the output feedback**: the learner inserts a
#   `prediction_concat` transform that concatenates y onto the input channels;
#   all input channels, including the attached y, are z-score normalized.
# - **FranSys separates diagnosis from prognosis**: the diagnosis network
#   estimates the prognosis network's hidden state from an initialization window
#   of real measurements, and the prognosis network predicts forward from that
#   state.
# - **`init_sz` controls the initialization window**: more timesteps give better
#   state estimates but leave fewer timesteps for evaluated predictions.
# - **Prediction requires measured outputs at inference time** (to fill the
#   initialization window) -- if you only have the input signal, use simulation
#   mode (Example 02) instead.
