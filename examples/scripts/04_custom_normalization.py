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
# # Custom Normalization
#
# All Learner functions accept a `Scaler` class for `input_norm` and `output_norm`.
#
# Built-in options: `StandardScaler1D` (default), `MinMaxScaler1D`, `MaxAbsScaler1D`, `UnitNormScaler`.
#
# You can also define your own by subclassing `Scaler`.

# %%
from tsfast.basics import *

# %% [markdown]
# ## Using built-in scalers

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %%
# Default: StandardScaler1D (z-score normalization)
lrn = RNNLearner(dls, rnn_type='lstm')

# MinMax normalization to [0, 1]
lrn = RNNLearner(dls, rnn_type='lstm', input_norm=MinMaxScaler1D)

# No normalization
lrn = RNNLearner(dls, rnn_type='lstm', input_norm=None)


# %% [markdown]
# ## Defining a custom scaler
#
# Subclass `Scaler` and implement `normalize`, `denormalize`, and the `from_stats` classmethod.
#
# Here's an example: a unit-norm scaler useful for quaternion data, which projects each vector to unit L2 norm.

# %%
class QuaternionUnitNormScaler(Scaler):
    'Normalize to unit L2 norm along the last dimension.'
    _epsilon = 1e-16
    def normalize(self, x):   return x / (x.norm(p=2, dim=-1, keepdim=True) + self._epsilon)
    def denormalize(self, x): return x  # unit quaternions are the natural representation
    @classmethod
    def from_stats(cls, stats): return cls()  # no dataset statistics needed


# %%
# Use it like any built-in scaler
lrn = RNNLearner(dls, rnn_type='lstm', input_norm=QuaternionUnitNormScaler)
lrn.fit_flat_cos(n_epoch=1)

# %%
