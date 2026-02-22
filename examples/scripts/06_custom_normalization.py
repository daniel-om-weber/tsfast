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
# # Example 06: Custom Normalization
#
# Neural networks train faster when inputs are properly scaled. TSFast
# normalizes inputs by default using z-score normalization. This example shows
# all built-in scalers, compares their effects on training, and demonstrates
# how to create a custom scaler.

# %% [markdown]
# ## Prerequisites
#
# - [Example 00: Your First Model](00_your_first_model.ipynb)
# - [Example 01: Understanding the Data Pipeline](01_data_pipeline.ipynb)

# %% [markdown]
# ## Setup

# %%
import torch

from tsfast.datasets.benchmark import create_dls_silverbox
from tsfast.models.rnn import RNNLearner
from tsfast.models.layers import Scaler, StandardScaler1D, MinMaxScaler1D, MaxAbsScaler1D
from tsfast.learner.losses import fun_rmse

# %% [markdown]
# ## Why Normalization Matters
#
# Neural networks learn best when input features are on similar scales. Without
# normalization, features with large magnitudes dominate gradient updates,
# causing slow or unstable training. TSFast automatically normalizes input
# signals by default.

# %% [markdown]
# ## Load the Dataset

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Built-in Scalers
#
# TSFast provides three built-in scalers:
#
# - **`StandardScaler1D`** (default): z-score normalization.
#   `x_norm = (x - mean) / std`
# - **`MinMaxScaler1D`**: scales to [0, 1].
#   `x_norm = (x - min) / (max - min)`
# - **`MaxAbsScaler1D`**: scales to [-1, 1].
#   `x_norm = x / max(|min|, |max|)`

# %% [markdown]
# ## Training with Different Scalers
#
# Train with each scaler for a fair comparison. All models use the same
# architecture and training schedule so the only difference is the scaler.

# %%
lrn_std = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_std.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"StandardScaler1D: {lrn_std.validate()}")

# %%
lrn_mm = RNNLearner(dls, rnn_type='lstm', input_norm=MinMaxScaler1D, metrics=[fun_rmse])
lrn_mm.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"MinMaxScaler1D:   {lrn_mm.validate()}")

# %%
lrn_ma = RNNLearner(dls, rnn_type='lstm', input_norm=MaxAbsScaler1D, metrics=[fun_rmse])
lrn_ma.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"MaxAbsScaler1D:   {lrn_ma.validate()}")

# %%
lrn_none = RNNLearner(dls, rnn_type='lstm', input_norm=None, metrics=[fun_rmse])
lrn_none.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"No normalization: {lrn_none.validate()}")

# %% [markdown]
# ## Output Normalization
#
# By default, only inputs are normalized and outputs stay in physical units.
# For multi-output systems where outputs have very different scales, you can
# also normalize outputs. Predictions are automatically denormalized back to
# physical units.

# %%
lrn_out = RNNLearner(dls, rnn_type='lstm', output_norm=StandardScaler1D, metrics=[fun_rmse])
lrn_out.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## Creating a Custom Scaler
#
# To create a custom scaler, subclass `Scaler` and implement three methods:
# `normalize`, `denormalize`, and the `from_stats` classmethod. Here is an
# example that clips values to a fixed range and then scales to [-1, 1].

# %%
class ClipScaler(Scaler):
    """Clips values to [-clip_val, clip_val] then scales to [-1, 1]."""

    def __init__(self, clip_val: torch.Tensor):
        super().__init__()
        self.register_buffer('clip_val', clip_val)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -self.clip_val, self.clip_val) / self.clip_val

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.clip_val

    @classmethod
    def from_stats(cls, stats):
        clip_val = torch.max(torch.abs(torch.tensor(stats.min)),
                             torch.abs(torch.tensor(stats.max))).float()
        return cls(clip_val.unsqueeze(0).unsqueeze(0))


# %% [markdown]
# `from_stats` receives a `NormPair` object containing the dataset statistics
# (`mean`, `std`, `min`, `max` as 1-D arrays) and must return a configured
# scaler instance. `register_buffer` ensures the parameters move with the
# model to GPU/CPU automatically.

# %% [markdown]
# Train with the custom scaler:

# %%
lrn_custom = RNNLearner(dls, rnn_type='lstm', input_norm=ClipScaler, metrics=[fun_rmse])
lrn_custom.fit_flat_cos(n_epoch=5, lr=3e-3)
print(f"ClipScaler:       {lrn_custom.validate()}")

# %% [markdown]
# ## Visualize Results

# %%
lrn_custom.show_results(max_n=2)

# %% [markdown]
# ## Key Takeaways
#
# - **`StandardScaler1D`** (z-score) is the default and works well for most
#   problems.
# - **`MinMaxScaler1D`** and **`MaxAbsScaler1D`** are alternatives for bounded
#   signals.
# - **`input_norm=None`** disables normalization (useful for pre-normalized
#   data).
# - **`output_norm=StandardScaler1D`** normalizes outputs for multi-scale
#   training. Predictions are automatically denormalized.
# - Custom scalers subclass `Scaler` with `normalize`, `denormalize`, and
#   `from_stats`.
