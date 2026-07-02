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
# # Example 06: Custom Normalization
#
# Neural networks train best when inputs are properly scaled. TSFast
# normalizes inputs by default using z-score normalization. This example shows
# all built-in scalers, how to swap and compare them on a benchmark, and how
# to create a custom scaler.

# %% [markdown]
# ## Prerequisites
#
# - [Example 00: Your First Model](00_your_first_model.ipynb)
# - [Example 01: Understanding the Data Pipeline](01_data_pipeline.ipynb)

# %% [markdown]
# ## Setup

# %%
import torch

from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.models.scaling import Scaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from tsfast.training import RNNLearner, fun_rmse

# %% [markdown]
# ## Why Normalization Matters
#
# Neural networks learn best when input features are on similar scales. When
# features have large or mismatched magnitudes -- say, a pressure in pascals
# (~1e5) next to a valve position in [0, 1] -- the large features dominate
# gradient updates, causing slow or unstable training. TSFast therefore
# normalizes input signals by default.
#
# The Silverbox signals used below are already well-scaled (inputs within about
# +/-0.1, outputs within +/-0.22), so don't expect dramatic differences between
# scalers here. The comparison demonstrates the mechanism for swapping scalers
# and shows that on data like this the choice is minor -- the payoff comes when
# your signals arrive in raw physical units with very different magnitudes.

# %% [markdown]
# ## Load the Dataset

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Built-in Scalers
#
# TSFast provides three built-in scalers:
#
# - **`StandardScaler`** (default): z-score normalization.
#   `x_norm = (x - mean) / std`
# - **`MinMaxScaler`**: scales to [0, 1].
#   `x_norm = (x - min) / (max - min)`
# - **`MaxAbsScaler`**: scales to [-1, 1].
#   `x_norm = x / max(|min|, |max|)`

# %% [markdown]
# ## Training with Different Scalers
#
# Train with each scaler. All models use the same architecture and training
# schedule, so the only difference is the scaler. Each run's `validate()`
# result goes into a dict; a summary table at the end compares them all.

# %%
results = {}

# %%
lrn_std = RNNLearner(dls, rnn_type='lstm', metrics=[fun_rmse])
lrn_std.fit_flat_cos(n_epoch=5, lr=3e-3)
results['StandardScaler'] = lrn_std.validate()

# %%
lrn_mm = RNNLearner(dls, rnn_type='lstm', input_norm=MinMaxScaler, metrics=[fun_rmse])
lrn_mm.fit_flat_cos(n_epoch=5, lr=3e-3)
results['MinMaxScaler'] = lrn_mm.validate()

# %%
lrn_ma = RNNLearner(dls, rnn_type='lstm', input_norm=MaxAbsScaler, metrics=[fun_rmse])
lrn_ma.fit_flat_cos(n_epoch=5, lr=3e-3)
results['MaxAbsScaler'] = lrn_ma.validate()

# %%
lrn_none = RNNLearner(dls, rnn_type='lstm', input_norm=None, metrics=[fun_rmse])
lrn_none.fit_flat_cos(n_epoch=5, lr=3e-3)
results['None'] = lrn_none.validate()

# %% [markdown]
# ## Output Normalization
#
# By default, only inputs are normalized and outputs stay in physical units.
# For multi-output systems where outputs have very different scales, you can
# also normalize outputs. Predictions are automatically denormalized back to
# physical units.

# %%
lrn_out = RNNLearner(dls, rnn_type='lstm', output_norm=StandardScaler, metrics=[fun_rmse])
lrn_out.fit_flat_cos(n_epoch=5, lr=3e-3)

# %% [markdown]
# ## Creating a Custom Scaler
#
# To create a custom scaler, subclass `Scaler` and implement three methods:
# `normalize`, `denormalize`, and the `from_stats` classmethod. Here is an
# example that saturates outliers: values beyond half the largest training
# amplitude are clipped before scaling to [-1, 1]. Placing the threshold
# inside the data range (rather than at the extremes, where `clamp` would
# never fire) makes the clipping actually do something.

# %%
class ClipScaler(Scaler):
    """Clips values to [-clip_val, clip_val] then scales to [-1, 1].

    Args:
        clip_val: saturation threshold, set to half the largest absolute
            value in the training data by ``from_stats``.
    """

    def __init__(self, clip_val: torch.Tensor):
        super().__init__()
        self.register_buffer('clip_val', clip_val)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -self.clip_val, self.clip_val) / self.clip_val

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.clip_val

    @classmethod
    def from_stats(cls, stats):
        max_abs = torch.max(torch.abs(torch.tensor(stats.min)),
                            torch.abs(torch.tensor(stats.max))).float()
        return cls((0.5 * max_abs).unsqueeze(0).unsqueeze(0))


# %% [markdown]
# `from_stats` receives a `NormPair` object containing the dataset statistics
# (`mean`, `std`, `min`, `max` as 1-D arrays) and must return a configured
# scaler instance. `register_buffer` ensures the parameters move with the
# model to GPU/CPU automatically.

# %% [markdown]
# Check how much the scaler actually clips. Building it from the input stats
# (`dls.norm_stats.u`) and applying it to a raw validation batch counts the
# values that saturate:

# %%
scaler = ClipScaler.from_stats(dls.norm_stats.u)
xb, yb = next(iter(dls.valid))
n_clipped = int((xb.abs() > scaler.clip_val).sum())
print(f"clip_val: {scaler.clip_val.item():.4f}")
print(f"clipped:  {n_clipped} of {xb.numel()} values ({100 * n_clipped / xb.numel():.1f}%)")

# %% [markdown]
# Train with the custom scaler:

# %%
lrn_custom = RNNLearner(dls, rnn_type='lstm', input_norm=ClipScaler, metrics=[fun_rmse])
lrn_custom.fit_flat_cos(n_epoch=5, lr=3e-3)
results['ClipScaler'] = lrn_custom.validate()

# %% [markdown]
# ## Visualize Results

# %%
lrn_custom.show_results(max_n=2)

# %% [markdown]
# ## Comparison
#
# All five runs share the same architecture and training schedule, so the
# scaler is the only difference:

# %%
for name, (loss, metrics) in results.items():
    rmse = metrics.get('fun_rmse', float('nan'))
    print(f"{name:16s}: loss={loss:.4f}, RMSE={rmse:.4f}")

# %% [markdown]
# The built-in scalers -- and even disabling normalization -- land at
# essentially the same RMSE: Silverbox is insensitive to the choice because
# its signals are already near unit scale. Even the lossy `ClipScaler` stays
# close, since only a small fraction of values exceed its threshold. Treat
# this table as evidence that on well-scaled data the scaler is not the
# bottleneck -- and as the template for running this comparison on your own
# data, where mismatched physical units can make the differences substantial.

# %% [markdown]
# ## Key Takeaways
#
# - **`StandardScaler`** (z-score) is the default and works well for most
#   problems.
# - **`MinMaxScaler`** and **`MaxAbsScaler`** are alternatives for bounded
#   signals.
# - **`input_norm=None`** disables normalization (useful for pre-normalized
#   data).
# - **`output_norm=StandardScaler`** normalizes outputs for multi-scale
#   training. Predictions are automatically denormalized.
# - Custom scalers subclass `Scaler` with `normalize`, `denormalize`, and
#   `from_stats`.
# - On data that is already near unit scale, the scaler choice makes little
#   difference; normalization pays off when features have large or mismatched
#   magnitudes.
