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
# # Example 12: Sequence-to-Scalar Prediction
#
# Not all system identification tasks predict a full output sequence. Sometimes
# the target is a single scalar -- a physical parameter, a classification label,
# or a summary statistic. This example builds a custom data pipeline with
# `HDF5Attrs` for scalar targets and uses `SeqAggregation` to reduce RNN output
# to a scalar prediction.

# %% [markdown]
# ## Prerequisites
#
# This example builds on [Example 00](00_your_first_model.py) and
# [Example 01](01_data_pipeline.py). Make sure the library is installed:
#
# ```bash
# uv sync --extra dev
# ```

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from tsfast.tsdata import (
    WindowedDataset, HDF5Signals, HDF5Attrs, FileEntry,
    DataLoaders, get_hdf_files, split_by_parent,
)
from tsfast.models.rnn import SimpleRNN
from tsfast.models.layers import SeqAggregation
from tsfast.training import Learner, fun_rmse

# %% [markdown]
# ## The Task: Estimating Initial Conditions
#
# We have spring-damper trajectories with different initial positions (`x0`) and
# velocities (`v0`). Given the full time series `[u, x, v]` -- force input,
# position, and velocity -- can a neural network estimate what the initial
# conditions were?
#
# This is a **sequence-to-scalar regression** problem: the input is a multi-
# channel time series and the output is a fixed-size vector `[x0, v0]`.

# %% [markdown]
# ## Load the Dataset
#
# The `pinn_var_ic` test data contains HDF5 files organized into `train/`,
# `valid/`, and `test/` directories. Each file stores a 500-step spring-damper
# trajectory with the force input `u`, position `x`, and velocity `v` as
# datasets, plus scalar attributes like `x0` (initial position) and `v0`
# (initial velocity).

# %%
def _find_project_root(marker: str = "test_data") -> Path:
    """Walk up from script/notebook location to find the project root."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path(".").resolve()
    p = start
    while p != p.parent:
        if (p / marker).is_dir():
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not find '{marker}' directory above {start}")


_root = _find_project_root()
data_path = _root / "test_data" / "pinn_var_ic"

files = get_hdf_files(data_path)
print(f"Found {len(files)} HDF5 files")
for f in files[:5]:
    print(f"  {f.parent.name}/{f.name}")

# %% [markdown]
# ## Building a Custom Data Pipeline
#
# Unlike `create_dls` which creates sequence-to-sequence pipelines, here we need
# a custom pipeline with:
#
# - **Input:** `HDF5Signals` reading `[u, x, v]` columns as a multi-channel
#   time series
# - **Target:** `HDF5Attrs` reading `[x0, v0]` from HDF5 file attributes

# %%
inputs_block = HDF5Signals(['u', 'x', 'v'])
targets_block = HDF5Attrs(['x0', 'v0'])

# %% [markdown]
# Create `FileEntry` objects from the discovered files. Each entry points to
# an HDF5 file with a default resampling factor of 1.0.

# %%
entries = [FileEntry(path=str(f)) for f in files]
train_idx, valid_idx = split_by_parent(files)

train_entries = [entries[i] for i in train_idx]
valid_entries = [entries[i] for i in valid_idx]

# %% [markdown]
# Build `WindowedDataset` for train and validation splits. Using
# `win_sz=500, stp_sz=500` gives one window per file (the full trajectory).

# %%
train_ds = WindowedDataset(train_entries, inputs=inputs_block, targets=targets_block, win_sz=500, stp_sz=500)
valid_ds = WindowedDataset(valid_entries, inputs=inputs_block, targets=targets_block, win_sz=500, stp_sz=500)

print(f"Train samples: {len(train_ds)}")
print(f"Valid samples: {len(valid_ds)}")

# %% [markdown]
# Wrap in DataLoaders. We construct standard PyTorch DataLoaders and wrap them
# in `DataLoaders` for compatibility with the Learner.

# %%
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False)
dls = DataLoaders(train_dl, valid_dl)

# %% [markdown]
# Each component of the pipeline:
#
# - **`HDF5Signals(['u', 'x', 'v'])`** -- extracts named datasets from each
#   HDF5 file as a 3-channel time series with shape `(seq_len, 3)`.
# - **`HDF5Attrs(['x0', 'v0'])`** -- reads named attributes from HDF5
#   metadata as a scalar target vector of shape `(2,)`.
# - **`WindowedDataset`** -- combines the blocks and slices files into
#   windows. With `win_sz=500, stp_sz=500`, each file produces exactly one
#   sample.
# - **`split_by_parent`** -- uses the directory structure (`train/`, `valid/`)
#   to split files into train and validation sets.

# %% [markdown]
# ## Modifying the Model for Scalar Output
#
# A standard `SimpleRNN` produces a sequence output with shape
# `(batch, seq_len, n_outputs)`. To predict scalars, we append `SeqAggregation`
# which selects the last timestep of the RNN output, reducing it to
# `(batch, n_outputs)`.
#
# We build the model manually with `SimpleRNN` and wrap it in a `Learner`
# because `RNNLearner` expects `dls.norm_stats` (computed by `create_dls`),
# which our custom pipeline does not provide.

# %%
input_size = 3   # u, x, v
output_size = 2  # x0, v0
hidden_size = 40

model = nn.Sequential(
    SimpleRNN(input_size, output_size, hidden_size=hidden_size, rnn_type='lstm'),
    SeqAggregation(),
)

lrn = Learner(
    model,
    dls,
    loss_func=nn.L1Loss(),
    metrics=[fun_rmse],
    lr=1e-3,
)

# %% [markdown]
# The model pipeline:
#
# 1. **`SimpleRNN`** processes the 3-channel input sequence and produces
#    `(batch, seq_len, 2)`.
# 2. **`SeqAggregation()`** selects the last timestep, yielding
#    `(batch, 2)` -- one prediction for `x0` and one for `v0`.

# %% [markdown]
# ## Train and Evaluate

# %%
lrn.fit_flat_cos(n_epoch=10, lr=1e-3)

# %% [markdown]
# ## Inspect Predictions
#
# After training, we extract predictions and targets from the validation set to
# compare them side by side.

# %%
preds, targs = lrn.get_preds(ds_idx=1)
print(f"Predictions shape: {preds.shape}")  # (n_samples, 2)
print(f"Targets shape:     {targs.shape}")  # (n_samples, 2)

for i in range(min(5, len(preds))):
    print(
        f"  Pred: x0={preds[i, 0]:.3f}, v0={preds[i, 1]:.3f}  |  "
        f"True: x0={targs[i, 0]:.3f}, v0={targs[i, 1]:.3f}"
    )

# %% [markdown]
# ## Key Takeaways
#
# - **Sequence-to-scalar tasks** predict a single value (or vector) from a time
#   series -- useful for parameter estimation, classification, and condition
#   monitoring.
# - **`HDF5Attrs`** reads scalar targets from HDF5 file attributes,
#   complementing `HDF5Signals` for the input.
# - **`SeqAggregation`** reduces RNN sequence output to a scalar by selecting
#   the last timestep. It is a standard `nn.Module` that can be appended to any
#   sequence model via `nn.Sequential`.
# - **Custom pipelines** combine `WindowedDataset`, `HDF5Signals`, `HDF5Attrs`,
#   and `split_by_parent` for flexible task definitions beyond the standard
#   `create_dls` workflow.
# - The same RNN architectures work for both sequence-to-sequence and
#   sequence-to-scalar problems -- only the output reduction changes.
