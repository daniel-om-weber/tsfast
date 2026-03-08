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
# # Example 13: Custom Data Pipelines with HDF5
#
# When your dataset doesn't fit the standard `create_dls` pattern -- custom
# readers, different signal combinations, fixed batch counts -- you need a
# custom pipeline. This example shows how to compose tsfast's building readers
# to create flexible data loading for any HDF5 dataset. You will learn how
# each primitive works on its own, and then combine them into a complete
# training pipeline.

# %% [markdown]
# ## Prerequisites
#
# - [Example 00: Your First Model](00_your_first_model.ipynb)
# - [Example 01: Understanding the Data Pipeline](01_data_pipeline.ipynb)

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

from torch.utils.data import DataLoader

from tsfast.tsdata import (
    WindowedDataset, HDF5Signals, FileEntry, DataLoaders,
    create_dls, create_dls_from_readers, get_hdf_files, split_by_parent,
)
from tsfast.training import RNNLearner, fun_rmse

# %% [markdown]
# ## Finding HDF5 Files
#
# The first step in any custom pipeline is discovering which files exist on
# disk. `get_hdf_files` recursively searches a directory for `.hdf5` and `.h5`
# files and returns them as a sorted list of `Path` objects.

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

data_path = _root / "test_data" / "WienerHammerstein"

files = get_hdf_files(data_path)
print(f"Found {len(files)} HDF5 files:")
for f in files:
    print(f"  {f.parent.name}/{f.name}")

# %% [markdown]
# ## The Standard Approach
#
# Before building anything custom, let's see the standard `create_dls` call for
# reference. It handles file discovery, windowing, splitting, and normalization
# in a single function. All the primitives we explore below are composed
# internally by `create_dls`.
#
# Parameters:
#
# - **`u=['u']`** -- input signal column names in the HDF5 files.
# - **`y=['y']`** -- output signal column names the model learns to predict.
# - **`dataset`** -- path to a directory with `train/`, `valid/`, and `test/`
#   subdirectories containing HDF5 files.
# - **`win_sz=200`** -- window size in time steps. Each training sample is a
#   200-step slice.
# - **`stp_sz=50`** -- step size (stride) between consecutive windows.
# - **`bs=32`** -- batch size.

# %%
dls_standard = create_dls(
    u=['u'], y=['y'],
    dataset=data_path,
    win_sz=200, stp_sz=50,
    bs=32,
)

# %% [markdown]
# ## Building a Custom Pipeline Step by Step
#
# Now let's rebuild the same pipeline manually to understand each component.
# This knowledge lets you customize any part of the pipeline when the standard
# approach doesn't fit your needs.

# %% [markdown]
# ### Step 1: Split Files by Directory
#
# `split_by_parent` inspects each file's parent directory name and returns
# `(train_indices, valid_indices)`. Files under a `train/` directory go to
# training, files under `valid/` go to validation.

# %%
train_idx, valid_idx = split_by_parent(files)
train_files = [files[i] for i in train_idx]
valid_files = [files[i] for i in valid_idx]
print(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}")

# %% [markdown]
# ### Step 2: Define Signal Readers
#
# `HDF5Signals` defines which datasets to read from each HDF5 file. The first
# reader reads input signals, the second reads target signals.

# %%
inputs = HDF5Signals(['u'])
targets = HDF5Signals(['y'])

# %% [markdown]
# ### Step 3: Create WindowedDatasets
#
# `WindowedDataset` takes a list of `FileEntry` objects and the signal readers,
# then creates overlapping windows of the specified size. Each sample is a
# `(input_tensor, target_tensor)` tuple.
#
# - **`win_sz=200`** -- each window is 200 timesteps long
# - **`stp_sz=50`** -- windows overlap with a stride of 50 timesteps

# %%
train_entries = [FileEntry(path=str(f)) for f in train_files]
valid_entries = [FileEntry(path=str(f)) for f in valid_files]

train_ds = WindowedDataset(train_entries, inputs=inputs, targets=targets, win_sz=200, stp_sz=50)
valid_ds = WindowedDataset(valid_entries, inputs=inputs, targets=targets, win_sz=200, stp_sz=50)

print(f"Train windows: {len(train_ds)}")
print(f"Valid windows: {len(valid_ds)}")

# %% [markdown]
# ### Step 4: Create DataLoaders
#
# Wrap the datasets in standard PyTorch DataLoaders, then bundle them in
# tsfast's `DataLoaders` container for compatibility with the Learner.

# %%
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False)
dls_custom = DataLoaders(train_dl, valid_dl)

# %% [markdown]
# ## Using create_dls_from_readers
#
# For a more concise approach, `create_dls_from_readers` handles the
# `WindowedDataset` and `DataLoader` construction for you. It also supports
# `n_batches_train` to control the number of training batches per epoch.

# %%
dls_readers = create_dls_from_readers(
    inputs=inputs, targets=targets,
    train_files=train_files, valid_files=valid_files,
    win_sz=200, stp_sz=50, bs=32,
)

# %% [markdown]
# ## Fixed Batch Counts
#
# When datasets have very different sizes, you may want a fixed number of
# batches per epoch regardless of how many windows exist. Pass
# `n_batches_train` to `create_dls_from_readers` or `create_dls`.
#
# This uses `RandomSampler` with `replacement=True` to oversample when there
# are fewer windows than requested, ensuring consistent training time across
# datasets of varying size.

# %%
dls_nbatch = create_dls_from_readers(
    inputs=inputs, targets=targets,
    train_files=train_files, valid_files=valid_files,
    win_sz=200, stp_sz=50, bs=32,
    n_batches_train=100,
)
print(f"Training batches per epoch: {len(dls_nbatch.train)}")
print(f"Validation batches per epoch: {len(dls_nbatch.valid)}")

# %% [markdown]
# You can also build fixed-batch DataLoaders manually using PyTorch's
# `RandomSampler`:
#
# ```python
# n_samples = 100 * 32  # 100 batches * 32 batch size
# sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples)
# train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler)
# ```

# %% [markdown]
# ## Training with the Custom Pipeline
#
# Let's train an LSTM on the custom DataLoaders to verify everything works
# end-to-end. `RNNLearner` creates a recurrent neural network wrapped in a
# Learner.
#
# Parameters:
#
# - **`dls_custom`** -- the DataLoaders we built manually above.
# - **`rnn_type='lstm'`** -- use Long Short-Term Memory cells.
# - **`hidden_size=40`** -- number of hidden units in the LSTM.
# - **`metrics=[fun_rmse]`** -- track root mean squared error during training.

# %%
lrn = RNNLearner(dls_custom, rnn_type='lstm', hidden_size=40, metrics=[fun_rmse])
lrn.fit_flat_cos(n_epoch=5, lr=3e-3)

# %%
lrn.show_results(max_n=3)

# %% [markdown]
# ## Key Takeaways
#
# - **`get_hdf_files`** discovers HDF5 files recursively in a directory tree.
# - **`split_by_parent`** splits files into train/validation sets based on
#   parent directory names (e.g., `train/` vs `valid/`).
# - **`HDF5Signals`** defines which datasets to extract from HDF5 files.
# - **`WindowedDataset`** creates overlapping windows from HDF5 files, with
#   configurable window size and step size.
# - **`FileEntry`** wraps a file path with optional resampling metadata.
# - **`DataLoaders`** bundles train/valid DataLoaders for the Learner.
# - **`create_dls_from_readers`** handles dataset + DataLoader construction
#   from readers and file lists, including `n_batches_train` for fixed batch
#   counts.
# - The standard **`create_dls`** composes these same primitives internally --
#   understanding them lets you customize any part of the pipeline.
