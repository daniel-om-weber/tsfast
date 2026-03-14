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
# # Example 14: Hyperparameter Optimization with Ray Tune
#
# Manually tuning hyperparameters -- learning rate, hidden size, model type --
# is tedious and error-prone. TSFast integrates with
# [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to automate the
# search. This example runs a small hyperparameter search to find the best
# model configuration for the Silverbox benchmark.

# %% [markdown]
# ## Prerequisites
#
# This example builds on concepts from:
#
# - **Example 00** -- data loading and model training basics
# - **Example 04** -- model architectures and `rnn_type`
#
# Make sure Ray Tune is installed:
#
# ```bash
# uv sync --extra dev
# ```

# %% [markdown]
# ## Setup

# %%
from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.tune import ray_device, report_metrics, resume_checkpoint
from tsfast.training import RNNLearner, fun_rmse
from ray import tune

# %% [markdown]
# ## Why Hyperparameter Optimization?
#
# Model performance depends heavily on hyperparameters: learning rate, hidden
# size, architecture choice, and regularization strength. Finding the right
# combination by hand requires many experiments and careful record-keeping.
#
# Automated approaches help:
#
# - **Grid search** evaluates every combination -- thorough but expensive.
# - **Random search** samples randomly and is surprisingly effective in
#   high-dimensional spaces.
# - **Population-based training** evolves configurations during training,
#   combining exploration with exploitation.
#
# Ray Tune provides all of these strategies (and more) behind a unified API.
# TSFast provides three helpers that bridge a Learner and Ray Tune:
#
# - **`ray_device()`** -- detects the GPU assigned to the current Ray worker.
# - **`report_metrics(lrn)`** -- patches `log_epoch` to report metrics and
#   checkpoints to Ray Tune after every epoch.
# - **`resume_checkpoint(lrn)`** -- restores from a Ray Tune checkpoint when
#   resuming a trial (e.g. population-based training).

# %% [markdown]
# ## Prepare the DataLoaders
#
# We use the Silverbox benchmark with a small batch size and window size to
# keep the example lightweight.

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Define a Training Function
#
# Ray Tune calls a training function once per trial, each time with a different
# hyperparameter configuration sampled from the search space. The DataLoaders
# are captured via closure.

# %%
def train(config):
    lrn = RNNLearner(
        dls,
        rnn_type=config["rnn_type"],
        hidden_size=config["hidden_size"],
        n_skip=50,
        metrics=[fun_rmse],
        device=ray_device(),
    )
    resume_checkpoint(lrn)
    with lrn.no_bar(), report_metrics(lrn):
        lrn.fit_flat_cos(config["n_epoch"], lr=config.get("lr"))


# %% [markdown]
# ## Define the Search Space
#
# The search space is a plain dictionary where values are Ray Tune sampling
# primitives:
#
# - **`tune.choice`** -- samples uniformly from a list of discrete options.
#   Good for categorical parameters like architecture type or layer count.
# - **`tune.loguniform`** -- samples uniformly on a logarithmic scale. Ideal
#   for parameters that span orders of magnitude, such as learning rate.
#
# Training parameters (`n_epoch`, `lr`) go in the same dict --
# they are read by the training function and logged by Ray Tune.

# %% [markdown]
# ## Run the Optimization
#
# Pass the training function to `tune.Tuner` together with the search space.
# `num_samples=4` runs 4 independent trials, each with a different
# hyperparameter combination.

# %%
tuner = tune.Tuner(
    tune.with_resources(train, {"cpu": 1, "gpu": 0}),
    param_space={
        "rnn_type": tune.choice(["gru", "lstm"]),
        "hidden_size": tune.choice([32, 40]),
        "n_epoch": 3,
        "lr": 3e-3,
    },
    tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=4),
)

results = tuner.fit()

# %% [markdown]
# ## Analyze Results
#
# `tuner.fit()` returns a `ResultGrid`. You can query it for the best trial
# configuration, inspect per-trial metrics, or export data for further
# analysis.

# %%
best = results.get_best_result()
print("Best config:")
for key in ["rnn_type", "hidden_size", "lr"]:
    print(f"  {key}: {best.config[key]}")

# %%
result_df = results.get_dataframe()
print("\nAll trial results:")
result_df[["config/rnn_type", "config/hidden_size", "valid_loss"]]

# %% [markdown]
# ## Using tune.loguniform for Learning Rate
#
# In the first search we fixed the learning rate. A more thorough search treats
# `lr` as a tunable parameter using `tune.loguniform`. This samples on a
# logarithmic scale between the given bounds -- appropriate because the
# difference between `1e-4` and `1e-3` matters more than between `1e-2` and
# `1.1e-2`.

# %%
tuner_v2 = tune.Tuner(
    tune.with_resources(train, {"cpu": 1, "gpu": 0}),
    param_space={
        "rnn_type": tune.choice(["gru", "lstm"]),
        "hidden_size": tune.choice([32, 40]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "n_epoch": 3,
    },
    tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=4),
)

results_v2 = tuner_v2.fit()

# %%
best_v2 = results_v2.get_best_result()
print("Best config (with lr search):")
for key in ["rnn_type", "hidden_size", "lr"]:
    print(f"  {key}: {best_v2.config[key]}")

# %% [markdown]
# ## Key Takeaways
#
# - **`ray_device()`** detects the GPU assigned to the current Ray worker --
#   pass it as `device` when constructing your Learner.
# - **`report_metrics(lrn)`** patches `log_epoch` to report metrics and
#   checkpoints to Ray Tune after every epoch.
# - **`resume_checkpoint(lrn)`** restores from a Ray Tune checkpoint when
#   resuming a trial (needed for population-based training).
# - **`tune.choice`** for categorical parameters (architecture, layer count);
#   **`tune.loguniform`** for continuous parameters on a log scale (learning
#   rate). No custom samplers needed.
# - **`tune.Tuner`** gives you full control over scheduling, stopping
#   criteria, and resource allocation.
# - **Start small** -- few trials, few epochs -- to validate the pipeline
#   before scaling up.
