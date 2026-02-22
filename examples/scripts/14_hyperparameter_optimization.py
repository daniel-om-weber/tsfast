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
from tsfast.datasets.benchmark import create_dls_silverbox
from tsfast.models.rnn import RNNLearner
from tsfast.tune import HPOptimizer, log_uniform
from tsfast.learner.losses import fun_rmse
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
# TSFast's `HPOptimizer` wraps Ray Tune so you can search over model
# configurations with minimal boilerplate.

# %% [markdown]
# ## Prepare the DataLoaders
#
# We use the Silverbox benchmark with a small batch size and window size to
# keep the example lightweight.

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)

# %% [markdown]
# ## Define a Learner Factory
#
# `HPOptimizer` needs a factory function that takes `(dls, config)` and returns
# a configured Learner. Ray Tune calls this function once per trial, each time
# with a different hyperparameter configuration sampled from the search space.

# %%
def create_learner(dls, config):
    """Create a configured RNNLearner from hyperparameter config."""
    return RNNLearner(
        dls,
        rnn_type=config["rnn_type"],
        hidden_size=config["hidden_size"],
        n_skip=50,
        metrics=[fun_rmse],
    )


# %% [markdown]
# ## Define the Search Space
#
# The search space is a plain dictionary where values are Ray Tune sampling
# primitives:
#
# - **`tune.choice`** -- samples uniformly from a list of discrete options.
#   Good for categorical parameters like architecture type or layer count.
# - **`log_uniform`** -- samples uniformly on a logarithmic scale. Ideal for
#   parameters that span orders of magnitude, such as learning rate.
#
# We start with a small search over two parameters: RNN cell type and hidden
# size.

# %%
search_config = {
    "rnn_type": tune.choice(["gru", "lstm"]),
    "hidden_size": tune.choice([32, 40]),
    "n_epoch": 3,
    "lr": 3e-3,
}

# %% [markdown]
# The config also contains fixed training parameters:
#
# - **`n_epoch=3`** -- each trial trains for 3 epochs (enough to compare
#   configurations, not enough for final training).
# - **`lr=3e-3`** -- fixed learning rate for all trials in this first search.

# %% [markdown]
# ## Run the Optimization
#
# `HPOptimizer` takes the learner factory and the DataLoaders. Calling
# `optimize` launches the search: `num_samples=4` runs 4 independent trials,
# each with a different hyperparameter combination drawn from `search_config`.
#
# The default training function uses `fit_flat_cos` and reports training loss,
# validation loss, and metrics to Ray Tune after every epoch.

# %%
optimizer = HPOptimizer(
    create_lrn=create_learner,
    dls=dls,
)

results = optimizer.optimize(
    config=search_config,
    num_samples=4,
    resources_per_trial={"cpu": 1, "gpu": 0},
)

# %% [markdown]
# ## Analyze Results
#
# The `optimize` call returns a Ray Tune `ExperimentAnalysis` object stored in
# `optimizer.analysis`. You can query it for the best trial configuration,
# inspect per-trial results, or export data for further analysis.

# %%
best = optimizer.analysis.get_best_config(metric="valid_loss", mode="min")
print("Best config:")
for key in ["rnn_type", "hidden_size", "lr"]:
    print(f"  {key}: {best[key]}")

# %%
result_df = optimizer.analysis.results_df
print("\nAll trial results:")
result_df[["config/rnn_type", "config/hidden_size", "valid_loss"]]

# %% [markdown]
# ## Using log_uniform for Learning Rate
#
# In the first search we fixed the learning rate. A more thorough search treats
# `lr` as a tunable parameter using `log_uniform`. This samples on a
# logarithmic scale between the given bounds -- appropriate because the
# difference between `1e-4` and `1e-3` matters more than between `1e-2` and
# `1.1e-2`.

# %%
search_config_v2 = {
    "rnn_type": tune.choice(["gru", "lstm"]),
    "hidden_size": tune.choice([32, 40]),
    "lr": log_uniform(1e-4, 1e-2),
    "n_epoch": 3,
}

# %% [markdown]
# When `lr` is a callable sampler in the config, the training function samples
# a fresh value for each trial. This overrides any fixed learning rate.

# %%
optimizer_v2 = HPOptimizer(
    create_lrn=create_learner,
    dls=dls,
)

results_v2 = optimizer_v2.optimize(
    config=search_config_v2,
    num_samples=4,
    resources_per_trial={"cpu": 1, "gpu": 0},
)

# %%
best_v2 = optimizer_v2.analysis.get_best_config(metric="valid_loss", mode="min")
print("Best config (with lr search):")
for key in ["rnn_type", "hidden_size", "lr"]:
    print(f"  {key}: {best_v2[key]}")

# %% [markdown]
# ## Key Takeaways
#
# - **`HPOptimizer`** wraps Ray Tune for easy hyperparameter search with
#   TSFast. Pass a learner factory and DataLoaders, then call `optimize`.
# - **Learner factory** -- a function `(dls, config) -> Learner` that builds a
#   fresh model from the hyperparameter config each trial.
# - **`tune.choice`** for categorical parameters (architecture, layer count);
#   **`log_uniform`** for continuous parameters on a log scale (learning rate).
# - **Start small** -- few trials, few epochs -- to validate the pipeline
#   before scaling up.
# - **`optimizer.analysis`** gives access to the full Ray Tune
#   `ExperimentAnalysis` for querying best configs, exporting results, and
#   loading the best checkpoint.
