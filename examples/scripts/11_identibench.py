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
# # Example 11: Benchmarking with IdentiBench
#
# IdentiBench provides standardized benchmarks for comparing system
# identification methods. This example shows how to run your TSFast models
# on IdentiBench benchmarks for fair, reproducible comparison with other
# methods.

# %% [markdown]
# ## Prerequisites
#
# - [Example 00: Your First Model](00_your_first_model.ipynb)
# - [Example 01: Understanding the Data Pipeline](01_data_pipeline.ipynb)
# - [Example 02: Simulation](02_simulation.ipynb)
# - [Example 04: Benchmark RNN](04_benchmark_rnn.py)

# %% [markdown]
# ## Setup

# %%
import identibench as idb

from tsfast.datasets.benchmark import create_dls_from_spec
from tsfast.models.rnn import RNNLearner
from tsfast.inference import InferenceWrapper
from tsfast.learner.losses import fun_rmse

# %% [markdown]
# ## What is IdentiBench?
#
# IdentiBench is a benchmarking framework that provides standardized
# datasets, evaluation protocols, and metrics for system identification.
# Each benchmark defines:
#
# - A **dataset** with specified train/validation/test splits
# - **Input and output column names** (e.g., voltage in, displacement out)
# - **Evaluation metrics** (typically NRMSE -- normalized root mean square
#   error)
# - A **standard API** that all methods must follow, ensuring fair
#   comparison
#
# The `workshop_benchmarks` dictionary contains the benchmarks used in the
# IdentiBench workshop -- a curated set covering different system types and
# difficulties.

# %% [markdown]
# ## The Build Model Function
#
# IdentiBench requires a `build_model` function that takes a
# `TrainingContext` and returns a callable model for evaluation. The context
# provides:
#
# - **`context.spec`** -- the benchmark specification (dataset path, column
#   names, window sizes, metric function)
# - **`context.hyperparameters`** -- your model's hyperparameters, passed
#   through from the benchmark runner
#
# The returned model must accept numpy arrays: `model(u_test, y_init)` for
# simulation benchmarks, where `u_test` is the full input signal and
# `y_init` is the initial output window.

# %%
def build_model(context: idb.TrainingContext):
    """Build and train a TSFast model for an IdentiBench benchmark."""
    dls = create_dls_from_spec(context.spec)

    lrn = RNNLearner(
        dls,
        rnn_type=context.hyperparameters.get('model_type', 'lstm'),
        num_layers=context.hyperparameters.get('num_layers', 1),
        hidden_size=context.hyperparameters.get('hidden_size', 100),
        n_skip=context.spec.init_window,
        metrics=[fun_rmse],
    )

    lrn.fit_flat_cos(n_epoch=10, lr=3e-3)
    return InferenceWrapper(lrn)


# %% [markdown]
# Key details:
#
# - **`create_dls_from_spec`** automatically extracts column names, window
#   sizes, and prediction settings from the benchmark spec. It also applies
#   benchmark-specific DataLoader defaults (e.g., batch size, step size)
#   from TSFast's `BENCHMARK_DL_KWARGS` table.
# - **`n_skip=context.spec.init_window`** uses the benchmark-defined
#   initialization window to skip the initial transient in the loss. This
#   matches IdentiBench's evaluation protocol, which discards the first
#   `init_window` timesteps.
# - **`InferenceWrapper`** wraps the trained learner into a numpy-in,
#   numpy-out callable that IdentiBench's evaluation harness can call
#   directly.

# %% [markdown]
# ## Configure and Run Benchmarks
#
# We define a hyperparameter dictionary and pass it along with the
# benchmarks to `idb.run_benchmarks`. The runner:
#
# 1. Downloads each dataset (on first use)
# 2. Calls `build_model` with the spec and hyperparameters
# 3. Evaluates the returned model on the held-out test set
# 4. Collects metrics into a pandas DataFrame

# %%
model_config = {
    'model_type': 'lstm',
    'num_layers': 1,
    'hidden_size': 100,
}

benchmarks = list(idb.workshop_benchmarks.values())
results = idb.run_benchmarks(benchmarks, build_model, model_config)

# %% [markdown]
# ## Analyze Results
#
# The results DataFrame shows the benchmark name, metric score, and
# training/test times for each benchmark.

# %%
print(results)

# %% [markdown]
# ## Trying Different Configurations
#
# One of IdentiBench's strengths is making it easy to compare different
# model architectures on the same benchmarks. Here we try a GRU with 2
# layers instead of a single-layer LSTM.

# %%
model_config_v2 = {
    'model_type': 'gru',
    'num_layers': 2,
    'hidden_size': 100,
}

results_v2 = idb.run_benchmarks(benchmarks, build_model, model_config_v2)

# %%
print(results_v2)

# %% [markdown]
# ## Key Takeaways
#
# - **IdentiBench provides standardized, reproducible benchmarks** for fair
#   comparison across system identification methods.
# - The **`build_model` function** follows a simple API: receive a training
#   context, build and train a model, return an `InferenceWrapper`.
# - **`create_dls_from_spec`** handles dataset-specific configuration
#   automatically -- column names, window sizes, and prediction settings
#   are all extracted from the benchmark spec.
# - **Compare different architectures** (LSTM vs. GRU, depth, width) on
#   the same benchmarks with minimal code changes.
# - Results are **directly comparable** with other methods in the
#   IdentiBench ecosystem.
