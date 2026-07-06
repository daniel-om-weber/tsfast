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
# # Example 15: Hyperparameter Optimization with Ray Tune
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
#
# Ray's uv integration requires the working directory to contain the project's
# `pyproject.toml` when Ray workers are launched. Notebook kernels start in the
# notebook's directory, so we move to the project root first.

# %%
import os
from pathlib import Path

_root = Path.cwd()
while not (_root / "pyproject.toml").is_file() and _root != _root.parent:
    _root = _root.parent
if (_root / "pyproject.toml").is_file():
    os.chdir(_root)

# %%
import identibench as idb

from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.tune import LearnerTrainable, ray_device, report_metrics, resume_checkpoint
from tsfast.training import RNNLearner, fun_rmse
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

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
# TSFast provides helpers that bridge a Learner and Ray Tune:
#
# - **`ray_device()`** -- detects the GPU assigned to the current Ray worker.
# - **`report_metrics(lrn)`** -- patches `log_epoch` to report metrics and
#   checkpoints to Ray Tune after every epoch.
# - **`resume_checkpoint(lrn)`** -- restores from a Ray Tune checkpoint when
#   resuming a trial (e.g. population-based training).
# - **`LearnerTrainable`** -- a class-based Trainable that wraps a Learner for
#   schedulers like Population-Based Training that need checkpoint/restore and
#   actor reuse.

# %% [markdown]
# ## Prepare the DataLoaders
#
# We use the Silverbox benchmark with a small batch size and window size to
# keep the example lightweight. The benchmark spec defines an initialization
# window that its evaluation protocol discards; using it as `n_skip` excludes
# the same initial transient from the training loss.

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
n_skip = idb.BenchmarkSilverbox_Simulation.task.init_window

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
        n_skip=n_skip,
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
# ## Population-Based Training
#
# The examples above run each trial independently. **Population-Based Training
# (PBT)** takes a different approach: it trains a population of models in
# parallel, periodically copying weights from the best performers and perturbing
# their hyperparameters. This combines the exploration of random search with the
# exploitation of hand-tuning.
#
# PBT requires the class-based Trainable API so that Ray can checkpoint, restore,
# and mutate trials. `LearnerTrainable` provides this -- you supply a
# `create_learner` factory and an optional `apply_config` callback for actor
# reuse.

# %%
def create_learner(config):
    return RNNLearner(
        dls,
        rnn_type="gru",
        hidden_size=32,
        n_skip=n_skip,
        metrics=[fun_rmse],
        device=ray_device(),
    )


def apply_config(lrn, config):
    for pg in lrn.opt.param_groups:
        pg["lr"] = config["lr"]


# %%
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=2,
    hyperparam_mutations={"lr": tune.loguniform(1e-4, 1e-2)},
)

tuner_pbt = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(
            LearnerTrainable,
            create_learner=create_learner,
            apply_config=apply_config,
        ),
        {"cpu": 1, "gpu": 0},
    ),
    param_space={"lr": 3e-3},
    tune_config=tune.TuneConfig(
        metric="valid_loss",
        mode="min",
        num_samples=4,
        scheduler=pbt_scheduler,
    ),
    run_config=tune.RunConfig(stop={"training_iteration": 6}),
)

results_pbt = tuner_pbt.fit()

# %%
best_pbt = results_pbt.get_best_result()
print("Best PBT config:")
print(f"  lr: {best_pbt.config['lr']}")
print(f"  valid_loss: {best_pbt.metrics['valid_loss']:.4f}")

# %% [markdown]
# ## Packing Multiple Trials per GPU
#
# The searches above ran on CPU. On a GPU the picture changes: a small sequence
# model reserves well under 1 GB of a 24 GB card and keeps the device mostly
# idle, so giving each trial a whole GPU wastes most of it. Ray Tune happily
# co-locates trials on one device (`{"gpu": 1/k}`) -- the only question is
# *which k* is safe and worthwhile. TSFast answers it with four functions, one
# per step of a **measure → decide → verify → enforce** workflow. They are
# separate on purpose:
#
# 1. **Measure -- `probe_gpu_saturation(make_learner, configs)`** collects the
#    facts about your workload: per-config memory footprint (including the
#    ~0.5 GB CUDA context every co-located process pays) and how busy a single
#    trial keeps the device. It runs a few real training steps per config --
#    seconds each -- and you run it once per workload (model family × dataset).
# 2. **Decide -- `recommend_trials_per_gpu(probe)`** turns those measurements
#    into a number: the memory ceiling `k_mem`, a saturation prior `k_compute`,
#    and the headline `rec.k = min` of the two. It is pure math on the probe
#    result -- instant, GPU-free -- so you can re-derive k under a different
#    memory margin or `max_k` without measuring again.
# 3. **Verify -- `measure_packing_curve(make_learner, configs, ks)`** is the
#    ground truth the recommendation is checked against: it launches k real
#    training processes on one GPU and measures aggregate steps/s. The compute
#    side of step 2 is only a prior (power draw does not see cache or memory-
#    bandwidth contention), so before freezing k for a long tuning campaign,
#    validate it on a few anchor configs with the curve. This is the expensive
#    step -- minutes, not seconds -- which is exactly why it is not folded into
#    the probe.
# 4. **Enforce -- `trial_resources(k)` and `apply_gpu_quota(share, ...)`**
#    wire the frozen k into Ray Tune. Fractional GPUs in Ray are purely
#    logical -- k trials get scheduled onto one device with no isolation -- so
#    `apply_gpu_quota`, called as the first line of the trainable, caps each
#    process's CUDA allocator to its slice. A config that grows too large then
#    fails deterministically against its own quota instead of taking down its
#    neighbors.

# %% [markdown]
# ### Measure and decide
#
# `sample_ray_space` draws random probe configs from the same search space you
# tune over, so the probe sees the sizes a real tuning session would sample.
# Plain values pass through as constants.

# %%
import torch

from tsfast.training.profiling import (
    probe_gpu_saturation,
    recommend_trials_per_gpu,
    sample_ray_space,
)
from tsfast.tune import apply_gpu_quota, trial_resources

search_space = {
    "rnn_type": tune.choice(["gru", "lstm"]),
    "hidden_size": tune.choice([32, 40]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "n_epoch": 1,
}

# %%
if torch.cuda.is_available():

    def make_learner(config):
        return RNNLearner(
            dls,
            rnn_type=config["rnn_type"],
            hidden_size=config["hidden_size"],
            n_skip=n_skip,
            metrics=[fun_rmse],
        )

    probe_configs = sample_ray_space(search_space, n=8, seed=0)
    probe = probe_gpu_saturation(make_learner, probe_configs)
    rec = recommend_trials_per_gpu(probe, mem_margin=0.9, max_k=8)
    print(f"k={rec.k}  (memory ceiling {rec.k_mem}, saturation prior {rec.k_compute})")

# %% [markdown]
# ### Verify
#
# Before a multi-day tuning campaign, check the recommendation against the
# measured packing curve on a few anchor configs. `measure_packing_curve`
# spawns fresh worker processes that must *import* `make_learner`, so it runs
# from a script (module-level factory), not from a notebook cell:
#
# ```python
# curve = measure_packing_curve(make_learner, probe_configs[:3], ks=(1, 2, 4))
# k = min(rec.k, curve.knee)   # freeze this number for the campaign
# ```
#
# On 2× RTX 4090 a small GRU workload measured 139/212/375 aggregate steps/s at
# k=1/2/4 -- co-locating four trials nearly tripled tuning throughput.

# %% [markdown]
# ### Enforce
#
# `trial_resources(k)` makes Ray schedule k trials per GPU; `apply_gpu_quota`
# at the top of the trainable makes each trial's memory slice binding. The
# context overhead measured by the probe is passed along so the k allocator
# quotas plus the k CUDA contexts fit the device.

# %%
if torch.cuda.is_available():

    def train_packed(config):
        apply_gpu_quota(1 / rec.k, context_bytes=rec.context_overhead_bytes)
        lrn = RNNLearner(
            dls,
            rnn_type=config["rnn_type"],
            hidden_size=config["hidden_size"],
            n_skip=n_skip,
            metrics=[fun_rmse],
            device=ray_device(),
        )
        with lrn.no_bar(), report_metrics(lrn, checkpoint_every=None):
            lrn.fit_flat_cos(config["n_epoch"], lr=config.get("lr"))

    tuner_packed = tune.Tuner(
        tune.with_resources(train_packed, trial_resources(rec.k)),
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=4),
    )
    results_packed = tuner_packed.fit()
    print(f"Best packed-run loss: {results_packed.get_best_result().metrics['valid_loss']:.4f}")

# %% [markdown]
# ## Key Takeaways
#
# - **`ray_device()`** detects the GPU assigned to the current Ray worker --
#   pass it as `device` when constructing your Learner.
# - **`report_metrics(lrn)`** patches `log_epoch` to report metrics and
#   checkpoints to Ray Tune after every epoch.
# - **`resume_checkpoint(lrn)`** restores from a Ray Tune checkpoint when
#   resuming a trial (needed for population-based training).
# - **`LearnerTrainable`** wraps a Learner as a class-based Trainable for
#   schedulers like PBT that need checkpoint/restore and actor reuse. Supply a
#   `create_learner` factory and an optional `apply_config` callback via
#   `tune.with_parameters`.
# - **`tune.choice`** for categorical parameters (architecture, layer count);
#   **`tune.loguniform`** for continuous parameters on a log scale (learning
#   rate). No custom samplers needed.
# - **`tune.Tuner`** gives you full control over scheduling, stopping
#   criteria, and resource allocation.
# - **GPU packing** follows measure → decide → verify → enforce:
#   `probe_gpu_saturation` measures footprints and saturation,
#   `recommend_trials_per_gpu` derives k from them, `measure_packing_curve`
#   validates k against ground-truth aggregate throughput, and
#   `trial_resources` + `apply_gpu_quota` wire the frozen k into Ray Tune.
# - **Start small** -- few trials, few epochs -- to validate the pipeline
#   before scaling up.
