"""Hyperparameter optimization with Ray Tune integration."""

__all__ = [
    "log_uniform",
    "LearnerTrainable",
    "stop_shared_memory_managers",
    "learner_optimize",
    "sample_config",
    "HPOptimizer",
]

import gc
import os
import tempfile
from collections.abc import Callable
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np
import torch
from torch import nn

import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.experiment.trial import ExportFormat
from ray.tune.schedulers import PopulationBasedTraining


def _worker_device() -> torch.device:
    """Detect the appropriate device inside a Ray worker."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def log_uniform(min_bound: float, max_bound: float, base: float = 10) -> Callable:
    """Sample uniformly in an exponential (log) range.

    Args:
        min_bound: lower bound of the sampling range.
        max_bound: upper bound of the sampling range.
        base: logarithm base for the exponential scale.
    """
    logmin = np.log(min_bound) / np.log(base)
    logmax = np.log(max_bound) / np.log(base)

    def _sample():
        return base ** (np.random.uniform(logmin, logmax))

    return _sample


class LearnerTrainable(tune.Trainable):
    """Ray Tune Trainable wrapper for Learners.

    Args:
        config: Ray Tune config dict containing 'create_lrn' and 'dls' references.
    """

    def setup(self, config: dict):
        self.create_lrn = ray.get(config["create_lrn"])
        dls = ray.get(config["dls"])
        self.dls = dls.to(_worker_device()) if hasattr(dls, "to") else dls

        self.lrn = self.create_lrn(self.dls, config)

    def step(self) -> dict:
        with self.lrn.no_bar():
            self.lrn.fit(1)
        row = self.lrn.recorder.values[-1]
        result = {"train_loss": row[0], "valid_loss": row[1]}
        metric_names = [getattr(m, "__name__", type(m).__name__) for m in self.lrn.metrics]
        for name, val in zip(metric_names, row[2:]):
            result[name] = val
        return result

    def save_checkpoint(self, tmp_checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.lrn.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir: str):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.lrn.model.load_state_dict(torch.load(checkpoint_path))

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = os.path.join(export_dir, "exported_model")
            torch.save(self.lrn.model.state_dict(), path)
            return {ExportFormat.MODEL: path}
        else:
            raise ValueError("unexpected formats: " + str(export_formats))

    # the learner class will be recreated with every perturbation, saving the model
    # that way the new hyperparameter will be applied
    def reset_config(self, new_config: dict) -> bool:
        self.lrn = self.create_lrn(self.dls, new_config)
        self.config = new_config
        return True


def stop_shared_memory_managers(obj: object):
    """Find and stop all SharedMemoryManager instances within an object.

    Args:
        obj: root object to traverse for SharedMemoryManager instances.
    """
    visited = set()  # Track visited objects to avoid infinite loops
    stack = [obj]  # Use a stack to manage objects to inspect

    while stack:
        current_obj = stack.pop()
        obj_id = id(current_obj)

        if obj_id in visited:
            continue  # Skip already visited objects
        visited.add(obj_id)

        # Check if the current object is a SharedMemoryManager and stop it
        if isinstance(current_obj, SharedMemoryManager):
            current_obj.shutdown()
            continue

        # If it's a collection, add its items to the stack. Otherwise, add its attributes.
        if isinstance(current_obj, dict):
            stack.extend(current_obj.keys())
            stack.extend(current_obj.values())
        elif isinstance(current_obj, (list, set, tuple)):
            stack.extend(current_obj)
        elif hasattr(current_obj, "__dict__"):  # Check for custom objects with attributes
            stack.extend(vars(current_obj).values())


def _prepare_dls_for_serialization(dls):
    """Remove transient iterator state from DataLoaders for Ray serialization.

    fastai DataLoaders accumulate generator attributes during iteration
    that cannot be pickled.  This strips them so ``ray.put(dls)`` succeeds.
    """
    for dl in dls.loaders:
        if hasattr(dl, "it"):
            del dl.it


def learner_optimize(config: dict):
    """Training function for Ray Tune function-based API.

    Args:
        config: Ray Tune config dict containing 'create_lrn', 'dls',
            'fit_method', and hyperparameters.
    """
    try:
        create_lrn = ray.get(config["create_lrn"])
        dls = ray.get(config["dls"])
        dls = dls.to(_worker_device()) if hasattr(dls, "to") else dls

        # Scheduling Parameters for training the Model
        lrn_kwargs = {"n_epoch": 100, "pct_start": 0.5}
        for attr in ["n_epoch", "pct_start"]:
            if attr in config:
                lrn_kwargs[attr] = config[attr]

        lrn = create_lrn(dls, config)

        # load checkpoint data if provided
        checkpoint: tune.Checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                lrn.model.load_state_dict(torch.load(checkpoint_dir + "model.pth"))

        lr = config["lr"] if "lr" in config else 3e-3
        lrn.lr = lr() if callable(lr) else lr
        _attach_ray_reporter(lrn)
        fit_method = ray.get(config["fit_method"]) if "fit_method" in config else None
        with lrn.no_bar():
            if fit_method is not None:
                fit_method(lrn, **lrn_kwargs)
            else:
                lrn.fit_flat_cos(**lrn_kwargs)
    finally:
        # cleanup shared memory even when earlystopping occurs
        if "lrn" in locals():
            stop_shared_memory_managers(lrn)
            del lrn
            gc.collect()


def sample_config(config: dict) -> dict:
    """Sample concrete values from a config of callables.

    Args:
        config: dict mapping keys to callable samplers.
    """
    ret_conf = config.copy()
    for k in ret_conf:
        ret_conf[k] = ret_conf[k]()
    return ret_conf


def _attach_ray_reporter(lrn):
    """Patch ``_log_epoch`` on a Learner to report metrics and checkpoints to Ray Tune.

    The new Learner calls ``self._log_epoch(epoch, train_loss, val_loss, metrics, pbar)``
    after each epoch.  Assigning a plain function to the instance attribute overrides
    the method without ``self`` being passed, so the signature matches directly.
    """

    def _ray_log_epoch(epoch, train_loss, val_loss, metrics, pbar):
        result = {"train_loss": train_loss, "valid_loss": val_loss}
        result.update(metrics)
        with tempfile.TemporaryDirectory() as d:
            torch.save(lrn.model.state_dict(), os.path.join(d, "model.pth"))
            ray.tune.report(result, checkpoint=Checkpoint.from_directory(d))

    lrn._log_epoch = _ray_log_epoch


class HPOptimizer:
    """High-level interface for hyperparameter optimization with Ray Tune.

    Args:
        create_lrn: factory function that creates a Learner from (dls, config).
        dls: DataLoaders to use for training.
    """

    def __init__(self, create_lrn: Callable, dls):
        self.create_lrn = create_lrn
        self.dls = dls
        self.analysis = None

    @staticmethod
    def _find_project_root() -> str | None:
        """Find the project root containing pyproject.toml."""
        cwd = Path.cwd()
        for p in [cwd, *cwd.parents]:
            if (p / "pyproject.toml").exists():
                return str(p)
        return None

    def _ensure_ray(self):
        """Initialize Ray if not already running, setting working_dir to the project root."""
        if ray.is_initialized():
            return
        project_root = self._find_project_root()
        kwargs = {}
        if project_root and project_root != str(Path.cwd()):
            kwargs["runtime_env"] = {"working_dir": project_root}
        ray.init(**kwargs)

    def start_ray(self, **kwargs):
        """Initialize Ray runtime."""
        ray.shutdown()
        ray.init(**kwargs)

    def stop_ray(self):
        """Shut down Ray runtime."""
        ray.shutdown()

    def optimize(
        self,
        config: dict,
        optimize_func: Callable = learner_optimize,
        resources_per_trial: dict = {"gpu": 1.0},
        verbose: int = 1,
        **kwargs,
    ):
        """Run hyperparameter optimization using the function-based API.

        Args:
            config: Ray Tune search space configuration dict.
            optimize_func: training function to optimize.
            resources_per_trial: resource dict per trial (e.g. GPU/CPU counts).
            verbose: Ray Tune verbosity level.
        """
        self._ensure_ray()
        config["create_lrn"] = ray.put(self.create_lrn)
        # dls are large objects, letting ray handle the copying process makes it much faster
        _prepare_dls_for_serialization(self.dls)
        config["dls"] = ray.put(self.dls)

        self.analysis = tune.run(
            optimize_func, config=config, resources_per_trial=resources_per_trial, verbose=verbose, **kwargs
        )
        return self.analysis

    def optimize_pbt(
        self,
        opt_name: str,
        num_samples: int,
        config: dict,
        mut_conf: dict,
        perturbation_interval: int = 2,
        stop: dict = {"training_iteration": 40},
        resources_per_trial: dict = {"gpu": 1},
        resample_probability: float = 0.25,
        quantile_fraction: float = 0.25,
        **kwargs,
    ):
        """Run Population Based Training optimization.

        Args:
            opt_name: experiment name for Ray Tune.
            num_samples: number of parallel trials.
            config: initial hyperparameter configuration dict.
            mut_conf: mutable hyperparameter space for PBT mutations.
            perturbation_interval: epochs between PBT perturbations.
            stop: stopping criteria dict.
            resources_per_trial: resource dict per trial.
            resample_probability: probability of resampling vs. perturbing.
            quantile_fraction: fraction of trials to exploit/explore.
        """
        self.mut_conf = mut_conf

        self._ensure_ray()
        config["create_lrn"] = ray.put(self.create_lrn)
        # dls are large objects, letting ray handle the copying process makes it much faster
        _prepare_dls_for_serialization(self.dls)
        config["dls"] = ray.put(self.dls)

        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="mean_loss",
            mode="min",
            perturbation_interval=perturbation_interval,
            resample_probability=resample_probability,
            quantile_fraction=quantile_fraction,
            hyperparam_mutations=mut_conf,
        )

        self.analysis = tune.run(
            LearnerTrainable,
            name=opt_name,
            scheduler=scheduler,
            reuse_actors=True,
            verbose=1,
            stop=stop,
            checkpoint_score_attr="mean_loss",
            num_samples=num_samples,
            resources_per_trial=resources_per_trial,
            config=config,
            **kwargs,
        )
        return self.analysis

    def best_model(self) -> nn.Module:
        """Load and return the best model from the optimization run."""
        if self.analysis is None:
            raise Exception
        model = self.create_lrn(self.dls, sample_config(self.mut_conf)).model
        f_path = ray.get(self.analysis.get_best_trial("mean_loss", mode="min").checkpoint.value)
        model.load_state_dict(torch.load(f_path))
        return model
