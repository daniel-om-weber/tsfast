"""Ray Tune integration — bridge between Learner and Ray Tune's training API."""

__all__ = [
    "LearnerTrainable",
    "ray_device",
    "report_metrics",
    "resume_checkpoint",
    "trial_resources",
    "apply_gpu_quota",
]

import os
import tempfile
from contextlib import contextmanager

import torch

try:
    import ray
    import ray.tune
    from ray.tune import Checkpoint
except ImportError:
    raise ImportError("ray[tune] is required for hyperparameter optimization. Install it with: uv sync --extra tune")


@contextmanager
def report_metrics(lrn, checkpoint_every: int | None = 1):
    """Wrap a Learner's ``log_epoch`` to also report metrics and checkpoints to Ray Tune.

    The original ``log_epoch`` (progress bar, custom loggers, etc.) still runs;
    Ray Tune reporting is added after it.

    Args:
        lrn: a Learner instance whose ``log_epoch`` will be temporarily wrapped.
        checkpoint_every: save a checkpoint every N epochs. ``None`` disables
            checkpointing entirely (only metrics are reported).
    """
    had_instance_attr = "log_epoch" in lrn.__dict__
    orig_log_epoch = lrn.log_epoch

    def _ray_log_epoch(epoch, n_epoch, train_loss, val_loss, metrics, pbar):
        orig_log_epoch(epoch, n_epoch, train_loss, val_loss, metrics, pbar)
        result = {"train_loss": train_loss, "valid_loss": val_loss}
        result.update(metrics)
        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            with tempfile.TemporaryDirectory() as d:
                lrn.save_checkpoint(os.path.join(d, "checkpoint.pt"))
                ray.tune.report(result, checkpoint=Checkpoint.from_directory(d))
        else:
            ray.tune.report(result)

    lrn.log_epoch = _ray_log_epoch
    try:
        yield
    finally:
        if had_instance_attr:
            lrn.log_epoch = orig_log_epoch
        else:
            del lrn.log_epoch


def ray_device() -> torch.device:
    """Detect the device assigned to this Ray worker."""
    gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    if gpu_ids and torch.cuda.is_available():
        return torch.device("cuda", int(gpu_ids[0]))
    return torch.device("cpu")


def resume_checkpoint(lrn) -> None:
    """Load checkpoint into Learner if Ray Tune is resuming this trial."""
    checkpoint = ray.tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as d:
            lrn.load_checkpoint(os.path.join(d, "checkpoint.pt"))


class LearnerTrainable(ray.tune.Trainable):
    """Ray Tune Trainable wrapping a tsfast Learner.

    Enables Population-Based Training and other schedulers that require
    the class-based Trainable API.

    Pass ``create_learner`` and optionally ``apply_config`` and
    ``scheduler_fn`` via ``tune.with_parameters``.

    Args (via tune.with_parameters):
        create_learner: factory ``(config) -> Learner``
        apply_config: optional ``(lrn, config) -> None`` that applies
            mutated hyperparameters to an existing Learner for actor reuse.
            If not provided, ``reset_config`` returns False (actor rebuilt).
        scheduler_fn: optional factory ``(optimizer, total_steps) -> scheduler``
            for LR scheduling across training iterations. When provided,
            ``config["n_iter"]`` must specify the total number of iterations.
    """

    def setup(self, config, create_learner=None, apply_config=None, scheduler_fn=None):
        self._create_learner = create_learner
        self._apply_config = apply_config
        self.lrn = create_learner(config)
        self.lrn.setup(
            lr=config.get("lr"),
            scheduler_fn=scheduler_fn,
            n_epoch=config.get("n_iter"),
        )
        self.lrn._show_bar = False

    def step(self):
        lrn = self.lrn
        train_loss = lrn.train_one_epoch(
            epoch=self.iteration,
            n_epoch=self.config.get("n_iter", 1),
        )
        val_loss, metrics = lrn.validate()
        lrn.recorder.append([train_loss, val_loss] + [metrics[k] for k in sorted(metrics)])
        return {"train_loss": train_loss, "valid_loss": val_loss, **metrics}

    def save_checkpoint(self, checkpoint_dir):
        self.lrn.save_checkpoint(os.path.join(checkpoint_dir, "checkpoint.pt"))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        self.lrn.load_checkpoint(os.path.join(checkpoint_dir, "checkpoint.pt"))

    def reset_config(self, new_config):
        if self._apply_config is None:
            return False
        self._apply_config(self.lrn, new_config)
        return True

    def cleanup(self):
        if hasattr(self, "lrn"):
            self.lrn._teardown_composables()


def trial_resources(k: int, n_cpus: float = 1.0) -> dict:
    """Ray Tune resources dict packing *k* trials per GPU: ``{"cpu": n_cpus, "gpu": 1/k}`` (packing step 4: enforce).

    *k* comes from the measure → decide → verify workflow in
    :mod:`tsfast.training.profiling` (``probe_gpu_saturation`` →
    ``recommend_trials_per_gpu`` → ``measure_packing_curve``).

    Fractional GPUs are a purely logical resource — Ray bin-packs k trials onto one
    device and sets CUDA_VISIBLE_DEVICES, with no memory or compute isolation. Pair
    with :func:`apply_gpu_quota` inside the trainable for actual memory enforcement.
    """
    return {"cpu": n_cpus, "gpu": 1.0 / k}


def apply_gpu_quota(share: float, context_bytes: int = 0, margin: float = 0.9, device=None) -> None:
    """Cap this process's CUDA caching allocator to its slice of the device (packing step 4: enforce).

    Call at the top of the trainable. Makes OOM deterministic: a config fails iff it
    exceeds its own slice, independent of which neighbors happen to run — so a sampled
    (non-guaranteed) worst-case footprint degrades gracefully instead of taking down
    co-located trials.

    The CUDA context (~0.5 GB per process on consumer cards) lives outside the torch
    allocator, so it is subtracted from each slice: with k = 1/share co-located
    processes the allocator caps plus the k contexts then sum to ``margin`` of device
    memory. With the default ``context_bytes=0`` the sum can exceed the device at
    high k — pass the probed ``SaturationProbe.context_overhead_bytes`` (or
    ``PackingRecommendation.context_overhead_bytes``).

    Args:
        share: this trial's GPU fraction, e.g. ``1/k``, or read at runtime via
            ``ray.tune.get_context().get_trial_resources().required_resources["GPU"]``
        context_bytes: per-process CUDA context overhead from the saturation probe
        margin: fraction of device memory budgeted across all co-located trials
        device: CUDA device (defaults to the current one)
    """
    total = torch.cuda.get_device_properties(device if device is not None else "cuda").total_memory
    fraction = margin * share - context_bytes / total
    if fraction <= 0:
        raise ValueError(
            f"GPU share {share} with context overhead {context_bytes / 2**30:.2f} GB"
            f" and margin {margin} leaves no allocator budget"
        )
    torch.cuda.set_per_process_memory_fraction(fraction, device)
