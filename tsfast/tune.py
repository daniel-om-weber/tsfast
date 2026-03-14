"""Ray Tune integration — bridge between Learner and Ray Tune's training API."""

__all__ = ["ray_device", "report_metrics", "resume_checkpoint"]

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
def report_metrics(lrn):
    """Patch a Learner's ``log_epoch`` to report metrics and checkpoints to Ray Tune.

    Use this context manager when you need full control over the training loop
    but still want per-epoch Ray Tune reporting and checkpointing.

    Args:
        lrn: a Learner instance whose ``log_epoch`` will be temporarily replaced.
    """
    had_instance_attr = "log_epoch" in lrn.__dict__
    orig_log_epoch = lrn.__dict__.get("log_epoch")

    def _ray_log_epoch(epoch, n_epoch, train_loss, val_loss, metrics, pbar):
        result = {"train_loss": train_loss, "valid_loss": val_loss}
        result.update(metrics)
        with tempfile.TemporaryDirectory() as d:
            lrn.save_checkpoint(os.path.join(d, "checkpoint.pt"))
            ray.tune.report(result, checkpoint=Checkpoint.from_directory(d))

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
