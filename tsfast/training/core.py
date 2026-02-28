"""Learner, TbpttLearner, and Recorder — pure-PyTorch training loop."""

__all__ = [
    "Learner",
    "TbpttLearner",
    "Recorder",
]

import math
import os
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .viz import layout_samples, plot_sequence


# ──────────────────────────────────────────────────────────────────────────────
#  DataLoaders adapter
# ──────────────────────────────────────────────────────────────────────────────


class _DlsAdapter:
    """Wraps old-style fastai DataLoaders to expose ``.train``, ``.valid``, ``.test``."""

    def __init__(self, dls):
        self._dls = dls

    @property
    def train(self):
        return self._dls.loaders[0]

    @property
    def valid(self):
        return self._dls.loaders[1]

    @property
    def test(self):
        return self._dls.loaders[2] if len(self._dls.loaders) > 2 else None

    @property
    def norm_stats(self):
        return self._dls.norm_stats

    def one_batch(self):
        return self._dls.one_batch()


def _wrap_dls(dls):
    """If *dls* already has ``.train``/``.valid`` attributes, return as-is; otherwise wrap."""
    if hasattr(dls, "train") and hasattr(dls, "valid"):
        return dls
    return _DlsAdapter(dls)


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _auto_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU.

    MPS is only used when the ``TSFAST_ENABLE_MPS`` environment variable is set,
    since MPS support is still buggy for many operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("TSFAST_ENABLE_MPS") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_flat_cos_scheduler(optimizer, total_steps: int, pct_start: float = 0.75) -> LambdaLR:
    """Flat LR for pct_start fraction, then cosine decay to zero. Stepped per batch."""
    flat_steps = int(total_steps * pct_start)

    def _lr_lambda(step):
        if step < flat_steps:
            return 1.0
        progress = (step - flat_steps) / max(1, total_steps - flat_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, _lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
#  Recorder
# ──────────────────────────────────────────────────────────────────────────────


class Recorder:
    """Stores training history: ``values[epoch] = [train_loss, valid_loss, *metrics]``."""

    def __init__(self):
        self.values: list[list[float]] = []

    def append(self, row: list[float]):
        self.values.append(row)


# ──────────────────────────────────────────────────────────────────────────────
#  Learner
# ──────────────────────────────────────────────────────────────────────────────


class Learner:
    """Pure-PyTorch training loop for time-series models.

    Args:
        model: the model to train
        dls: DataLoaders container with ``.train`` and ``.valid`` attributes
        loss_func: primary loss function
        metrics: list of metric functions ``(pred, targ) -> scalar``
        lr: default learning rate
        opt_func: optimizer class
        transforms: list of ``(xb, yb) -> (xb, yb)`` applied to train + valid
        augmentations: list of ``(xb, yb) -> (xb, yb)`` applied to train only
        aux_losses: list of ``(pred, yb, xb) -> loss_term`` added to primary loss
        n_skip: number of initial time steps to skip in loss computation
        grad_clip: maximum gradient norm (None disables clipping)
        plot_fn: plotting function for show_batch/show_results
        device: target device (auto-detected if None)
    """

    def __init__(
        self,
        model: nn.Module,
        dls,
        loss_func: Callable,
        metrics: list[Callable] | None = None,
        lr: float = 3e-3,
        opt_func: type = torch.optim.Adam,
        transforms: list | None = None,
        augmentations: list | None = None,
        aux_losses: list | None = None,
        n_skip: int = 0,
        grad_clip: float | None = None,
        plot_fn: Callable | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.dls = _wrap_dls(dls)
        self.loss_func = loss_func
        self.metrics = metrics or []
        self.lr = lr
        self.opt_func = opt_func
        self.transforms = transforms or []
        self.augmentations = augmentations or []
        self.aux_losses = aux_losses or []
        self.n_skip = n_skip
        self.grad_clip = grad_clip
        self.plot_fn = plot_fn or plot_sequence
        self.device = device or _auto_device()
        self.recorder = Recorder()
        self._pct_train: float = 0.0
        self._show_bar: bool = True

    # ── post-construction helpers ─────────────────────────────────────────

    def add_aux_loss(self, obj):
        """Append an auxiliary loss composable."""
        self.aux_losses.append(obj)

    def add_transform(self, obj):
        """Append a transform composable (applied train + valid)."""
        self.transforms.append(obj)

    def add_augmentation(self, obj):
        """Append an augmentation composable (applied train only)."""
        self.augmentations.append(obj)

    # ── properties ────────────────────────────────────────────────────────

    @property
    def pct_train(self) -> float:
        return self._pct_train

    @pct_train.setter
    def pct_train(self, value: float):
        self._pct_train = value

    # ── context managers ──────────────────────────────────────────────────

    @contextmanager
    def no_bar(self):
        """Suppress tqdm progress bars (useful for Ray Tune)."""
        prev = self._show_bar
        self._show_bar = False
        try:
            yield
        finally:
            self._show_bar = prev

    # ── composable setup/teardown ─────────────────────────────────────────

    def _setup_composables(self):
        for obj in self.transforms + self.augmentations + self.aux_losses:
            if hasattr(obj, "setup"):
                obj.setup(self)

    def _teardown_composables(self):
        for obj in self.transforms + self.augmentations + self.aux_losses:
            if hasattr(obj, "teardown"):
                obj.teardown(self)

    # ── device helpers ────────────────────────────────────────────────────

    def _to_device(self, batch) -> tuple[Tensor, ...]:
        # Strip custom tensor subclasses (e.g. TensorSequencesInput) so that
        # standard loss functions like nn.L1Loss work without __torch_function__
        return tuple(t.to(self.device).as_subclass(Tensor) for t in batch)

    def _get_dl(self, ds_idx: int):
        """Get DataLoader by index: 0=train, 1=valid, 2+=test, -1=last."""
        if ds_idx == 0:
            return self.dls.train
        elif ds_idx == 1:
            return self.dls.valid
        elif ds_idx == -1:
            return self.dls.test if self.dls.test is not None else self.dls.valid
        elif ds_idx >= 2 and self.dls.test is not None:
            return self.dls.test
        return self.dls.valid

    # ── training step ─────────────────────────────────────────────────────

    def training_step(self, batch: tuple[Tensor, Tensor], optimizer, state=None) -> tuple[float | None, object]:
        """Single training step: forward, loss, backward, step.

        Returns:
            (loss_value or None if NaN, new_state or None)
        """
        xb, yb = batch

        # Apply transforms then augmentations
        for t in self.transforms:
            xb, yb = t(xb, yb)
        for a in self.augmentations:
            xb, yb = a(xb, yb)

        # Forward
        if state is not None:
            result = self.model(xb, state=state)
        else:
            result = self.model(xb)

        if isinstance(result, tuple):
            pred, new_state = result
        else:
            pred, new_state = result, None

        # Primary loss with n_skip
        pred_skip = pred[:, self.n_skip :] if self.n_skip > 0 else pred
        yb_skip = yb[:, self.n_skip :] if self.n_skip > 0 else yb
        loss = self.loss_func(pred_skip, yb_skip)

        # Aux losses
        for aux in self.aux_losses:
            loss = loss + aux(pred, yb, xb)

        # NaN check
        if torch.isnan(loss):
            optimizer.zero_grad()
            return None, None

        # Backward + step
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        from ..models.layers import _detach_state

        return loss.item(), _detach_state(new_state)

    # ── validation ────────────────────────────────────────────────────────

    def validate(self, dl=None) -> tuple[float, dict[str, float]]:
        """Run validation and compute loss + metrics on concatenated predictions.

        Returns:
            (val_loss, {metric_name: value})
        """
        dl = dl or self.dls.valid
        self.model.eval()
        all_preds, all_targs = [], []

        with torch.no_grad():
            for batch in dl:
                xb, yb = self._to_device(batch)
                for t in self.transforms:
                    xb, yb = t(xb, yb)

                result = self.model(xb)
                pred = result[0] if isinstance(result, tuple) else result

                all_preds.append(pred.cpu())
                all_targs.append(yb.cpu())

        preds = torch.cat(all_preds, dim=0)
        targs = torch.cat(all_targs, dim=0)

        pred_skip = preds[:, self.n_skip :] if self.n_skip > 0 else preds
        targ_skip = targs[:, self.n_skip :] if self.n_skip > 0 else targs

        val_loss = self.loss_func(pred_skip, targ_skip).item()

        metrics_dict = {}
        for m in self.metrics:
            name = getattr(m, "__name__", type(m).__name__)
            metrics_dict[name] = m(pred_skip, targ_skip).item()

        return val_loss, metrics_dict

    # ── fit methods ───────────────────────────────────────────────────────

    def fit(
        self,
        n_epoch: int,
        lr: float | None = None,
        make_scheduler: Callable | None = None,
    ):
        """Train for n_epoch epochs.

        Args:
            n_epoch: number of epochs
            lr: learning rate (uses self.lr if None)
            make_scheduler: factory ``(optimizer, total_steps) -> scheduler`` (None = no scheduler)
        """
        lr = lr or self.lr
        self.model.to(self.device)
        optimizer = self.opt_func(self.model.parameters(), lr=lr)

        n_batches = len(self.dls.train)
        total_steps = n_epoch * n_batches
        scheduler = make_scheduler(optimizer, total_steps) if make_scheduler is not None else None

        self._setup_composables()
        try:
            step = 0
            for epoch in range(n_epoch):
                # Train
                self.model.train()
                train_losses = []
                with tqdm(
                    total=n_batches, desc=f"Epoch {epoch + 1}/{n_epoch}",
                    disable=not self._show_bar, mininterval=0.5,
                ) as pbar:
                    for batch in self.dls.train:
                        xb, yb = self._to_device(batch)
                        self._pct_train = step / max(1, total_steps)
                        loss_val, _ = self.training_step((xb, yb), optimizer)
                        if loss_val is not None:
                            train_losses.append(loss_val)
                        if scheduler is not None:
                            scheduler.step()
                        step += 1
                        pbar.update(1)

                    train_loss = sum(train_losses) / max(1, len(train_losses))

                    # Validate
                    val_loss, metrics_dict = self.validate()

                    # Record
                    row = [train_loss, val_loss] + [metrics_dict[k] for k in sorted(metrics_dict)]
                    self.recorder.append(row)
                    self._log_epoch(epoch, train_loss, val_loss, metrics_dict, pbar)
        finally:
            self._teardown_composables()

    def fit_flat_cos(self, n_epoch: int, lr: float | None = None, pct_start: float = 0.75):
        """Convenience: flat LR then cosine decay, matching fastai fit_flat_cos."""
        self.fit(
            n_epoch,
            lr=lr,
            make_scheduler=lambda opt, steps: _make_flat_cos_scheduler(opt, steps, pct_start),
        )

    # ── logging ───────────────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: dict, pbar):
        """Log epoch results. Override for Ray Tune or custom logging."""
        parts = [f"train={train_loss:.4f}", f"valid={val_loss:.4f}"]
        for k, v in sorted(metrics.items()):
            parts.append(f"{k}={v:.4f}")
        pbar.set_postfix_str(" | ".join(parts))

    # ── predictions ───────────────────────────────────────────────────────

    def get_preds(self, ds_idx: int = 1) -> tuple[Tensor, Tensor]:
        """Batch-concatenated predictions and targets.

        Args:
            ds_idx: DataLoader index (0=train, 1=valid)
        """
        dl = self._get_dl(ds_idx)
        self.model.to(self.device)
        self.model.eval()
        all_preds, all_targs = [], []

        with torch.no_grad():
            for batch in dl:
                xb, yb = self._to_device(batch)
                for t in self.transforms:
                    xb, yb = t(xb, yb)

                result = self.model(xb)
                pred = result[0] if isinstance(result, tuple) else result

                all_preds.append(pred.cpu())
                all_targs.append(yb.cpu())

        return torch.cat(all_preds, dim=0), torch.cat(all_targs, dim=0)

    # ── visualization ─────────────────────────────────────────────────────

    def show_batch(self, max_n: int = 4, dl=None):
        """Plot a batch of input/target pairs."""
        dl = dl or self.dls.valid
        batch = next(iter(dl))
        xb, yb = self._to_device(batch)
        for t in self.transforms:
            xb, yb = t(xb, yb)

        n_samples = min(xb.shape[0], max_n)
        n_targ = yb.shape[-1]
        samples = [(xb[i].cpu(), yb[i].cpu()) for i in range(n_samples)]
        layout_samples(n_samples, n_targ, samples, self.plot_fn)

    def show_results(self, max_n: int = 4, ds_idx: int = 1):
        """Plot predictions vs targets."""
        dl = self._get_dl(ds_idx)
        self.model.to(self.device)
        self.model.eval()

        batch = next(iter(dl))
        xb, yb = self._to_device(batch)
        for t in self.transforms:
            xb, yb = t(xb, yb)

        with torch.no_grad():
            result = self.model(xb)
            pred = result[0] if isinstance(result, tuple) else result

        n_samples = min(xb.shape[0], max_n)
        n_targ = yb.shape[-1]
        samples = [(xb[i].cpu(), yb[i].cpu()) for i in range(n_samples)]
        outs = [(pred[i].cpu(),) for i in range(n_samples)]
        layout_samples(n_samples, n_targ, samples, self.plot_fn, outs)

    def show_worst(self, max_n: int = 4, ds_idx: int = 1):
        """Plot samples with highest per-sample loss."""
        dl = self._get_dl(ds_idx)
        self.model.to(self.device)
        self.model.eval()

        all_preds, all_targs, all_inputs = [], [], []
        with torch.no_grad():
            for batch in dl:
                xb, yb = self._to_device(batch)
                for t in self.transforms:
                    xb, yb = t(xb, yb)
                result = self.model(xb)
                pred = result[0] if isinstance(result, tuple) else result
                all_preds.append(pred.cpu())
                all_targs.append(yb.cpu())
                all_inputs.append(xb.cpu())

        preds = torch.cat(all_preds, dim=0)
        targs = torch.cat(all_targs, dim=0)
        inputs = torch.cat(all_inputs, dim=0)

        # Per-sample loss
        per_sample = torch.tensor(
            [self.loss_func(preds[i : i + 1], targs[i : i + 1]).item() for i in range(preds.shape[0])]
        )
        idxs = per_sample.argsort(descending=True)[:max_n]

        n_targ = targs.shape[-1]
        samples = [(inputs[i], targs[i]) for i in idxs]
        outs = [(preds[i],) for i in idxs]
        layout_samples(len(idxs), n_targ, samples, self.plot_fn, outs)


# ──────────────────────────────────────────────────────────────────────────────
#  TbpttLearner
# ──────────────────────────────────────────────────────────────────────────────


class TbpttLearner(Learner):
    """Learner with truncated backpropagation through time (TBPTT).

    Full sequences are loaded from the DataLoader, then split into
    sub-sequences of ``sub_seq_len``. Hidden state is carried across
    sub-sequences within a batch but reset between batches.

    Args:
        sub_seq_len: length of each sub-sequence chunk
    """

    def __init__(self, *args, sub_seq_len: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_seq_len = sub_seq_len

    def fit(
        self,
        n_epoch: int,
        lr: float | None = None,
        make_scheduler: Callable | None = None,
    ):
        lr = lr or self.lr
        self.model.to(self.device)
        optimizer = self.opt_func(self.model.parameters(), lr=lr)

        n_batches = len(self.dls.train)
        # Estimate total optimizer steps (each batch may have multiple sub-seq steps)
        # Use n_batches as a rough estimate for scheduler; actual step count varies
        total_steps = n_epoch * n_batches
        scheduler = make_scheduler(optimizer, total_steps) if make_scheduler is not None else None

        self._setup_composables()
        try:
            step = 0
            for epoch in range(n_epoch):
                self.model.train()
                train_losses = []
                with tqdm(
                    total=n_batches, desc=f"Epoch {epoch + 1}/{n_epoch}",
                    disable=not self._show_bar, mininterval=0.5,
                ) as pbar:
                    for batch in self.dls.train:
                        xb, yb = self._to_device(batch)

                        # Apply transforms + augmentations on full sequence
                        for t in self.transforms:
                            xb, yb = t(xb, yb)
                        for a in self.augmentations:
                            xb, yb = a(xb, yb)

                        # Split into sub-sequences
                        xb_chunks = xb.split(self.sub_seq_len, dim=1)
                        yb_chunks = yb.split(self.sub_seq_len, dim=1)

                        state = None  # reset state at each new batch
                        self._pct_train = step / max(1, total_steps)

                        for xb_sub, yb_sub in zip(xb_chunks, yb_chunks):
                            loss_val, state = self.training_step((xb_sub, yb_sub), optimizer, state)
                            if loss_val is not None:
                                train_losses.append(loss_val)

                        if scheduler is not None:
                            scheduler.step()
                        step += 1
                        pbar.update(1)

                    train_loss = sum(train_losses) / max(1, len(train_losses))
                    val_loss, metrics_dict = self.validate()

                    row = [train_loss, val_loss] + [metrics_dict[k] for k in sorted(metrics_dict)]
                    self.recorder.append(row)
                    self._log_epoch(epoch, train_loss, val_loss, metrics_dict, pbar)
        finally:
            self._teardown_composables()
