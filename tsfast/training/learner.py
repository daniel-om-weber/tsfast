"""Learner and TbpttLearner — pure-PyTorch training loop."""

__all__ = [
    "Learner",
    "TbpttLearner",
]

import math
import os
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..models.state import detach_state
from ..tsdata.pipeline import DataLoaders, get_signal_names
from . import viz
from .schedulers import sched_flat_cos


def _auto_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU.

    MPS is only used when the ``TSFAST_ENABLE_MPS`` environment variable is set,
    since MPS support is still buggy for many operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if os.environ.get("TSFAST_ENABLE_MPS") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
#  Learner
# ──────────────────────────────────────────────────────────────────────────────


class Learner:
    """Pure-PyTorch training loop for time-series models.

    Args:
        model: the model to train
        dls: train/valid/test DataLoaders
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
        show_bar: whether to show tqdm progress bars
    """

    def __init__(
        self,
        model: nn.Module,
        dls: DataLoaders,
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
        show_bar: bool = True,
    ):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.metrics = metrics or []
        self.lr = lr
        self.opt_func = opt_func
        self.transforms = transforms or []
        self.augmentations = augmentations or []
        self.aux_losses = aux_losses or []
        self.n_skip = n_skip
        self.grad_clip = grad_clip
        self.plot_fn = plot_fn or viz.plot_sequence
        dev = device or _auto_device()
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self.recorder: list[list[float]] = []
        self.opt = None
        self.sched = None
        self.pct_train: float = 0.0
        self._show_bar: bool = show_bar
        self._chunked_equiv_checked: set[int] = set()

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

    # ── setup ─────────────────────────────────────────────────────────────

    def setup(self, lr: float | None = None, scheduler_fn: Callable | None = None, n_epoch: int | None = None):
        """Create optimizer, move model to device, setup composables.

        Args:
            lr: learning rate (uses self.lr if None)
            scheduler_fn: factory ``(optimizer, total_steps) -> scheduler``
            n_epoch: total epochs — required when *scheduler_fn* is provided

        Enables manual training loops without calling ``fit()``.
        """
        self._setup_composables()
        self.model.to(self.device)
        self.opt = self.opt_func(self.model.parameters(), lr=lr or self.lr)
        pending = getattr(self, "_pending_opt_state", None)
        if pending is not None:
            self.opt.load_state_dict(pending)
            self._pending_opt_state = None
        if scheduler_fn is not None:
            n_batches = len(self.dls.train)
            self.sched = scheduler_fn(self.opt, n_epoch * n_batches)
        else:
            self.sched = None

    # ── save / load ────────────────────────────────────────────────────────

    def save_model(self, path: str | Path):
        """Save model for inference. Includes weights and normalization state."""
        torch.save(self.model, path)

    def save(self, path: str | Path):
        """Save entire learner state for training resume.

        Pickles everything except ``dls`` (DataLoaders cannot be serialized).
        If pickling fails (e.g. lambda loss functions), use
        ``save_checkpoint`` / ``load_checkpoint`` instead.
        """
        state = {k: v for k, v in self.__dict__.items() if k != "dls"}
        state["_class"] = type(self)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, dls: "DataLoaders") -> "Learner":
        """Load a saved learner to resume training.

        Args:
            path: checkpoint file saved by ``save()``
            dls: DataLoaders (must match the original training data layout)
        """
        state = torch.load(path, map_location=_auto_device(), weights_only=False)
        klass = state.pop("_class", cls)
        lrn = klass.__new__(klass)
        lrn.__dict__.update(state)
        lrn.dls = dls
        return lrn

    def save_checkpoint(self, path: str | Path):
        """Save model weights, optimizer state, and training history.

        Fallback for learners with unpicklable components (e.g. lambda losses).
        Use ``load_checkpoint`` to restore into a manually constructed Learner.
        """
        state = {"model": self.model.state_dict(), "recorder": self.recorder}
        if self.opt is not None:
            state["opt"] = self.opt.state_dict()
        if self.sched is not None:
            state["sched"] = self.sched.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path):
        """Load model weights, optimizer state, and training history.

        Restores state saved by ``save_checkpoint`` into this Learner.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.recorder = ckpt.get("recorder", [])
        opt_state = ckpt.get("opt")
        if opt_state is not None:
            if self.opt is not None:
                self.opt.load_state_dict(opt_state)
            else:
                self._pending_opt_state = opt_state
        sched_state = ckpt.get("sched")
        if sched_state is not None and self.sched is not None:
            self.sched.load_state_dict(sched_state)

    # ── batch preparation ─────────────────────────────────────────────────

    def prepare_batch(self, batch, training: bool = True) -> tuple[Tensor, Tensor]:
        """Device transfer + transforms + augmentations (if training)."""
        xb, yb = (t.to(self.device) for t in batch)
        for t in self.transforms:  # feature: transforms
            xb, yb = t(xb, yb)
        if training:
            for a in self.augmentations:  # feature: augmentations
                xb, yb = a(xb, yb)
        return xb, yb

    # ── loss computation ──────────────────────────────────────────────────

    def compute_loss(self, pred: Tensor, yb: Tensor, xb: Tensor, n_skip: int | None = None) -> Tensor:
        """Primary loss with n_skip + auxiliary losses."""
        if n_skip is None:
            n_skip = self.n_skip

        pred_skip = pred[:, n_skip:] if n_skip > 0 else pred  # feature: n_skip
        yb_skip = yb[:, n_skip:] if n_skip > 0 else yb  # feature: n_skip
        loss = self.loss_func(pred_skip, yb_skip)

        for aux in self.aux_losses:  # feature: auxiliary losses
            loss = loss + aux(pred, yb, xb)

        return loss

    # ── backward + optimizer step ─────────────────────────────────────────

    def backward_step(self, loss: Tensor):
        """Backward + grad_clip + optimizer step + zero_grad."""
        loss.backward()
        if self.grad_clip is not None:  # feature: gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()

    # ── training step ─────────────────────────────────────────────────────

    def training_step(self, xb: Tensor, yb: Tensor) -> float:
        """Forward + compute_loss + NaN check + backward_step.

        Returns:
            loss value, or NaN if loss was NaN (step is skipped)
        """
        result = self.model(xb)
        if isinstance(result, tuple):
            pred = result[0]
        else:
            pred = result

        loss = self.compute_loss(pred, yb, xb)

        if torch.isnan(loss):  # feature: NaN guard
            self.opt.zero_grad()
            return float("nan")

        self.backward_step(loss)
        return loss.item()

    # ── epoch loop ────────────────────────────────────────────────────────

    def train_one_epoch(self, pbar=None, epoch: int = 0, n_epoch: int = 1) -> float:
        """Run one training epoch.

        Args:
            pbar: optional tqdm progress bar
            epoch: current epoch index (0-based)
            n_epoch: total number of epochs

        Returns:
            mean training loss for the epoch
        """
        self.model.train()
        train_losses = []
        n_batches = len(self.dls.train)
        total_steps = max(1, n_epoch * n_batches)

        for batch_idx, batch in enumerate(self.dls.train):
            xb, yb = self.prepare_batch(batch, training=True)
            self.pct_train = (epoch * n_batches + batch_idx) / total_steps  # feature: training progress

            loss_val = self.training_step(xb, yb)
            if not math.isnan(loss_val):
                train_losses.append(loss_val)

            if self.sched is not None:  # feature: LR scheduling
                self.sched.step()
            if pbar is not None:
                pbar.update(1)

        return sum(train_losses) / max(1, len(train_losses))

    # ── validation ────────────────────────────────────────────────────────

    def validate(self, dl=None, chunk_sz: int | None = None) -> tuple[float, dict[str, float]]:
        """Run validation and compute loss + metrics on concatenated predictions.

        Args:
            dl: DataLoader to evaluate (defaults to validation set)
            chunk_sz: forwarded to :meth:`get_preds` for chunked evaluation

        Returns:
            (val_loss, {metric_name: value})
        """
        dl = dl or self.dls.valid
        preds, targs = self.get_preds(dl=dl, chunk_sz=chunk_sz)

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
        scheduler_fn: Callable | None = None,
    ):
        """Train for n_epoch epochs.

        Args:
            n_epoch: number of epochs
            lr: learning rate (uses self.lr if None)
            scheduler_fn: factory ``(optimizer, total_steps) -> scheduler`` (None = no scheduler)
        """
        self.setup(lr=lr, scheduler_fn=scheduler_fn, n_epoch=n_epoch)

        n_batches = len(self.dls.train)

        try:
            for epoch in range(n_epoch):
                with tqdm(
                    total=n_batches,
                    desc=f"Epoch {epoch + 1}/{n_epoch}",
                    disable=not self._show_bar,
                    mininterval=0.5,
                ) as pbar:
                    train_loss = self.train_one_epoch(pbar=pbar, epoch=epoch, n_epoch=n_epoch)

                    # Validate
                    val_loss, metrics_dict = self.validate()

                    # Record
                    row = [train_loss, val_loss] + [metrics_dict[k] for k in sorted(metrics_dict)]
                    self.recorder.append(row)
                    self.log_epoch(epoch, n_epoch, train_loss, val_loss, metrics_dict, pbar)
        finally:
            self._teardown_composables()

    def fit_flat_cos(self, n_epoch: int, lr: float | None = None, pct_start: float = 0.75):
        """Convenience: flat LR then cosine decay."""
        self.fit(
            n_epoch,
            lr=lr,
            scheduler_fn=lambda opt, steps: LambdaLR(opt, lambda s: sched_flat_cos(s / steps, pct_start)),
        )

    # ── logging ───────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, n_epoch: int, train_loss: float, val_loss: float, metrics: dict, pbar):
        """Log epoch results. Override for custom logging."""
        parts = [f"train={train_loss:.4f}", f"valid={val_loss:.4f}"]
        for k, v in sorted(metrics.items()):
            parts.append(f"{k}={v:.4f}")
        pbar.set_postfix_str(" | ".join(parts))

    # ── chunked-evaluation helpers ─────────────────────────────────────────

    def _check_chunked_equivalence(self, chunk_sz: int, dl):
        """Probe whether chunked evaluation matches full forward pass.

        Runs a small synthetic input through the model both as a single forward
        pass and as two chunks, then compares results.  Warns if they diverge —
        typically because the model is stateful but ``return_state`` is not
        enabled, or because the model uses convolutions whose receptive field
        spans the chunk boundary.  Results are cached per *chunk_sz* so the
        probe (and the DataLoader iteration it requires) only runs once.
        """
        if chunk_sz in self._chunked_equiv_checked:
            return
        self._chunked_equiv_checked.add(chunk_sz)

        n_in = next(iter(dl))[0].shape[-1]
        x = torch.randn(1, 2 * chunk_sz, n_in, device=self.device)
        with torch.no_grad():
            full_result = self.model(x)
            full = full_result[0] if isinstance(full_result, tuple) else full_result

            r1 = self.model(x[:, :chunk_sz, :])
            if isinstance(r1, tuple):
                p1, state = r1
                r2 = self.model(x[:, chunk_sz:, :], state=state)
            else:
                p1 = r1
                r2 = self.model(x[:, chunk_sz:, :])
            p2 = r2[0] if isinstance(r2, tuple) else r2
            chunked = torch.cat([p1, p2], dim=1)

        if not torch.allclose(full, chunked, atol=1e-4):
            warnings.warn(
                "Chunked evaluation produces different results than a full forward pass "
                "for this model.  For RNNs, set return_state=True so hidden state is "
                "carried across chunks.  For convolutional models (TCN/CNN), use a larger "
                "chunk_sz or avoid chunking — receptive-field effects at chunk boundaries "
                "cause small numerical differences.",
                stacklevel=2,
            )

    # ── predictions ───────────────────────────────────────────────────────

    def get_preds(self, dl=None, with_inputs: bool = False, chunk_sz: int | None = None):
        """Batch-concatenated predictions and targets.

        Args:
            dl: DataLoader to evaluate (defaults to validation set)
            with_inputs: if True, also return concatenated inputs
            chunk_sz: when set, split each batch's sequence into chunks of this
                size along the time axis and forward them sequentially, carrying
                model state across chunks (for RNNs). Keeps GPU memory bounded
                for very long sequences.
        """
        dl = dl or self.dls.valid
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        self.model.eval()
        all_preds, all_targs, all_inputs = [], [], []

        if chunk_sz is not None:
            self._check_chunked_equivalence(chunk_sz, dl)

        with torch.no_grad():
            for batch in dl:
                xb, yb = self.prepare_batch(batch, training=False)

                if chunk_sz is None or xb.shape[1] <= chunk_sz:
                    result = self.model(xb)
                    pred = result[0] if isinstance(result, tuple) else result
                else:
                    xb_chunks = xb.split(chunk_sz, dim=1)
                    chunk_preds = []
                    state = None
                    for xb_sub in xb_chunks:
                        if state is not None:
                            result = self.model(xb_sub, state=state)
                        else:
                            result = self.model(xb_sub)
                        if isinstance(result, tuple):
                            p, state = result
                        else:
                            p, state = result, None
                        chunk_preds.append(p)
                    pred = torch.cat(chunk_preds, dim=1)

                all_preds.append(pred.cpu())
                all_targs.append(yb.cpu())
                if with_inputs:
                    all_inputs.append(xb.cpu())

        preds = torch.cat(all_preds, dim=0)
        targs = torch.cat(all_targs, dim=0)
        if with_inputs:
            return preds, targs, torch.cat(all_inputs, dim=0)
        return preds, targs

    # ── worst-sample selection ──────────────────────────────────────────

    def get_worst(self, max_n: int = 4, dl=None) -> tuple[Tensor, Tensor, Tensor]:
        """Inputs, targets, and predictions for the samples with highest loss.

        Returns:
            (inputs, targets, predictions) sliced to the ``max_n`` worst samples
        """
        preds, targs, inputs = self.get_preds(dl=dl, with_inputs=True)
        if hasattr(self.loss_func, "reduction"):
            orig = self.loss_func.reduction
            self.loss_func.reduction = "none"
            raw = self.loss_func(preds, targs)
            self.loss_func.reduction = orig
            per_sample = raw.reshape(len(preds), -1).mean(dim=1)
        else:
            per_sample = torch.tensor(
                [self.loss_func(preds[i : i + 1], targs[i : i + 1]).item() for i in range(len(preds))]
            )
        idxs = per_sample.argsort(descending=True)[:max_n]
        return inputs[idxs], targs[idxs], preds[idxs]

    # ── visualization ─────────────────────────────────────────────────────

    def show_batch(self, max_n: int = 4, dl=None):
        """Plot a batch of input/target pairs."""
        dl = dl or self.dls.valid
        batch = next(iter(dl))
        xb, yb = self.prepare_batch(batch, training=False)

        n = min(xb.shape[0], max_n)
        samples = [(xb[i].cpu(), yb[i].cpu()) for i in range(n)]
        viz.layout_samples(n, yb.shape[-1], samples, self.plot_fn, signal_names=get_signal_names(dl))

    def show_results(self, max_n: int = 4, dl=None):
        """Plot predictions vs targets."""
        dl = dl or self.dls.valid
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        self.model.eval()

        batch = next(iter(dl))
        xb, yb = self.prepare_batch(batch, training=False)

        with torch.no_grad():
            result = self.model(xb)
            pred = result[0] if isinstance(result, tuple) else result

        n = min(xb.shape[0], max_n)
        samples = [(xb[i].cpu(), yb[i].cpu()) for i in range(n)]
        outs = [(pred[i].cpu(),) for i in range(n)]
        viz.layout_samples(n, yb.shape[-1], samples, self.plot_fn, outs, signal_names=get_signal_names(dl))

    def show_worst(self, max_n: int = 4, dl=None):
        """Plot samples with highest per-sample loss."""
        dl = dl or self.dls.valid
        inputs, targs, preds = self.get_worst(max_n=max_n, dl=dl)
        samples = [(inputs[i], targs[i]) for i in range(len(inputs))]
        outs = [(preds[i],) for i in range(len(preds))]
        viz.layout_samples(len(inputs), targs.shape[-1], samples, self.plot_fn, outs, signal_names=get_signal_names(dl))


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

    def training_step(self, xb: Tensor, yb: Tensor) -> float:
        """TBPTT training step: chunk input, forward/backward per chunk with carried state."""
        xb_chunks = xb.split(self.sub_seq_len, dim=1)
        yb_chunks = yb.split(self.sub_seq_len, dim=1)

        state = None
        losses = []
        for i, (xb_sub, yb_sub) in enumerate(zip(xb_chunks, yb_chunks)):
            # n_skip only applies to the first sub-sequence (RNN warmup);
            # subsequent chunks already have a warmed-up hidden state.
            skip = self.n_skip if i == 0 else 0

            # Forward
            if state is not None:
                result = self.model(xb_sub, state=state)
            else:
                result = self.model(xb_sub)

            if isinstance(result, tuple):
                pred, new_state = result
            else:
                pred, new_state = result, None

            loss = self.compute_loss(pred, yb_sub, xb_sub, n_skip=skip)

            if torch.isnan(loss):  # feature: NaN guard
                self.opt.zero_grad()
                state = None
                continue

            self.backward_step(loss)
            losses.append(loss.item())
            state = detach_state(new_state)

        if not losses:
            return float("nan")
        return sum(losses) / len(losses)

    def get_preds(self, dl=None, with_inputs: bool = False, chunk_sz: int | None = None):
        """Defaults ``chunk_sz`` to ``sub_seq_len`` so validation reuses CUDA graph shapes."""
        if chunk_sz is None:
            chunk_sz = self.sub_seq_len
        return super().get_preds(dl=dl, with_inputs=with_inputs, chunk_sz=chunk_sz)
