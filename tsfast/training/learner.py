"""Learner, TbpttLearner, and Recorder — pure-PyTorch training loop."""

__all__ = [
    "Learner",
    "TbpttLearner",
    "CudaGraphTbpttLearner",
    "Recorder",
]

import os
import warnings
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..tsdata.pipeline import DataLoaders, get_signal_names
from . import viz
from .schedulers import sched_flat_cos


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _detach_state(state):
    """Recursively detach tensors from the computation graph."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (list, tuple)):
        return type(state)(_detach_state(s) for s in state)
    if isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    return state


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
        self.device = device or _auto_device()
        self.recorder = Recorder()
        self.pct_train: float = 0.0
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

    def training_step(
        self, batch: tuple[Tensor, Tensor], optimizer, state=None, n_skip: int | None = None
    ) -> tuple[float | None, object]:
        """Single training step: forward, loss, backward, step.

        Args:
            n_skip: timesteps to skip in loss (defaults to ``self.n_skip``)

        Returns:
            (loss_value or None if NaN, new_state or None)
        """
        if n_skip is None:
            n_skip = self.n_skip
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
        pred_skip = pred[:, n_skip:] if n_skip > 0 else pred
        yb_skip = yb[:, n_skip:] if n_skip > 0 else yb
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

        return loss.item(), _detach_state(new_state)

    # ── validation ────────────────────────────────────────────────────────

    def validate(self, dl=None) -> tuple[float, dict[str, float]]:
        """Run validation and compute loss + metrics on concatenated predictions.

        Returns:
            (val_loss, {metric_name: value})
        """
        dl = dl or self.dls.valid
        preds, targs = self.get_preds(dl=dl)

        pred_skip = preds[:, self.n_skip :] if self.n_skip > 0 else preds
        targ_skip = targs[:, self.n_skip :] if self.n_skip > 0 else targs

        val_loss = self.loss_func(pred_skip, targ_skip).item()

        metrics_dict = {}
        for m in self.metrics:
            name = getattr(m, "__name__", type(m).__name__)
            metrics_dict[name] = m(pred_skip, targ_skip).item()

        return val_loss, metrics_dict

    # ── fit methods ───────────────────────────────────────────────────────

    def _train_one_batch(self, batch, optimizer, step: int, total_steps: int) -> list[float]:
        """Process one training batch. Override for custom batch processing (e.g. TBPTT)."""
        xb, yb = self._to_device(batch)
        self.pct_train = step / max(1, total_steps)
        loss_val, _ = self.training_step((xb, yb), optimizer)
        return [loss_val] if loss_val is not None else []

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
                self.model.train()
                train_losses = []
                with tqdm(
                    total=n_batches,
                    desc=f"Epoch {epoch + 1}/{n_epoch}",
                    disable=not self._show_bar,
                    mininterval=0.5,
                ) as pbar:
                    for batch in self.dls.train:
                        train_losses.extend(self._train_one_batch(batch, optimizer, step, total_steps))
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
        """Convenience: flat LR then cosine decay."""
        self.fit(
            n_epoch,
            lr=lr,
            make_scheduler=lambda opt, steps: LambdaLR(opt, lambda s: sched_flat_cos(s / steps, pct_start)),
        )

    # ── logging ───────────────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: dict, pbar):
        """Log epoch results. Override for Ray Tune or custom logging."""
        parts = [f"train={train_loss:.4f}", f"valid={val_loss:.4f}"]
        for k, v in sorted(metrics.items()):
            parts.append(f"{k}={v:.4f}")
        pbar.set_postfix_str(" | ".join(parts))

    # ── predictions ───────────────────────────────────────────────────────

    def get_preds(self, ds_idx: int = 1, dl=None, with_inputs: bool = False):
        """Batch-concatenated predictions and targets.

        Args:
            ds_idx: DataLoader index (0=train, 1=valid)
            dl: explicit DataLoader (overrides ds_idx)
            with_inputs: if True, also return concatenated inputs
        """
        dl = dl or self._get_dl(ds_idx)
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
                if with_inputs:
                    all_inputs.append(xb.cpu())

        preds = torch.cat(all_preds, dim=0)
        targs = torch.cat(all_targs, dim=0)
        if with_inputs:
            return preds, targs, torch.cat(all_inputs, dim=0)
        return preds, targs

    # ── worst-sample selection ──────────────────────────────────────────

    def get_worst(self, max_n: int = 4, ds_idx: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        """Inputs, targets, and predictions for the samples with highest loss.

        Returns:
            (inputs, targets, predictions) sliced to the ``max_n`` worst samples
        """
        preds, targs, inputs = self.get_preds(ds_idx=ds_idx, with_inputs=True)
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
        xb, yb = self._to_device(batch)
        for t in self.transforms:
            xb, yb = t(xb, yb)

        n = min(xb.shape[0], max_n)
        samples = [(xb[i].cpu(), yb[i].cpu()) for i in range(n)]
        viz.layout_samples(n, yb.shape[-1], samples, self.plot_fn, signal_names=get_signal_names(dl))

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

        n = min(xb.shape[0], max_n)
        samples = [(xb[i].cpu(), yb[i].cpu()) for i in range(n)]
        outs = [(pred[i].cpu(),) for i in range(n)]
        viz.layout_samples(n, yb.shape[-1], samples, self.plot_fn, outs, signal_names=get_signal_names(dl))

    def show_worst(self, max_n: int = 4, ds_idx: int = 1):
        """Plot samples with highest per-sample loss."""
        inputs, targs, preds = self.get_worst(max_n=max_n, ds_idx=ds_idx)
        dl = self._get_dl(ds_idx)
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

    def _prepare_chunks(self, batch) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Move batch to device, apply transforms + augmentations, split into sub-sequences."""
        xb, yb = self._to_device(batch)
        for t in self.transforms:
            xb, yb = t(xb, yb)
        for a in self.augmentations:
            xb, yb = a(xb, yb)
        return xb.split(self.sub_seq_len, dim=1), yb.split(self.sub_seq_len, dim=1)

    def _train_one_batch(self, batch, optimizer, step: int, total_steps: int) -> list[float]:
        xb_chunks, yb_chunks = self._prepare_chunks(batch)

        state = None
        self.pct_train = step / max(1, total_steps)
        losses = []
        for i, (xb_sub, yb_sub) in enumerate(zip(xb_chunks, yb_chunks)):
            # n_skip only applies to the first sub-sequence (RNN warmup);
            # subsequent chunks already have a warmed-up hidden state.
            skip = self.n_skip if i == 0 else 0
            loss_val, state = self.training_step((xb_sub, yb_sub), optimizer, state, n_skip=skip)
            if loss_val is not None:
                losses.append(loss_val)
        return losses


# ──────────────────────────────────────────────────────────────────────────────
#  CudaGraphTbpttLearner
# ──────────────────────────────────────────────────────────────────────────────


class CudaGraphTbpttLearner(TbpttLearner):
    """TbpttLearner accelerated with CUDA Graphs.

    Captures the forward + backward + optimizer step for a single TBPTT chunk
    into a CUDA graph and replays it, eliminating per-kernel CPU launch overhead.

    When ``n_skip > 0``, a second graph is captured for the first chunk (which
    has different loss tensor shapes due to skip-slicing).

    Constraints:
        - Requires CUDA device
        - ``win_sz % sub_seq_len == 0`` (all chunks must have equal shape)
        - Model must use ``return_state=True``
        - Loss function must have static tensor shapes (reduce with ``nan_reduce()``, not ``ignore_nan``)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_skip: torch.cuda.CUDAGraph | None = None
        self._s_xb: Tensor | None = None
        self._s_yb: Tensor | None = None
        self._s_state: list[Tensor] | None = None
        self._s_new_state: list[Tensor] | None = None
        self._s_new_state_skip: list[Tensor] | None = None
        self._s_loss: Tensor | None = None
        self._s_loss_skip: Tensor | None = None

    def fit(self, n_epoch, lr=None, make_scheduler=None):
        # Reset captured graphs so shapes are re-discovered each fit() call
        self._graph = None
        self._graph_skip = None

        orig_opt = self.opt_func

        def _capturable_opt(params, **kw):
            return orig_opt(params, capturable=True, **kw)

        self.opt_func = _capturable_opt
        try:
            super().fit(n_epoch, lr=lr, make_scheduler=make_scheduler)
        finally:
            self.opt_func = orig_opt

    def _captured_step(self, n_skip: int = 0) -> tuple[Tensor, list[Tensor]]:
        """Forward + loss + backward + optimizer step on static buffers.

        Returns graph-owned (pred_state, loss) tensors — only meaningful
        during CUDA graph capture (the returned references become the
        static output tensors replayed on each ``graph.replay()``).
        """
        pred, new_state = self.model(self._s_xb, state=self._s_state)
        pred_l = pred[:, n_skip:] if n_skip > 0 else pred
        yb_l = self._s_yb[:, n_skip:] if n_skip > 0 else self._s_yb
        loss = self.loss_func(pred_l, yb_l)
        for aux in self.aux_losses:
            loss = loss + aux(pred, self._s_yb, self._s_xb)
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self._optimizer.step()
        self._optimizer.zero_grad()
        return new_state, loss

    def _init_graph(self, xb_chunk, yb_chunk, optimizer):
        """Allocate static buffers and capture TBPTT-chunk graph(s)."""
        assert self.device.type == "cuda", "CudaGraphTbpttLearner requires a CUDA device"
        self._optimizer = optimizer

        # Allocate static input/target buffers
        self._s_xb = torch.empty_like(xb_chunk)
        self._s_yb = torch.empty_like(yb_chunk)

        # Allocate static state buffers — discover shape from a warmup forward
        with torch.no_grad():
            _, warmup_state = self.model(xb_chunk, state=None)
        self._s_state = [torch.zeros_like(s) for s in warmup_state]

        # Suppress the harmless AccumulateGrad stream-mismatch warning that
        # fires during side-stream warmup backward.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")
            self._warmup_and_capture(xb_chunk, yb_chunk, n_skip=0)
            if self.n_skip > 0:
                self._warmup_and_capture(xb_chunk, yb_chunk, n_skip=self.n_skip)

    def _warmup_and_capture(self, xb_chunk, yb_chunk, n_skip: int):
        """Run warmup iterations then capture a graph for the given n_skip."""
        # Warmup on a side stream (required before CUDA graph capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._s_xb.copy_(xb_chunk)
                self._s_yb.copy_(yb_chunk)
                for st in self._s_state:
                    st.zero_()
                self._captured_step(n_skip)
        torch.cuda.current_stream().wait_stream(s)

        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        for st in self._s_state:
            st.zero_()
        with torch.cuda.graph(graph):
            new_state, loss = self._captured_step(n_skip)

        if n_skip > 0:
            self._graph_skip = graph
            self._s_new_state_skip = new_state
            self._s_loss_skip = loss
        else:
            self._graph = graph
            self._s_new_state = new_state
            self._s_loss = loss

    def _train_one_batch(self, batch, optimizer, step: int, total_steps: int) -> list[float]:
        xb_chunks, yb_chunks = self._prepare_chunks(batch)

        if self._graph is None:
            self._init_graph(xb_chunks[0], yb_chunks[0], optimizer)

        # Zero state for first chunk
        for s in self._s_state:
            s.zero_()

        self.pct_train = step / max(1, total_steps)

        losses = []
        for i, (xc, yc) in enumerate(zip(xb_chunks, yb_chunks)):
            self._s_xb.copy_(xc)
            self._s_yb.copy_(yc)

            if i == 0 and self._graph_skip is not None:
                self._graph_skip.replay()
                losses.append(self._s_loss_skip.item())
                for s_in, s_out in zip(self._s_state, self._s_new_state_skip):
                    s_in.copy_(s_out)
            else:
                self._graph.replay()
                losses.append(self._s_loss.item())
                for s_in, s_out in zip(self._s_state, self._s_new_state):
                    s_in.copy_(s_out)

        return losses
