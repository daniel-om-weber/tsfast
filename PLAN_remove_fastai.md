# Plan: Remove fastai from tsfast

## Context

tsfast is deeply coupled to fastai (23/31 files, ~34% of code). fastai's implicit, convention-heavy style (callback ordering, shared mutable state, magic dispatch) conflicts with the goal of making tsfast transparent and extensible for researchers. The goal is to replace fastai with a minimal pure-PyTorch training framework that is easy to use for simple cases and easy to extend for complex ones (like custom PINNs).

## Development Strategy

### Branching
Work on a `remove-fastai` branch. Master stays as the known-good baseline. Commit frequently
at sub-milestones (e.g., "Learner core works with RNN", "explicit state on all models") so
`git bisect` works if something breaks. Merge back to master only when notebooks pass.

### tsdata: subpackage now, extract later
Build the new data pipeline as `tsfast/tsdata/` — a subpackage within tsfast. This gives
rapid iteration (single repo, single `uv sync`, no version coordination) while keeping a
clean boundary for future extraction into a standalone `tsdata` package.

**The one rule:** `tsfast.tsdata` never imports from the rest of `tsfast`. `tsfast` freely
imports from `tsfast.tsdata`. If this rule holds, extraction is mechanical: copy the directory,
add a `pyproject.toml`, and replace `from tsfast.tsdata` with `from tsdata` everywhere.

**Allowed dependencies for `tsfast/tsdata/`:**
- stdlib, numpy, h5py, scipy, torch, matplotlib

**What goes into `tsfast/tsdata/`:**
- Blocks — signal readers (`HDF5Signals`, `HDF5Attrs`) + `Resampled` wrapper
- `WindowedDataset` — formula-based windowing, `(inputs, targets)` tuples
- `FileEntry` — per-file metadata (path, resampling_factor)
- `DataLoaders` — container with `.train` / `.valid` / `.test` + on-demand `.stats()`
- `create_dls()` — convenience factory for the common case
- Signal processing (resampling, downsampling, filtering)
- Normalization (`NormStats` frozen dataclass, `compute_stats()`, disk caching)
- File discovery and splitting (`discover_split_files()`, explicit file lists)
- Benchmark dataset wrappers (`Silverbox`, `WienerHammerstein`, etc. via identibench)

**What stays in `tsfast` (depends on models/training):**
- Learner factories (`RNNLearner`, `TCNLearner`, etc.) — compose data + model + trainer
- Prediction mode temporal shifts — `prediction_concat` batch_tfm, not data pipeline
- Data augmentation (noise, bias, etc.) — `batch_tfms` in the training loop
- Plotting dispatch — `plot_fn` is set by learner factories, not by the data pipeline
- Custom tensor types — deleted entirely (plain `torch.Tensor` everywhere)

## Architecture: tsdata — Blocks + WindowedDataset

### Design principles
- **Simple cases stay simple:** `create_dls(u=["u"], y=["y"], dataset="path", win_sz=200)`
- **Flexible cases are explicit:** compose blocks, entries, samplers directly
- **No hidden machinery:** blocks are just readers with a `read()` method
- **Pure PyTorch:** `torch.utils.data.Dataset` + `DataLoader`, standard samplers
- **Tuple convention:** `__getitem__` always returns `(inputs, targets)`

### Blocks — what to read

A block reads one group of signals from a file. Temporal blocks read a window
slice: `read(path, l_slc, r_slc) → np.ndarray`. Scalar blocks read per-file
constants: `read(path) → np.ndarray`. No shared store object — blocks handle
their own HDF5 access and caching internally.

```python
# Sequences from HDF5 datasets — temporal block
HDF5Signals(["u1", "u2"])                    # → (seq_len, 2)

# Scalars from HDF5 attributes — scalar block (no slicing)
HDF5Attrs(["mass", "stiffness"])             # → (2,)

# Signals with fs/dt columns that scale during resampling
HDF5Signals(["u", "fs", "dt"],
            fs_idx=1, dt_idx=2)              # fs *= factor, dt /= factor
```

Temporal blocks also expose `file_len(path) → int` for the original sequence
length (cached after first call). The dataset queries this at init to compute
window counts. Since blocks only read raw signals (no transforms that change
the temporal dimension), all temporal blocks for the same file have consistent
lengths.

**Custom blocks are trivial to write.** No base class or registration — just
duck typing. A scalar block needs `read(path)`, a temporal block needs
`read(path, l_slc, r_slc)` + `file_len(path)`:

```python
# Custom scalar block: extract a value from the filename
class FilenameScalar:
    def __init__(self, pattern=r'(\d+)C'):
        self.pattern = pattern
    def read(self, path):
        m = re.search(self.pattern, Path(path).stem)
        return np.array([float(m.group(1))])

# Custom temporal block: read from CSV instead of HDF5
class CSVSignals:
    def __init__(self, columns):
        self.columns = columns
    def read(self, path, l_slc, r_slc):
        df = pd.read_csv(path)
        return df[self.columns].values[l_slc:r_slc]
    def file_len(self, path):
        return sum(1 for _ in open(path)) - 1  # row count minus header

# Use like any built-in block
ds = WindowedDataset(entries,
    inputs=(HDF5Signals(["u"]), FilenameScalar(r'(\d+)C')),
    targets=HDF5Signals(["y"]),
    win_sz=200)
```

HDF5 file handle caching and worker safety (re-opening files in DataLoader worker
processes) are implementation details inside the block. The optimal strategy
(mmap vs memoize vs raw h5py) will be determined by benchmarking during Phase 1.

### Resampled — wrapper for coordinate conversion + interpolation

Resampling is a cross-cutting concern: it affects both **reading** (interpolate
the signal) and **indexing** (more samples → more windows). Putting it inside
every block's `read()` would force every block to handle coordinate conversion,
and the dataset would still need the effective length for windowing.

Instead, `Resampled` is a single wrapper that sits between the dataset and a
temporal block. It handles coordinate-space conversion (target space → original
file space), reads raw data via the inner block, resamples, and trims:

```python
class Resampled:
    """Wraps a temporal block. Reads in original space, resamples to target rate."""

    def __init__(self, block):
        self.block = block

    def read(self, path: str, l_slc: int, r_slc: int, factor: float) -> np.ndarray:
        if factor == 1.0:
            return self.block.read(path, l_slc, r_slc)
        # Convert target-space slice to original file space
        l_orig = math.floor(l_slc / factor)
        r_orig = math.ceil(r_slc / factor) + 2   # margin for interpolation edges
        r_orig = min(r_orig, self.file_len(path))
        raw = self.block.read(path, l_orig, r_orig)
        resampled = resample_interp(raw, factor)
        return resampled[:r_slc - l_slc]          # trim to exact window

    def file_len(self, path: str) -> int:
        return self.block.file_len(path)
```

The dataset passes `entry.resampling_factor` to `Resampled.read()`. Plain
(unwrapped) blocks never see a resampling factor — their `read(path, l_slc, r_slc)`
is called directly. `HDF5Attrs` has no temporal axis and is never wrapped.

`create_dls()` wraps blocks automatically when `targ_fs` / `src_fs` are provided,
so the common case stays simple. Advanced users who compose blocks manually wrap
explicitly: `Resampled(HDF5Signals(["u"]))`.

### FileEntry — per-file metadata

```python
@dataclass
class FileEntry:
    path: str
    resampling_factor: float = 1.0
```

Minimal — just a path and an optional resampling factor. File lengths are not
stored on `FileEntry`. Instead, the `WindowedDataset` queries the first temporal
input block's `file_len(path)` at init to compute window counts (see below).
This avoids requiring every `FileEntry` construction site to perform HDF5 I/O.

`win_sz` and `stp_sz` are always in target (resampled) sample space. Upsampling
2× doubles the effective length and roughly doubles the window count.

Built from file discovery + optional multi-rate expansion:

```python
# Single rate (most common)
entries = [FileEntry(p) for p in train_files]

# Multi-rate: same file at multiple target sampling rates
entries = [
    FileEntry(p, resampling_factor=targ / src_fs)
    for p in train_files
    for targ in [50, 100, 300]
]

# Per-file rates (files recorded at different frequencies)
entries = [
    FileEntry(p, resampling_factor=target_fs / get_fs(p))
    for p in train_files
]
```

This replaces the DataFrame pipeline (`DfResamplingFactor` expanding rows). Adding a
new resampling factor is just appending entries — no DataFrame transform chain to re-run.

### WindowedDataset — indexing + assembly

One class handles both windowed (train/valid) and full-file (test) modes:

```python
class WindowedDataset(Dataset):
    def __init__(self, entries: list[FileEntry],
                 inputs: Block | tuple[Block, ...],
                 targets: Block | tuple[Block, ...],
                 win_sz: int | None = None,     # None = full-file mode
                 stp_sz: int = 1):
        # Query the first temporal input block for file lengths.
        # block.file_len() caches internally, so repeated calls (multi-rate
        # entries sharing the same file) are free.
        ref_block = self._find_temporal(inputs, targets)
        for e in entries:
            seq_len = ref_block.file_len(e.path)
            eff_len = int(seq_len * e.resampling_factor)
            n_win = (eff_len - win_sz) // stp_sz + 1
            ...
        # Store cumulative sum for O(log n) flat index → (entry, offset) lookup
        ...

    def __len__(self):
        # Windowed: total windows across all entries
        # Full-file: number of entries
        ...

    def __getitem__(self, flat_idx) -> tuple:
        entry, l_slc, r_slc = self._resolve(flat_idx)  # in target space
        inp = self._read_blocks(self._inputs, entry, l_slc, r_slc)
        tgt = self._read_blocks(self._targets, entry, l_slc, r_slc)
        return inp, tgt
        # Single block → plain tensor
        # Multiple blocks → tuple of tensors

    def _read_block(self, block, entry, l_slc, r_slc):
        """Dispatch to the right read() signature."""
        if isinstance(block, Resampled):
            return block.read(entry.path, l_slc, r_slc, entry.resampling_factor)
        elif hasattr(block, 'file_len'):       # temporal block
            return block.read(entry.path, l_slc, r_slc)
        else:                                   # scalar block (HDF5Attrs)
            return block.read(entry.path)
```

Usage:

```python
# Common case: one input block, one target block (no resampling)
ds = WindowedDataset(entries, inputs=HDF5Signals(["u"]), targets=HDF5Signals(["y"]),
                     win_sz=200, stp_sz=50)
# → (xb, yb)  both plain tensors

# With resampling: blocks wrapped by Resampled
ds = WindowedDataset(entries,
    inputs=Resampled(HDF5Signals(["u"])),
    targets=Resampled(HDF5Signals(["y"])),
    win_sz=200, stp_sz=50)
# → (xb, yb)  resampled to entry.resampling_factor

# Multi-input: sequences + scalar attributes
ds = WindowedDataset(entries,
    inputs=(Resampled(HDF5Signals(["u"])), HDF5Attrs(["mass"])),
    targets=Resampled(HDF5Signals(["y"])),
    win_sz=200, stp_sz=50)
# → ((xb_seq, xb_scalar), yb)

# Full-file mode for test (no windowing)
test_ds = WindowedDataset(test_entries, inputs=HDF5Signals(["u"]),
                          targets=HDF5Signals(["y"]))
# → one sample per file, variable length, use with bs=1
```

### Samplers — iteration control

Standard PyTorch samplers, no custom DataLoader subclasses:

```python
# Default: shuffle, full epoch
sampler = RandomSampler(train_ds)

# Fixed batch count per epoch (replaces NBatches)
sampler = RandomSampler(train_ds, replacement=True, num_samples=n_batches * bs)

# Adaptive weighted sampling: trainer updates weights between epochs
# based on per-sample loss for hard example mining
weights = torch.ones(len(train_ds))  # mutable — trainer updates this
sampler = WeightedRandomSampler(weights, num_samples=n_batches * bs)

# Validation: sequential
sampler = SequentialSampler(valid_ds)
```

### DataLoaders — container with on-demand stats

```python
@dataclass
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader | None
    # blocks not stored — no downstream consumer yet; add when needed

    @functools.cache
    def stats(self, n_batches: int = 10) -> list[NormStats]:
        """Compute stats on demand from validation set. One NormStats per block."""
        ...

    def stats_from_files(self, cache_id: str | None = None) -> list[NormStats]:
        """Exact stats from full HDF5 scan. Disk-cached via cache_id."""
        ...
```

Stats are computed on demand (not eagerly stored on `dls`). The `cache_id` parameter
enables disk caching at `.tsfast_cache/{cache_id}.pkl` for reproducibility.

### create_dls() — convenience factory

Wraps all of the above into a one-liner for the common case:

```python
dls = create_dls(
    u=["u1", "u2"], y=["y1"],
    dataset="path/to/data",      # auto-discovers train/valid/test splits
    win_sz=200, stp_sz=50,
    bs=64,
)
# Internally:
#   blocks = (HDF5Signals(u), HDF5Signals(y))
#   entries from discover_split_files() with seq_len from HDF5 metadata
#   WindowedDataset + DataLoader + DataLoaders
```

Advanced usage with multi-rate and scalars:

```python
dls = create_dls(
    inputs=HDF5Signals(["u"]),
    targets=HDF5Signals(["y"]),
    extra_inputs=HDF5Attrs(["mass", "stiffness"]),
    dataset="path/to/data",
    win_sz=200, stp_sz=50, bs=64,
    targ_fs=[50, 100, 300], src_fs=100,      # multi-rate expansion
    n_batches=300,                             # fixed batch count
)
# Internally: blocks are wrapped with Resampled() when targ_fs is provided,
# entries are expanded (one per file × target rate) with resampling_factor
```

### What moves from data pipeline to training loop

These features currently live in the data pipeline but belong in the training loop.
They become `batch_tfms` in the Learner:

| Feature | Old location | New location |
|---|---|---|
| Prediction mode concat | `create_dls(prediction=True)` | `prediction_concat()` transform (train + valid) |
| Noise injection | `SeqNoiseInjection` (RandTransform) | `noise()` augmentation (train only) |
| Bias injection | `SeqBiasInjection` (RandTransform) | `bias()` augmentation (train only) |
| Sequence truncation | `VarySeqLen` (RandTransform) | `vary_seq_len()` augmentation (train only) |
| Progressive curriculum | `CB_TruncateSequence` (callback) | `truncate_sequence()` augmentation (train only) |
| STFT | `STFT` block wrapper / `spectogram.py` TransformBlock | Model layer or transform (`torch.stft` on GPU, applied to all data) |

The data pipeline returns raw, unaugmented data. All augmentation and transforms
that change the temporal dimension happen in the training loop, where they run on
GPU and have access to training state (`trainer.pct_train`, train vs eval mode, etc.).

## Architecture: 2-Level Override + Composable Callables

### Design principles
- **Simple cases stay simple:** `RNNLearner(dls, hidden_size=40).fit_flat_cos(5, lr=3e-3)`
- **Common customizations are composable:** pass `transforms`, `augmentations`, and `aux_losses` as lists of callables
- **Complex customizations override methods:** `training_step` for batch-level, `fit` for epoch-level
- **No hidden machinery:** the default `training_step` is ~15 lines of readable Python
- **Pure PyTorch:** zero framework dependency. Fabric can be added later (3 lines of change)

### The Learner class

```python
class Learner:
    def __init__(self, model, dls, loss_func, metrics=None,
                 transforms=None, augmentations=None, aux_losses=None,
                 plot_fn=plot_sequence, config=None): ...

    def fit_flat_cos(self, n_epoch, lr=3e-3, pct_start=0.75):
        """Convenience: flat LR then cosine decay."""
        scheduler = make_flat_cos_scheduler(...)
        self.fit(n_epoch, lr, scheduler)

    def fit(self, n_epoch, lr=3e-3, scheduler=None):
        """The training loop. Override for epoch-level control (curriculum, TBPTT).
        Short enough (~15 lines) to copy-paste-modify."""
        optimizer = Adam(self.model.parameters(), lr=lr)
        self._setup_composables()
        for epoch in range(n_epoch):
            self.pct_train = epoch / n_epoch
            self.model.train()
            train_loss = 0
            state = None
            for batch in self.dls.train:
                loss, state = self.training_step(batch, optimizer, state)
                if loss is not None: train_loss += loss
                if scheduler: scheduler.step()
                state = None  # default: no state carry between batches
            val_loss, metrics = self.validate()
            self._log_epoch(epoch, train_loss, val_loss, metrics)

    def _setup_composables(self):
        """Call setup(trainer) on transforms, augmentations, and aux_losses that support it."""
        for obj in [*self.transforms, *self.augmentations, *self.aux_losses]:
            if hasattr(obj, 'setup'): obj.setup(self)

    def training_step(self, batch, optimizer, state=None):
        """One training step. Override for batch-level control."""
        xb, yb = self.unpack(batch)
        for tfm in self.transforms: xb, yb = tfm(xb, yb)       # always
        for aug in self.augmentations: xb, yb = aug(xb, yb)     # train only
        optimizer.zero_grad()
        result = self.model(xb, state)
        if isinstance(result, tuple):
            pred, new_state = result
        else:
            pred, new_state = result, None
        loss = self.loss_func(pred[:, self.n_skip:], yb[:, self.n_skip:])
        for aux in self.aux_losses: loss = loss + aux(pred, yb, xb)
        if torch.isnan(loss): return None, None
        loss.backward()
        if self.grad_clip: clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()
        return loss.item(), detach_state(new_state)

    def validate(self):
        """Collect all predictions, compute loss + metrics on full validation set."""
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in self.dls.valid:
                xb, yb = self.unpack(batch)
                for tfm in self.transforms: xb, yb = tfm(xb, yb)   # always
                # no augmentations during validation
                result = self.model(xb)
                pred = result[0] if isinstance(result, tuple) else result
                all_preds.append(pred[:, self.n_skip:].cpu())
                all_targets.append(yb[:, self.n_skip:].cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_loss = self.loss_func(all_preds, all_targets).item()
        metrics = {m.__name__: m(all_preds, all_targets) for m in self.metrics}
        return val_loss, metrics

    def show_batch(self, n=4): ...
    def show_results(self, n=4): ...
    def show_worst(self, n=4): ...
    def plot_dataset(self, dl=None): ...
```

### Visualization: explicit `plot_fn`, not type dispatch

fastai uses Plum multi-dispatch on tensor types to auto-select the right plot function. This is hidden machinery — you have to know your data is wrapped in `TensorQuaternionInclination`, and debug through a registry + MRO walk when the wrong plot appears.

Instead, `plot_fn` is an explicit parameter on the Learner, just like `loss_func`. It plots **one sample on one axes** — the only thing that varies by data type:

```python
def plot_fn(ax, x, y, pred=None, **kwargs):
    """Plot a single sample. Line plot for time series, heatmap for spectrograms, etc."""
```

The Learner methods handle **which samples** and **layout** — the part that varies by task:

```python
class Learner:
    def __init__(self, ..., plot_fn=plot_sequence): ...

    def show_batch(self, n=4):
        """Show n samples from the training set."""
        samples = self._get_batch_samples(n)
        fig, axes = plt.subplots(...)
        for ax, (x, y) in zip(axes, samples):
            self.plot_fn(ax, x, y)

    def show_results(self, n=4):
        """Show predictions vs targets."""
        samples, preds = self._get_predictions(n)
        for ax, (x, y), pred in zip(axes, samples, preds):
            self.plot_fn(ax, x, y, pred=pred)

    def show_worst(self, n=4):
        """Show the n worst predictions by loss."""
        samples, preds = self._get_predictions_sorted_by_loss(n)
        for ax, (x, y), pred in zip(axes, samples, preds):
            self.plot_fn(ax, x, y, pred=pred)

    def plot_dataset(self, dl=None):
        """Plot predictions for an entire dataset."""
        all_preds, all_targets = self._predict_all(dl)
        self.plot_fn(ax, x_full, y_full, pred=all_preds)
```

Learner factories set a sensible default:

```python
def RNNLearner(dls, **kw):
    return Learner(model, dls, ..., plot_fn=plot_sequence, **kw)

def QuaternionLearner(dls, **kw):
    return Learner(model, dls, ..., plot_fn=plot_quaternion_inclination, **kw)
```

**Why not auto-dispatch:**
- Custom tensor subclasses exist only to carry type tags — without dispatch, plain `torch.Tensor` everywhere
- `plot_fn` is visible, overridable, and requires zero machinery (no registry, no MRO walk)
- Same pattern as `loss_func` — nobody expects the loss to be auto-detected from tensor type

### Researcher experience by complexity

**Simple (unchanged from today):**
```python
lrn = RNNLearner(dls, hidden_size=40)
lrn.fit_flat_cos(5, lr=3e-3)
```

**Common customizations (composable callables):**
```python
lrn = RNNLearner(dls, hidden_size=40,
    transforms=[prediction_concat(t_offset=1)],
    augmentations=[noise(std=0.1)],
    aux_losses=[physics_loss(my_ode, weight=1.0)])
```

**Complex regularization (stateful aux_losses with setup):**
```python
lrn = FranSysLearner(dls, init_sz=50,
    aux_losses=[FranSysRegularizer(p_state_sync=1e7, p_osp_loss=0.1)])
lrn.fit_flat_cos(10, lr=3e-3)
```

**Batch-level override (training_step):**
```python
class MyLearner(Learner):
    def training_step(self, batch, optimizer):
        # full control over one training step
```

**Epoch-level override (fit):**
```python
class CurriculumLearner(Learner):
    def fit(self, n_epoch, lr=3e-3, scheduler=None):
        # full control over epoch iteration, validation timing, etc.
```

## Key Design Decisions

### Why not Lightning?
Full Lightning (`LightningModule` + `Learner`) owns the epoch loop, which is incompatible with the `fit()` override level. Lightning Fabric is compatible (it's just `fabric.backward()` + device management) and can be added later with ~3 lines of change if mixed precision or multi-GPU is needed.

### Explicit state passing, not stateful models

Models in tsfast currently store hidden state internally (`self.hidden`, `self.y_init`, `self.x_init`,
`self.seq_idx`) and require `reset_state()` calls + `TbpttResetCB` to manage it. This has been a
recurring source of bugs: silent state leaks between sequences, batch size mismatches discarding
state, scattered `reset_state()` calls across 4 model classes.

Instead, **all model state is explicit**: models accept `state` as input and return it as output.
The training loop owns the state and decides when to carry or reset it.

**Model convention:** `forward(x, state=None) → (output, new_state)`
- tsfast models (`RNN`, `TCN`, `FranSys`, `AR_Model`) always return `(output, state)`
- Stateless layers return `(output, None)`
- Wrappers (`NormalizedModel`) pass `state` through opaquely without inspecting it
- External models (plain `nn.Module`) can return just a tensor — the Learner handles both:
  ```python
  result = self.model(xb, state)
  if isinstance(result, tuple):
      pred, new_state = result
  else:
      pred, new_state = result, None
  ```
  This `isinstance` check lives in exactly two places: `training_step` and `validate`.

**State is opaque.** Nothing outside the model inspects state — wrappers pass it
through, `detach_state` recurses structurally, the Learner just carries or resets it.
State can be any nested combination of tensors, tuples, lists, dicts, or None.

Simple models return whatever is natural:
- GRU: a single tensor `h`
- LSTM: a tuple `(h, c)`
- Stateless: `None`

**Composite models should use dicts** for self-documenting, extensible state:
```python
class CRNN(nn.Module):
    def forward(self, x, state=None):
        tcn_state = state["tcn"] if state else None
        rnn_state = state["rnn"] if state else None
        x, tcn_state = self.tcn(x, tcn_state)
        out, rnn_state = self.rnn(x, rnn_state)
        return out, {"tcn": tcn_state, "rnn": rnn_state}

class AR_Model(nn.Module):
    def forward(self, x, state=None):
        h = state["h"] if state else None
        y_init = state["y_init"] if state else None
        ...
        return output, {"h": new_hidden, "y_init": last_pred}
```

Dict keys are readable when debugging, stable when adding submodules, and map
naturally to named ONNX inputs/outputs. This is a recommendation, not enforced —
`detach_state` handles any structure.

**Wrappers pass state through opaquely:**
```python
class NormalizedModel(nn.Module):
    def forward(self, x, state=None):
        x = self.input_norm.normalize(x)
        out, state = self.model(x, state)  # pass through, don't inspect
        return self.output_norm.denormalize(out), state
```

**What this eliminates:**
- `self.hidden`, `self.y_init`, `self.x_init`, `self.seq_idx` on model classes
- `reset_state()` methods on all models
- `reset_model_state()` utility that walks the module tree
- `TbpttResetCB` callback
- The `stateful` flag on `RNN`, `CausalConv1d`, `BatchNorm_1D_Stateful`, `AR_Model`
- `Sequential_RNN` (which existed only to discard state — now callers just ignore it)

**`detach_state` is the TBPTT truncation point:**
```python
def detach_state(state):
    if state is None: return None
    if isinstance(state, Tensor): return state.detach()
    if isinstance(state, (list, tuple)): return type(state)(detach_state(s) for s in state)
    if isinstance(state, dict): return {k: detach_state(v) for k, v in state.items()}
```

### TBPTT lives in `fit`, not in the DataLoader

The current `TbpttDl` (130 lines) splits sequences into sub-windows inside the DataLoader,
manages worker queue reordering, and signals sequence boundaries via a `rnn_reset` flag that
a callback reads. This is complex machinery in the wrong place.

With explicit state, TBPTT is just a `fit` override. The DataLoader yields full sequences.
The training loop splits them and carries state:

```python
class TbpttLearner(Learner):
    def __init__(self, *args, sub_seq_len, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_seq_len = sub_seq_len

    def fit(self, n_epoch, lr=3e-3, scheduler=None):
        optimizer = Adam(self.model.parameters(), lr=lr)
        self._setup_composables()
        for epoch in range(n_epoch):
            self.pct_train = epoch / n_epoch
            self.model.train()
            train_loss = 0
            for batch in self.dls.train:
                xb, yb = self.unpack(batch)
                state = None  # reset at each new sequence
                for xb_sub, yb_sub in zip(xb.split(self.sub_seq_len, dim=1),
                                           yb.split(self.sub_seq_len, dim=1)):
                    loss, state = self.training_step(
                        (xb_sub, yb_sub), optimizer, state)
                    if loss is not None: train_loss += loss
                    # state carries across sub-sequences (detached in training_step)
                if scheduler: scheduler.step()
            val_loss, metrics = self.validate()
            self._log_epoch(epoch, train_loss, val_loss, metrics)
```

Each sub-sequence is a full optimization step (zero_grad → forward → backward → step).
`detach_state()` in `training_step` truncates gradients at sub-sequence boundaries while
carrying hidden state values forward. Setting `state = None` at the start of each batch
resets state at sequence boundaries.

`RNNLearner(..., stateful=True)` returns a `TbpttLearner` instead of a `Learner` — the
researcher doesn't need to know about the override.

**What this eliminates:**
- `TbpttDl` — the entire 130-line class with worker queue reordering
- `TbpttResetCB` — the callback
- The `rnn_reset` side-channel between DataLoader and callback

### Why not a callback system?
Callbacks (fastai-style) add indirection: researchers must learn lifecycle hooks, understand execution ordering, and debug through dispatch. The composable callables (`transforms`, `augmentations`, `aux_losses`) provide the same composition without the complexity. For anything beyond that, overriding `training_step` or `fit` gives full control with zero framework fighting.

### The `setup(trainer)` / `teardown(trainer)` protocol

All three composable lists (`transforms`, `augmentations`, `aux_losses`) follow the same duck-typed protocol. Simple cases are just callables. Complex cases can optionally implement `setup(trainer)` and/or `teardown(trainer)`. The Learner calls `setup` once before training and `teardown` once after (in a `finally` block, so cleanup happens even on error).

**Simple augmentation — plain callable:**
```python
def noise(std=0.1):
    def _aug(xb, yb):
        return xb + torch.randn_like(xb) * std, yb
    return _aug
```

**Stateful augmentation — callable with setup (progressive curriculum):**
```python
class truncate_sequence:
    """Progressively increase sequence length during training."""

    def __init__(self, initial_len):
        self.initial_len = initial_len

    def setup(self, trainer):
        self.trainer = trainer

    def __call__(self, xb, yb):
        full_len = xb.shape[1]
        target = int(self.initial_len + (full_len - self.initial_len) * self.trainer.pct_train)
        return xb[:, :target], yb[:, :target]
```

**Simple aux loss — plain callable:**
```python
def physics_loss(ode_fn, weight=1.0):
    def _loss(pred, yb, xb):
        residual = ode_fn(pred, xb)
        return weight * residual.pow(2).mean()
    return _loss
```

**Stateful aux loss — callable with setup:**
```python
class FranSysRegularizer:
    """Regularizes FranSys by syncing diagnosis/prognosis hidden states."""

    def __init__(self, p_state_sync=1e7, p_diag_loss=0, p_osp_loss=0,
                 p_osp_sync=0, p_tar_loss=0, sync_type='mse'):
        self.p_state_sync = p_state_sync
        # ... store all params

    def setup(self, trainer):
        """Called once before training. Register hooks, grab model ref."""
        self.model = unwrap_model(trainer.model)
        self._hook_diag = self.model.rnn_diagnosis.register_forward_hook(self._capture_diag)
        self._hook_prog = self.model.rnn_prognosis.register_forward_hook(self._capture_prog)

    def _capture_diag(self, m, i, o): self._out_diag = o[0]
    def _capture_prog(self, m, i, o): self._out_prog = o[0]

    def __call__(self, pred, yb, xb):
        diag, prog = self._out_diag, self._out_prog
        self._out_diag = self._out_prog = None  # clear for next batch

        loss = torch.tensor(0.0, device=pred.device)
        if self.p_state_sync > 0:
            loss += self.p_state_sync * ((prog - diag) / diag.norm()).pow(2).mean()
        if self.p_diag_loss > 0:
            y_diag = self.model.final(diag[-1])
            loss += self.p_diag_loss * mae(y_diag, yb)
        if self.p_osp_loss > 0:
            # extra forward pass through rnn_prognosis for one-step prediction
            inp = xb[:, self.model.init_sz:, :self.model.n_u]
            h_init = diag[:, :, self.model.init_sz - 1:-1]
            # ... reshape, run model.rnn_prognosis(inp, h_init), compute loss
            loss += self.p_osp_loss * osp_loss
        if self.p_tar_loss > 0:
            h = torch.cat([diag[:, :, :self.model.init_sz], prog], 2)
            loss += self.p_tar_loss * (h[:, :, 1:] - h[:, :, :-1]).pow(2).mean()
        return loss
```

**PINN collocation — same pattern:**
```python
class CollocationLoss:
    """Physics loss on synthetic collocation points."""

    def __init__(self, ode_fn, excitation_gen, n_points=256, weight=1.0): ...

    def setup(self, trainer):
        self.model = trainer.model
        self._colloc_dl = DataLoader(...)  # synthetic data generator
        self._colloc_iter = iter(self._colloc_dl)

    def __call__(self, pred, yb, xb):
        colloc_batch = next(self._colloc_iter)
        colloc_pred = self.model(colloc_batch)
        residual = self.ode_fn(colloc_pred, colloc_batch)
        return self.weight * residual.pow(2).mean()
```

**Why this isn't a callback system:**
- No lifecycle ordering — transforms/augmentations compose left-to-right, aux_losses are independent (all additive)
- No shared mutable state — transforms/augmentations return `(xb, yb)`, aux_losses return a scalar; the caller owns assignment
- No dispatch/registration — just duck typing (`hasattr(obj, 'setup')`)
- Two extension points: optional `setup(trainer)` and `teardown(trainer)` for all three lists
- Transform/augmentation contract: `__call__(xb, yb) → (xb, yb)`
- Aux_loss contract: `__call__(pred, yb, xb) → loss_term`
- Simple callables never see `setup`/`teardown` — a plain `lambda` still works

### Transforms vs augmentations

**Transforms** prepare data into the representation the model expects. Applied to all data
(train, valid, test). Examples: `prediction_concat(t_offset)`, `ar_init()`.

**Augmentations** add randomized perturbations for regularization. Applied during training
only. Examples: `noise(std)`, `bias(std)`, `vary_seq_len(min_len)`, `truncate_sequence(initial_len)`.

STFT is a representation change that can live in either place depending on preference:
as a **model layer** (model accepts raw signals, applies `torch.stft` internally) or
as a **transform** (data is converted before the model sees it). Either way, it runs
on GPU via `torch.stft` and is applied to all data (train, valid, test).

### Data-altering operations belong in the training loop
Operations like prediction concat, noise injection, sequence truncation, and TBPTT reset
depend on training state or model expectations. They live as `transforms` or
`augmentations` in the Learner, not in the data pipeline. The tsdata architecture
section has a full table of what moves from data pipeline to training loop.

### Relationship with tsjax
Full unification of tsfast and tsjax is not practical (HF Transformers dropped multi-backend
support for this reason). However, both libraries can share:
- `tsdata` — built first as `tsfast/tsdata/` subpackage, extracted to standalone package when
  stable. Adopts architectural patterns from tsjax's tsdata (formula-based windowing, on-demand
  stats, file entry model) but uses `torch.utils.data.DataLoader` instead of tsjax's
  `ThreadedLoader`, and adds tsfast-specific features (multi-rate `FileEntry`, block-based
  signal readers, `(inputs, targets)` tuple convention).
- Consistent naming conventions (`RNNLearner`, `create_dls`, `fit_flat_cos`)
- The same 2-level override philosophy

## Feature coverage analysis

How every current callback/customization maps to the new architecture:

### Default loop (transforms + augmentations + aux_losses) — ~70% of use cases

| Current callback | New mechanism | Notes |
|---|---|---|
| `PredictionCallback` | `transforms` callable | `prediction_concat(t_offset)` — applied train + valid |
| `ARInitCB` | `transforms` callable | `ar_init()` — concat y onto x, applied train + valid |
| `SeqNoiseInjection` | `augmentations` callable | `noise(std)` — train only |
| `SeqBiasInjection` | `augmentations` callable | `bias(std)` — train only |
| `VarySeqLen` | `augmentations` callable | `vary_seq_len(min_len)` — train only |
| `CB_TruncateSequence` | Stateful `augmentations` with `setup()` | `truncate_sequence(initial_len)` — reads `trainer.pct_train`, train only |
| `SkipFirstNCallback` | Built into `training_step` | `self.n_skip` already in plan |
| `GradientClipping` | Built into `training_step` | `self.grad_clip` already in plan |
| `SkipNaN` | Built into `training_step` | `if torch.isnan(loss): return None` already in plan |
| `CB_AddLoss` | Simple `aux_losses` callable | Direct replacement |
| `TimeSeriesRegularizer` | Stateful `aux_losses` with `setup()` | Hooks into RNN layers for AR/TAR |
| `FranSysCallback` | Stateful `aux_losses` with `setup()` | `FranSysRegularizer` — hooks + optional extra forward pass |
| `PhysicsLossCallback` | Simple `aux_losses` callable | Physics loss on training data |
| `CollocationPointsCB` | Stateful `aux_losses` with `setup()` | `CollocationLoss` — manages its own DataLoader |
| `ConsistencyCallback` | Stateful `aux_losses` with `setup()` | Hooks for hidden state consistency |
| `TransitionSmoothnessCallback` | Simple `aux_losses` callable | Curvature penalty on predictions |
| `SkipNLoss`, `CutLoss`, `NormLoss` | `loss_func` wrappers | Unchanged — wrap the primary loss |

### Custom `training_step` override — ~15% of use cases

| Current callback | Why it needs override |
|---|---|
| `GradientBatchFiltering` | Conditionally skips `optimizer.step()` based on gradient norm |
| `WeightClipping` | Clamps weights after optimizer step |
| `BatchLossFilter` | Computes per-sample loss, filters to hardest %, recomputes |

### Custom `fit` override — ~10% of use cases

| Current callback | Why it needs override |
|---|---|
| `TbpttResetCB` + `TbpttDl` | `TbpttLearner.fit()` — splits sequences, carries explicit state (see design decision above) |
| `CancelNaNCallback` | Stops all training (not just one batch) on NaN |
| Ray Tune (`LearnerTrainable`) | Calls `fit(1)` per trial step, interleaves with checkpointing |

### Diagnostics and reporting — `_log_epoch` override or standalone functions

Diagnostics (`plot_grad_flow`, `GradientNormPrint`) and reporting (`CBRayReporter`) are currently
fastai callbacks that hook into `after_backward` or `after_epoch`. They don't need a callback
system — they're either pure functions or simple `_log_epoch` overrides.

**Pure diagnostic functions** (no framework dependency, call from notebook or training_step override):

| Current callback | New form | Notes |
|---|---|---|
| `plot_grad_flow` | Standalone function | Already pure matplotlib — `plot_grad_flow(model.named_parameters())` |
| `GradientNormPrint` | Standalone function | `grad_norm(model.parameters()) → float` |
| `CB_PlotGradient` | Call `plot_grad_flow` in a `training_step` override | Researcher controls when/how often to plot |

**Reporting via `_log_epoch` override:**

| Current callback | New form | Notes |
|---|---|---|
| `CBRayReporter` | `TuneLearner._log_epoch()` override | Reports metrics + checkpoint to Ray Tune |

```python
class Learner:
    def _log_epoch(self, epoch, train_loss, val_loss, metrics):
        """Called after each epoch. Override for custom reporting."""
        # default: tqdm/print progress

class TuneLearner(Learner):
    def _log_epoch(self, epoch, train_loss, val_loss, metrics):
        super()._log_epoch(epoch, train_loss, val_loss, metrics)
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(self.model.state_dict(), os.path.join(tmpdir, "model.pth"))
            ray.train.report(
                {"train_loss": train_loss, "valid_loss": val_loss, **metrics},
                checkpoint=Checkpoint.from_directory(tmpdir),
            )
```

### Outside the training loop — unchanged or moved to tsdata

| Feature | New location |
|---|---|
| `WeightedDL` | `WeightedRandomSampler` (standard PyTorch) |
| `NBatches` | `RandomSampler(replacement=True, num_samples=n*bs)` |
| `AlternatingEncoderCB` | Model training-time randomization (see below) |
| `FranSysCallback_variable_init` | Model training-time randomization (see below) |
| `InferenceWrapper`, `OnnxInferenceWrapper` | Post-training inference |

**Training-time randomization lives in the model, not the training loop.**
`AlternatingEncoderCB` and `FranSysCallback_variable_init` currently mutate model
attributes per-batch via callbacks. Instead, models handle this internally using
`self.training` — the same pattern as `nn.Dropout`:

```python
class FranSys(nn.Module):
    def __init__(self, ..., init_sz=50, init_sz_range=None):
        self.init_sz = init_sz
        self.init_sz_range = init_sz_range  # e.g., (30, 70)

    def forward(self, x, state=None):
        init_sz = self.init_sz
        if self.training and self.init_sz_range:
            init_sz = random.randint(*self.init_sz_range)
        # use init_sz for this forward pass
        ...
```

No external mutation, no side effects from transforms or callbacks. The
randomization range is a model config parameter set at construction. The Learner
doesn't need to know about it — `model.train()` / `model.eval()` already
controls the behavior via standard PyTorch.

## Migration Plan

All work happens on a `remove-fastai` branch. Master stays as the known-good baseline.
Commit frequently at sub-milestones so `git bisect` works if something breaks. Merge
back to master only when notebooks pass.

### Guiding principles

1. **Existing tests are the safety net.** The 150+ tests covering all learner types,
   callbacks, data pipeline, normalization, ONNX, inference, TBPTT, and PINN are the
   most valuable asset during migration. Keep them passing after every sub-phase.
2. **Don't rewrite tests first.** Make the new code pass the old tests, then update
   tests only when the old API surface genuinely changes.
3. **Coexist, then cut over.** New modules (`tsdata/`, `training/`) live alongside old
   modules (`data/`, `datasets/`, `learner/`). Delete old modules only after all consumers
   are migrated and tests pass against the new code.
4. **One model at a time.** Refactor each model class to explicit state individually,
   with a commit and test run between each.
5. **Validate against the working training loop.** Do explicit model state *before* the
   Learner, so state bugs are caught with the fastai Learner still functional.

### Phase 0: Capture golden baselines — DONE

Golden baselines captured in `tests/golden/`.

### Phase 1: Build `tsfast/tsdata/` — pure-PyTorch data pipeline — DONE

Implemented in `tsfast/tsdata/`. See `tests/test_tsdata.py`.

### Phase 2: Make model state explicit — DONE

All models refactored to explicit state passing:
- `RNN`: `forward(inp, state=None) → (output, new_hidden)`
- `AR_Model`: accepts dict/list/None state via match/case, returns `{"h", "y_init"}` when `return_state=True`
- `NormalizedModel`: handles tuple returns from inner model
- `ARProg`/`ARProg_Init`: replaced y_init mutation with explicit state dict passing
- `CausalConv1d`/`SeperateTCN`: internal state kept (deferred to Phase 5, nn.Sequential nesting)
- `BatchNorm_1D_Stateful`: `BN_start` param provides explicit override, `seq_idx` kept as simple counter
- Temporary `reset_state()` compat shims kept for `TbpttResetCB` (deleted in Phase 6)
- Removed `to_detach`/`one_param` imports from fastai in model files
- Added `_detach_state()` utility in `layers.py`

### Phase 3: Build the new training framework — DONE

Pure-PyTorch training framework built in `tsfast/training/`. Coexists with old fastai code.
See `tests/test_training.py` (39 tests, all passing).

**Files created:**

1. `tsfast/training/losses.py` — loss functions, metrics, schedulers, aux loss composables
   - Pure losses: `mse`, `mse_nan`, `ignore_nan`, `float64_func`, `SkipNLoss`, `CutLoss`,
     `NormLoss`, `weighted_mae`, `RandSeqLenLoss`, `fun_rmse`, `cos_sim_loss`,
     `cos_sim_loss_pow`, `nrmse`, `nrmse_std`, `mean_vaf`, `zero_loss`
   - Schedulers: `sched_lin_p`, `sched_ramp`
   - Simple aux losses (`__call__(pred, yb, xb) -> loss_term`):
     `add_loss`, `physics_loss`, `transition_smoothness`
   - Stateful aux losses (with `setup(trainer)` / `teardown(trainer)`):
     `TimeSeriesRegularizerLoss` (AR + TAR via hooks),
     `FranSysRegularizer` (diag/prog sync, OSP, TAR via hooks)
   - Deferred to Phase 5: `CollocationLoss`, `ConsistencyLoss` (PINN-specific)

2. `tsfast/training/transforms.py` — transforms + augmentations
   - Transforms (train + valid, `__call__(xb, yb) -> (xb, yb)`):
     `prediction_concat`, `ar_init`
   - Augmentations (train only): `noise`, `noise_varying`, `noise_grouped`,
     `bias`, `vary_seq_len`, `truncate_sequence`

3. `tsfast/training/viz.py` — visualization
   - `plot_sequence`, `plot_seqs_single_figure`, `plot_seqs_multi_figures`,
     `layout_samples`, `plot_grad_flow`, `grad_norm`

4. `tsfast/training/core.py` — Learner + TbpttLearner + Recorder
   - `Learner`: `fit()`, `fit_flat_cos()`, `training_step()`, `validate()`,
     `get_preds()`, `show_batch()`, `show_results()`, `show_worst()`, `no_bar()`
   - `TbpttLearner`: overrides `fit()` for truncated backprop through time
   - `Recorder`: `values[epoch] = [train_loss, valid_loss, *metrics]`
   - Helpers: `_auto_device()`, `_make_flat_cos_scheduler()`
   - Composable protocol: `setup(trainer)` / `teardown(trainer)` on transforms,
     augmentations, and aux_losses (called automatically by `fit()`)

5. `tsfast/training/__init__.py` — public re-exports

**Key design decisions:**
- Composables use `__call__` protocol, not callback hooks — explicit data flow
- Stateful composables (`TimeSeriesRegularizerLoss`, `FranSysRegularizer`,
  `truncate_sequence`) register hooks in `setup()`, remove in `teardown()`
- `Recorder.values` matches fastai convention for test compatibility
- `_detach_state` reused from `tsfast/models/layers.py` (not duplicated)
- Old vs new Learner comparison test deferred to Phase 4 (when factories switch)

### Phase 4: Update learner factories + switch to new data pipeline

Update factory functions to return new `Learner` instead of fastai `Learner`, and
switch from old `data/`+`datasets/` to new `tsdata/`.

- `tsfast/models/rnn.py` — `RNNLearner()`, `AR_RNNLearner()`.
  `RNNLearner(..., stateful=True)` returns `TbpttLearner` with `sub_seq_len`.
- `tsfast/models/cnn.py` — `TCNLearner()`, `CRNNLearner()`, `AR_TCNLearner()`
- `tsfast/prediction/fransys.py` — `FranSysLearner()`
- `tsfast/pinn/pirnn.py` — `PIRNNLearner()`

Keep the same function signatures and return type duck-typing (`.fit_flat_cos()`,
`.show_results()`, etc.).

**This is where existing tests break.** Fix them one by one, using the golden baselines
from Phase 0 to verify that training behavior hasn't regressed.

**Commit checkpoints:**
1. `RNNLearner returns Learner — test_models.py passes`
2. `TCNLearner/CRNNLearner returns Learner — test_models.py passes`
3. `AR_RNNLearner/AR_TCNLearner returns Learner — test_prediction.py passes`
4. `FranSysLearner returns Learner — test_prediction.py passes`
5. `PIRNNLearner returns Learner — test_pinn.py passes`

### Phase 5: Update inference and tuning

- `tsfast/inference/core.py` — update `InferenceWrapper`: replace `reset_model_state()`
  with `state=None`. Explicit state enables streaming inference (call model per-timestep).
- `tsfast/inference/onnx.py` — explicit state aligns with ONNX's stateless graph model.
  State tensors become explicit ONNX inputs/outputs.
- `tsfast/tune.py` — update Ray Tune integration to use new Learner. Ensure
  `DataLoaders` is picklable for Ray workers (cf. commit `cf93447`).

**Verification:** `test_inference.py`, `test_onnx.py`, `test_tune.py` all pass.

**Commit checkpoints:**
1. `InferenceWrapper uses explicit state — test_inference.py passes`
2. `ONNX export uses explicit state — test_onnx.py passes`
3. `Ray Tune uses new Learner — test_tune.py passes`

### Phase 6: Cleanup — remove fastai

- Delete old `tsfast/data/`, `tsfast/datasets/` (replaced by `tsfast/tsdata/`)
- Delete `tsfast/learner/callbacks.py` (replaced by `tsfast/training/`)
- Remove all remaining `from fastai.*` and `from fastcore.*` imports
- Remove `fastai` and `fastcore` from `pyproject.toml` dependencies
- Replace `delegates` (fastcore) with explicit signatures
- Replace `L` with plain lists
- Replace `retain_types`, `to_detach`, `one_param` with direct PyTorch equivalents
- Delete all `TensorBase` subclasses — plain `torch.Tensor` everywhere
- Replace fastai's `Adam` with `torch.optim.Adam`
- Copy `RNNDropout`/`WeightDropout` from fastai (~50 lines, self-contained)
- Update `tsfast/__init__.py` — remove fastai MPS patch
- Delete `TbpttDl`, `TbpttResetCB`, `reset_model_state()`, `Sequential_RNN`
- Delete all temporary `reset_state()` compatibility shims from Phase 2

**Verification:** `grep -r 'from fastai\|from fastcore' tsfast/` returns nothing.
Full test suite passes: `pytest tests/ -v`.

### Phase 7: Update notebooks and final testing

- Update all 17 example notebooks to use new API
- One commit per notebook so regressions are bisectable
- Run full test suite: `pytest tests/ -v`
- Run all notebooks: `pytest --nbmake examples/notebooks/*.ipynb`
- Compare training curves on Silverbox benchmark against golden baselines

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| **TBPTT state carry** | Current `TbpttDl` + `TbpttResetCB` has subtle multi-worker ordering. Test with `num_workers=0` and `num_workers=2` explicitly. |
| **Normalization semantics** | `prediction=True` mode concatenates `[u_norm, y_raw]`. Ensure the new pipeline preserves this mixed-normalization behavior. Golden baselines catch regressions. |
| **FranSys hooks** | `FranSysRegularizer` registers forward hooks. Test that hook outputs match the old callback's captured activations. |
| **Ray serialization** | DataLoaders must be picklable for Ray workers. Commit `cf93447` stripped transient iterator state — make sure new `DataLoaders` handles this too. |
| **Custom tensor type removal** | Currently `show_batch`/`show_results` dispatch on `TensorSequencesInput` etc. When switching to plain tensors + explicit `plot_fn`, verify all 17 notebooks still produce plots. |
| **`delegates` removal** | fastcore's `delegates` forwards `**kwargs` to parent classes. When replacing with explicit signatures, don't accidentally drop parameters that notebooks rely on. Grep all notebook scripts for every kwarg used. |
| **Long-lived branch divergence** | Commit frequently at sub-milestones. Keep the branch small by coexisting old/new rather than deleting old code early. Merge to master as soon as all tests + notebooks pass. |

## Files to modify (summary)

**New subpackage — `tsfast/tsdata/`:**
- `__init__.py` — public API surface
- `blocks.py` — signal readers (`HDF5Signals`, `HDF5Attrs`) + `Resampled` wrapper
- `dataset.py` — `WindowedDataset`, `FileEntry`
- `signal.py` — resampling, filtering (from `data/core.py`)
- `norm.py` — `NormStats`, `compute_stats()`, disk caching
- `split.py` — `discover_split_files()`, splitter functions
- `pipeline.py` — `DataLoaders` container, `create_dls()` factory
- `benchmark.py` — benchmark dataset wrappers (from `datasets/benchmark.py`)

**New subpackage — `tsfast/training/`:**
- `__init__.py` — public API surface
- `core.py` — Learner, TbpttLearner, detach_state
- `transforms.py` — transforms (always) and augmentations (train-only)
- `losses.py` — aux losses and loss wrappers
- `viz.py` — visualization functions (plot_sequence, layout helpers)

**Major rewrites:**
- `tsfast/models/rnn.py` — explicit state: remove `self.hidden`/`stateful`/`reset_state`,
  remove `Sequential_RNN`, update `RNNLearner` factory
- `tsfast/models/cnn.py` — explicit state: remove `self.x_init`/`stateful`/`reset_state`,
  update TCN/CRNN factories
- `tsfast/models/layers.py` — explicit state: `AR_Model` returns state instead of
  mutating `self.y_init`, `NormalizedModel` passes state through,
  `BatchNorm_1D_Stateful` accepts/returns `seq_idx` (or replace with LayerNorm)

**Moderate changes:**
- `tsfast/prediction/fransys.py` — FranSysLearner, remove `ARProg` y_init mutation
- `tsfast/pinn/pirnn.py` — PIRNNLearner
- `tsfast/pinn/core.py` — PINN callbacks → aux_losses
- `tsfast/inference/core.py` — InferenceWrapper: state=None replaces reset_model_state()
- `tsfast/inference/onnx.py` — state tensors as explicit ONNX inputs/outputs
- `tsfast/tune.py` — Ray Tune integration
- `tsfast/spectogram.py` — replace Transform/TransformBlock
- `tsfast/__init__.py` — remove fastai MPS patch

**Deleted (Phase 6, after migration complete):**
- `tsfast/data/` — entire directory (replaced by `tsfast/tsdata/`)
- `tsfast/datasets/` — entire directory (replaced by `tsfast/tsdata/`)
- `tsfast/learner/callbacks.py` — replaced by `tsfast/training/`
- `TbpttDl` class — replaced by TbpttLearner.fit()
- `TbpttResetCB` — replaced by explicit state=None
- `Sequential_RNN` — callers that don't need state just ignore it
- `reset_model_state()` — no internal state to reset
- All custom tensor subclasses (`TensorSequences`, `TensorScalars`, etc.) — plain tensors
- All temporary `reset_state()` compatibility shims

## Verification

**After every sub-phase:**
- `pytest tests/ -v` — full test suite

**After Phase 3:**
- Old vs new Learner comparison: same model, same data, similar validation loss

**After Phase 4:**
- All learner factory tests pass against new Learner
- Golden baseline comparison: validation losses within tolerance

**After Phase 5:**
- `test_inference.py`, `test_onnx.py`, `test_tune.py` all pass

**After Phase 6:**
- `grep -r 'from fastai\|from fastcore' tsfast/` returns nothing
- `grep -r 'reset_state\|self\.hidden' tsfast/models/` returns nothing
- Full test suite passes

**After Phase 7:**
- `pytest --nbmake examples/notebooks/*.ipynb` — all 17 notebooks pass
- Compare Silverbox training curves against golden baselines
- Visual inspection of `show_batch` / `show_results` plots in notebooks
