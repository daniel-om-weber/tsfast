"""DataLoaders container and create_dls factory for pure-PyTorch data pipeline."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .blocks import HDF5Signals, Resampled
from .dataset import FileEntry, WindowedDataset
from .norm import (
    NormStats,
    _load_norm_stats,
    _save_norm_stats,
    compute_stats,
    compute_stats_from_files,
)
from .split import discover_split_files, get_hdf_files, split_by_parent


class DataLoaders:
    """Container for train/valid/test DataLoaders with on-demand normalization stats.

    Args:
        train: training DataLoader
        valid: validation DataLoader
        test: test DataLoader, or None if no test split
    """

    def __init__(
        self,
        train: DataLoader,
        valid: DataLoader,
        test: DataLoader | None = None,
    ):
        self.train = train
        self.valid = valid
        self.test = test
        self._train_files: list | None = None
        self._signal_names: tuple[list[str], list[str]] | None = None
        self._dls_id: str | None = None
        self._cached_stats: NormStats | None = None

    @property
    def norm_stats(self) -> NormStats:
        """Normalization stats, computed lazily on first access.

        Uses exact file-based stats when dls_id was provided to create_dls,
        otherwise estimates from the first 10 training batches.
        """
        if self._cached_stats is None:
            if self._dls_id is not None:
                self._cached_stats = self.stats_from_files(self._dls_id)
            else:
                self._cached_stats = self.stats()
        return self._cached_stats

    def stats(self, n_batches: int = 10) -> NormStats:
        """Estimate normalization stats from training batches."""
        input_stats, output_stats = compute_stats(self.train, n_batches)
        return NormStats(u=input_stats, y=output_stats)

    def stats_from_files(self, cache_id: str | None = None) -> NormStats:
        """Compute exact stats from full HDF5 scan, with optional disk caching."""
        if cache_id is not None:
            cached = _load_norm_stats(cache_id)
            if cached is not None:
                return cached
        if self._signal_names is None or self._train_files is None:
            raise ValueError(
                "File-based stats require train_files and signal_names. "
                "Use create_dls() or set _train_files and _signal_names manually."
            )
        u_names, y_names = self._signal_names
        norm_u = compute_stats_from_files(self._train_files, u_names)
        norm_y = compute_stats_from_files(self._train_files, y_names)
        result = NormStats(norm_u, norm_y)
        if cache_id is not None:
            _save_norm_stats(cache_id, result)
        return result

    @property
    def loaders(self) -> list[DataLoader]:
        """List of all non-None loaders."""
        return [dl for dl in [self.train, self.valid, self.test] if dl is not None]

    def one_batch(self) -> tuple:
        """Return one batch from the training DataLoader."""
        return next(iter(self.train))


def _compute_resampling_factors(
    files: list[Path],
    targ_fs: float | list[float],
    src_fs: float | str | None,
) -> dict[str, list[tuple[float, float]]]:
    """Compute per-file resampling factors from source/target sampling rates.

    Returns dict mapping path -> list of (resampling_factor, targ_fs) pairs.
    When targ_fs is a list, each file is expanded into multiple entries.
    """
    if isinstance(targ_fs, (int, float)):
        targ_fs = [targ_fs]

    result = {}
    for f in files:
        path = str(f)
        if isinstance(src_fs, str):
            # src_fs is an HDF5 attribute name — read from file
            import h5py

            with h5py.File(path, "r") as hf:
                file_fs = float(hf.attrs[src_fs])
        elif src_fs is not None:
            file_fs = float(src_fs)
        else:
            file_fs = 1.0

        result[path] = [(tf / file_fs, tf) for tf in targ_fs]
    return result


def _wrap_one(block):
    """Wrap a single temporal block in Resampled if needed."""
    if isinstance(block, Resampled):
        return block
    if hasattr(block, "file_len"):
        return Resampled(block)
    return block


def _wrap_resampled(blocks):
    """Wrap temporal blocks in Resampled, leave scalar/already-wrapped blocks untouched."""
    if not isinstance(blocks, tuple):
        return _wrap_one(blocks)
    return tuple(_wrap_one(b) for b in blocks)


def create_dls_from_blocks(
    inputs,
    targets,
    train_files: list[Path | str],
    valid_files: list[Path | str],
    test_files: list[Path | str] | None = None,
    win_sz: int = 100,
    stp_sz: int = 1,
    bs: int = 64,
    valid_stp_sz: int | None = None,
    num_workers: int = 5,
    n_batches_train: int | None = 300,
    n_batches_valid: int | None = None,
    targ_fs: list[float] | float | None = None,
    src_fs: float | str | None = None,
) -> DataLoaders:
    """Create DataLoaders from user-provided blocks and file lists.

    Args:
        inputs: input block or tuple of blocks
        targets: target block or tuple of blocks
        train_files: training HDF5 files
        valid_files: validation HDF5 files
        test_files: test HDF5 files, or None
        win_sz: window size in (resampled) samples
        stp_sz: step size between consecutive training windows
        bs: batch size
        valid_stp_sz: step size between consecutive validation windows, defaults to win_sz
        num_workers: number of worker processes for the DataLoader
        n_batches_train: exact number of training batches per epoch, None for all
        n_batches_valid: exact number of validation batches per epoch, None for all
        targ_fs: target sampling frequency/frequencies for resampling
        src_fs: source sampling frequency (number or HDF5 attribute name)
    """
    if valid_stp_sz is None:
        valid_stp_sz = win_sz
    if test_files is None:
        test_files = []

    # Handle resampling
    if targ_fs is not None:
        rs_map = _compute_resampling_factors(train_files + valid_files + test_files, targ_fs, src_fs)
        input_block = _wrap_resampled(inputs)
        target_block = _wrap_resampled(targets)
    else:
        rs_map = None
        input_block = inputs
        target_block = targets

    # --- Build entries (expand for multi-frequency resampling) ---
    def _make_entries(files):
        entries = []
        for f in files:
            path = str(f)
            if rs_map is not None and path in rs_map:
                for factor, _ in rs_map[path]:
                    entries.append(FileEntry(path=path, resampling_factor=factor))
            else:
                entries.append(FileEntry(path=path))
        return entries

    train_entries = _make_entries(train_files)
    valid_entries = _make_entries(valid_files)
    test_entries = _make_entries(test_files)

    # --- Build datasets ---
    train_ds = WindowedDataset(train_entries, input_block, target_block, win_sz=win_sz, stp_sz=stp_sz)
    valid_ds = WindowedDataset(valid_entries, input_block, target_block, win_sz=win_sz, stp_sz=valid_stp_sz)

    # --- Build samplers ---
    if n_batches_train is not None:
        n_samples_train = n_batches_train * bs
        train_sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples_train)
    else:
        train_sampler = RandomSampler(train_ds)

    if n_batches_valid is not None:
        n_samples_valid = n_batches_valid * bs
        valid_sampler = RandomSampler(valid_ds, replacement=True, num_samples=n_samples_valid)
    else:
        valid_sampler = SequentialSampler(valid_ds)

    # --- Build DataLoaders ---
    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=bs,
        sampler=valid_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Test DataLoader: full-file mode, sequential, bs=1
    test_dl = None
    if test_entries:
        test_ds = WindowedDataset(test_entries, input_block, target_block, win_sz=None)
        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            sampler=SequentialSampler(test_ds),
            num_workers=0,
        )

    # --- Build DataLoaders container ---
    dls = DataLoaders(train=train_dl, valid=valid_dl, test=test_dl)
    dls._train_files = train_files
    return dls


def create_dls(
    u: list[str],
    y: list[str],
    dataset: Path | str | list | dict,
    win_sz: int = 100,
    stp_sz: int = 1,
    bs: int = 64,
    valid_stp_sz: int | None = None,
    num_workers: int = 5,
    n_batches_train: int | None = 300,
    n_batches_valid: int | None = None,
    dls_id: str | None = None,
    targ_fs: list[float] | float | None = None,
    src_fs: float | str | None = None,
) -> DataLoaders:
    """Create DataLoaders from HDF5 time-series files.

    Args:
        u: list of input signal names
        y: list of output signal names
        dataset: path to dataset, list of filepaths, or {'train':[], 'valid':[], 'test':[]} dict
        win_sz: window size in (resampled) samples
        stp_sz: step size between consecutive training windows
        bs: batch size
        valid_stp_sz: step size between consecutive validation windows, defaults to win_sz
        num_workers: number of worker processes for the DataLoader
        n_batches_train: exact number of training batches per epoch, None for all
        n_batches_valid: exact number of validation batches per epoch, None for all
        dls_id: cache id: when provided, computes exact stats from full training set and caches to disk
        targ_fs: target sampling frequency/frequencies for resampling
        src_fs: source sampling frequency (number or HDF5 attribute name)
    """
    # --- Resolve files ---
    if isinstance(dataset, dict):
        train_files = list(dataset.get("train", []))
        valid_files = list(dataset.get("valid", []))
        test_files = list(dataset.get("test", []))
        if not train_files:
            raise ValueError("dataset dict must contain 'train' files")
        if not valid_files:
            raise ValueError("dataset dict must contain 'valid' files")
    elif isinstance(dataset, (Path, str)):
        split = discover_split_files(dataset)
        train_files = split["train"]
        valid_files = split["valid"]
        test_files = split["test"]
        if not train_files and not valid_files:
            all_files = get_hdf_files(dataset)
            if not all_files:
                raise FileNotFoundError(
                    f"No HDF5 files found in '{dataset}'. Check that the path exists and contains .hdf5/.h5 files."
                )
            train_idxs, valid_idxs = split_by_parent(all_files)
            train_files = [all_files[i] for i in train_idxs]
            valid_files = [all_files[i] for i in valid_idxs]
            test_files = [f for f in all_files if f not in train_files and f not in valid_files]
    elif isinstance(dataset, (list, tuple)):
        train_idxs, valid_idxs = split_by_parent(dataset)
        train_files = [dataset[i] for i in train_idxs]
        valid_files = [dataset[i] for i in valid_idxs]
        test_files = [f for f in dataset if Path(f).parent.name == "test"]
    else:
        raise ValueError(f"dataset must be a Path, list, or dict. {type(dataset)} was given.")

    dls = create_dls_from_blocks(
        inputs=HDF5Signals(u),
        targets=HDF5Signals(y),
        train_files=train_files,
        valid_files=valid_files,
        test_files=test_files,
        win_sz=win_sz,
        stp_sz=stp_sz,
        bs=bs,
        valid_stp_sz=valid_stp_sz,
        num_workers=num_workers,
        n_batches_train=n_batches_train,
        n_batches_valid=n_batches_valid,
        targ_fs=targ_fs,
        src_fs=src_fs,
    )
    dls._signal_names = (u, y)
    dls._dls_id = dls_id
    return dls
