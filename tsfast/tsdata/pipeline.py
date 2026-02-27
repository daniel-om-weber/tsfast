"""DataLoaders container and create_dls factory for pure-PyTorch data pipeline."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .blocks import HDF5Signals, Resampled
from .dataset import FileEntry, WindowedDataset
from .norm import (
    NormPair,
    NormStats,
    _load_norm_stats,
    _save_norm_stats,
    compute_stats,
    compute_stats_from_files,
)  # NormPair used in batch-estimation path
from .split import discover_split_files, get_hdf_files, split_by_parent


@dataclass
class DataLoaders:
    """Container for train/valid/test DataLoaders with normalization stats.

    Args:
        train: training DataLoader
        valid: validation DataLoader
        test: test DataLoader, or None if no test split
        norm_stats: per-signal normalization statistics
    """

    train: DataLoader
    valid: DataLoader
    test: DataLoader | None = None
    norm_stats: NormStats | None = None

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
    if valid_stp_sz is None:
        valid_stp_sz = win_sz

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

    # --- Build blocks ---
    input_block = HDF5Signals(u)
    target_block = HDF5Signals(y)

    # Handle resampling
    if targ_fs is not None:
        rs_map = _compute_resampling_factors(train_files + valid_files + test_files, targ_fs, src_fs)
        input_block_r = Resampled(input_block)
        target_block_r = Resampled(target_block)
    else:
        rs_map = None
        input_block_r = input_block
        target_block_r = target_block

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
    train_ds = WindowedDataset(train_entries, input_block_r, target_block_r, win_sz=win_sz, stp_sz=stp_sz)
    valid_ds = WindowedDataset(valid_entries, input_block_r, target_block_r, win_sz=win_sz, stp_sz=valid_stp_sz)

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
        test_ds = WindowedDataset(test_entries, input_block_r, target_block_r, win_sz=None)
        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            sampler=SequentialSampler(test_ds),
            num_workers=0,
        )

    # --- Compute normalization stats ---
    if dls_id is not None:
        norm_stats = _load_norm_stats(dls_id)
        if norm_stats is None:
            norm_u = compute_stats_from_files(train_files, u)
            norm_y = compute_stats_from_files(train_files, y)
            norm_stats = NormStats(norm_u, norm_y)
            _save_norm_stats(dls_id, norm_stats)
    else:
        input_stats, output_stats = compute_stats(train_dl)
        n_u = len(u)
        norm_u = NormPair(input_stats.mean[:n_u], input_stats.std[:n_u], input_stats.min[:n_u], input_stats.max[:n_u])
        norm_y = output_stats
        norm_stats = NormStats(norm_u, norm_y)

    return DataLoaders(train=train_dl, valid=valid_dl, test=test_dl, norm_stats=norm_stats)
