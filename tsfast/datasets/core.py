"""DataLoader creation and normalization utilities for HDF5 time-series datasets."""

__all__ = [
    "create_dls_test",
    "extract_mean_std_from_dls",
    "dict_file_save",
    "dict_file_load",
    "extract_mean_std_from_hdffiles",
    "extract_mean_std_from_dataset",
    "extract_norm_from_hdffiles",
    "is_dataset_directory",
    "create_dls",
    "get_default_dataset_path",
    "get_dataset_path",
    "clean_default_dataset_path",
    "create_dls_downl",
    "estimate_norm_stats",
    "NormPair",
    "NormStats",
    "split_by_parent",
]

import os
import pickle
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import rmtree
from typing import NamedTuple

import h5py
import numpy as np
import torch

from fastcore.foundation import L, mask2idxs
from fastcore.meta import delegates
from fastai.data.block import DataBlock
from fastai.data.core import TfmdDL

from ..data.block import SequenceBlock
from ..data.core import (
    CreateDict,
    DfApplyFuncSplit,
    DfHDFCreateWindows,
    TensorSequencesInput,
    TensorSequencesOutput,
    get_hdf_files,
)
from ..data.loader import TbpttDl, NBatches_Factory, BatchLimit_Factory
from ..data.split import ParentSplitter


@dataclass
class NormPair:
    """Per-signal normalization statistics (mean, std, min, max as 1-D numpy arrays).

    Args:
        mean: per-feature mean values
        std: per-feature standard deviation values
        min: per-feature minimum values
        max: per-feature maximum values
    """

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    def __add__(self, other: "NormPair") -> "NormPair":
        "Concatenate two NormPairs feature-wise (e.g. norm_u + norm_y)."
        return NormPair(*(np.hstack([a, b]) for a, b in zip(self, other)))

    def __iter__(self):
        return iter((self.mean, self.std, self.min, self.max))

    def __getitem__(self, idx):
        return (self.mean, self.std, self.min, self.max)[idx]


class NormStats(NamedTuple):
    """Normalization statistics for input, state, and output signals.

    Args:
        u: normalization stats for input signals
        x: normalization stats for state signals, or None if no states
        y: normalization stats for output signals
    """

    u: NormPair
    x: NormPair | None
    y: NormPair


def extract_mean_std_from_dls(dls) -> NormStats:
    "Extract normalization statistics stored on the DataLoaders object."
    if not hasattr(dls, "norm_stats"):
        raise AttributeError("DataLoaders missing norm_stats. Use create_dls to create them.")
    return dls.norm_stats


def dict_file_save(key: str, value, f_path: str | Path = "dls_normalize.p"):
    "Save value to a dictionary file, appends if it already exists."

    # use the absolute path, so seperate processes refer to the same file
    f_path = Path(f_path)
    try:
        with open(f_path, "rb") as f:
            dictionary = pickle.load(f)
    except OSError:
        dictionary = {}
    dictionary[key] = value

    with open(f_path, "wb") as f:
        pickle.dump(dictionary, f)


def dict_file_load(key: str, f_path: str | Path = "dls_normalize.p"):
    "Load value from a dictionary file."

    # use the absolute path, so seperate processes refer to the same file
    f_path = Path(f_path)
    try:
        with open(f_path, "rb") as f:
            dictionary = pickle.load(f)
        if key in dictionary:
            return dictionary[key]
    except OSError:
        print(f"{f_path} not found")
    return None


def _cache_path(dls_id):
    return Path(f".tsfast_cache/{dls_id}.pkl")


def _save_norm_stats(dls_id, norm_stats):
    p = _cache_path(dls_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(norm_stats, f)


def _load_norm_stats(dls_id):
    p = _cache_path(dls_id)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def extract_mean_std_from_hdffiles(
    lst_files: list,
    lst_signals: list[str],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Calculate the mean and standard deviation of the signals from the provided HDF5 files.

    Args:
        lst_files: paths to HDF5 files
        lst_signals: signal names, each a dataset within the HDF5 files
    """
    if len(lst_signals) == 0:
        return (None, None)

    # Initialize accumulators for mean and squared mean
    sums = np.zeros(len(lst_signals))
    squares = np.zeros(len(lst_signals))
    counts = 0

    # Process each file
    for file in lst_files:
        with h5py.File(file, "r") as f:
            for i, signal in enumerate(lst_signals):
                data = f[signal][:]
                if data.ndim > 1:
                    raise ValueError(f"Each dataset in a file has to be 1d. {signal} is {data.ndim}.")
                sums[i] += np.sum(data)
                squares[i] += np.sum(data**2)

        counts += data.size

    # Calculate the mean and standard deviation across all files
    means = sums / counts
    stds = np.sqrt((squares / counts) - (means**2))

    return means.astype(np.float32), stds.astype(np.float32)


def extract_mean_std_from_dataset(lst_files: list, u: list[str], x: list[str], y: list[str]) -> tuple:
    "Extract mean/std normalization statistics for u, x, and y signals from training files."
    train_files = lst_files[ParentSplitter()(lst_files)[0]]
    norm_u = extract_mean_std_from_hdffiles(train_files, u)
    norm_x = extract_mean_std_from_hdffiles(train_files, x)
    norm_y = extract_mean_std_from_hdffiles(train_files, y)
    return (norm_u, norm_x, norm_y)


def extract_norm_from_hdffiles(lst_files: list, lst_signals: list[str]) -> NormPair | None:
    "Compute NormPair (mean, std, min, max) from all samples in HDF5 files."
    if len(lst_signals) == 0:
        return None
    sums = np.zeros(len(lst_signals))
    squares = np.zeros(len(lst_signals))
    mins = np.full(len(lst_signals), np.inf)
    maxs = np.full(len(lst_signals), -np.inf)
    counts = 0
    for file in lst_files:
        with h5py.File(file, "r") as f:
            for i, signal in enumerate(lst_signals):
                data = f[signal][:]
                if data.ndim > 1:
                    raise ValueError(f"Each dataset in a file has to be 1d. {signal} is {data.ndim}.")
                sums[i] += np.sum(data)
                squares[i] += np.sum(data**2)
                mins[i] = min(mins[i], np.min(data))
                maxs[i] = max(maxs[i], np.max(data))
            counts += data.size
    means = sums / counts
    stds = np.sqrt((squares / counts) - (means**2))
    return NormPair(means.astype(np.float32), stds.astype(np.float32), mins.astype(np.float32), maxs.astype(np.float32))


def estimate_norm_stats(dls, n_batches: int = 10) -> tuple[NormPair, ...]:
    "Estimate per-feature mean/std/min/max from training batches."
    acc = None
    for i, batch in enumerate(dls.train):
        if i >= n_batches:
            break
        if acc is None:
            acc = [[t] for t in batch]
        else:
            for j, t in enumerate(batch):
                acc[j].append(t)

    stats = []
    for tensors in acc:
        t = torch.cat(tensors).flatten(0, -2)  # [total_samples, features]
        stats.append(
            NormPair(
                mean=t.mean(0).cpu().numpy().astype(np.float32),
                std=t.std(0).cpu().numpy().astype(np.float32),
                min=t.min(0).values.cpu().numpy().astype(np.float32),
                max=t.max(0).values.cpu().numpy().astype(np.float32),
            )
        )
    return tuple(stats)


def _FileListSplitter(train_set, valid_set):
    "Split items by membership in train/valid file path sets."

    def _inner(o, **kwargs):
        if len(o) > 0 and isinstance(o.iloc[0] if hasattr(o, "iloc") else o[0], dict):
            o = [d["path"] for d in o]
        train_idxs = mask2idxs(str(p) in train_set for p in o)
        valid_idxs = mask2idxs(str(p) in valid_set for p in o)
        return train_idxs, valid_idxs

    return _inner


def split_by_parent(
    files_or_path: Path | str | list, train_name: str = "train", valid_name: str = "valid", test_name: str = "test"
) -> dict[str, L]:
    "Collect HDF files and group into train/valid/test dict by parent directory name."
    if isinstance(files_or_path, (Path, str)):
        files = get_hdf_files(files_or_path)
    else:
        files = L(files_or_path)
    return {
        "train": files.filter(lambda f: Path(f).parent.name == train_name),
        "valid": files.filter(lambda f: Path(f).parent.name == valid_name),
        "test": files.filter(lambda f: Path(f).parent.name == test_name),
    }


def is_dataset_directory(ds_path: Path | str) -> bool:
    """Check if the given directory path is a dataset with HDF5 files.

    Args:
        ds_path: path to the directory to check

    Returns:
        True if the directory contains 'train', 'valid', and 'test' subdirectories,
        each with at least one HDF5 file.
    """
    required_dirs = ["train", "valid", "test"]

    # Check if all required directories exist and contain HDF5 files
    for dir_name in required_dirs:
        dir_path = os.path.join(ds_path, dir_name)
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return False  # Directory missing
        hdf_files = get_hdf_files(dir_path, recurse=True)
        if not hdf_files:
            return False  # No HDF5 files found in this directory

    return True  # All checks passed


def create_dls(
    u: list[str],
    y: list[str],
    dataset: Path | list | dict,
    win_sz: int = 100,
    x: list[str] = [],
    stp_sz: int = 1,
    sub_seq_len: int | None = None,
    bs: int = 64,
    prediction: bool = False,
    input_delay: bool = False,
    valid_stp_sz: int | None = None,
    cached: bool = True,
    num_workers: int = 5,
    n_batches_train: int | None = 300,
    n_batches_valid: int | None = None,
    max_batches_training: int | None = None,
    max_batches_valid: int | None = None,
    dls_id: str | None = None,
):
    """Create DataLoaders from HDF5 time-series files with normalization statistics.

    Args:
        u: list of input signal names
        y: list of output signal names
        dataset: path to dataset, list of filepaths, or {'train':[], 'valid':[], 'test':[]} dict
        win_sz: initial window size
        x: optional list of state signal names
        stp_sz: step size between consecutive windows
        sub_seq_len: if provided uses truncated backpropagation through time with this sub sequence length
        bs: batch size
        prediction: if true, the output is concatenated to the input, mainly for prediction tasks
        input_delay: if true, the input is delayed by one step
        valid_stp_sz: step size between consecutive validation windows, defaults to win_sz
        cached: if true, the data is cached in RAM
        num_workers: number of processes for the dataloader, 0 for no multiprocessing
        n_batches_train: exact number of training batches per epoch
        n_batches_valid: exact number of validation batches per epoch
        max_batches_training: DEPRECATED: limits the number of training batches in a single epoch
        max_batches_valid: DEPRECATED: limits the number of validation batches in a single epoch
        dls_id: cache id: when provided, computes exact stats from full training set and caches to disk
    """
    if valid_stp_sz is None:
        valid_stp_sz = win_sz

    # resolve dataset into hdf_files, splitter, and test_files
    if isinstance(dataset, dict):
        train_files = L(dataset.get("train", []))
        valid_files = L(dataset.get("valid", []))
        test_files = L(dataset.get("test", []))
        if len(train_files) == 0:
            raise ValueError("dataset dict must contain 'train' files")
        if len(valid_files) == 0:
            raise ValueError("dataset dict must contain 'valid' files")
        hdf_files = train_files + valid_files + test_files
        splitter = _FileListSplitter({str(f) for f in train_files}, {str(f) for f in valid_files})
    elif isinstance(dataset, (Path, str)):
        hdf_files = get_hdf_files(dataset)
        splitter = ParentSplitter()
        train_files = hdf_files[splitter(hdf_files)[0]]
        test_files = None
    elif isinstance(dataset, (list, tuple, L)):
        hdf_files = dataset
        splitter = ParentSplitter()
        train_files = hdf_files[splitter(hdf_files)[0]]
        test_files = None
    else:
        raise ValueError(f"dataset must be a Path, list, or dict. {type(dataset)} was given.")

    # choose input and output signal blocks
    if prediction:
        if input_delay:
            blocks = (
                SequenceBlock.from_hdf(u + x + y, TensorSequencesInput, clm_shift=[-1] * len(u + x + y), cached=cached),
                SequenceBlock.from_hdf(y, TensorSequencesOutput, clm_shift=[1] * len(y), cached=cached),
            )
        else:
            blocks = (
                SequenceBlock.from_hdf(
                    u + x + y, TensorSequencesInput, clm_shift=([0] * len(u) + [-1] * len(x + y)), cached=cached
                ),
                SequenceBlock.from_hdf(y, TensorSequencesOutput, clm_shift=[1] * len(y), cached=cached),
            )
    else:
        blocks = (
            SequenceBlock.from_hdf(u, TensorSequencesInput, cached=cached),
            SequenceBlock.from_hdf(y, TensorSequencesOutput, cached=cached),
        )

    seq = DataBlock(
        blocks=blocks,
        get_items=CreateDict(
            [
                DfApplyFuncSplit(
                    splitter,
                    DfHDFCreateWindows(win_sz=win_sz, stp_sz=stp_sz, clm=u[0]),
                    DfHDFCreateWindows(win_sz=win_sz, stp_sz=valid_stp_sz, clm=u[0]),
                )
            ]
        ),
        splitter=splitter,
    )

    # Determine which factory to use based on parameters
    use_old_factory = False
    if max_batches_training is not None or max_batches_valid is not None:
        # Old parameters provided
        if n_batches_train is not None or n_batches_valid is not None:
            # Both old and new provided - new takes precedence
            warnings.warn(
                "Both old ('max_batches_training', 'max_batches_valid') and new ('n_batches_train', 'n_batches_valid') "
                "parameters provided. Using new parameters. Old parameters are deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            # Only old parameters - use old factory with warning
            warnings.warn(
                "Parameters 'max_batches_training' and 'max_batches_valid' are deprecated. "
                "Use 'n_batches_train' and 'n_batches_valid' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_old_factory = True

    if use_old_factory:
        # Use deprecated BatchLimit_Factory for backwards compatibility
        if sub_seq_len is None:
            dl_kwargs = [{"max_batches": max_batches_training}, {"max_batches": max_batches_valid}]
            dl_type = BatchLimit_Factory(TfmdDL)
        else:
            dl_kwargs = [
                {"sub_seq_len": sub_seq_len, "max_batches": max_batches_training},
                {"sub_seq_len": sub_seq_len, "max_batches": max_batches_valid},
            ]
            dl_type = BatchLimit_Factory(TbpttDl)
    else:
        # Use new NBatches_Factory
        if sub_seq_len is None:
            dl_kwargs = [{"n_batches": n_batches_train}, {"n_batches": n_batches_valid}]
            dl_type = NBatches_Factory(TfmdDL)
        else:
            dl_kwargs = [
                {"sub_seq_len": sub_seq_len, "n_batches": n_batches_train},
                {"sub_seq_len": sub_seq_len, "n_batches": n_batches_valid},
            ]
            dl_type = NBatches_Factory(TbpttDl)

    dls = seq.dataloaders(hdf_files, bs=bs, num_workers=num_workers, dl_type=dl_type, dl_kwargs=dl_kwargs)

    # compute normalization stats
    if dls_id is not None:
        # exact stats from full training set, with file caching
        norm_stats = _load_norm_stats(dls_id)
        if norm_stats is None:
            norm_u = extract_norm_from_hdffiles(train_files, u)
            norm_x = extract_norm_from_hdffiles(train_files, x) if len(x) > 0 else None
            norm_y = extract_norm_from_hdffiles(train_files, y)
            norm_stats = NormStats(norm_u, norm_x, norm_y)
            _save_norm_stats(dls_id, norm_stats)
        dls.norm_stats = norm_stats
    else:
        # estimate from training batches
        input_stats, output_stats = estimate_norm_stats(dls)
        n_u, n_x = len(u), len(x)
        norm_u = NormPair(input_stats.mean[:n_u], input_stats.std[:n_u], input_stats.min[:n_u], input_stats.max[:n_u])
        norm_x = (
            NormPair(
                input_stats.mean[n_u : n_u + n_x],
                input_stats.std[n_u : n_u + n_x],
                input_stats.min[n_u : n_u + n_x],
                input_stats.max[n_u : n_u + n_x],
            )
            if n_x > 0
            else None
        )
        norm_y = output_stats
        dls.norm_stats = NormStats(norm_u, norm_x, norm_y)

    # add the test dataloader
    if test_files is None:
        test_files = hdf_files.filter(lambda o: Path(o).parent.name == "test")
    if len(test_files) > 0:
        if prediction:
            items = CreateDict([DfHDFCreateWindows(win_sz=win_sz, stp_sz=win_sz, clm=u[0])])(test_files)
            test_dl = dls.test_dl(items, bs=min(bs, len(items)), with_labels=True)
        else:
            items = CreateDict()(test_files)
            test_dl = dls.test_dl(items, bs=1, with_labels=True)
        dls.loaders.append(test_dl)

    return dls


def _get_project_root():
    """Walk up from this file to find the project root (directory containing test_data/)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "test_data").is_dir():
            return d
        d = d.parent
    raise FileNotFoundError("Could not find project root containing test_data/")


create_dls_test = partial(
    create_dls, u=["u"], y=["y"], dataset=_get_project_root() / "test_data/WienerHammerstein", win_sz=100, stp_sz=100
)
create_dls_test.__doc__ = "Create a DataLoader from the small test dataset bundled with tsfast."


def get_default_dataset_path() -> Path:
    "Create and return the default directory for storing datasets in the user's home."
    data_dir = Path.home() / ".tsfast" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_dataset_path() -> Path:
    "Return the dataset directory from TSFAST_PATH env var, or the default path."
    env_var_name = "TSFAST_PATH"
    env_path = os.getenv(env_var_name)

    if env_path:
        return Path(env_path)
    else:
        return get_default_dataset_path()


def clean_default_dataset_path() -> None:
    "Remove the default directory where datasets are stored."
    rmtree(get_default_dataset_path())


@delegates(create_dls, keep=True)
def create_dls_downl(
    dataset: Path | str | None = None,
    download_function: Callable | None = None,
    **kwargs,
):
    """Create DataLoaders, downloading the dataset first if needed.

    Args:
        dataset: path to the dataset directory, if not provided uses default
        download_function: callable that downloads the dataset to a given path
    """
    if dataset is None and download_function is not None:
        dataset = get_dataset_path() / download_function.__name__
    else:
        dataset = Path(dataset)

    if not is_dataset_directory(dataset):
        if download_function is not None:
            print(f'Dataset not found. Downloading it to "{dataset}"')
            download_function(dataset)
        else:
            raise ValueError(f"{dataset} does not contain a dataset. Check the path or activate the download flag.")

    return create_dls(dataset=dataset, **kwargs)
