"""Signal reader blocks for time series data."""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from scipy.signal import resample as fft_resample

from .signal import resample_interp


@dataclass
class _MmapInfo:
    """Byte-level metadata for a single contiguous HDF5 dataset."""

    offset: int
    dtype: np.dtype
    shape: tuple[int, ...]


class HDF5Signals:
    """Temporal block: reads named 1-D datasets from HDF5 files.

    Uses np.memmap for contiguous datasets (~2x faster than h5py),
    falls back to h5py for chunked/compressed datasets.

    Args:
        names: dataset column names to extract
        dataset: HDF5 group name containing the datasets, None for root
        fs_idx: index of frequency column, scaled by resampling_factor
        dt_idx: index of time-step column, scaled by resampling_factor
    """

    def __init__(
        self,
        names: list[str],
        dataset: str | None = None,
        fs_idx: int | None = None,
        dt_idx: int | None = None,
    ):
        self.names = names
        self.dataset = dataset
        self.fs_idx = fs_idx
        self.dt_idx = dt_idx
        self._len_cache: dict[str, int] = {}
        self._mmap_info: dict[str, dict[str, _MmapInfo | None]] = {}

    def _probe(self, path: str) -> None:
        """Probe HDF5 datasets for contiguous layout; cache byte offsets."""
        if path in self._mmap_info:
            return
        info = {}
        with h5py.File(path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            for name in self.names:
                dataset = ds[name]
                if dataset.chunks is not None:
                    info[name] = None
                    continue
                byte_offset = dataset.id.get_offset()
                if byte_offset is None or byte_offset == 0:
                    info[name] = None
                    continue
                info[name] = _MmapInfo(
                    offset=byte_offset,
                    dtype=dataset.dtype,
                    shape=dataset.shape,
                )
            if path not in self._len_cache:
                self._len_cache[path] = ds[self.names[0]].shape[0]
        self._mmap_info[path] = info

    def read(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        """Read columns and stack -> [seq_len, n_features]."""
        self._probe(path)
        path_info = self._mmap_info[path]
        arrays = []
        h5py_names = []
        for name in self.names:
            mi = path_info[name]
            if mi is not None:
                mm = np.memmap(path, dtype=mi.dtype, mode="r", offset=mi.offset, shape=mi.shape)
                arrays.append((name, np.array(mm[l_slc:r_slc])))
            else:
                h5py_names.append(name)
        if h5py_names:
            with h5py.File(path, "r") as f:
                ds = f if self.dataset is None else f[self.dataset]
                for name in h5py_names:
                    arrays.append((name, ds[name][l_slc:r_slc]))
        arrays.sort(key=lambda pair: self.names.index(pair[0]))
        return np.stack([a for _, a in arrays], axis=-1)

    def file_len(self, path: str) -> int:
        """Length of first named dataset. Cached per path."""
        if path not in self._len_cache:
            self._probe(path)
        return self._len_cache[path]

    @property
    def n_features(self) -> int:
        return len(self.names)


class HDF5Attrs:
    """Scalar block: reads named HDF5 attributes.

    Args:
        names: attribute names to extract
        dataset: HDF5 group name containing the attributes, None for root
        dtype: output data type
    """

    def __init__(
        self,
        names: list[str],
        dataset: str | None = None,
        dtype: np.dtype = np.float32,
    ):
        self.names = names
        self.dataset = dataset
        self.dtype = dtype

    def read(self, path: str) -> np.ndarray:
        with h5py.File(path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            return np.array([self.dtype(ds.attrs[n]).item() for n in self.names])

    @property
    def n_features(self) -> int:
        return len(self.names)


class Resampled:
    """Wraps a temporal block, reading in original space and resampling to target rate.

    Args:
        block: temporal block with read(path, l_slc, r_slc) and file_len(path)
        fast_resample: use linear interpolation (True) or FFT resampling (False)
    """

    def __init__(self, block: HDF5Signals, fast_resample: bool = True):
        self.block = block
        self.fast_resample = fast_resample

    def read(self, path: str, l_slc: int, r_slc: int, factor: float) -> np.ndarray:
        """Read and resample a window. l_slc/r_slc are in resampled coordinates."""
        if factor == 1.0:
            return self.block.read(path, l_slc, r_slc)

        target_len = r_slc - l_slc
        l_orig = math.floor(l_slc / factor)
        r_orig = min(math.ceil(r_slc / factor) + 2, self.file_len(path))
        raw = self.block.read(path, l_orig, r_orig)

        if self.fast_resample:
            resampled = resample_interp(raw, factor)
        else:
            resampled = fft_resample(raw, int(raw.shape[0] * factor), window=("kaiser", 14.0))

        if hasattr(self.block, "fs_idx") and self.block.fs_idx is not None:
            resampled[:, self.block.fs_idx] = raw[0, self.block.fs_idx] * factor
        if hasattr(self.block, "dt_idx") and self.block.dt_idx is not None:
            resampled[:, self.block.dt_idx] = raw[0, self.block.dt_idx] / factor

        return resampled[:target_len]

    def file_len(self, path: str) -> int:
        """Length in original (un-resampled) coordinates."""
        return self.block.file_len(path)

    @property
    def n_features(self) -> int:
        return self.block.n_features


class CSVSignals:
    """Temporal block: reads named columns from CSV files.

    Args:
        columns: column names to extract
        delimiter: CSV delimiter character
    """

    def __init__(self, columns: list[str], delimiter: str = ","):
        self.columns = columns
        self.delimiter = delimiter
        self._data_cache: dict[str, np.ndarray] = {}
        self._len_cache: dict[str, int] = {}

    def _load(self, path: str) -> np.ndarray:
        """Load CSV file and cache the result as [rows, n_columns] array."""
        if path not in self._data_cache:
            with open(path, newline="") as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                rows = [[float(row[col]) for col in self.columns] for row in reader]
            arr = np.array(rows, dtype=np.float32)
            self._data_cache[path] = arr
            self._len_cache[path] = arr.shape[0]
        return self._data_cache[path]

    def read(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        """Read columns and slice -> [seq_len, n_features]."""
        return self._load(path)[l_slc:r_slc]

    def file_len(self, path: str) -> int:
        """Row count excluding header. Cached per path."""
        if path not in self._len_cache:
            self._load(path)
        return self._len_cache[path]

    @property
    def n_features(self) -> int:
        return len(self.columns)


class Cached:
    """Wrapper that caches full file data in memory on first read.

    Args:
        block: any signal reader block to wrap
    """

    def __init__(self, block):
        self.block = block
        self._data_cache: dict[str, np.ndarray] = {}

    def read(self, path: str, l_slc: int | None = None, r_slc: int | None = None) -> np.ndarray:
        if path not in self._data_cache:
            if hasattr(self.block, "file_len"):
                self._data_cache[path] = self.block.read(path, 0, self.block.file_len(path))
            else:
                self._data_cache[path] = self.block.read(path)
        if l_slc is not None:
            return self._data_cache[path][l_slc:r_slc]
        return self._data_cache[path]

    def __getattr__(self, name):
        return getattr(self.block, name)


class FilenameScalar:
    """Scalar block: extracts numbers from filenames via regex.

    Args:
        pattern: regex with capture groups to extract from the filename stem
    """

    def __init__(self, pattern: str = r"(\d+\.?\d*)"):
        self._pattern = re.compile(pattern)
        self._n_features = self._pattern.groups

    def read(self, path: str) -> np.ndarray:
        """Search filename stem and return captured groups as float32 array."""
        stem = Path(path).stem
        m = self._pattern.search(stem)
        if m is None:
            raise ValueError(f"Pattern {self._pattern.pattern!r} did not match filename {stem!r}")
        return np.array([float(g) for g in m.groups()], dtype=np.float32)

    @property
    def n_features(self) -> int:
        return self._n_features
