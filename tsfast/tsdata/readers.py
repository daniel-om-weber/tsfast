"""Signal readers for time series data."""

import math
import re
import warnings
from pathlib import Path

import h5py
import numpy as np

from scipy.signal import resample as fft_resample

from .signal import resample_interp


class HDF5Signals:
    """Temporal reader: reads named 1-D datasets from HDF5 files.

    Uses np.memmap for contiguous datasets (~2x faster than h5py),
    falls back to h5py for chunked/compressed datasets.

    Args:
        names: dataset column names to extract
        dataset: HDF5 group name containing the datasets, None for root
        use_mmap: if True (default), use np.memmap for contiguous datasets;
            if False, use direct seek+read to avoid kernel VMA pressure
            with many files / DataLoader workers
    """

    def __init__(
        self,
        names: list[str],
        dataset: str | None = None,
        use_mmap: bool = True,
    ):
        self.names = names
        self.dataset = dataset
        self._use_mmap = use_mmap
        self._len_cache: dict[str, int] = {}
        self._probe_cache: dict[str, dict[str, np.ndarray | int | None]] = {}
        self._dtype: np.dtype | None = None
        self._widths: dict[str, int] = {}

    def _probe(self, path: str) -> None:
        """Probe HDF5 datasets and cache access info for contiguous ones."""
        if path in self._probe_cache:
            return
        entries = {}
        with h5py.File(path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            for name in self.names:
                dataset = ds[name]
                if self._dtype is None:
                    self._dtype = dataset.dtype
                if name not in self._widths:
                    self._widths[name] = dataset.shape[1] if dataset.ndim == 2 else 1
                if dataset.chunks is not None:
                    entries[name] = None
                    continue
                byte_offset = dataset.id.get_offset()
                if byte_offset is None or byte_offset == 0:
                    entries[name] = None
                    continue
                if self._use_mmap:
                    entries[name] = np.memmap(
                        path,
                        dtype=dataset.dtype,
                        mode="r",
                        offset=byte_offset,
                        shape=dataset.shape,
                    )
                else:
                    entries[name] = byte_offset
            if path not in self._len_cache:
                self._len_cache[path] = ds[self.names[0]].shape[0]
        self._probe_cache[path] = entries

    def read(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        """Read columns into pre-allocated array -> [seq_len, n_features]."""
        self._probe(path)
        entries = self._probe_cache[path]
        count = r_slc - l_slc
        out = np.empty((count, self.n_features), dtype=self._dtype)
        h5py_deferred: list[tuple[str, int, int]] = []
        col = 0
        if self._use_mmap:
            for name in self.names:
                w = self._widths[name]
                mm = entries[name]
                if mm is not None:
                    if w == 1:
                        out[:, col] = mm[l_slc:r_slc]
                    else:
                        out[:, col : col + w] = mm[l_slc:r_slc]
                else:
                    h5py_deferred.append((name, col, w))
                col += w
        else:
            itemsize = self._dtype.itemsize
            with open(path, "rb") as fh:
                for name in self.names:
                    w = self._widths[name]
                    off = entries[name]
                    if off is not None:
                        fh.seek(off + l_slc * w * itemsize)
                        buf = np.frombuffer(fh.read(count * w * itemsize), dtype=self._dtype)
                        if w == 1:
                            out[:, col] = buf
                        else:
                            out[:, col : col + w] = buf.reshape(count, w)
                    else:
                        h5py_deferred.append((name, col, w))
                    col += w
        if h5py_deferred:
            with h5py.File(path, "r") as f:
                ds = f if self.dataset is None else f[self.dataset]
                for name, c, w in h5py_deferred:
                    data = ds[name][l_slc:r_slc]
                    if w == 1:
                        out[:, c] = data
                    else:
                        out[:, c : c + w] = data
        return out

    def file_len(self, path: str) -> int:
        """Length of first named dataset. Cached per path."""
        if path not in self._len_cache:
            self._probe(path)
        return self._len_cache[path]

    @property
    def signal_names(self) -> list[str]:
        return self.names

    @property
    def n_features(self) -> int:
        if not self._widths:
            warnings.warn(
                "n_features accessed before probing; may be wrong for 2D datasets. Call read() or file_len() first.",
                stacklevel=2,
            )
            return len(self.names)
        return sum(self._widths.values())


class HDF5Attrs:
    """Reader for named HDF5 attributes (scalar or array).

    Scalar and array attributes are flattened into a single 1-D output vector.
    For example, reading a scalar ``dt`` and a (3,) array ``ja_rr`` produces
    a 4-element vector ``[dt, ja_rr[0], ja_rr[1], ja_rr[2]]``.

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
        self._n_features: int | None = None

    def probe(self, path: str) -> None:
        """Probe attribute sizes from a file."""
        if self._n_features is not None:
            return
        with h5py.File(path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            total = 0
            for n in self.names:
                val = np.atleast_1d(np.asarray(ds.attrs[n], dtype=self.dtype))
                total += val.size
        self._n_features = total

    def read(self, path: str) -> np.ndarray:
        with h5py.File(path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            parts = []
            for n in self.names:
                val = np.atleast_1d(np.asarray(ds.attrs[n], dtype=self.dtype))
                parts.append(val.ravel())
            result = np.concatenate(parts)
        if self._n_features is None:
            self._n_features = len(result)
        return result

    @property
    def signal_names(self) -> list[str]:
        return self.names

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            warnings.warn(
                "n_features accessed before probing; may be wrong for array attributes. Call read() or probe() first.",
                stacklevel=2,
            )
            return len(self.names)
        return self._n_features


class Resampled:
    """Wraps a temporal reader, reading in original space and resampling to target rate.

    Args:
        block: temporal reader with read(path, l_slc, r_slc) and file_len(path)
        fs_idx: column index of sampling rate, scaled by resampling factor
        dt_idx: column index of time step, scaled by resampling factor
        fast_resample: use linear interpolation (True) or FFT resampling (False)
    """

    def __init__(
        self,
        block: HDF5Signals,
        fs_idx: int | None = None,
        dt_idx: int | None = None,
        fast_resample: bool = True,
    ):
        self.block = block
        self.fs_idx = fs_idx
        self.dt_idx = dt_idx
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

        if self.fs_idx is not None:
            resampled[:, self.fs_idx] = raw[0, self.fs_idx] * factor
        if self.dt_idx is not None:
            resampled[:, self.dt_idx] = raw[0, self.dt_idx] / factor

        return resampled[:target_len]

    def file_len(self, path: str) -> int:
        """Length in original (un-resampled) coordinates."""
        return self.block.file_len(path)

    @property
    def signal_names(self) -> list[str]:
        return self.block.signal_names

    @property
    def n_features(self) -> int:
        return self.block.n_features


class CSVSignals:
    """Temporal reader: reads named columns from CSV files.

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
            with open(path) as f:
                header = f.readline().strip().split(self.delimiter)
                col_indices = [header.index(col) for col in self.columns]
                arr = np.loadtxt(f, delimiter=self.delimiter, usecols=col_indices, dtype=np.float32, ndmin=2)
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
    def signal_names(self) -> list[str]:
        return self.columns

    @property
    def n_features(self) -> int:
        return len(self.columns)


class Cached:
    """Wrapper that caches full file data in memory on first read.

    Args:
        block: any signal reader to wrap
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
    """Scalar reader: extracts numbers from filenames via regex.

    Args:
        pattern: regex with capture groups to extract from the filename stem
    """

    def __init__(self, pattern: str = r"(\d+\.?\d*)"):
        self._pattern = re.compile(pattern)
        self._n_features = self._pattern.groups
        # Reverse groupindex (name→number) to (number→name)
        idx_to_name = {v: k for k, v in self._pattern.groupindex.items()}
        self._signal_names = [idx_to_name.get(i + 1, f"scalar_{i}") for i in range(self._n_features)]

    def read(self, path: str) -> np.ndarray:
        """Search filename stem and return captured groups as float32 array."""
        stem = Path(path).stem
        m = self._pattern.search(stem)
        if m is None:
            raise ValueError(f"Pattern {self._pattern.pattern!r} did not match filename {stem!r}")
        return np.array([float(g) for g in m.groups()], dtype=np.float32)

    @property
    def signal_names(self) -> list[str]:
        return self._signal_names

    @property
    def n_features(self) -> int:
        return self._n_features
