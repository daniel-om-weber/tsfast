"""Signal readers for time series data."""

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from scipy.signal import resample as fft_resample

from .signal import resample_interp


@dataclass(frozen=True)
class SourceEntry:
    """A source signal to read: a file path plus the rate it is sampled at.

    This is the unit a :class:`~tsfast.tsdata.dataset.WindowedDataset` enumerates
    and the key :class:`Cached` stores under. Because it is frozen (hashable), one
    file resampled to several rates is several entries — one per ``factor`` — that
    never collide in the cache.

    Args:
        path: filesystem path to the source file
        factor: resampling factor in resampled coordinates (1.0 = no resampling)
    """

    path: str
    factor: float = 1.0


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

    def probe(self, entry: SourceEntry) -> None:
        """Probe HDF5 datasets and cache access info for contiguous ones."""
        path = entry.path
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

    def read(self, entry: SourceEntry, l_slc: int, r_slc: int) -> np.ndarray:
        """Read columns into pre-allocated array -> [seq_len, n_features]."""
        path = entry.path
        self.probe(entry)
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

    def file_len(self, entry: SourceEntry) -> int:
        """Length of first named dataset. Cached per path."""
        path = entry.path
        if path not in self._len_cache:
            self.probe(entry)
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

    def probe(self, entry: SourceEntry) -> None:
        """Probe attribute sizes from a file."""
        if self._n_features is not None:
            return
        with h5py.File(entry.path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            total = 0
            for n in self.names:
                val = np.atleast_1d(np.asarray(ds.attrs[n], dtype=self.dtype))
                total += val.size
        self._n_features = total

    def read(self, entry: SourceEntry) -> np.ndarray:
        with h5py.File(entry.path, "r") as f:
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
    """Wraps a temporal reader, reading in original space and resampling to a target rate.

    A pure resampling view over an inner reader. The factor is read per call from
    the :class:`SourceEntry` (``entry.factor``), so ``read``/``file_len`` keep the
    *same signature* as every other temporal reader — there is no factor to store
    and no marker to dispatch on, and a single ``Resampled`` instance serves every
    file at every rate. ``file_len`` is reported in *resampled* coordinates, which
    makes ``read(entry, 0, file_len)`` the whole-file resample; wrapping the view in
    :class:`Cached` therefore caches one resampled copy per ``(path, factor)`` and
    serves windows as slices (``create_dls(..., cache_resampled=True)`` does this).

    Reading the whole file once and slicing is also more correct than resampling
    each window independently: per-window resampling rebuilds the interpolation
    grid and lowpass-filter state from each window's own length, drifting the
    sample grid and leaving filter transients at window boundaries.  One resample
    of the whole signal yields a single, consistent grid.

    Args:
        block: temporal reader with read(entry, l_slc, r_slc) and file_len(entry)
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

    def probe(self, entry: SourceEntry) -> None:
        self.block.probe(entry)

    def read(self, entry: SourceEntry, l_slc: int, r_slc: int) -> np.ndarray:
        """Read a window in resampled coordinates.

        Reads only the raw span covering ``[l_slc, r_slc]`` and resamples it.
        Called with the whole resampled range (by :class:`Cached`) it reads and
        resamples the whole file; called per-window (cache disabled) it resamples
        just that span.
        """
        factor = entry.factor
        if factor == 1.0:
            return self.block.read(entry, l_slc, r_slc)
        target_len = r_slc - l_slc
        l_orig = math.floor(l_slc / factor)
        r_orig = min(math.ceil(r_slc / factor) + 2, self.block.file_len(entry))
        raw = self.block.read(entry, l_orig, r_orig)
        return self._resample(raw, factor)[:target_len]

    def _resample(self, raw: np.ndarray, factor: float) -> np.ndarray:
        if self.fast_resample:
            resampled = resample_interp(raw, factor)
        else:
            resampled = fft_resample(raw, int(raw.shape[0] * factor), window=("kaiser", 14.0))
        if self.fs_idx is not None:
            resampled[:, self.fs_idx] = raw[0, self.fs_idx] * factor
        if self.dt_idx is not None:
            resampled[:, self.dt_idx] = raw[0, self.dt_idx] / factor
        # float32 (the dtype the dataset casts to) to halve cache memory.
        return resampled.astype(np.float32, copy=False)

    def file_len(self, entry: SourceEntry) -> int:
        """Length in *resampled* coordinates (``int(orig_len * factor)``)."""
        return int(self.block.file_len(entry) * entry.factor)

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

    def read(self, entry: SourceEntry, l_slc: int, r_slc: int) -> np.ndarray:
        """Read columns and slice -> [seq_len, n_features]."""
        return self._load(entry.path)[l_slc:r_slc]

    def file_len(self, entry: SourceEntry) -> int:
        """Row count excluding header. Cached per path."""
        path = entry.path
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
    """Wrapper that caches a reader's whole-file output in memory on first read.

    The single caching mechanism in the pipeline, keyed by :class:`SourceEntry`.
    It is reader-agnostic: it asks the inner reader for its whole-file output once
    and serves windows as slices. Wrapping a :class:`Resampled` view therefore
    caches the resampled signal — ``Cached`` does the caching, ``Resampled`` only
    the resampling — and because the entry carries the factor, one file at several
    rates is several keys in the same cache that never collide.

    The inner reader supplies its whole-file output via ``read(entry, 0, file_len)``
    when it is temporal (has ``file_len``), else ``read(entry)`` (scalars).

    Args:
        block: any signal reader to wrap
    """

    def __init__(self, block):
        self.block = block
        self._data_cache: dict[SourceEntry, np.ndarray] = {}

    def read(self, entry: SourceEntry, l_slc: int | None = None, r_slc: int | None = None) -> np.ndarray:
        full = self._data_cache.get(entry)
        if full is None:
            full = self._read_full(entry)
            self._data_cache[entry] = full
        if l_slc is not None:
            return full[l_slc:r_slc]
        return full

    def _read_full(self, entry: SourceEntry) -> np.ndarray:
        if hasattr(self.block, "file_len"):
            return self.block.read(entry, 0, self.block.file_len(entry))
        return self.block.read(entry)

    def __getstate__(self):
        # Don't ship the (potentially large) cache to DataLoader workers on spawn;
        # they rebuild it lazily. On fork it is shared copy-on-write regardless.
        state = self.__dict__.copy()
        state["_data_cache"] = {}
        return state

    def __getattr__(self, name):
        # During unpickling __dict__ is empty; raise instead of recursing on self.block.
        if name == "block":
            raise AttributeError(name)
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

    def read(self, entry: SourceEntry) -> np.ndarray:
        """Search filename stem and return captured groups as float32 array."""
        stem = Path(entry.path).stem
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
