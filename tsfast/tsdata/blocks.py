"""HDF5 signal readers for time series data blocks."""

import math

import h5py
import numpy as np

from scipy.signal import resample as fft_resample

from .signal import resample_interp


class HDF5Signals:
    """Temporal block: reads named 1-D datasets from HDF5 files.

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

    def read(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        """Read columns and stack -> [seq_len, n_features]."""
        with h5py.File(path, "r") as f:
            ds = f if self.dataset is None else f[self.dataset]
            arrays = [ds[name][l_slc:r_slc] for name in self.names]
        return np.stack(arrays, axis=-1)

    def file_len(self, path: str) -> int:
        """Length of first named dataset. Cached per path."""
        if path not in self._len_cache:
            with h5py.File(path, "r") as f:
                ds = f if self.dataset is None else f[self.dataset]
                self._len_cache[path] = ds[self.names[0]].shape[0]
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
