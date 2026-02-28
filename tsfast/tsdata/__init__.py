"""Pure-PyTorch data pipeline for time series datasets."""

from .blocks import HDF5Attrs, HDF5Signals, Resampled
from .dataset import FileEntry, WindowedDataset
from .norm import NormPair, NormStats, compute_stats, compute_stats_from_files
from .pipeline import DataLoaders, create_dls, create_dls_from_blocks
from .signal import downsample_mean, resample_interp, running_mean
from .split import (
    discover_split_files,
    get_hdf_files,
    is_dataset_directory,
    split_by_parent,
    split_by_percentage,
)
