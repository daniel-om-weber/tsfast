"""Normalization statistics computation for time series datasets."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import torch


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
        """Concatenate two NormPairs feature-wise."""
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


def _cache_path(dls_id: str) -> Path:
    return Path(f".tsfast_cache/{dls_id}.pkl")


def _save_norm_stats(dls_id: str, norm_stats: NormStats) -> None:
    p = _cache_path(dls_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(norm_stats, f)


def _load_norm_stats(dls_id: str) -> NormStats | None:
    p = _cache_path(dls_id)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def compute_stats_from_files(files: list, signals: list[str]) -> NormPair | None:
    """Compute exact NormPair (mean, std, min, max) from all samples in HDF5 files.

    Args:
        files: paths to HDF5 files
        signals: signal dataset names within each file
    """
    if len(signals) == 0:
        return None

    sums = np.zeros(len(signals))
    squares = np.zeros(len(signals))
    mins = np.full(len(signals), np.inf)
    maxs = np.full(len(signals), -np.inf)
    counts = 0

    for file in files:
        with h5py.File(file, "r") as f:
            for i, signal in enumerate(signals):
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
    return NormPair(
        means.astype(np.float32),
        stds.astype(np.float32),
        mins.astype(np.float32),
        maxs.astype(np.float32),
    )


def compute_stats(dl, n_batches: int = 10) -> tuple[NormPair, ...]:
    """Estimate per-feature mean/std/min/max from training batches.

    Args:
        dl: DataLoader to sample from
        n_batches: number of batches to use for estimation
    """
    acc = None
    for i, batch in enumerate(dl):
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
