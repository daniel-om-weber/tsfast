"""Benchmark HDF5 loading strategies for tsfast.

Replicates tsjax benchmark methodology in tsfast's context:
- Experiment 1: Raw read speed (h5py vs mmap, contiguous vs chunked)
- Experiment 2: DataLoader throughput with num_workers scaling
- Experiment 3: Heavy workload (CPU-bound transforms)
"""

import gc
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

# ── tsfast imports ──────────────────────────────────────────────────────────
from tsfast.tsdata.blocks import HDF5Signals
from tsfast.tsdata.dataset import FileEntry, WindowedDataset
from tsfast.tsdata.prefetch import PrefetchLoader

# ── Config ──────────────────────────────────────────────────────────────────
N_FILES = 20
N_SAMPLES = 500_000
SIGNALS = ["u", "y"]
WIN_SZ = 1000
STP_SZ = 100
BS = 64
MEASURE_SECONDS = 3
WARMUP_READS = 1000
WARMUP_BATCHES = 10
FFT_ROUNDS = 10


# ── Synthetic Data Generation ───────────────────────────────────────────────

def create_data(base_dir: Path, contiguous: bool) -> list[str]:
    """Create synthetic HDF5 files. Returns list of file paths."""
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(N_FILES):
        path = base_dir / f"file_{i:03d}.h5"
        with h5py.File(str(path), "w") as f:
            for name in SIGNALS:
                data = rng.standard_normal(N_SAMPLES).astype(np.float32)
                if contiguous:
                    f.create_dataset(name, data=data)
                else:
                    f.create_dataset(name, data=data, chunks=(1000,))
        paths.append(str(path))
    return paths


# ── MmapHDF5Signals (local benchmark variant) ──────────────────────────────

@dataclass
class _MmapInfo:
    offset: int
    dtype: np.dtype
    shape: tuple[int, ...]


class MmapHDF5Signals:
    """Drop-in for HDF5Signals using np.memmap for contiguous datasets."""

    def __init__(self, names: list[str], dataset: str | None = None):
        self.names = names
        self.dataset = dataset
        self._len_cache: dict[str, int] = {}
        self._mmap_info: dict[str, dict[str, _MmapInfo | None]] = {}

    def _probe(self, path: str) -> None:
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
        self._probe(path)
        path_info = self._mmap_info[path]
        arrays = []
        for name in self.names:
            mi = path_info[name]
            if mi is not None:
                mm = np.memmap(path, dtype=mi.dtype, mode="r",
                               offset=mi.offset, shape=mi.shape)
                arrays.append(np.array(mm[l_slc:r_slc]))
            else:
                with h5py.File(path, "r") as f:
                    ds = f if self.dataset is None else f[self.dataset]
                    arrays.append(ds[name][l_slc:r_slc])
        return np.stack(arrays, axis=-1)

    def file_len(self, path: str) -> int:
        if path not in self._len_cache:
            self._probe(path)
        return self._len_cache[path]

    @property
    def n_features(self) -> int:
        return len(self.names)


# ── Heavy-workload wrapper ──────────────────────────────────────────────────

class HeavyHDF5Signals(HDF5Signals):
    """HDF5Signals with CPU-bound FFT transforms per read."""

    def read(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        arr = super().read(path, l_slc, r_slc)
        for _ in range(FFT_ROUNDS):
            arr = np.fft.rfft(arr, axis=0)
            arr = np.fft.irfft(arr, n=r_slc - l_slc, axis=0)
        return arr.real.astype(np.float32) if np.iscomplexobj(arr) else arr


class HeavyMmapSignals(MmapHDF5Signals):
    """MmapHDF5Signals with CPU-bound FFT transforms per read."""

    def read(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        arr = super().read(path, l_slc, r_slc)
        for _ in range(FFT_ROUNDS):
            arr = np.fft.rfft(arr, axis=0)
            arr = np.fft.irfft(arr, n=r_slc - l_slc, axis=0)
        return arr.real.astype(np.float32) if np.iscomplexobj(arr) else arr


# ── Benchmark Utilities ─────────────────────────────────────────────────────

def sustained_reads(block, paths: list[str], duration: float = MEASURE_SECONDS) -> tuple[int, float]:
    """Sustained random-window reads. Returns (count, elapsed)."""
    rng = np.random.default_rng(42)
    max_offset = N_SAMPLES - WIN_SZ
    n_paths = len(paths)

    for _ in range(WARMUP_READS):
        p = paths[int(rng.integers(0, n_paths))]
        off = int(rng.integers(0, max_offset))
        block.read(p, off, off + WIN_SZ)

    gc.collect()
    count = 0
    start = time.perf_counter()
    deadline = start + duration
    while time.perf_counter() < deadline:
        for _ in range(500):
            p = paths[int(rng.integers(0, n_paths))]
            off = int(rng.integers(0, max_offset))
            block.read(p, off, off + WIN_SZ)
        count += 500
    elapsed = time.perf_counter() - start
    return count, elapsed


def sustained_batches(loader, duration: float = MEASURE_SECONDS) -> tuple[int, int, float]:
    """Sustained batch iteration. Returns (n_batches, n_samples, elapsed).

    Single-pass: skips first WARMUP_BATCHES then measures the rest.
    One iterator, no reuse, no dangling worker state.
    """
    n_batches = 0
    n_samples = 0
    start = None

    for i, batch in enumerate(loader):
        if i < WARMUP_BATCHES:
            continue
        if start is None:
            gc.collect()
            start = time.perf_counter()
        n_batches += 1
        n_samples += batch[0].shape[0]
        if time.perf_counter() - start >= duration:
            break

    elapsed = time.perf_counter() - start
    return n_batches, n_samples, elapsed


def make_dataset(paths: list[str], block_cls, **block_kwargs) -> WindowedDataset:
    entries = [FileEntry(path=p) for p in paths]
    inp = block_cls(SIGNALS, **block_kwargs)
    tgt = block_cls(SIGNALS, **block_kwargs)
    return WindowedDataset(entries, inp, tgt, win_sz=WIN_SZ, stp_sz=STP_SZ)


def make_loader(ds: WindowedDataset, num_workers: int, prefetch: bool = False) -> DataLoader:
    sampler = RandomSampler(ds, replacement=True, num_samples=20000)
    dl = DataLoader(
        ds, batch_size=BS, sampler=sampler, num_workers=num_workers,
        drop_last=True,
    )
    if prefetch:
        return PrefetchLoader(dl)
    return dl


# ── Experiment 1: Raw Read Speed ────────────────────────────────────────────

def experiment_1(contig_paths: list[str], chunked_paths: list[str]):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Raw Read Speed (h5py vs mmap)")
    print("=" * 70)
    print(f"Config: {N_FILES} files × {len(SIGNALS)} signals × {N_SAMPLES:,} samples")
    print(f"Window: {WIN_SZ}, random offsets, sustained {MEASURE_SECONDS}s")
    print()
    print(f"{'Layout':<12} {'Method':<18} {'k reads/s':>10}  {'vs h5py':>8}")
    print("-" * 52)

    baselines = {}
    for layout, paths in [("contiguous", contig_paths), ("chunked", chunked_paths)]:
        for label, block in [
            ("h5py (current)", HDF5Signals(SIGNALS)),
            ("mmap", MmapHDF5Signals(SIGNALS)),
        ]:
            count, elapsed = sustained_reads(block, paths)
            rate = count / elapsed / 1000
            if "h5py" in label:
                baselines[layout] = rate
            speedup = rate / baselines[layout]
            print(f"{layout:<12} {label:<18} {rate:>10.1f}  {speedup:>7.2f}x")


# ── Experiment 2: DataLoader Throughput ─────────────────────────────────────

def experiment_2(contig_paths: list[str]):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: DataLoader Throughput (contiguous data)")
    print("=" * 70)
    print(f"Config: win_sz={WIN_SZ}, stp_sz={STP_SZ}, bs={BS}")
    print(f"Sustained {MEASURE_SECONDS}s per config")
    print()
    print(f"{'Config':<22} {'batch/s':>8} {'k samp/s':>10}  {'vs base':>8}")
    print("-" * 52)

    configs = [
        # (label, block_cls, num_workers, prefetch)
        ("h5py, w=0", HDF5Signals, 0, False),
        ("h5py, w=0+prefetch", HDF5Signals, 0, True),
        ("h5py, w=4", HDF5Signals, 4, False),
        ("mmap, w=0", MmapHDF5Signals, 0, False),
        ("mmap, w=0+prefetch", MmapHDF5Signals, 0, True),
        ("mmap, w=4", MmapHDF5Signals, 4, False),
    ]

    base = None
    for label, block_cls, nw, prefetch in configs:
        ds = make_dataset(contig_paths, block_cls)
        loader = make_loader(ds, nw, prefetch)
        nb, ns, elapsed = sustained_batches(loader, MEASURE_SECONDS)
        batch_rate = nb / elapsed
        sample_rate = ns / elapsed / 1000
        if base is None:
            base = sample_rate
        speedup = sample_rate / base
        print(f"{label:<22} {batch_rate:>8.0f} {sample_rate:>10.1f}  {speedup:>7.2f}x")
        del loader, ds
        gc.collect()


# ── Experiment 3: Heavy Workload ────────────────────────────────────────────

def experiment_3(contig_paths: list[str]):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Heavy Workload ({FFT_ROUNDS}× FFT round-trips)")
    print("=" * 70)
    print(f"Config: win_sz={WIN_SZ}, stp_sz={STP_SZ}, bs={BS}")
    print(f"Sustained {MEASURE_SECONDS}s per config")
    print()
    print(f"{'Config':<22} {'batch/s':>8} {'k samp/s':>10}  {'vs base':>8}")
    print("-" * 52)

    configs = [
        ("h5py heavy, w=0", HeavyHDF5Signals, 0, False),
        ("h5py heavy, w=4", HeavyHDF5Signals, 4, False),
        ("mmap heavy, w=0", HeavyMmapSignals, 0, False),
        ("mmap heavy, w=4", HeavyMmapSignals, 4, False),
    ]

    base = None
    for label, block_cls, nw, prefetch in configs:
        ds = make_dataset(contig_paths, block_cls)
        loader = make_loader(ds, nw, prefetch)
        nb, ns, elapsed = sustained_batches(loader, MEASURE_SECONDS)
        batch_rate = nb / elapsed
        sample_rate = ns / elapsed / 1000
        if base is None:
            base = sample_rate
        speedup = sample_rate / base
        print(f"{label:<22} {batch_rate:>8.0f} {sample_rate:>10.1f}  {speedup:>7.2f}x")
        del loader, ds
        gc.collect()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "tsfast_bench"

    contig_dir = tmp / "contiguous"
    chunked_dir = tmp / "chunked"

    print("Generating synthetic data...")
    contig_paths = create_data(contig_dir, contiguous=True)
    chunked_paths = create_data(chunked_dir, contiguous=False)
    total_mb = N_FILES * len(SIGNALS) * N_SAMPLES * 4 / 1e6
    print(f"  {N_FILES} files × {len(SIGNALS)} signals × {N_SAMPLES:,} samples = {total_mb:.0f} MB (per layout)")

    try:
        experiment_1(contig_paths, chunked_paths)
        experiment_2(contig_paths)
        experiment_3(contig_paths)
    finally:
        print(f"\nCleaning up {tmp}...")
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nDone.")
