"""Pure PyTorch Dataset for windowed time series from HDF5 files."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .readers import Resampled


@dataclass
class FileEntry:
    """A single HDF5 file with optional resampling.

    Args:
        path: filesystem path to the HDF5 file
        resampling_factor: scaling factor for the sequence length
    """

    path: str
    resampling_factor: float = 1.0


class WindowedDataset(Dataset):
    """Pure PyTorch Dataset for windowed time series from HDF5 files.

    Args:
        entries: list of FileEntry (path + resampling_factor)
        inputs: single reader or tuple of readers for input signals
        targets: single reader or tuple of readers for target signals
        win_sz: window size in (resampled) samples, None = full-file mode
        stp_sz: step size between windows
    """

    def __init__(
        self,
        entries: list[FileEntry],
        inputs,
        targets,
        win_sz: int | None = None,
        stp_sz: int = 1,
    ):
        self.entries = entries
        self._inputs = (inputs,) if not isinstance(inputs, tuple) else inputs
        self._targets = (targets,) if not isinstance(targets, tuple) else targets
        self._single_input = not isinstance(inputs, tuple)
        self._single_target = not isinstance(targets, tuple)
        self.win_sz = win_sz
        self.stp_sz = stp_sz
        self._ref_block = self._find_temporal(*self._inputs, *self._targets)

        if win_sz is not None:
            ref_block = self._ref_block
            counts = []
            for e in entries:
                raw_len = ref_block.file_len(e.path)
                eff_len = int(raw_len * e.resampling_factor)
                n = max(0, (eff_len - win_sz) // stp_sz + 1)
                counts.append(n)
            self._cumsum = np.cumsum(counts)
            self._counts = np.array(counts)

    def __len__(self) -> int:
        if self.win_sz is None:
            return len(self.entries)
        return int(self._cumsum[-1]) if len(self._cumsum) > 0 else 0

    def __getitem__(self, idx: int) -> tuple:
        if self.win_sz is None:
            entry = self.entries[idx]
            eff_len = int(self._ref_block.file_len(entry.path) * entry.resampling_factor)
            l_slc, r_slc = 0, eff_len
        else:
            entry_idx = int(np.searchsorted(self._cumsum, idx, side="right"))
            entry = self.entries[entry_idx]
            offset = idx - (int(self._cumsum[entry_idx - 1]) if entry_idx > 0 else 0)
            l_slc = offset * self.stp_sz
            r_slc = l_slc + self.win_sz

        inp = self._read_readers(self._inputs, entry, l_slc, r_slc)
        tgt = self._read_readers(self._targets, entry, l_slc, r_slc)

        if self._single_input:
            inp = inp[0]
        if self._single_target:
            tgt = tgt[0]
        return inp, tgt

    def _read_readers(self, readers: tuple, entry: FileEntry, l_slc: int, r_slc: int) -> tuple[torch.Tensor, ...]:
        results = []
        for block in readers:
            if isinstance(block, Resampled):
                arr = block.read(entry.path, l_slc, r_slc, entry.resampling_factor)
            elif hasattr(block, "file_len"):
                arr = block.read(entry.path, l_slc, r_slc)
            else:
                arr = block.read(entry.path)
            results.append(torch.from_numpy(arr.astype(np.float32)))
        return tuple(results)

    @staticmethod
    def _find_temporal(*readers):
        """Find first temporal reader (has file_len method)."""
        for b in readers:
            if isinstance(b, Resampled):
                return b
            if hasattr(b, "file_len"):
                return b
        raise ValueError("At least one temporal reader required")
