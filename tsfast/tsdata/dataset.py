"""Pure PyTorch Dataset for windowed time series from source files."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .readers import SourceEntry


class WindowedDataset(Dataset):
    """Pure PyTorch Dataset for windowed time series from source files.

    Args:
        sources: list of SourceEntry (path + resampling factor); one file at
            several rates is several entries, one per factor
        inputs: single reader or tuple of readers for input signals
        targets: single reader or tuple of readers for target signals
        win_sz: window size in (resampled) samples, None = full-file mode
        stp_sz: step size between windows
    """

    def __init__(
        self,
        sources: list[SourceEntry],
        inputs,
        targets,
        win_sz: int | None = None,
        stp_sz: int = 1,
    ):
        self.sources = sources
        self._inputs = (inputs,) if not isinstance(inputs, tuple) else inputs
        self._targets = (targets,) if not isinstance(targets, tuple) else targets
        self._single_input = not isinstance(inputs, tuple)
        self._single_target = not isinstance(targets, tuple)
        self.win_sz = win_sz
        self.stp_sz = stp_sz
        self._ref_block = self._find_temporal(*self._inputs, *self._targets)

        if sources:
            for block in (*self._inputs, *self._targets):
                if hasattr(block, "probe"):
                    block.probe(sources[0])

        if win_sz is not None:
            ref_block = self._ref_block
            counts = []
            for e in sources:
                eff_len = ref_block.file_len(e)  # already in resampled coords
                n = max(0, (eff_len - win_sz) // stp_sz + 1)
                counts.append(n)
            self._cumsum = np.cumsum(counts)
            self._counts = np.array(counts)

    def __len__(self) -> int:
        if self.win_sz is None:
            return len(self.sources)
        return int(self._cumsum[-1]) if len(self._cumsum) > 0 else 0

    def __getitem__(self, idx: int) -> tuple:
        if self.win_sz is None:
            entry = self.sources[idx]
            eff_len = self._ref_block.file_len(entry)  # already in resampled coords
            l_slc, r_slc = 0, eff_len
        else:
            entry_idx = int(np.searchsorted(self._cumsum, idx, side="right"))
            entry = self.sources[entry_idx]
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

    def _read_readers(self, readers: tuple, entry: SourceEntry, l_slc: int, r_slc: int) -> tuple[torch.Tensor, ...]:
        results = []
        for block in readers:
            if hasattr(block, "file_len"):  # temporal reader (incl. Resampled views)
                arr = block.read(entry, l_slc, r_slc)
            else:
                arr = block.read(entry)  # scalar reader
            results.append(torch.from_numpy(arr.astype(np.float32)))
        return tuple(results)

    @staticmethod
    def _find_temporal(*readers):
        """Find the first temporal reader (one with a file_len method)."""
        for b in readers:
            if hasattr(b, "file_len"):
                return b
        raise ValueError("At least one temporal reader required")
