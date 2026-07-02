"""Tests for PrefetchLoader producer-thread lifecycle."""

import gc
import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsfast.tsdata.prefetch import PrefetchLoader


def wait_for(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def make_loader(n: int = 64, bs: int = 4) -> PrefetchLoader:
    ds = TensorDataset(torch.arange(n, dtype=torch.float32).unsqueeze(1))
    return PrefetchLoader(DataLoader(ds, batch_size=bs))


def test_yields_all_batches_in_order():
    batches = [b[0] for b in make_loader(n=16, bs=4)]
    assert len(batches) == 4
    assert torch.equal(torch.cat(batches).squeeze(1), torch.arange(16, dtype=torch.float32))


def test_full_iteration_stops_thread():
    before = threading.active_count()
    for _ in make_loader():
        pass
    assert wait_for(lambda: threading.active_count() <= before)


def test_abandoned_iterator_stops_thread():
    before = threading.active_count()
    it = iter(make_loader())
    next(it)
    del it
    gc.collect()
    assert wait_for(lambda: threading.active_count() <= before)


def test_close_stops_thread():
    it = iter(make_loader())
    next(it)
    it.close()
    assert not it._thread.is_alive()


def test_next_after_close_ends_iteration():
    it = iter(make_loader())
    next(it)
    it.close()
    with pytest.raises(StopIteration):
        for _ in range(10):
            next(it)


def test_dataset_exception_propagates():
    class Boom(TensorDataset):
        def __getitem__(self, i):
            if i >= 8:
                raise ValueError("boom")
            return super().__getitem__(i)

    ds = Boom(torch.zeros(16, 1))
    dl = PrefetchLoader(DataLoader(ds, batch_size=4))
    with pytest.raises(ValueError, match="boom"):
        for _ in dl:
            pass
