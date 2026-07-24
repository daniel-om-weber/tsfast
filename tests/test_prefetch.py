"""Tests for PrefetchLoader producer-thread lifecycle."""

import gc
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsfast.tsdata.prefetch import PrefetchLoader, _cleanup_iterators, _producers


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


def test_producer_is_registered_and_joined_by_cleanup():
    """The atexit cleanup tracks running producers and waits for them."""
    it = iter(make_loader(n=4096, bs=4))
    next(it)
    thread = it._thread
    assert thread in _producers

    _cleanup_iterators()
    assert not thread.is_alive()
    assert thread not in _producers


def test_producer_unregisters_after_full_iteration():
    it = iter(make_loader(n=16, bs=4))
    thread = it._thread
    for _ in it:
        pass
    assert wait_for(lambda: thread not in _producers)


@pytest.mark.slow
def test_interpreter_exits_cleanly_with_abandoned_producers(wh_path):
    """Producers orphaned by one_batch() must not deadlock or crash shutdown.

    Each one_batch() abandons a prefetch iterator whose producer may still be inside a
    native DataLoader call. Daemon threads are never joined by the interpreter, so
    without the atexit join this exits with SIGSEGV or hangs forever.
    """
    code = textwrap.dedent(f"""
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=r"{wh_path}", win_sz=100, stp_sz=100,
                         num_workers=0, n_batches_train=2)
        for _ in range(6):
            dls.one_batch()
    """)
    for _ in range(3):
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr[-1500:]}"


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
