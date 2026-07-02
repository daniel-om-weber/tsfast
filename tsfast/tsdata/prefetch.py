"""Thread-based batch prefetcher for DataLoader(num_workers=0)."""

import atexit
import queue
import threading
import weakref

from torch.utils.data import DataLoader

_DONE = object()
_live_iterators: set[weakref.ref] = set()
_lock = threading.Lock()


def _cleanup_iterators():
    """Stop all live prefetch threads at interpreter shutdown."""
    with _lock:
        refs = list(_live_iterators)
        _live_iterators.clear()
    for ref in refs:
        it = ref()
        if it is not None:
            it.close()


atexit.register(_cleanup_iterators)


def _put_until_stop(q: queue.Queue, item, stop: threading.Event) -> bool:
    """Put `item` on `q`, polling `stop` so an abandoned producer can exit."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _produce(dl_iter, q: queue.Queue, stop: threading.Event):
    # Module-level function on purpose: the thread must not hold a reference
    # to the _PrefetchIterator, or the iterator could never be garbage
    # collected and an abandoned iterator would leak its producer thread.
    try:
        for batch in dl_iter:
            if not _put_until_stop(q, batch, stop):
                return
            batch = None
        _put_until_stop(q, _DONE, stop)
    except Exception as exc:
        _put_until_stop(q, exc, stop)
    finally:
        stop.set()
        # Release the DataLoader iterator (and any open file handles) in the
        # thread that owns it rather than at interpreter teardown.
        del dl_iter


class _PrefetchIterator:
    """Iterator that prefetches batches from a DataLoader in a background daemon thread."""

    def __init__(self, dl_iter, prefetch: int):
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=_produce, args=(dl_iter, self._queue, self._stop), daemon=True)
        self._thread.start()
        ref = weakref.ref(self, _remove_ref)
        with _lock:
            _live_iterators.add(ref)
        self._ref = ref

    def close(self, timeout: float = 2.0):
        """Stop the producer thread and wait for it to exit."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        with _lock:
            _live_iterators.discard(self._ref)

    def __next__(self):
        while True:
            try:
                item = self._queue.get(timeout=1.0)
                break
            except queue.Empty:
                if self._thread.is_alive():
                    continue
                if self._stop.is_set():
                    raise StopIteration from None
                raise RuntimeError("prefetch producer thread exited without a result") from None
        if item is _DONE:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __iter__(self):
        return self

    def __del__(self):
        try:
            self._stop.set()
        except Exception:
            pass


def _remove_ref(ref):
    with _lock:
        _live_iterators.discard(ref)


class PrefetchLoader:
    """Proxy around DataLoader that prefetches batches in a background thread.

    Args:
        dl: a DataLoader (typically with num_workers=0)
        prefetch: number of batches to buffer ahead
    """

    def __init__(self, dl: DataLoader, prefetch: int = 2):
        object.__setattr__(self, "_dl", dl)
        object.__setattr__(self, "_prefetch", prefetch)

    def __iter__(self):
        return _PrefetchIterator(iter(self._dl), self._prefetch)

    def __len__(self):
        return len(self._dl)

    def __getattr__(self, name):
        return getattr(self._dl, name)

    def __setattr__(self, name, value):
        if name in ("_dl", "_prefetch"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._dl, name, value)

    def __reduce__(self):
        return (PrefetchLoader, (self._dl, self._prefetch))
