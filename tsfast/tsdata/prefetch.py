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
        for ref in list(_live_iterators):
            it = ref()
            if it is not None:
                it._stop.set()
                it._thread.join(timeout=2.0)
        _live_iterators.clear()


atexit.register(_cleanup_iterators)


class _PrefetchIterator:
    """Iterator that prefetches batches from a DataLoader in a background daemon thread."""

    def __init__(self, dl_iter, prefetch: int):
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, args=(dl_iter,), daemon=True)
        self._thread.start()
        ref = weakref.ref(self, _remove_ref)
        with _lock:
            _live_iterators.add(ref)
        self._ref = ref

    def _produce(self, dl_iter):
        try:
            for batch in dl_iter:
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
            self._queue.put(_DONE)
        except Exception as exc:
            if not self._stop.is_set():
                self._queue.put(exc)
        finally:
            self._stop.set()

    def __next__(self):
        item = self._queue.get()
        if item is _DONE:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __iter__(self):
        return self

    def __del__(self):
        self._stop.set()


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
