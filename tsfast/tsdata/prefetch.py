"""Thread-based batch prefetcher for DataLoader(num_workers=0)."""

import queue
import threading

from torch.utils.data import DataLoader


_DONE = object()


class _PrefetchIterator:
    """Iterator that prefetches batches from a DataLoader in a background daemon thread."""

    def __init__(self, dl_iter, prefetch: int):
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, args=(dl_iter,), daemon=True)
        self._thread.start()

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

    def one_batch(self) -> tuple:
        """Return one batch from this loader."""
        return next(iter(self))

    def __getattr__(self, name):
        return getattr(self._dl, name)

    def __setattr__(self, name, value):
        if name in ("_dl", "_prefetch"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._dl, name, value)

    def __reduce__(self):
        return (PrefetchLoader, (self._dl, self._prefetch))
