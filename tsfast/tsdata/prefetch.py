"""Thread-based batch prefetcher for DataLoader(num_workers=0)."""

import atexit
import queue
import threading

from torch.utils.data import DataLoader

_DONE = object()
_JOIN_TIMEOUT = 2.0
_producers: dict[threading.Thread, threading.Event] = {}
_lock = threading.Lock()


def _unregister(thread: threading.Thread) -> None:
    with _lock:
        _producers.pop(thread, None)


def _cleanup_iterators():
    """Stop every producer thread and wait for it at interpreter shutdown.

    Producers are daemon threads, so nothing else joins them. Tracking the threads
    rather than the iterators is what makes this reachable: an abandoned iterator is
    collected as soon as its caller drops it, and its producer may still be inside a
    native DataLoader call (pinning memory, reading HDF5) when finalization starts,
    which deadlocks or crashes the interpreter.
    """
    with _lock:
        producers = list(_producers.items())
        _producers.clear()
    for _, stop in producers:
        stop.set()
    for thread, _ in producers:
        thread.join(timeout=_JOIN_TIMEOUT)


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
        _unregister(threading.current_thread())


class _PrefetchIterator:
    """Iterator that prefetches batches from a DataLoader in a background daemon thread."""

    def __init__(self, dl_iter, prefetch: int):
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=_produce, args=(dl_iter, self._queue, self._stop), daemon=True)
        # Registered before start so a producer is never running unregistered; holds
        # only the thread and its event, never the iterator, so an abandoned iterator
        # stays collectable.
        with _lock:
            _producers[self._thread] = self._stop
        self._thread.start()

    def close(self, timeout: float = _JOIN_TIMEOUT):
        """Stop the producer thread and wait for it to exit."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        _unregister(self._thread)

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
