"""Safe DataLoader iteration that prevents worker deadlocks on early exit."""

import queue

from torch.utils.data import DataLoader


def drain_worker_queue(iterator) -> None:
    """Drain a DataLoader iterator's worker queues to unblock workers stuck in put().

    On macOS (spawn start method), discarding a multi-worker DataLoader iterator
    before consuming all batches can deadlock: workers block on data_queue.put()
    because the pipe buffer is full, and never see the shutdown signal. This drains
    the queues so workers can exit cleanly.

    No-op for num_workers=0 or if the expected attributes are missing.
    """
    if not hasattr(iterator, "_workers_done_event"):
        return

    # Signal workers to stop producing new items
    iterator._workers_done_event.set()

    # Drain the result queue to unblock workers stuck in put()
    for attr in ("_worker_result_queue", "_data_queue"):
        q = getattr(iterator, attr, None)
        if q is None:
            continue
        while True:
            try:
                q.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break


class SafeDataLoaderIterator:
    """Wraps a multi-worker DataLoader iterator to drain queues before shutdown."""

    def __init__(self, iterator):
        self._iterator = iterator

    def __next__(self):
        return next(self._iterator)

    def __iter__(self):
        return self

    def __del__(self):
        drain_worker_queue(self._iterator)
        # Let the original iterator's __del__ call _shutdown_workers
        del self._iterator


class SafeDataLoader:
    """Thin proxy around a DataLoader that returns SafeDataLoaderIterator for multi-worker loaders."""

    def __init__(self, dl: DataLoader):
        object.__setattr__(self, "_dl", dl)

    def __iter__(self):
        it = iter(self._dl)
        if getattr(self._dl, "num_workers", 0) > 0 and not getattr(self._dl, "persistent_workers", False):
            return SafeDataLoaderIterator(it)
        return it

    def __len__(self):
        return len(self._dl)

    def __getattr__(self, name):
        return getattr(self._dl, name)

    def __setattr__(self, name, value):
        if name == "_dl":
            object.__setattr__(self, name, value)
        else:
            setattr(self._dl, name, value)

    def __reduce__(self):
        return (SafeDataLoader, (self._dl,))
