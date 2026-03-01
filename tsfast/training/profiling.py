"""DataLoader profiling tools for diagnosing data pipeline bottlenecks."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager

__all__ = ["DataProfiler", "benchmark_dataloaders"]


class _TimedIterator:
    """Transparent DataLoader wrapper that measures time spent in ``__next__``."""

    def __init__(self, dl, profiler: DataProfiler):
        self._dl = dl
        self._profiler = profiler

    def __iter__(self):
        self._inner = iter(self._dl)
        return self

    def __next__(self):
        t0 = time.perf_counter()
        batch = next(self._inner)
        self._profiler._record_data_time(time.perf_counter() - t0)
        return batch

    def __len__(self) -> int:
        return len(self._dl)

    def __getattr__(self, name):
        return getattr(self._dl, name)


class DataProfiler:
    """Records per-batch data loading and training step timings.

    Args:
        stall_threshold: seconds above which a data wait counts as a stall
    """

    def __init__(self, stall_threshold: float = 0.005):
        self.stall_threshold = stall_threshold
        self.data_times: list[float] = []
        self.step_times: list[float] = []

    def _record_data_time(self, t: float):
        self.data_times.append(t)

    def _record_step_time(self, t: float):
        self.step_times.append(t)

    @property
    def n_stalls(self) -> int:
        """Number of batches where data wait exceeded the stall threshold."""
        return sum(1 for t in self.data_times if t > self.stall_threshold)

    @property
    def stall_pct(self) -> float:
        """Percentage of batches that stalled."""
        if not self.data_times:
            return 0.0
        return 100.0 * self.n_stalls / len(self.data_times)

    @property
    def data_wait_fraction(self) -> float:
        """Fraction of total time spent waiting for data."""
        total_data = sum(self.data_times)
        total_step = sum(self.step_times)
        total = total_data + total_step
        if total == 0:
            return 0.0
        return total_data / total

    def summary(self) -> str:
        """Formatted profiling summary with actionable diagnostics."""
        n = len(self.data_times)
        if n == 0:
            return "=== DataLoader Profile ===\nNo batches recorded."

        d = self.data_times
        s = self.step_times

        def _fmt(times: list[float]) -> str:
            mn = 1000 * statistics.mean(times)
            md = 1000 * statistics.median(times)
            mx = 1000 * max(times)
            return f"mean {mn:6.2f}ms | median {md:6.2f}ms | max {mx:6.2f}ms"

        wait_pct = 100.0 * self.data_wait_fraction
        thresh_ms = self.stall_threshold * 1000

        lines = [
            "=== DataLoader Profile ===",
            f"Batches: {n}",
            "",
            f"Data loading:  {_fmt(d)}",
            f"Training step: {_fmt(s)}" if s else "Training step: (not recorded)",
            "",
            f"Data wait: {wait_pct:.1f}% of total time",
            f"Stalls (>{thresh_ms:.1f}ms): {self.n_stalls} / {n} batches ({self.stall_pct:.1f}%)",
            "",
        ]

        if wait_pct > 50:
            lines.append(
                "Verdict: Data loading IS a bottleneck!"
                " Try: increase num_workers, enable cache=True, or increase prefetch depth."
            )
        elif wait_pct > 20:
            lines.append(
                "Verdict: Data loading is significant. Consider increasing num_workers or enabling cache=True."
            )
        else:
            lines.append("Verdict: No bottleneck detected.")

        return "\n".join(lines)

    @classmethod
    @contextmanager
    def profile(cls, learner, stall_threshold: float = 0.005):
        """Instrument a Learner's training loop to measure data vs step time.

        Args:
            learner: the Learner to instrument
            stall_threshold: seconds above which a data wait counts as a stall
        """
        self = cls(stall_threshold=stall_threshold)
        original_dl = learner.dls.train
        had_instance_method = "_train_one_batch" in learner.__dict__
        original_method = learner._train_one_batch

        learner.dls.train = _TimedIterator(original_dl, self)

        def _timed_train_one_batch(batch, optimizer, step, total_steps):
            t0 = time.perf_counter()
            result = original_method(batch, optimizer, step, total_steps)
            self._record_step_time(time.perf_counter() - t0)
            return result

        learner._train_one_batch = _timed_train_one_batch

        try:
            yield self
        finally:
            learner.dls.train = original_dl
            if had_instance_method:
                learner._train_one_batch = learner.__dict__.get("_train_one_batch")
            else:
                learner.__dict__.pop("_train_one_batch", None)
            print(self.summary())


def benchmark_dataloaders(
    dls_factory: Callable,
    num_workers_list: list[int] | None = None,
    n_batches: int = 100,
    **factory_kwargs,
) -> dict[int, dict[str, float]]:
    """Benchmark data loading speed across different ``num_workers`` values.

    Creates DataLoaders with each ``num_workers`` setting, iterates *n_batches*,
    and reports throughput.

    Args:
        dls_factory: callable that returns a DataLoaders (e.g. ``create_dls``)
        num_workers_list: values to test (default ``[0, 1, 2, 4]``)
        n_batches: batches to iterate per setting
        **factory_kwargs: forwarded to *dls_factory*
    """
    if num_workers_list is None:
        num_workers_list = [0, 1, 2, 4]

    results: dict[int, dict[str, float]] = {}
    header = f"{'num_workers':>11} | {'mean (ms)':>9} | {'median (ms)':>11} | {'throughput (batch/s)':>20}"
    print("=== DataLoader Benchmark ===")
    print(header)
    print("-" * len(header))

    for nw in num_workers_list:
        dls = dls_factory(num_workers=nw, **factory_kwargs)
        times: list[float] = []
        it = iter(dls.train)
        for _ in range(n_batches):
            t0 = time.perf_counter()
            try:
                next(it)
            except StopIteration:
                break
            times.append(time.perf_counter() - t0)

        if not times:
            continue

        mean_ms = 1000 * statistics.mean(times)
        median_ms = 1000 * statistics.median(times)
        throughput = len(times) / sum(times) if sum(times) > 0 else float("inf")

        results[nw] = {"mean_ms": mean_ms, "median_ms": median_ms, "throughput": throughput}
        print(f"{nw:>11} | {mean_ms:>9.2f} | {median_ms:>11.2f} | {throughput:>20.1f}")

    return results
