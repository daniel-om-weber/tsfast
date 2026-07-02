"""Speed measurement tools for data pipelines, models, and training loops."""

from __future__ import annotations

import copy
import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "DataProfiler",
    "benchmark_dataloaders",
    "time_inference",
    "time_training_module",
    "time_training_learner",
]


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
        had_instance_method = "training_step" in learner.__dict__
        original_method = learner.training_step

        learner.dls.train = _TimedIterator(original_dl, self)

        def _timed_training_step(xb, yb):
            t0 = time.perf_counter()
            result = original_method(xb, yb)
            self._record_step_time(time.perf_counter() - t0)
            return result

        learner.training_step = _timed_training_step

        try:
            yield self
        finally:
            learner.dls.train = original_dl
            if had_instance_method:
                learner.training_step = learner.__dict__.get("training_step")
            else:
                learner.__dict__.pop("training_step", None)
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


def _device_available(name: str) -> bool:
    match torch.device(name).type:
        case "cuda":
            return torch.cuda.is_available()
        case "mps":
            return torch.backends.mps.is_available()
        case _:
            return True


def _device_sync(device: torch.device) -> Callable | None:
    match device.type:
        case "cuda":
            return torch.cuda.synchronize
        case "mps":
            return torch.mps.synchronize
        case _:
            return None


def _timed_loop(step: Callable, n_warmup: int, min_seconds: float, sync: Callable | None) -> list[float]:
    """Run *step* untimed *n_warmup* times, then timed until *min_seconds* elapsed."""
    for _ in range(n_warmup):
        step()
    if sync:
        sync()
    times: list[float] = []
    total = 0.0
    while total < min_seconds:
        t0 = time.perf_counter()
        step()
        if sync:
            sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
    return times


def _speed_stats(times: list[float], n_samples: int, n_timesteps: int) -> dict[str, float]:
    total = sum(times)
    return {
        "median_ms": 1000 * statistics.median(times),
        "mean_ms": 1000 * statistics.mean(times),
        "samples_per_s": len(times) * n_samples / total,
        "timesteps_per_s": len(times) * n_samples * n_timesteps / total,
    }


def _print_device_table(title: str, run_device: Callable, devices: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Run *run_device* per available device and print results as an aligned table."""
    results: dict[str, dict[str, float]] = {}
    header = f"{'device':>8} | {'median (ms)':>11} | {'mean (ms)':>9} | {'samples/s':>10}"
    print(f"=== {title} ===")
    print(header)
    print("-" * len(header))

    for name in devices:
        if not _device_available(name):
            print(f"{name:>8} | (not available, skipped)")
            continue
        stats = run_device(torch.device(name))
        results[name] = stats
        print(f"{name:>8} | {stats['median_ms']:>11.2f} | {stats['mean_ms']:>9.2f} | {stats['samples_per_s']:>10.1f}")

    return results


def time_inference(
    model: nn.Module,
    xb: Tensor,
    devices: tuple[str, ...] = ("cpu", "cuda"),
    n_warmup: int = 3,
    min_seconds: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Measure forward-pass speed of a model on one batch, per device.

    The model is deep-copied per device, so the caller's model is untouched.
    Unavailable devices are skipped.

    Args:
        model: model to time
        xb: sample input batch ``[batch, seq_len, features]``
        devices: device names to measure on
        n_warmup: untimed iterations before measurement (absorbs cuDNN autotune, lazy init)
        min_seconds: measurement time budget per device

    Returns:
        ``{device: {median_ms, mean_ms, samples_per_s, timesteps_per_s}}``
    """

    def run(device: torch.device) -> dict[str, float]:
        m = copy.deepcopy(model).to(device).eval()
        x = xb.to(device)
        with torch.inference_mode():
            times = _timed_loop(lambda: m(x), n_warmup, min_seconds, _device_sync(device))
        return _speed_stats(times, xb.shape[0], xb.shape[1])

    return _print_device_table("Inference Timing", run, devices)


def time_training_module(
    model: nn.Module,
    xb: Tensor,
    yb: Tensor,
    loss_fn: Callable = F.mse_loss,
    devices: tuple[str, ...] = ("cpu", "cuda"),
    n_warmup: int = 3,
    min_seconds: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Measure raw training-step speed (forward + loss + backward + SGD step), per device.

    Times pure model compute on a fixed batch — no dataloader, transforms, or
    TBPTT chunking. For real training-loop speed use :func:`time_training_learner`;
    the gap between the two is pipeline overhead. SGD is used regardless of how
    the model will actually be trained, so optimizer state does not affect timing.
    The model is deep-copied per device, so the caller's model is untouched.

    Args:
        model: model to time
        xb: sample input batch ``[batch, seq_len, features]``
        yb: sample target batch
        loss_fn: loss ``(pred, target) -> scalar``
        devices: device names to measure on
        n_warmup: untimed iterations before measurement (absorbs cuDNN autotune, lazy init)
        min_seconds: measurement time budget per device

    Returns:
        ``{device: {median_ms, mean_ms, samples_per_s, timesteps_per_s}}``
    """

    def run(device: torch.device) -> dict[str, float]:
        m = copy.deepcopy(model).to(device).train()
        x, y = xb.to(device), yb.to(device)
        opt = torch.optim.SGD(m.parameters(), lr=1e-4)

        def step():
            result = m(x)
            pred = result[0] if isinstance(result, tuple) else result
            loss_fn(pred, y).backward()
            opt.step()
            opt.zero_grad()

        times = _timed_loop(step, n_warmup, min_seconds, _device_sync(device))
        return _speed_stats(times, xb.shape[0], xb.shape[1])

    return _print_device_table("Training Step Timing", run, devices)


def time_training_learner(learner, n_batches: int = 20, n_warmup: int = 3) -> dict[str, float]:
    """Measure real training-loop speed and extrapolate epoch time.

    Runs the learner's actual training loop — dataloader, transforms,
    augmentations, TBPTT chunking — for a few batches on the learner's own
    device. Model weights and optimizer state are restored afterwards, so the
    learner is left untrained. The extrapolation slightly underestimates true
    epoch time since validation and per-epoch overheads are not measured.

    Args:
        learner: Learner to measure
        n_batches: number of timed batches (dataloader is re-iterated if shorter)
        n_warmup: untimed batches before measurement

    Returns:
        ``{batch_ms_median, batch_ms_mean, samples_per_s, sec_per_epoch}``
    """
    model_state = copy.deepcopy(learner.model.state_dict())
    prev_opt, prev_sched = learner.opt, learner.sched

    learner.setup()
    learner.model.train()
    sync = _device_sync(learner.device)

    times: list[float] = []
    n_samples = 0
    it = iter(learner.dls.train)
    try:
        for i in range(n_warmup + n_batches):
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(learner.dls.train)
                batch = next(it)
            xb, yb = learner.prepare_batch(batch, training=True)
            learner.training_step(xb, yb)
            if sync:
                sync()
            if i >= n_warmup:
                times.append(time.perf_counter() - t0)
                n_samples += xb.shape[0]
    finally:
        learner._teardown_composables()
        learner.model.load_state_dict(model_state)
        learner.opt, learner.sched = prev_opt, prev_sched

    median_s = statistics.median(times)
    results = {
        "batch_ms_median": 1000 * median_s,
        "batch_ms_mean": 1000 * statistics.mean(times),
        "samples_per_s": n_samples / sum(times),
        "sec_per_epoch": median_s * len(learner.dls.train),
    }

    print("=== Training Loop Timing ===")
    print(f"Device: {learner.device} | {len(times)} batches measured")
    print(f"Batch: median {results['batch_ms_median']:.2f}ms | mean {results['batch_ms_mean']:.2f}ms")
    print(f"Throughput: {results['samples_per_s']:.1f} samples/s")
    print(f"Extrapolated epoch: {results['sec_per_epoch']:.2f}s ({len(learner.dls.train)} batches/epoch)")
    return results
