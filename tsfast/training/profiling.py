"""Speed measurement tools for data pipelines, models, and training loops, plus GPU trial-packing probes."""

from __future__ import annotations

import copy
import gc
import math
import multiprocessing
import os
import statistics
import threading
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "DataProfiler",
    "benchmark_dataloaders",
    "time_inference",
    "time_training_module",
    "time_training_learner",
    "ConfigProbe",
    "SaturationProbe",
    "PackingRecommendation",
    "PackingCurve",
    "sample_ray_space",
    "sample_optuna_space",
    "probe_gpu_saturation",
    "recommend_trials_per_gpu",
    "measure_packing_curve",
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


# ── GPU trial packing ─────────────────────────────────────────────────────────
#
# HPO trials of small sequence models leave big GPUs mostly idle; co-locating k
# trials per GPU multiplies throughput nearly linearly until kernels queue.
# Finding a safe, worthwhile k is a four-step workflow, one function per step:
#
#   1. measure — probe_gpu_saturation: per-config footprints + saturation facts
#   2. decide  — recommend_trials_per_gpu: pure math on the probe, derives k
#   3. verify  — measure_packing_curve: ground-truth aggregate throughput of
#                k real co-located processes; validates the step-2 prior
#   4. enforce — tsfast.tune.trial_resources / apply_gpu_quota: wire k into
#                Ray Tune and make each trial's memory slice binding
#
# The steps are separate functions because they run at different costs and
# frequencies: measuring is seconds per config and done once per workload,
# deciding is instant and re-run freely under different margins, verifying is
# minutes and needed only on a few anchor configs. The resulting k is frozen
# before the tuning run — deliberately not an adaptive controller, so effective
# compute stays reproducible.


def _nvml_init():
    """Return the initialized pynvml module, or None if NVML is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _nvml_handle(pynvml, device: torch.device):
    """NVML handle for a torch CUDA device.

    NVML enumerates devices in PCI order while torch indices follow
    CUDA_VISIBLE_DEVICES, so the two indexings can differ on multi-GPU hosts;
    matching by UUID is authoritative.
    """
    try:
        uuid = torch.cuda.get_device_properties(device).uuid
        return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
    except Exception:
        return pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)


def _nvml_process_used_bytes(pynvml, handle) -> int | None:
    """This process's device memory as NVML sees it (torch allocator + CUDA context), or None if unreported."""
    try:
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if p.pid == os.getpid() and p.usedGpuMemory:
                return p.usedGpuMemory
    except Exception:
        pass
    return None


class _NvmlSampler(threading.Thread):
    """Samples NVML utilization and power draw on a background thread (~20 Hz)."""

    def __init__(self, pynvml, handle, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self._pynvml, self._handle, self._interval = pynvml, handle, interval_s
        self._stop_evt = threading.Event()
        self._util: list[float] = []
        self._power_mw: list[float] = []

    def run(self):
        # sample before the first wait so even sub-interval windows record once
        while True:
            try:
                self._util.append(self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu / 100.0)
                self._power_mw.append(self._pynvml.nvmlDeviceGetPowerUsage(self._handle))
            except Exception:
                return
            if self._stop_evt.wait(self._interval):
                return

    def stop(self) -> tuple[float | None, float | None]:
        """Stop sampling; return (mean busy fraction, mean power / enforced limit) or Nones if nothing sampled."""
        self._stop_evt.set()
        self.join()
        if not self._util:
            return None, None
        limit_mw = self._pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle)
        return statistics.mean(self._util), statistics.mean(self._power_mw) / limit_mw


@dataclass
class ConfigProbe:
    """Measurements for one sampled config.

    Args:
        config: the sampled hyperparameter dict
        reserved_bytes: peak torch caching-allocator reservation during training steps
        footprint_bytes: ``reserved_bytes`` plus the per-process CUDA context overhead —
            the number a co-located process actually costs the device
        step_time_s: steady-state seconds per training step (busy subset only)
        busy_fraction: mean NVML utilization over the steady-state window (over-reads
            for launch-bound models: it counts "any kernel in the sample window")
        power_fraction: mean power draw / enforced power limit (under-reads; the
            two bracket the true saturation)
    """

    config: dict
    reserved_bytes: int
    footprint_bytes: int
    step_time_s: float | None = None
    busy_fraction: float | None = None
    power_fraction: float | None = None


@dataclass
class SaturationProbe:
    """Result of :func:`probe_gpu_saturation`; input to :func:`recommend_trials_per_gpu`.

    Footprints are conservative upper estimates: configs are probed sequentially in one
    process, so caches warmed by earlier configs (cuDNN workspaces, Triton autotune) can
    inflate later readings, and the max sampled footprint is not a guaranteed supremum
    over the search space. The per-trial quota (:func:`tsfast.tune.apply_gpu_quota`)
    keeps that safe: a config fails deterministically against its own slice instead of
    taking down neighbors.
    """

    per_config: list[ConfigProbe]
    total_mem_bytes: int
    context_overhead_bytes: int
    device: str
    driver_version: str | None = None

    @property
    def max_footprint(self) -> int:
        return max(p.footprint_bytes for p in self.per_config)

    @property
    def median_footprint(self) -> float:
        return statistics.median(p.footprint_bytes for p in self.per_config)

    @property
    def footprint_ratio(self) -> float:
        """max/median footprint — below ~2 a uniform k loses little to config-size spread."""
        return self.max_footprint / self.median_footprint

    @property
    def median_busy_fraction(self) -> float | None:
        vals = [p.busy_fraction for p in self.per_config if p.busy_fraction is not None]
        return statistics.median(vals) if vals else None

    @property
    def median_power_fraction(self) -> float | None:
        vals = [p.power_fraction for p in self.per_config if p.power_fraction is not None]
        return statistics.median(vals) if vals else None


def sample_ray_space(space: dict, n: int = 20, seed: int = 0) -> list[dict]:
    """Draw *n* random configs from a Ray Tune declarative search space.

    Plain values in *space* pass through as constants; nested dicts are sampled
    recursively. ``tune.sample_from`` entries depend on other resolved values and
    cannot be sampled standalone — pass explicit ``configs`` to the probe instead.

    Args:
        space: dict mapping keys to ``ray.tune`` Domain objects or constants
        n: number of configs to draw
        seed: seed for reproducible sampling
    """
    import numpy as np
    from ray.tune.search.sample import Domain, Function

    rng = np.random.RandomState(seed)

    def sample_value(key: str, value):
        match value:
            case Function():
                raise ValueError(
                    f"space[{key!r}] uses tune.sample_from, which depends on other resolved"
                    " config values and cannot be sampled standalone; pass explicit configs= instead"
                )
            case Domain():
                return value.sample(random_state=rng)
            case dict():
                return {k: sample_value(f"{key}.{k}", v) for k, v in value.items()}
            case _:
                return value

    return [{k: sample_value(k, v) for k, v in space.items()} for _ in range(n)]


def sample_optuna_space(space_fn: Callable, n: int = 20, seed: int = 0) -> list[dict]:
    """Draw *n* random configs from an Optuna define-by-run space function.

    Runs *space_fn* against throwaway trials from a ``RandomSampler`` study. If
    *space_fn* returns a dict it is used directly; otherwise the config is read
    from ``trial.params`` (the usual define-by-run style, where the function only
    calls ``trial.suggest_*``).

    Args:
        space_fn: ``(trial) -> config dict | None``
        n: number of configs to draw
        seed: seed for the RandomSampler
    """
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    configs = []
    for _ in range(n):
        trial = study.ask()
        result = space_fn(trial)
        configs.append(dict(result) if isinstance(result, dict) else dict(trial.params))
    return configs


def _batch_stream(lrn):
    """Yield prepared training batches forever, re-iterating the dataloader as needed."""
    while True:
        for batch in lrn.dls.train:
            yield lrn.prepare_batch(batch, training=True)


def _resolve_cuda_device(device) -> torch.device:
    device = torch.device(device if device is not None else "cuda")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def probe_gpu_saturation(
    make_learner: Callable,
    configs: list[dict],
    n_steps_mem: int = 5,
    n_steps_busy: int = 50,
    warmup: int = 10,
    busy_subset: int = 3,
    device=None,
) -> SaturationProbe:
    """Measure memory footprint and GPU saturation of a workload's config distribution (packing step 1: measure).

    Per config: builds the learner, runs a few real training steps
    (``learner.training_step``, so TBPTT chunking and transforms are reflected), and
    records the peak allocator reservation plus the CUDA context overhead measured via
    NVML per-process memory. For *busy_subset* configs nearest the median footprint it
    additionally samples NVML utilization and power over a steady-state window.

    Intended workflow: run this across the whole model x dataset grid, validate a few
    anchor pairs with :func:`measure_packing_curve` (tier 2), freeze the numbers.
    A probe samples the space at probe time — a TPE/BO search may concentrate on larger
    configs later; the per-trial quota keeps that safe, the throughput estimate may drift.
    Without NVML the probe degrades to memory-only (context overhead 0, no busy/power).

    Args:
        make_learner: factory ``(config) -> Learner``
        configs: configs to probe; from :func:`sample_ray_space` /
            :func:`sample_optuna_space`, or hand-picked (include a known worst case)
        n_steps_mem: max training steps per config for the footprint (stops early once
            the peak reservation is stable; optimizer state and workspaces appear
            within the first steps)
        n_steps_busy: training steps in the timed steady-state window
        warmup: untimed steps before the busy window (absorbs autotune and lazy init)
        busy_subset: number of configs to measure busy/power on (0 disables)
        device: target CUDA device (defaults to the current one)
    """
    device = _resolve_cuda_device(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    pynvml = _nvml_init()
    handle = _nvml_handle(pynvml, device) if pynvml else None
    driver = None
    if pynvml:
        v = pynvml.nvmlSystemGetDriverVersion()
        driver = v.decode() if isinstance(v, bytes) else v
    else:
        warnings.warn(
            "pynvml unavailable (install nvidia-ml-py): footprints exclude the CUDA context"
            " (~0.5 GB/process) and the recommendation will be memory-only.",
            stacklevel=2,
        )

    def run_steps(lrn, n: int, stop_when_stable: bool = False) -> None:
        stream = _batch_stream(lrn)
        prev = -1
        for i in range(n):
            xb, yb = next(stream)
            lrn.training_step(xb, yb)
            if stop_when_stable:
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_reserved(device)
                if i >= 2 and peak == prev:
                    break
                prev = peak
        torch.cuda.synchronize(device)

    context_overhead = 0
    probes: list[ConfigProbe] = []
    for config in configs:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        lrn = make_learner(config)
        lrn.device = device
        lrn.setup()
        lrn.model.train()
        try:
            run_steps(lrn, n_steps_mem, stop_when_stable=True)
            reserved = torch.cuda.max_memory_reserved(device)
            if handle is not None:
                used = _nvml_process_used_bytes(pynvml, handle)
                if used is None:
                    # driver reports no per-process memory: fall back to device-level used,
                    # which overstates on GPUs shared with a display server or other jobs
                    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                context_overhead = max(context_overhead, used - torch.cuda.memory_reserved(device))
        finally:
            lrn._teardown_composables()
        del lrn
        probes.append(ConfigProbe(config=config, reserved_bytes=reserved, footprint_bytes=reserved))

    for p in probes:
        p.footprint_bytes = p.reserved_bytes + context_overhead

    if handle is not None and busy_subset > 0:
        med = statistics.median(p.reserved_bytes for p in probes)
        subset = sorted(probes, key=lambda p: abs(p.reserved_bytes - med))[:busy_subset]
        for p in subset:
            gc.collect()
            torch.cuda.empty_cache()
            lrn = make_learner(p.config)
            lrn.device = device
            lrn.setup()
            lrn.model.train()
            try:
                run_steps(lrn, warmup)
                sampler = _NvmlSampler(pynvml, handle)
                sampler.start()
                t0 = time.perf_counter()
                run_steps(lrn, n_steps_busy)
                dt = time.perf_counter() - t0
                p.busy_fraction, p.power_fraction = sampler.stop()
                p.step_time_s = dt / n_steps_busy
            finally:
                lrn._teardown_composables()
            del lrn

    if pynvml:
        pynvml.nvmlShutdown()

    probe = SaturationProbe(
        per_config=probes,
        total_mem_bytes=total_mem,
        context_overhead_bytes=context_overhead,
        device=str(device),
        driver_version=driver,
    )
    gb = 1024**3
    print("=== GPU Saturation Probe ===")
    print(f"Device: {probe.device} ({total_mem / gb:.1f} GB) | {len(probes)} configs")
    print(
        f"Footprint: median {probe.median_footprint / gb:.2f} GB | max {probe.max_footprint / gb:.2f} GB"
        f" | ratio {probe.footprint_ratio:.2f} | context {context_overhead / gb:.2f} GB"
    )
    if probe.median_busy_fraction is not None:
        print(f"Busy fraction: {probe.median_busy_fraction:.2f} | power fraction: {probe.median_power_fraction:.2f}")
    return probe


@dataclass
class PackingRecommendation:
    """How many concurrent trials one GPU supports for the probed workload.

    Args:
        k: recommended trials per GPU — ``min(k_mem, k_compute)`` clipped to ``[1, max_k]``
        k_mem: hard memory ceiling from the footprint quantile
        k_compute: saturation prior from power fraction (optimistic side; None without NVML).
            A prior, not ground truth — it ignores bandwidth/L2 contention; validate anchors
            with :func:`measure_packing_curve`.
        k_compute_conservative: saturation prior from NVML utilization (pessimistic side:
            utilization over-reads for launch-bound models, so this can read k=1 on exactly
            the small models that pack best)
        quota_fraction: per-process allocator fraction for ``k`` co-located trials, with the
            k CUDA contexts already subtracted — pass to
            ``torch.cuda.set_per_process_memory_fraction`` or use
            :func:`tsfast.tune.apply_gpu_quota`
        context_overhead_bytes: per-process CUDA context overhead carried over from the probe
        mem_margin: fraction of device memory budgeted (headroom for fragmentation)
    """

    k: int
    k_mem: int
    k_compute: int | None
    k_compute_conservative: int | None
    quota_fraction: float
    context_overhead_bytes: int
    mem_margin: float


def recommend_trials_per_gpu(
    probe: SaturationProbe,
    mem_margin: float = 0.9,
    max_k: int = 8,
    footprint_quantile: float = 1.0,
) -> PackingRecommendation:
    """Derive a trials-per-GPU recommendation from a :class:`SaturationProbe` (packing step 2: decide).

    Pure math on the probe result — instant and GPU-free, so it can be re-run under
    different margins or budgets without re-measuring.

    Memory is a feasibility constraint (unused VRAM has no performance value), so
    ``k_mem`` is a hard ceiling; the compute priors only cap k below it when the single
    trial already saturates the device. Footprints include the CUDA context, so ``k_mem``
    bounds true device usage of k co-located processes.

    Args:
        probe: measurements from :func:`probe_gpu_saturation`
        mem_margin: fraction of device memory to budget
        max_k: upper clip for the recommendation
        footprint_quantile: size ``k_mem`` to this footprint quantile instead of the max.
            Below 1.0 rarer, larger configs will fail deterministically against their
            quota — this constrains the effective search space. Legitimate when tuning
            your own model; wrong for fairness-sensitive benchmarking.
    """
    footprints = sorted(p.footprint_bytes for p in probe.per_config)
    idx = max(0, math.ceil(footprint_quantile * len(footprints)) - 1)
    sizing_footprint = footprints[idx]
    total = probe.total_mem_bytes

    k_mem = max(1, math.floor(mem_margin * total / sizing_footprint))

    pf, bf = probe.median_power_fraction, probe.median_busy_fraction
    k_compute = max(1, round(1 / pf)) if pf else None
    k_compute_conservative = max(1, round(1 / bf)) if bf else None
    if k_compute is None:
        warnings.warn("probe has no NVML data: recommendation is memory-only", stacklevel=2)

    k = min(k_mem, k_compute) if k_compute is not None else k_mem
    k = max(1, min(k, max_k))
    quota_fraction = (mem_margin * total - k * probe.context_overhead_bytes) / (k * total)

    return PackingRecommendation(
        k=k,
        k_mem=k_mem,
        k_compute=k_compute,
        k_compute_conservative=k_compute_conservative,
        quota_fraction=quota_fraction,
        context_overhead_bytes=probe.context_overhead_bytes,
        mem_margin=mem_margin,
    )


def _packing_worker(worker_idx, make_learner, config, device_index, warmup, measure_seconds, barrier, result_q):
    """Tier-2 worker: warm up, rendezvous at the barrier, then count steps in a shared wall-clock window."""
    try:
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
        lrn = make_learner(config)
        lrn.device = device
        lrn.setup()
        lrn.model.train()
        stream = _batch_stream(lrn)
        for _ in range(warmup):
            xb, yb = next(stream)
            lrn.training_step(xb, yb)
        torch.cuda.synchronize(device)
        barrier.wait(timeout=600)
        t0 = time.monotonic()
        steps = 0
        while time.monotonic() - t0 < measure_seconds:
            xb, yb = next(stream)
            lrn.training_step(xb, yb)
            steps += 1
        torch.cuda.synchronize(device)
        rate = steps / (time.monotonic() - t0)
        # release dataloader/HDF5 handles while the interpreter is fully alive;
        # deallocation during shutdown prints spurious h5py tracebacks
        lrn._teardown_composables()
        del stream, lrn
        gc.collect()
        result_q.put((worker_idx, rate, None))
    except Exception:
        import traceback

        result_q.put((worker_idx, None, traceback.format_exc()))


@dataclass
class PackingCurve:
    """Measured aggregate-throughput packing curve from :func:`measure_packing_curve`.

    Args:
        aggregate: ``{k: summed steps/s across the k workers}``
        per_worker: ``{k: [each worker's steps/s]}``
    """

    aggregate: dict[int, float]
    per_worker: dict[int, list[float]] = field(default_factory=dict)

    @property
    def knee(self) -> int:
        """Smallest measured k achieving at least 95% of the best aggregate throughput."""
        best = max(self.aggregate.values())
        return min(k for k, v in self.aggregate.items() if v >= 0.95 * best)

    def slowdown(self, k: int) -> float:
        """Mean per-worker slowdown at k relative to the k=1 rate (requires k=1 in the curve)."""
        return statistics.mean(self.per_worker[1]) / statistics.mean(self.per_worker[k])


def measure_packing_curve(
    make_learner: Callable,
    configs: list[dict],
    ks: tuple[int, ...] = (1, 2, 4),
    warmup: int = 10,
    measure_seconds: float = 30.0,
    device=None,
) -> PackingCurve:
    """Measure ground-truth aggregate throughput of k co-located training processes (packing step 3: verify).

    For each k spawns k separate worker processes on one device (threads would serialize
    kernel launches on the GIL and CUDA streams share one allocator — both overstate
    capacity relative to real multi-process co-location). Each worker builds its learner,
    warms up its own config (so Triton/cuDNN autotune stays out of the timed window),
    rendezvouses at a barrier, then counts training steps inside a shared wall-clock
    window — staggered process startup therefore cannot inflate the aggregate.

    This is the validator (and no-NVML fallback) for the tier-1 heuristic: run it on a
    few anchor configs and compare :attr:`PackingCurve.knee` against the recommendation.

    Args:
        make_learner: factory ``(config) -> Learner``; must be picklable for the spawn
            context (a module-level function or ``functools.partial`` of one — not a lambda)
        configs: picklable configs, drawn round-robin across workers so the mix
            reflects a real tuning session
        ks: co-location levels to measure
        warmup: untimed steps per worker before the barrier
        measure_seconds: length of the timed window
        device: target CUDA device (defaults to the current one)
    """
    device = _resolve_cuda_device(device)
    ctx = multiprocessing.get_context("spawn")
    aggregate: dict[int, float] = {}
    per_worker: dict[int, list[float]] = {}

    print("=== GPU Packing Curve ===")
    for k in ks:
        barrier = ctx.Barrier(k)
        result_q = ctx.Queue()
        procs = [
            ctx.Process(
                target=_packing_worker,
                args=(
                    i,
                    make_learner,
                    configs[i % len(configs)],
                    device.index,
                    warmup,
                    measure_seconds,
                    barrier,
                    result_q,
                ),
                daemon=True,
            )
            for i in range(k)
        ]
        for p in procs:
            p.start()
        results = [result_q.get(timeout=600 + measure_seconds) for _ in range(k)]
        for p in procs:
            p.join(timeout=60)
        errors = [err for _, _, err in results if err is not None]
        if errors:
            raise RuntimeError(f"{len(errors)} of {k} packing workers failed; first error:\n{errors[0]}")
        rates = [rate for _, rate, _ in sorted(results)]
        per_worker[k] = rates
        aggregate[k] = sum(rates)
        print(f"k={k}: aggregate {aggregate[k]:.2f} steps/s | per-worker mean {statistics.mean(rates):.2f} steps/s")

    curve = PackingCurve(aggregate=aggregate, per_worker=per_worker)
    print(f"Knee: k={curve.knee}")
    return curve
