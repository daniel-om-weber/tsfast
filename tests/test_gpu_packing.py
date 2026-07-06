"""Tests for GPU trial-packing probes — sampling adapters, recommendation math, packing curve."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tsfast.training.profiling import (
    ConfigProbe,
    PackingCurve,
    SaturationProbe,
    probe_gpu_saturation,
    recommend_trials_per_gpu,
    sample_ray_space,
    measure_packing_curve,
)

GB = 1024**3


def _synthetic_probe(
    footprints_gb,
    total_gb=24.0,
    context_gb=0.5,
    busy=None,
    power=None,
) -> SaturationProbe:
    per_config = [
        ConfigProbe(
            config={"i": i},
            reserved_bytes=int((f - context_gb) * GB),
            footprint_bytes=int(f * GB),
            busy_fraction=busy,
            power_fraction=power,
        )
        for i, f in enumerate(footprints_gb)
    ]
    return SaturationProbe(
        per_config=per_config,
        total_mem_bytes=int(total_gb * GB),
        context_overhead_bytes=int(context_gb * GB),
        device="cuda:0",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Recommendation math (pure, no GPU)
# ──────────────────────────────────────────────────────────────────────────────


class TestSaturationProbeProperties:
    def test_footprint_stats(self):
        probe = _synthetic_probe([1.0, 2.0, 4.0])
        assert probe.max_footprint == 4 * GB
        assert probe.median_footprint == 2 * GB
        assert probe.footprint_ratio == pytest.approx(2.0)

    def test_busy_medians_none_without_nvml(self):
        probe = _synthetic_probe([1.0, 2.0])
        assert probe.median_busy_fraction is None
        assert probe.median_power_fraction is None


class TestRecommendTrialsPerGpu:
    def test_memory_only_path_warns_and_uses_k_mem(self):
        probe = _synthetic_probe([1.0, 2.0, 4.0])
        with pytest.warns(UserWarning, match="memory-only"):
            rec = recommend_trials_per_gpu(probe, mem_margin=0.9, max_k=8)
        # k_mem = floor(0.9 * 24 / 4) = 5
        assert rec.k_mem == 5
        assert rec.k_compute is None
        assert rec.k == 5

    def test_max_k_clip(self):
        probe = _synthetic_probe([0.5, 0.5])
        with pytest.warns(UserWarning):
            rec = recommend_trials_per_gpu(probe, max_k=8)
        assert rec.k_mem == 43
        assert rec.k == 8

    def test_compute_bound_from_power(self):
        probe = _synthetic_probe([0.5, 0.5], busy=0.9, power=0.25)
        rec = recommend_trials_per_gpu(probe, max_k=8)
        assert rec.k_compute == 4
        assert rec.k_compute_conservative == 1
        assert rec.k == 4

    def test_footprint_quantile_shrinks_sizing_footprint(self):
        probe = _synthetic_probe([1.0] * 9 + [8.0])
        with pytest.warns(UserWarning):
            rec_full = recommend_trials_per_gpu(probe)
            rec_q = recommend_trials_per_gpu(probe, footprint_quantile=0.9, max_k=100)
        assert rec_full.k_mem == math.floor(0.9 * 24 / 8)  # sized to the 8 GB outlier
        assert rec_q.k_mem == math.floor(0.9 * 24 / 1)  # outlier excluded

    def test_quota_fraction_subtracts_contexts(self):
        probe = _synthetic_probe([2.2 + 0.5] * 3, total_gb=24.0, context_gb=0.5)
        with pytest.warns(UserWarning):
            rec = recommend_trials_per_gpu(probe, mem_margin=0.9, max_k=8)
        assert rec.k == 8
        # (0.9 * 24 - 8 * 0.5) / (8 * 24)
        assert rec.quota_fraction == pytest.approx((0.9 * 24 - 8 * 0.5) / (8 * 24))
        # k allocator caps + k contexts stay within the margin
        per_proc = rec.quota_fraction * probe.total_mem_bytes + probe.context_overhead_bytes
        assert rec.k * per_proc <= 0.9 * probe.total_mem_bytes + 1e-6 * GB

    def test_huge_footprint_still_recommends_one(self):
        probe = _synthetic_probe([30.0], total_gb=24.0)
        with pytest.warns(UserWarning):
            rec = recommend_trials_per_gpu(probe)
        assert rec.k == 1


# ──────────────────────────────────────────────────────────────────────────────
#  Sampling adapters
# ──────────────────────────────────────────────────────────────────────────────


class TestSampleRaySpace:
    def _space(self):
        from ray import tune

        return {
            "lr": tune.loguniform(1e-4, 1e-1),
            "width": tune.choice([8, 16, 32]),
            "constant": "adam",
            "nested": {"depth": tune.randint(1, 4)},
        }

    def test_sampling_shape_and_constants(self):
        configs = sample_ray_space(self._space(), n=5, seed=0)
        assert len(configs) == 5
        for cfg in configs:
            assert 1e-4 <= cfg["lr"] <= 1e-1
            assert cfg["width"] in (8, 16, 32)
            assert cfg["constant"] == "adam"
            assert 1 <= cfg["nested"]["depth"] < 4

    def test_seeded_determinism(self):
        assert sample_ray_space(self._space(), n=5, seed=3) == sample_ray_space(self._space(), n=5, seed=3)
        assert sample_ray_space(self._space(), n=5, seed=0) != sample_ray_space(self._space(), n=5, seed=1)

    def test_sample_from_rejected(self):
        from ray import tune

        space = {"a": tune.uniform(0, 1), "b": tune.sample_from(lambda spec: spec.config.a * 2)}
        with pytest.raises(ValueError, match="sample_from"):
            sample_ray_space(space, n=1)


class TestSampleOptunaSpace:
    def test_define_by_run_sampling(self):
        pytest.importorskip("optuna")
        from tsfast.training.profiling import sample_optuna_space

        def space_fn(trial):
            trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            trial.suggest_categorical("width", [8, 16, 32])

        configs = sample_optuna_space(space_fn, n=4, seed=0)
        assert len(configs) == 4
        assert all(set(c) == {"lr", "width"} for c in configs)
        assert configs == sample_optuna_space(space_fn, n=4, seed=0)


# ──────────────────────────────────────────────────────────────────────────────
#  PackingCurve
# ──────────────────────────────────────────────────────────────────────────────


class TestPackingCurve:
    def test_knee_is_smallest_k_near_best(self):
        curve = PackingCurve(aggregate={1: 10.0, 2: 18.0, 4: 20.0, 8: 20.1})
        assert curve.knee == 4

    def test_slowdown(self):
        curve = PackingCurve(aggregate={1: 10.0, 2: 16.0}, per_worker={1: [10.0], 2: [8.0, 8.0]})
        assert curve.slowdown(2) == pytest.approx(1.25)


# ──────────────────────────────────────────────────────────────────────────────
#  GPU smoke tests
# ──────────────────────────────────────────────────────────────────────────────


class _TinyDls:
    """Minimal DataLoaders-like object; constructed inside make_learner so it never crosses process boundaries."""

    def __init__(self, bs=4, seq_len=50, n=16):
        ds = TensorDataset(torch.randn(n, seq_len, 1), torch.randn(n, seq_len, 1))
        self.train = DataLoader(ds, batch_size=bs, shuffle=True)
        self.valid = DataLoader(ds, batch_size=bs)
        self.test = None


def _make_tiny_learner(config):
    from tsfast.training import Learner

    width = config.get("width", 8)
    model = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, 1))
    return Learner(model, _TinyDls(), torch.nn.functional.mse_loss, show_bar=False)


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
@pytest.mark.slow
class TestProbeGpuSaturation:
    def test_smoke(self):
        configs = [{"width": 8}, {"width": 32}]
        probe = probe_gpu_saturation(
            _make_tiny_learner, configs, n_steps_mem=3, n_steps_busy=10, warmup=3, busy_subset=1
        )
        assert len(probe.per_config) == 2
        assert all(p.reserved_bytes > 0 for p in probe.per_config)
        assert probe.max_footprint >= probe.median_footprint
        assert probe.total_mem_bytes > GB
        measured = [p for p in probe.per_config if p.busy_fraction is not None]
        assert len(measured) == 1
        assert measured[0].step_time_s > 0
        assert 0 <= measured[0].busy_fraction <= 1
        assert 0 < measured[0].power_fraction <= 1

        rec = recommend_trials_per_gpu(probe)
        assert 1 <= rec.k <= 8
        assert rec.quota_fraction > 0

    def test_no_nvml_degrades_to_memory_only(self, monkeypatch):
        import tsfast.training.profiling as profiling

        monkeypatch.setattr(profiling, "_nvml_init", lambda: None)
        with pytest.warns(UserWarning, match="pynvml unavailable"):
            probe = probe_gpu_saturation(_make_tiny_learner, [{"width": 8}], n_steps_mem=2, busy_subset=1)
        assert probe.context_overhead_bytes == 0
        assert probe.median_busy_fraction is None
        with pytest.warns(UserWarning, match="memory-only"):
            rec = recommend_trials_per_gpu(probe)
        assert rec.k_compute is None
        assert rec.k >= 1


@needs_cuda
@pytest.mark.slow
class TestMeasurePackingCurve:
    def test_two_workers_do_not_collapse_throughput(self):
        curve = measure_packing_curve(_make_tiny_learner, [{"width": 8}], ks=(1, 2), warmup=2, measure_seconds=2.0)
        assert set(curve.aggregate) == {1, 2}
        assert len(curve.per_worker[2]) == 2
        assert all(rate > 0 for rate in curve.per_worker[2])
        # co-locating a second tiny trial must not cost aggregate throughput
        assert curve.aggregate[2] >= 0.9 * curve.aggregate[1]
        assert curve.knee in (1, 2)
