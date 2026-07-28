# Choosing trials per GPU: measured knees vs `recommend_trials_per_gpu`

Internal engineering analysis. Not user documentation — this is a baseline of measured packing
behaviour for one real workload, plus what it says about the tier-1 recommender, so a better
way to compute *k* can be designed against numbers rather than against intuition.

Every claim is labelled **measured**, **inferred** or **assumed**. All GPU numbers come from
one host: 2 × RTX 4090 (24 GiB, 450 W), driver 595.84, Compute Mode `Default`, MIG off,
32-core x86-64, torch 2.10.0+cu128, cuDNN 9.10.2, Python 3.14.3, tsfast 0.5.0.

The workload is a GRU attitude estimator (RIANN reproduction): `num_layers=2`, `hidden_size`
swept 10–300, 7-channel input, `bs=64`, 9000-sample windows, TBPTT `sub_seq_len=1800`,
`cuda_graph=True`, `torch.set_num_threads(1)`, one `PrefetchLoader` thread doing on-the-fly
resampling. One "step" is one optimizer step over a 64 × 9000 batch. A training run is 512
epochs × 300 steps.

---

## 1. Problem

`recommend_trials_per_gpu` returns a *k* that misses the measured knee at both ends of this
workload, and the miss at the expensive end is a 3× over-packing (**measured**):

| | measured knee | recommended `k` | `k_mem` | `k_compute` (power) | `k_compute_conservative` (busy) |
|---|---|---|---|---|---|
| `hidden_size=10` | **4** | 7 | 22 | 7 (power 0.14) | **4** (busy 0.25) |
| `hidden_size=200` | **1** | 3 | 6 | 3 (power 0.35) | **1** (busy 0.97) |

Both priors are computed and only the optimistic one reaches `k`, via
`k = min(k_mem, k_compute)`. The conservative prior — the one the docstring warns "can read
k = 1 on exactly the small models that pack best" — is **exactly right at both ends here**, and
its warned-about failure mode does not occur: at hidden 10 it reads 4, not 1.

This note argues the split is not luck. The two priors answer different physical questions, and
which one is admissible depends on a machine setting the recommender never inspects.

---

## 2. Method, and the one anchor that calibrates it

Two independent harnesses were used, and they agree:

- `measure_packing_curve` — spawns *k* workers, warms each past capture, rendezvouses at a
  `Barrier`, counts `training_step` calls in a shared window (30 s here).
- a hand-rolled bench that wraps `training_step` inside a real `fit_flat_cos` and starts each
  worker's clock after its own warmup (40 s window, warmup 15) — i.e. no barrier, staggered
  clocks. It additionally times one `validate()` call, which the wall-clock model needs.

Agreement, no MPS, same card (**measured**):

| | k=1 | k=2 | k=4 |
|---|---|---|---|
| hidden 10, `measure_packing_curve` | 9.303 | 17.800 | 32.144 |
| hidden 10, hand-rolled | 9.296 | 17.824 | 31.907 |
| hidden 200, `measure_packing_curve` | 3.957 | 3.875 | 3.873 |
| hidden 200, hand-rolled | 3.952 | 3.877 | — |

≤ 0.7 % at every level, so neither the barrier nor the staggered clocks matter at these window
lengths, and the numbers are not an artifact of either harness.

**The anchor.** At `hidden_size=200`, 3.952 steps/s is 75.9 s per 300-step epoch. Four
completed 512-epoch training runs of this configuration took **10.8 h**, i.e. ~76 s/epoch
(**measured**, independently, weeks earlier). The k=1 column is therefore calibrated against
real training, not just self-consistent.

Card-to-card spread is ~2 %: the same hidden-200 k=1 measurement gives 3.952 on GPU 0 and
4.023 on GPU 1 (**measured**). Any MPS-vs-baseline comparison must use the same card.

---

## 3. Baseline: no MPS

Aggregate steps/s at *k* co-located processes on one card, and the wall clock a 512-epoch run
implies (**measured**; `h/run` includes one validation pass per epoch):

| hidden | k=1 | valid | h/run | k=2 gain | k=4 gain | knee |
|---|---|---|---|---|---|---|
| 10 | 9.296 | 0.114 s | 4.61 | 1.92× | 3.43× | 4 |
| 20 | 9.392 | 0.106 s | 4.56 | 1.91× | 3.43× | 4 |
| 40 | 9.352 | 0.102 s | 4.58 | 1.91× | 2.89× | 4 |
| 70 | 9.517 | 0.114 s | 4.50 | 1.85× | — | 2 |
| 100 | 9.447 | 0.113 s | 4.53 | 1.88× | — | 2 |
| 200 | 3.952 | 0.288 s | 10.84 | 0.98× | 0.98× | 1 |
| 250 | 3.270 | 0.373 s | 13.10 | 0.97× | — | 1 |
| 300 | 2.934 | 0.384 s | 14.60 | 0.97× | — | 1 |

Two regimes with a sharp boundary between 100 and 200:

- **10 → 100 is size-independent.** 9.3–9.5 steps/s across a tenfold range of hidden units, with
  NVML utilization at 0.25 and each worker holding ~140 % CPU — one thread pegged plus the
  prefetch thread. The limit is per-process launch and Python overhead, not the device
  (**inferred** from the flat throughput, the low utilization and the pegged thread together).
  These sizes pack ~linearly: 1.9× at k=2, 3.4× at k=4.
- **200 and above is device-bound.** Per-worker slowdown is exactly *k*, aggregate is flat to
  within 3 %, and memory is nowhere near binding (3.3 GiB per process plus a 0.6 GiB context, on
  24 GiB).

Also **measured**, at hidden 200, on the same host under tsfast 0.4: with the dataloader removed
entirely (one batch replayed for the whole window) the aggregate is still flat —
3.969 / 3.818 / 3.816 at k = 1 / 2 / 4 — and the `DataProfiler` data-wait fraction is 0.0 % at
every *k*. The flat curve is device saturation, not loader starvation. Those figures sit within
2 % of the 0.5 numbers above, so the two versions agree on this workload.

---

## 4. Mechanism: idle time is sellable, idle width is not

The probe at hidden 200 reports **busy 0.97, power 0.35** (**measured**). Those are consistent,
not contradictory: the GRU occupies the timeline continuously while its kernels are narrow —
`bs=64 × hidden 200`, 1800 timesteps deep, each dependent on the last — so they leave most of
the 4090's SMs idle at any instant.

Without MPS every process gets its own CUDA context and the driver **time-slices** between
contexts; kernels from two processes are never co-resident. So a second process can only fill
idle *time*, never idle *width* (**inferred** — standard CUDA context behaviour — and confirmed
by §5).

That single fact predicts the whole baseline table:

- hidden 10: busy 0.25 → 75 % of the timeline is free → `1/0.25 = 4`, measured knee **4** ✓
- hidden 200: busy 0.97 → nothing free → `1/0.97 = 1`, measured knee **1** ✓

So `k_compute_conservative = round(1/busy_fraction)` is not "the pessimistic estimator". Under
Compute Mode `Default` with no MPS it is **the correct model**. And `k_compute =
round(1/power_fraction)` is an estimator for a machine configuration that is not in effect —
it prices width headroom that no second process can reach.

---

## 5. MPS changes the answer, and only where the mechanism says it should

Same card (GPU 1), same session, same harness, MPS daemon restricted to that GPU
(**measured**):

**hidden 200**

| k | no MPS | with MPS |
|---|---|---|
| 1 | 4.023 | 3.826 (−5 %) |
| 2 | 3.955 | 6.195 |
| 4 | 3.956 | 9.366 |
| 6 | — | **11.53** |
| knee | **1** | **≥ 6** |

11.53 / 3.956 = **2.91×** aggregate throughput on one card. At k=6 the six clients hold
19.8 GiB of 23.5, so memory — not compute — is what bounds *k* here; the curve was still rising
when it ran out (9.37 → 11.53).

**hidden 10**

| k | no MPS | with MPS |
|---|---|---|
| 1 | 9.303 | 9.26 |
| 2 | 17.800 | 18.13 |
| 4 | 32.144 | 31.84 |

**No effect**, within noise. Exactly as predicted: at hidden 10 the constraint is per-process
launch overhead, which sharing a context does not touch, and the idle *time* was already
harvestable without MPS.

Two secondary results, both **measured**: CUDA-graph capture succeeds under MPS (2/2 workers
graphed, 0 failures), and MPS costs ~5 % at k=1 — the price of routing through the server.

Under MPS the power prior becomes the better of the two (3 against a measured knee ≥ 6 — still
low, but on the right side), and the utilization prior becomes badly wrong (1 against ≥ 6). The
priors swap validity with the machine configuration.

---

## 6. What this suggests for the API

Ordered by cost, not by preference:

1. **Gate the prior on the execution mode.** Query `nvmlDeviceGetComputeMode` and detect a
   reachable MPS control daemon; with no MPS use `1/busy_fraction`, with MPS admit
   `1/power_fraction`. This is the only option that is correct in *both* configurations rather
   than tuned to one, and §4–5 is the evidence for it.
2. **Adaptive `find_packing_knee`.** Measure k=1, then double while the aggregate gain stays
   above a threshold (~1.15×), bounded by `k_mem`. All the machinery exists (`_packing_worker`,
   the barrier). It measures the objective directly, needs no NVML, and costs 2–3 windows —
   comparable to the probe it would replace.
3. **Escalate on disagreement.** When the two priors differ by more than ~2×, run a two-point
   curve for that config. On this workload that triggers at hidden 200 (3 vs 1) and not at
   hidden 10 (7 vs 4), i.e. it spends GPU time exactly where the cheap answer is ambiguous.
4. **Better width telemetry** — DCGM `SM_ACTIVE`/occupancy, or a kernel-time fraction from
   `torch.profiler` instead of NVML's coarse `utilization.gpu`. Only worth it under MPS, since
   width is what MPS unlocks.

Two design points that the current API cannot express, and that a redesign should:

- **The error is asymmetric, and the direction depends on the scheduler.** Over-packing a
  saturated device costs ~2 % of aggregate throughput but multiplies per-trial latency by *k*;
  under-packing an idle device costs up to 3.4× of throughput outright. For a fixed grid, the
  first is nearly free and the second is expensive. For ASHA/PBT, per-trial latency is what
  pruning depends on, so the ranking reverses. The recommender does not know which regime the
  caller is in.
- **`PackingCurve.knee` is relative to what was measured** — "smallest *k* within 95 % of the
  best aggregate" reports 6 for a curve that was still climbing at 6 because 6 was the last
  point. Under MPS that is exactly what happened. A knee that ran out of measured range should
  be distinguishable from one that flattened.

---

## 7. Limits of this baseline

- **One workload.** A 2-layer GRU with a 1800-step TBPTT loop is about as launch-bound and as
  narrow as a model gets. A CNN/TCN with wide kernels would saturate SM width at k=1 and should
  show MPS gaining nothing at any size — untested (**assumed**).
- **One host, one GPU model, `Default` compute mode.** No MIG, no multi-tenant, consumer cards.
- **n = 2 for the priors.** Utilization and power were probed at hidden 10 and 200 only. The
  intermediate sizes have measured *knees* but no measured prior to check them against, so
  "busy-fraction is the right prior" rests on the mechanism plus two anchor points, not on a fit.
- **k > 4 measured only at hidden 200 under MPS.** Neither the no-MPS small sizes nor
  hidden 250/300 were pushed past 4.
- **MPS was not run for a full training.** Throughput was measured in 30 s windows; nothing here
  says a 22 h run under MPS completes, and MPS clients share a context, so one client's fatal
  fault can take the server and its siblings with it.
- The hidden-10 knee of 4 is where the measurement stopped; per-worker was still only 13 % down
  at k=4, so the true knee may be higher.

---

## 8. Reproducing

Raw numbers behind every table are in `notes/data/`: `gpu-packing-size-sweep.json` (the no-MPS
per-size sweep, per-worker rates included) and `gpu-packing-mps.json` (the MPS comparison, the
cross-harness check and the probe outputs).

The scripts live outside this repo (in the RIANN reproduction that produced the workload), but
the measurement is three calls into `tsfast.training.profiling`:

```python
from tsfast.training.profiling import (
    measure_packing_curve, probe_gpu_saturation, recommend_trials_per_gpu)

# tier 1 + 2: the cheap recommendation (needs nvidia-ml-py, not a declared dependency)
probe = probe_gpu_saturation(make_learner, [{"hidden_size": 200}])
rec = recommend_trials_per_gpu(probe)      # .k, .k_mem, .k_compute, .k_compute_conservative

# tier 3: ground truth
curve = measure_packing_curve(make_learner, [{"hidden_size": 200}], ks=(1, 2, 4, 6))
```

`make_learner` must be a module-level function (spawn pickles it), and without `nvidia-ml-py`
the probe degrades to memory-only. At hidden 200 that leaves `k_mem` alone deciding: the
footprint then excludes the ~0.5 GiB CUDA context, so `floor(0.9 × 23.5 / 2.7) = 7`, against a
measured knee of 1 (**inferred** — the arithmetic of `recommend_trials_per_gpu` applied to the
probe's own no-NVML path, not run with `nvidia-ml-py` removed).

MPS, restricted to one GPU and one private pipe directory:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
CUDA_VISIBLE_DEVICES=1 nvidia-cuda-mps-control -d
# clients: select the card by UUID, not index -- inside the server's namespace the
# daemon's GPU 1 is index 0, so a client asking for index 1 gets "No CUDA GPUs are available"
CUDA_VISIBLE_DEVICES=GPU-f5f4dcda-... python train.py
echo quit | nvidia-cuda-mps-control
```

**Keep the pipe directory path short.** A UNIX socket path is capped at 108 bytes; a longer
`CUDA_MPS_PIPE_DIRECTORY` makes the control daemon exit rc=1 having created its lock and FIFO,
with nothing in `control.log` and no message on stderr. That failure looks exactly like "MPS
is unsupported on this card", and it is not.

---

## 9. Open questions

- Does the MPS gain hold for a whole 512-epoch run, or does something (allocator pressure,
  context contention, the 22 h per-run latency) erode it? Only 30 s windows were measured.
- Where is the real knee at hidden 200 under MPS? The curve was still rising at k=6 and memory
  bound it there. Per-client memory limits (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`) might buy
  another level.
- Do hidden 250/300 pack under MPS like 200 does? Bigger footprints cap *k* lower; unmeasured.
- Is `1/busy_fraction` predictive at the intermediate sizes (70, 100), where knees are measured
  but the prior is not?
- What does a width-saturating architecture (TCN, wide CNN) do to both priors? If MPS gains
  nothing there, the gating rule in §6.1 needs a width term, not just a mode check.
