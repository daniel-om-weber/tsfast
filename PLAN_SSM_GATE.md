# Implementation plan: gated NeuralStateSpace

Delivery plan for a gating mechanism on the state transition of `NeuralStateSpace`.

**Status.** All four phases landed plus a kernel extension, and are committed. Every
gate runs on the fused `triton` and `c` kernels as well as eager/compiled.
`tests/test_statespace.py` is 61 passed / 2 skipped (metal, no MPS on this machine).

Phase 2 is complete. The mechanism criterion passes for `leak` and `gru` and rules `residual`
out; the accuracy criterion ran on all four benchmarks. Gating earns its place — the best
gated variant cuts simulation NRMSE by 43%, 46% and 72% on WH, CascadedTanks and EMPS — but
the accuracy evaluation **cannot separate the three gated variants** on any benchmark (§4).

Metal is deliberately ungated (§3), and its `supports()` declines a gated spec explicitly —
accepting one would run the ungated kernel against a final layer twice as wide as its layout
assumes and return wrong results. It is now the only backend that declines any gate, which is
what the explicit-`gates` argument to `rollout_unsupported` exists to guarantee.

**Why `gru` is the recommended variant.** It is the only one that passes the mechanism
criterion *and* tracks `gate_tmax` closely (`T/tmax ∈ [0.78, 0.94]` against `leak`'s
`[0.59, 0.80]`); `residual` fails that criterion outright, its unit carry giving a gradient
horizon that is not a parameter of the model at all. Accuracy neither contradicts this nor
supports an alternative: the three gated variants land within 0.7–3.4% of each other on every
benchmark, inside the 3–6% that a change of floating-point summation order alone moves the
same variants (§4). A `gru` gate whose weight rows are held at zero reduces to `sigmoid(bias)`,
which is exactly `leak`, so `gru` also subsumes it; `tests/test_statespace.py` pins that
numerically. Speed is no longer a differentiator — all three are fused and cost within 4% of
each other (§4, Phase 3).

## Context

`NeuralStateSpace` steps `x_{k+1} = f(x_k, u_k)` with an MLP `f`, so

```
∂x_{k+1}/∂x_k = W_K D_{K-1} W_{K-1} … D_0 W_0[:, :nx]
```

is a product of `n_linear` dense matrices with no constraint on its spectrum, and BPTT
over a length-`L` rollout multiplies `L` of them. At the default init (`n_state=8`, two
64-wide `tanh` layers, `nn.Linear`'s U(±1/√fan_in)) the product's singular values sit well
below 1, so gradients from late in a long rollout never reach `x_0` — the practical failure
mode is vanishing, not exploding.

Gating is the standard fix: force the Jacobian to `diag(1−z) + diag(z)·∂f/∂x` so there is a
tunable near-identity path per state channel. The literature that establishes this is
settled — Tallec & Ollivier (arXiv:1804.11188) derive gates axiomatically from invariance
to time warping and give the bias initialization; AntisymmetricRNN (arXiv:1902.09689) Eq. 13
is the same construction around a nonlinear ODE discretization; GRU-ODE-Bayes
(arXiv:1905.12374) is its continuous-time limit; Mamba's input-dependent `Δ` and Griffin's
RG-LRU are the linear-recurrence versions, both already in this repo as `mamba` and `lru`.

What is **not** established is whether it pays off on the *physical* state of a black-box
neural SSM. The system-identification line (Forgione & Piga, Beintema's SUBNET, Gedon et
al.'s survey arXiv:2003.14162) attacks the same BPTT problem from the training side
(truncated simulation error, multiple shooting) and the stability side (RENs — ROADMAP A1),
never architecturally. So this is a real open question and Phase 2 is a genuine decision
point, not a formality.

Intended outcome: `NeuralStateSpace(..., gate="gru")` trains on long rollouts where the
ungated model stalls, keeps the physical-state contract that makes `TbpttLearner` chunking
exact, and runs on the fused CUDA/CPU kernels at the same speed class as today.

---

## 1. The one structural decision

**The gate rides on the existing final linear layer; it is not a second head.**

For the input-dependent variants the final linear emits `2·nx` instead of `nx`, split into
candidate `c_k` and gate pre-activation `s_k`. This is the whole reason the change stays
small:

- All gate parameters remain ordinary entries of `self.net`, so they cross the custom-op
  boundary in the existing `params` list and `mlp_param_grads` needed one change — the final
  adjoint width reads `spec.out_width` instead of a hardcoded `n_state` — rather than a new
  gradient path.
- The trunk's per-layer GEMV codegen is unchanged; only the final layer differs.
- The gate sees `(x_k, u_k)` through the shared trunk. When `hidden=()` (linear SSM) the
  final layer *is* the first layer, so the gate reads `[x;u]` directly — a textbook GRU gate.

A separate gate MLP would double the trunk cost, add a second parameter list to thread
through three backends, and buy nothing.

**Consequence:** `SSMSpec` gains the gate as a field. It is frozen and already keys the
compiled-kernel caches (`_EXTENSIONS`, `_KERNELS`), so gated and ungated specs cannot
collide by construction — no cache-invalidation work. `eps` is pinned to `1.0` for every
gate but `residual`, since it is also part of that key and would otherwise let two
numerically identical models compile separate kernels.

**What this bullet list understated:** "only the last layer's width changes" holds for the C
generator, which indexes freely. Triton cannot slice a padded register tensor at an arbitrary
offset, so its final layer became two `n_state`-wide GEMVs split on the host — see §4,
Phase 3. That is the single largest piece of unforeseen work in the plan.

---

## 2. Math contract

`nx = n_state`, `nu = n_input`, `K = n_linear`. `c_k ∈ R^nx` is the candidate (first `nx`
outputs of the final linear), `s_k ∈ R^nx` the gate pre-activation (last `nx`, absent for
`none`/`leak`). Backends cite this section; it is the source of truth for the kernels.

### 2.1 Forward

| `gate` | update | final-layer width |
|---|---|---|
| `none` | `x_{k+1} = c_k` | `nx` |
| `leak` | `x_{k+1} = x_k + a ⊙ (c_k − x_k)`, `a = σ(λ)`, `λ ∈ R^nx` a parameter | `nx` |
| `gru` | `x_{k+1} = x_k + z_k ⊙ (c_k − x_k)`, `z_k = σ(s_k)` | `2·nx` |
| `residual` | `x_{k+1} = x_k + ε·z_k ⊙ c_k`, `z_k = σ(s_k)` | `2·nx` |

`leak` and `gru` use the lerp (convex) form deliberately: `1−z` *is* the per-channel
discrete-time pole, so the gate directly encodes a time constant, matching the `LRU`'s
`exp(-exp(nu_log))` parametrization (`lru.py`). `residual` is the AntisymmetricRNN form — its
carry path is exactly `I`, giving a perfect gradient path but a marginal-stability bias,
which is the wrong prior for dissipative plants. It was in the candidate set to test that
claim, not because it was expected to win; §4 records that it failed the mechanism criterion
for precisely that reason.

### 2.2 Backward

With `g_{k+1} = ∂L/∂x_{k+1}` the *total* step adjoint (`gout[k] + carry`), let `gc`/`gs` be
the adjoints of `c_k`/`s_k` and `carry_direct` the part of `∂L/∂x_k` that bypasses the MLP:

| `gate` | `gc` | `gs` | `carry_direct` |
|---|---|---|---|
| `none` | `g` | — | `0` |
| `leak` | `a ⊙ g` | — | `(1−a) ⊙ g` |
| `gru` | `z ⊙ g` | `z(1−z) ⊙ g ⊙ (c_k − x_k)` | `(1−z) ⊙ g` |
| `residual` | `ε·z ⊙ g` | `ε·z(1−z) ⊙ g ⊙ c_k` | `g` |

The existing reverse chain then runs unchanged from `[gc; gs]` through layers `K−1…1` to
`carry_mlp`, and the recurrence closes with

```
carry = carry_direct + carry_mlp
```

For `leak`, `λ` is outside the MLP, so its gradient is a plain reduction done in torch:

```
∂L/∂λ = a(1−a) ⊙ Σ_{b,k} g_{k+1} ⊙ (c_k − x_k)
```

### 2.3 Extra stored tensors

`saved_widths(spec)` in `ssm/core.py` is the contract. A gated training forward appends to the
existing hidden activations the vector the gate scales, and — for the input-dependent gates
only — the gate pre-activation:

- `d = c_k − x_k` for `leak`/`gru`, the candidate *offset*; `c_k` itself for `residual`, whose
  `gs` row in §2.2 needs the candidate rather than the offset
- `s`, the gate pre-activation (`gru`/`residual`; `leak` has none)

Storing `d` rather than `c_k` is what lets the reverse sweep run without `out` or `x0`: every
term in the §2.2 `gru` row is a function of `d`, `s` and the incoming adjoint alone. It is
also free — the lerp forms `d` anyway. Storing `s` rather than `z = σ(s)` costs one sigmoid
per step in the backward and keeps the forward's stores symmetric.

For the input-dependent gates the backward kernel emits nothing extra: `gy` simply widens from
`nx` to `spec.out_width` and carries `[gc; gs]`, so the gate's gradients reach the optimizer
through the existing batched-GEMM stage.

`leak` is the asymmetric one, as §2.2 implies. It stores one tensor rather than two — its gate
is a parameter, so there is no per-step pre-activation to keep — and its `λ` gradient is a
reduction over *both* batch and time that no other gate needs. The kernels accumulate the time
axis in-register per program and write `[B, nx]` partials alongside `gx0`; the host sums the
batch axis. That keeps the sequential sweep free of atomics. `a = σ(λ)` crosses the op boundary
already squashed, so autograd differentiates the sigmoid outside the op and the kernels only
ever accumulate `∂L/∂a`.

Recomputing `c_k` from `out` instead of storing an offset (`c = x_k + (x_{k+1} − x_k)/z`) is
rejected: `z` is initialized near `1/T` and legitimately reaches `1e-3`.

### 2.4 Chrono initialization

Tallec & Ollivier initialize the *retention* gate bias so `σ(b) = T/(T+1)` for
`T ~ U[1, T_max−1]`. Our `z` is the *update* gate `= 1 − retention`, so the bias is
`−log T`:

- `leak`: `λ_i = −log(T_i)`
- `gru`/`residual`: the gate-bias block of the final linear gets `−log(T_i)`, and the
  gate-*weight* rows are zero-initialized

Zero-initializing the gate weights means `gru` and `leak` start from **identical dynamics**,
which is what makes the Phase-2 comparison measure the input-dependence and not the
initialization. `T_max` is the `gate_tmax` argument, default 100.

---

## 3. Module layout and API

```python
NeuralStateSpace(
    n_input, n_output, n_state=8, hidden_size=64, num_layers=2, act="tanh",
    gate="none",           # "none" | "leak" | "gru" | "residual"
    gate_tmax=100.0,       # chrono-init longest time constant
    eps=1.0,               # residual step size; ignored by other variants
    backend="auto", return_state=False,
)
```

`gate="none"` is the default, so every existing model, checkpoint and benchmark is
bit-identical. `SSMLearner` forwards `gate`/`gate_tmax` through its existing `**kwargs` — no
signature change; the docstring explains why you would reach for it. Every gate runs fused on
`c` and `triton`, so the choice among them is a modelling decision rather than a speed one.

Files as built:

- `ssm/core.py` — `SSMSpec` gains `gate`/`eps` and an `out_width` property; `gate_step` and
  `chrono_bias` helpers; `leak_logit` parameter; `saved_widths` (§2.3); the three custom ops
  gain a single `gate: str` field; `mlp_param_grads` reads the final adjoint width from
  `spec.out_width`; `rollout_unsupported` gains a required `gates` argument
- `ssm/backend_triton.py` — `_pdims`, split final-layer blocks in `_gen_source` and
  `_prep_weights`, `fits`/`_num_warps` off `_pdims`; a `split` flag (final layer doubled)
  distinct from `gated` (state not overwritten), since `leak` is the second without the first;
  declares `_GATES = ("none", "leak", "gru", "residual")`
- `ssm/backend_c.py` — per-gate epilogue in `ssm_fwd`, gate adjoints in `ssm_bwd`, the `λ`
  accumulator; declares `_GATES = ("none", "leak", "gru", "residual")`
- `ssm/backend_metal.py` — one line: declares `("none",)` to its screen
- `tsfast/training/learners.py` — `SSMLearner` docstring
- `benchmarks/gate_ssm_gating.py` (new), `benchmarks/benchmark_ssm.py` (gate row),
  `tests/test_statespace.py`

**Each backend declares the gates its generator emits, and passes them to
`rollout_unsupported` explicitly.** This is not decoration. The first cut screened on a
single shared constant in `core.py`; widening that constant to admit `gru` silently made
metal's `supports()` accept a gated spec it has no code for, which would have run the ungated
kernel against a `2·nx`-wide final layer and returned wrong results on any Mac. There is no
default for the `gates` argument, so a new backend cannot inherit the mistake.

That per-backend screen is now the only one. `core.py` also carried a `_FUSED_GATES` union
that let `_rollout_mode` short-circuit to the fallback for a gate no kernel served, keeping
`"auto"` quiet on the designed route; once every gate had a kernel on CPU and CUDA its
condition could no longer be true, so it and the branch were removed. A gated model on MPS now
reaches `resolve`, and metal's declining warns once per process before falling back — accurate,
since on that device there genuinely is no kernel.

Metal stays out of the kernel work: no MPS device here, and its `_build_jt` sequence-parallel
adjoint scan materializes explicit step Jacobians, which the gate changes. Landing that
unverified would be worse than the fallback.

---

## 4. Phases

### Phase 1 — eager gate, all four variants (S) — **landed**

`core.py` only, `backend="eager"`/`"compiled"`. All four variants behind `gate=`, chrono
init, `leak_logit` parameter. Gated specs were screened off every fused backend so
`backend="auto"` fell back — nothing broke before the kernels existed.

One thing the plan did not anticipate: routing that screen through `supports()` made
`resolve()` warn on a *supported* configuration, since it warns for every declining
candidate. `_rollout_mode` grew a short-circuit to the fallback for a gate no kernel served,
while an explicitly named fused family still warned — that is a request the library genuinely
cannot honour. The short-circuit was removed once every gate had a kernel (§3); the policy it
enforced is still pinned by `test_auto_fallback_is_not_warned`.

### Phase 2 — gate: does gating earn its place (S, wall-clock M) — **decision point, complete**

`benchmarks/gate_ssm_gating.py`, following `benchmarks/gate_ren.py`. Two criteria, both
required to proceed to kernels.

**Mechanism.** `‖∂L/∂x_0‖` with the loss restricted to the last 10 steps, so the adjoint has
to traverse the whole rollout. Measured in float64 at `L = 10…2000`, per variant.

Piloted before writing this criterion, and the result reframes it. At the default
(`n_state=8`, `hidden=64×2`, `tanh`) the ungated model decays at **~0.45 per step**: `1e-11`
at `L=20`, `5e-34` at `L=50`, `2e-70` at `L=100`, and underflowed to exactly zero beyond.
Not an fp32 artifact — float64 shows the same curve. The ungated `NeuralStateSpace` cannot
carry a gradient past roughly 20 steps, and only trains at all because sysid targets are
dominated by short-horizon response and `n_skip`/TBPTT hide the rest.

So a flat gradient-vs-horizon ratio is the wrong bar: only `residual`'s exactly-unit carry
achieves it, and an undecaying adjoint is the wrong prior for a dissipative plant anyway.
The criterion is instead that the gate turns the horizon into a **dial**: fit the per-step
decay rate over `L ∈ [200, 1000]` and check that the implied time constant tracks
`gate_tmax`. Piloted for `gru`: `tmax=10 → T=8.9`, `30 → 27.6`, `100 → 93.5`, `300 → 233.6`,
`1000 → 459.8` (rolling off at the top, where the chrono band saturates at
`(tmax-1)/tmax` and the MLP's own contraction dominates). `leak` tracks slightly lower.
A variant passes if `T` is monotone in `gate_tmax` and within 2× of it up to `tmax=300`.

Measured (`benchmarks/gate_ssm_gating.py --skip-accuracy`):

| variant | verdict | fitted `T` at `tmax` = 10 / 30 / 100 / 300 / 1000 |
|---|---|---|
| `none` | fail | dead at every `tmax` — no finite horizon to fit |
| `leak` | **pass** | 7.9 / 24.0 / 74.7 / 176.6 / 402.3 — `T/tmax ∈ [0.59, 0.80]` |
| `gru` | **pass** | 8.9 / 27.6 / 93.5 / 233.6 / 459.8 — `T/tmax ∈ [0.78, 0.94]` |
| `residual` | fail | `inf` at every `tmax` — the unit carry never decays |

`residual` fails the criterion it was added to test, and for the predicted reason: a carry
path of exactly `I` means the gradient horizon is not a parameter of the model at all. That is
why it is not the recommended gate despite scoring marginally best on two benchmarks — a model
whose adjoint never decays cannot express that a dissipative plant forgets its initial
condition, and the accuracy differences that flatter it are below the suite's resolution (§4).
It is fused and fully supported.

**Accuracy.** Simulation NRMSE at matched parameter count on WH, Silverbox, CascadedTanks and
EMPS, through the `create_dls_*` factories in `tsfast.tsdata` — the same route `gate_ren.py`
takes, which keeps the comparison under this script's control rather than the benchmark
runner's. Two baselines: ungated `NeuralStateSpace` and `RNNLearner` (a GRU). Same
honest-opponent argument as `gate_ren.py` — the GRU already has this gate, so a gated SSM
that does not at least close half the ungated→GRU gap has not bought anything the library did
not already give, and the physical-state contract is the only remaining justification.

Every variant is width-matched to the ungated model's parameter count, since the
input-dependent gates double the final layer and would otherwise get a free capacity bump.
Learning rate is swept per variant, for the reason `gate_ren.py` documents: one shared rate
measures the tuning, not the architecture.

CascadedTanks and EMPS are integrating plants, where a per-channel pole near 1 is exactly
what `leak`/`gru` can express and the ungated model must discover — the most informative
column.

Report per variant: gradient-ratio table, NRMSE table, parameter counts, wall-clock.
**Decision:** the best-scoring variant proceeds to Phase 3. If none passes both criteria,
the eager implementation still ships (it costs nothing when `gate="none"`) and the plan
stops here with the negative result written up.

Measured, 30 epochs, lr swept per variant, matched parameter count ~5329. WH ran first, with
`leak`/`residual` on the `compiled` fallback; the other three ran after the kernels landed,
with every variant fused (simulation NRMSE, lower is better):

| benchmark | `none` | `leak` | `gru` | `residual` | GRU | best gated vs `none` |
|---|---|---|---|---|---|---|
| WH | 0.0686 | 0.0396 | 0.0393 | 0.0394 | 0.0361 | −43% |
| Silverbox | 0.1539 | 0.1536 | 0.1525 | 0.1536 | 0.1527 | −0.9% |
| CascadedTanks | 0.9087 | 0.5070 | 0.5095 | 0.4946 | 0.5735 | −46% |
| EMPS | 0.8260 | 0.2406 | 0.2349 | 0.2328 | 0.2063 | −72% |

**Gating earns its place, and the size of the effect tracks how much memory the plant has.**
EMPS (−72%) and CascadedTanks (−46%) are the integrating plants, where a per-channel pole
near 1 is exactly what the gate expresses and the ungated model must discover; WH (−43%) is a
Wiener–Hammerstein cascade with a static nonlinearity between two short filters; Silverbox
(−0.9%) is a Duffing oscillator whose response is dominated by an instantaneous nonlinearity,
and there the gate has nothing to bite on. Nothing regresses — the worst case is a wash.

On CascadedTanks all three gated variants **beat** the GRU outright (1.19–1.24× of the
ungated→GRU gap) while keeping the physical-state contract the GRU does not have. Median
fraction of that gap closed across the three fused benchmarks: `leak` 0.94, `gru` 1.19,
`residual` 0.96. The Silverbox entries in that statistic are noise — its ungated→GRU gap is
0.0012 NRMSE, so the ratio amplifies differences in the fourth decimal and should be read as
"no gap to close", not as a variant ranking.

**The evaluation cannot separate the three gated variants, on any benchmark.** Their spread is
0.7% (Silverbox), 0.8% (WH), 3.0% (CascadedTanks) and 3.4% (EMPS). Re-running CascadedTanks
with `leak`/`residual` moved from the compiled fallback onto their new kernels — the same seed,
the same math to 5e-7 per step, only a different floating-point summation order — shifted
`leak` 0.4913→0.5070 and `residual` 0.4668→0.4946, i.e. 3–6%, and flipped the learning rate
the sweep selected for both. `none` and `gru`, whose execution path did not change, reproduced
bit-for-bit, which is what establishes that the harness itself is deterministic and that the
shift is numerical rather than noise in the measurement. Differences of a few percent between
gated variants are therefore below this benchmark suite's resolution at one seed, and the
choice among them rests on the mechanism criterion (§4, Phase 2) rather than on accuracy.

### Phase 3 — fused triton + C kernels (M) — **landed, then extended to every gate**

Both generators follow §2. The forward stores `d = c - x` and `s`; the backward recomputes
`z = σ(s)`, derives `gc`/`gs` from the §2.2 table, writes `[gc; gs]` into the now
`2·nx`-wide `gy` buffer, runs the existing reverse chain, and closes with
`carry = carry_direct + carry_mlp`. `mlp_param_grads` needed exactly one change — the final
adjoint width comes from `spec.out_width` rather than `n_state` — because folding the gate
into the final linear keeps its parameters inside the existing batched-GEMM stage.

The one non-obvious piece is Triton. It cannot slice a padded register tensor at an
arbitrary offset, so the gated final layer is emitted as **two separate `n_state`-wide
GEMVs** (candidate, gate) rather than one `2·n_state`-wide one, split on the host in
`_prep_weights`. That keeps every on-chip vector at `px` and makes `fits()` reason about
`_pdims` — `(px, *ph, px)` — rather than the logical `dims`. When `n_linear == 1` the final
layer is also layer 0, so it splits by input *and* output into four weight blocks.

Verification: for four ungated spec shapes the generated source of both backends is
**byte-identical to `HEAD`**, which is the strongest available guarantee that the default
path is untouched. Gated parity against eager is ≤6e-7 relative on outputs and all
gradients, across multi-layer, linear (`n_linear == 1`), non-power-of-two `n_state`, and all
three activations.

**Extension: `leak` and `residual`.** The plan originally fused `gru` alone, on the mechanism
criterion and the subsumption argument, with the accuracy evidence one benchmark deep. Once the
full evaluation showed the variants indistinguishable on accuracy (Phase 2), leaving two of
them 12× slower meant the library's recommendation would be driven by which one happened to
have a kernel. Both are now fused.

`residual` turned out to be a structural twin of `gru` — same `2·nx` final layer, same two
stored tensors, same `gy` layout — so each generator's gate epilogue became gate-specific
rather than a boolean: store `c` instead of `d`, carry the `ε` factor into `gc`/`gs`, and take
`carry_direct = g` instead of `(1−z)·g`. The only plumbing was `ε` itself, which `_spec_from`
had deliberately dropped ("no fused gate reads it"); it now crosses the op boundary and is
baked into the generated source, and since it was already an `SSMSpec` field it already keys
the kernel caches — pinned by a test asserting two `ε` values generate different source.

`leak` needed real plumbing, and not for the reason §2.3 anticipated. `leak_logit` is a
standalone `nn.Parameter`, but `_params_flat()` gathers only the linears, so `λ` never crossed
the op boundary at all. It is now an optional tensor input threaded through all three custom
ops, both `register_fake`s, the autograd wiring and `fused_rollout`, plus the `[B, nx]`
accumulator of §2.3. In Triton it also broke an assumption: the generator used one `gated`
flag for both "the state is not overwritten" and "the final layer is doubled", and `leak` is
the first gate that is the former without the latter — so those became `gated` and `split`,
with the two-block weight loading, reverse chain and carry all keyed off `split`.

Cost of the gate once fused, `benchmarks/benchmark_ssm.py` on an RTX 4090 confirmed idle
(`nvidia-smi` clean, no other compute process). `n_state=10`, `hidden=64×2`, `L=300`,
microseconds per trajectory per training step on the `triton` backend:

| gate | B=16 | B=64 | B=256 | vs `none` |
|---|---|---|---|---|
| `none` | 58.58 | 16.29 | 4.96 | — |
| `leak` | 61.96 | 16.22 | 5.03 | 1.06 / 1.00 / 1.01 |
| `gru` | 63.48 | 16.30 | 5.57 | 1.08 / 1.00 / 1.12 |
| `residual` | 63.43 | 16.40 | 5.59 | 1.08 / 1.01 / 1.13 |

**The gate is close to free once fused** — at worst 13%, and indistinguishable from the ungated
rollout at B=64. This supersedes the earlier estimate of 1.03–1.51×, which was taken on a
contended GPU and overstated the cost. The ordering matches the design: `leak` is consistently
the cheapest because its final layer stays `nx` wide, so it carries one fewer GEMV and one
fewer store per step than the input-dependent gates, which are identical to each other to
within noise as their identical structure predicts.

Fusion against the fallback is best measured on real training rather than a microbenchmark. On
CascadedTanks, 30 epochs with the learning rate swept, moving `leak` and `residual` off
`compiled` onto their kernels took them from 406.2 s and 401.5 s to 34.8 s and 35.4 s — **11.7×
and 11.3×** — landing both within 4% of `none` (33.2 s) and `gru` (34.4 s). The `8×` in the
Phase 2 table conflated two effects (gated-on-fallback vs ungated-on-fused) and should not be
read as the gate's cost. The same eager column of the benchmark shows what the gate costs
*without* a kernel: 1.48× at B=16, against 1.08× fused.

### Phase 4 — integration (S) — **landed**

`SSMLearner`'s docstring gains a paragraph on why the gate exists (the ungated adjoint dies
within tens of steps, and no `sub_seq_len` recovers a longer dependency). `benchmark_ssm.py`
grows a `gate=gru` block mirroring the ungated one, over the same backends, so the
gated/ungated delta stays on the record and the fallback penalty is visible where a backend
lacks a gated kernel. The `NeuralStateSpace` class docstring carries the §2.1 update table
and states which backend serves which gate.

---

## 5. Tests

Added to `tests/test_statespace.py`, reusing `_run`/`_rel`/`_assert_backend_parity`.

**Phase 1**
- shapes for all four variants × `hidden` of `64`, `[]`, `[8,16]`, asserting the final layer
  widens only for the input-dependent gates
- `gate="none"` is bit-identical to the ungated model at fixed seed (guards the default path)
- chrono init: sampled retention `1−z` spans `[1/2, (T_max−1)/T_max]`, with the
  long-time-constant tail populated
- one step against the §2.1 update equations in float64, independent of the rollout loop
- `leak` and `gru` produce identical trajectories once the candidate block is matched — the
  §2.4 claim that zeroed gate weights make them start from the same dynamics
- gradcheck in float64 for every variant — the §2.2 table is the thing most likely to be
  wrong, and eager is the reference the kernels are checked against
- exact chunked equivalence per variant: the gate must not break the property that
  distinguishes this from a GRU
- `SSMLearner(gate=...)` fits under `sub_seq_len` (state carrying is where a gate would
  plausibly break training rather than the forward pass)
- unknown `gate` raises `ValueError`, matching the `act` check

**Phase 3**
- `_assert_backend_parity` gains `gate`/`n_state`/`eps`, exercised for every gate on
  `("c", "cpu")` and `("triton", "cuda")` at `hidden=()`, `(16,)`, `(24,)`, `(48, 32)`,
  `(64, 64)`, `(128,)` across all three activations
- `ε` keys the kernel cache and reaches the kernel across the op boundary: two `ε` values
  generate different source, no other gate leaks an `EPS` constant into it, and `residual`
  holds parity at `ε ∈ {0.25, 2.0}`
- non-power-of-two `n_state` (7) and the widest spec `fits()` admits (`n_state=128`,
  `hidden=(128,)`) — the gate holds on-chip vectors the ungated envelope was sized without
- `torch.compile(fullgraph=True)` parity, mirroring `test_compile_fullgraph_*_parity`
- a `gru` gate with zeroed weight rows reproduces `leak` through the fused kernel, pinning
  the subsumption claim that justified fusing `gru` rather than `leak`
- the auto-fallback warning policy: `"auto"` stays silent for every gate, while an explicitly
  named family that cannot serve one still warns — exercised through `metal`, the only backend
  left that declines a gate, and reachable without an MPS device because its gate screen runs
  before its device screen
- metal declines a gated spec through `supports()` — checkable without MPS, since the gate
  screen runs before the device screen. This is the regression test for the bug in §3.

---

## 6. Risks — outcomes

- **The premise may not hold at the depths that matter.** *Retired.* It holds, and harder
  than assumed: the ungated adjoint dies within ~20 steps at the library default (§4).
- **Triton on-chip budget.** *Retired.* Splitting the final layer into two `n_state`-wide
  GEMVs keeps every on-chip vector at `px`, so the envelope is unchanged from ungated and
  `fits()` reasons about `_pdims`. Parity is pinned at the widest spec it admits
  (`n_state=128`, `hidden=(128,)`).
- **`gate="none"` regression.** *Retired.* Both generators emit byte-identical source to
  `HEAD` for four ungated spec shapes.
- **Convex-hull slew-rate limit.** *Retired, with a negative result.* `x_{k+1}` lies between
  `x_k` and `c_k`, so a stiff plant needs a large `c_k` for a fast transient. `residual` is the
  one variant with no such constraint, and EMPS — the stiff, integrating benchmark chosen to
  expose it — now ran: `residual` 0.2328 against `gru` 0.2349 and `leak` 0.2406, a 1–3% spread
  well inside the 3–6% that summation order alone moves these variants. Freedom from the convex
  hull buys nothing measurable on the benchmark designed to reward it.
- **A one-benchmark accuracy result.** *Retired, by running the other three.* The evaluation
  is now 4 of 4 (§4). It confirms gating pays — 43%, 46% and 72% NRMSE reductions on the
  benchmarks with any memory — and establishes that it *cannot* separate the three gated
  variants, whose spread is smaller than the shift produced by changing floating-point
  summation order alone. The recommendation therefore stands on the mechanism criterion, and
  the risk that accuracy would contradict it is resolved rather than merely hedged.

---

## 7. Out of scope

- Reset gate (`r ⊙ x_k` into `f`). MGU/LiGRU/JANET report single-gate variants matching full
  GRU accuracy, and "forget the state" is not meaningful for a physical state.
- LSTM-style separate cell (`x = o ⊙ tanh(c)`). It makes `c` the real state and `x` an
  observation, breaking the `n_state ≥ system order` contract the class is built on.
- Metal kernels — see §3.
- Stability guarantees. Gating bounds nothing; contraction by construction is ROADMAP A1
  (`PLAN_REN.md`). The two are complementary and a gated REN is a separate question.
