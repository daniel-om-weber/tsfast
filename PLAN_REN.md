# Implementation plan: REN and R2DN

Delivery plan for ROADMAP entry **A1**. The goal is a nonlinear architecture whose
contraction and incremental-Lipschitz certificates hold at *every* value of the free
parameters, trained by ordinary unconstrained SGD.

**Status: six planned phases delivered, plus a seventh the plan got wrong.** `REN`
(contracting, Lipschitz, `(Q,S,R)`-dissipative) with eager, C++ and Triton rollouts; `R2DN`
(contracting, Lipschitz) with eager and Triton rollouts; `RENLearner` and `R2DNLearner`;
`compare_ren.py` and `compare_r2dn.py`; `test_ren.py`, `test_ren_backends.py`, `test_r2dn.py`
and `test_r2dn_backends.py`; `benchmark_ren.py`, `benchmark_r2dn.py` and `gate_ren.py`, the
last now carrying `R2DN` as a third arm. What is *not* done: an `examples/` notebook (§4
cross-cutting deferred it past Phase 4, and it is still deferred). Sections below record
outcomes against what was planned; §7 is now a post-mortem rather than a forecast.

Phase 7 exists because Phase 6 declined a fused R2DN backend on a wrong premise, and running
the gate is what exposed it — the plan reasoned about arithmetic per step where the machine
cares about launches per step. The general lesson, already visible in Phase 5 and now twice
confirmed: for these models nothing about relative cost is knowable without a measurement,
because every quantity that matters is dispatch, not FLOPs.

Two claims from A1 shaped everything below. The first held; the second did not:

1. **The acyclic variant needs no fixed-point solve.** `D11` comes out strictly lower
   triangular from the parameterization itself, so the equilibrium layer resolves in one
   sequential sweep over the `n_nl` neurons. Confirmed — and the sweep being the expensive
   part is what motivated both Phase 5 and Phase 6.
2. ~~**R2DN is a parameterization swap, not a second architecture.**~~ Wrong: R2DN sets
   `D11 = 0`, so the sweep is *deleted* rather than swapped, its explicit realization has
   different fields, and it carries the same certificate axis the REN does — it is a second
   architecture, and `variant="r2dn"` would have collapsed two orthogonal axes. The
   *boundary* chosen in Phase 1 nevertheless paid off exactly as intended: the
   free-parameters → explicit-tensors → rollout seam carried over unchanged, so Phase 6 reused
   the certificate/rollout split, the stateful protocol, the learner shape and the test
   conventions, and shared the non-square Cayley transform outright. See Phase 6.

---

## 1. The one structural decision

Split the model in two, at a seam that does not exist in `PHNNCore`:

```
free parameters  ──(A)──>  explicit matrices  ──(B)──>  rollout over L steps
   X, Y1, B2, ...          A, B1, B2e, C1,              x⁺ = Ax + B1 w + B2e u
                           D11, D12e, C2, D21, D22      w  = sweep(C1 x + D12e u + bv)
```

**(A) runs once per forward, in plain autograd.** It contains the `XᵀX` product, the
partition, and the `E⁻¹`/`Λ⁻¹` solves. It is a handful of ops on matrices of size
`(2·n_state + n_nl)²` — irrelevant next to an `L`-step rollout, and never worth fusing.

**(B) is the only sequential part** and the only thing a fused backend ever needs to see.
It consumes *explicit matrices as tensors*, so a kernel needs no knowledge of the
certificate construction at all.

This is a large simplification over PHNN, and it is the reason the cost estimate is S–M
rather than M–L:

- The fused kernel's backward is BPTT through a linear recurrence plus one elementwise
  nonlinearity. There is no closed-form Hamiltonian gradient, no RK4 stage, and therefore
  no second-order term — nothing like the derivation `MATH.md` exists for.
- Gradients stop at the explicit matrices; autograd carries them back through `XᵀX` and
  the solves for free. A kernel that gets `∂L/∂A` right needs no opinion about `∂L/∂X`.
- The seam is what let Phase 6 be additive: R2DN rebuilds (A) and drops the `sweep` from (B)
  in its own module, and the state recurrence, the stateful protocol, the learner shape and
  the scalers carry over untouched. The custom-op plumbing is the one thing it does *not*
  reuse — it has no fused backend to plug in.

**Consequence for module layout:** `explicit()` must be a public, separately testable
method returning a plain dataclass of tensors — not an implementation detail folded into
`forward`. Held for both models, and it is what makes the certificate testable independently
of the rollout (§5).

---

## 2. Math contract

The REN's, as implemented. Source of truth: Revay, Wang & Manchester (arXiv:2104.05942)
§V-A, cross-read against `nic-barbara/R2DN` `robustnn/ren_base.py` (MIT). `MATH_REN.md` now
carries the fused rollout's forward and BPTT contract, written for Phase 5; this section
remains the contract for the *construction*, which no kernel sees. The R2DN's construction is
a different matrix of a different size and is documented where it lives — `r2dn.py`'s module
and class docstrings, with the derivations that have no reference implementation recorded
under Phase 6.

Dimensions: `nx = n_state`, `nu = n_input`, `ny = n_output`, `nv = n_nl` (neurons in the
equilibrium layer).

**Free parameters.** `X ∈ R^{(2nx+nv)×(2nx+nv)}`, `Y1 ∈ R^{nx×nx}`, `B2 ∈ R^{nx×nu}`,
`D12 ∈ R^{nv×nu}`, `C2 ∈ R^{ny×nx}`, `D21 ∈ R^{ny×nv}`, `D22`, biases `bx, bv, by`, and an
optional polar scalar `p`. All unconstrained.

**Contracting construction**, with contraction rate `ᾱ ∈ (0, 1]`:

```
H = XᵀX + εI                    (polar variant: H = (p²/‖X‖²)·XᵀX + εI)

partition H by (nx | nv | nx):
    P    = H33
    F    = H31
    E    = (H11 + P/ᾱ² + Y1 − Y1ᵀ) / 2
    B1~  = H32
    C1~  = −H21
    D11~ = −tril(H22, −1)                 ← strictly lower triangular: acyclic
    Λ⁻¹  = 2 / diag(H22)

explicit:
    A = E⁻¹F        B1 = E⁻¹B1~      B2e = E⁻¹B2
    C1 = Λ⁻¹C1~     D11 = Λ⁻¹D11~    D12e = Λ⁻¹D12
```

**Per-step evaluation**, with `σ` monotone and slope-restricted to `[0,1]` (`tanh`, `relu`):

```
b   = x C1ᵀ + u D12eᵀ + bv
w   = sweep(b, D11)             w_i = σ(b_i + Σ_{j<i} D11[i,j] w_j),  i = 0 .. nv−1
x⁺  = x Aᵀ + w B1ᵀ + u B2eᵀ + bx
y   = x C2ᵀ + w D21ᵀ + u D22ᵀ + by
```

**Variants.** Lipschitz-`γ` and `(Q,S,R)`-dissipative change *only* step (A): the LMI gains
the `(Q,S,R)` terms and `D22` stops being free (built from auxiliary `X3, Y3, Z3` through the
non-square Cayley transform now shared with R2DN). The sweep, the recurrence, and every
backend are unchanged. The constructions were read off `ren_base.py`'s subclasses rather than
re-derived, with Lipschitz implemented as the `(Q,S,R)` special case `Q = -I/γ, S = 0, R = γI`.

**Numerical notes.** `Λ⁻¹ = 2/diag(H22)` is the sharp edge — small diagonal entries blow it
up, so `ε` is load-bearing, not cosmetic. `E` is inverted every forward, so the code uses
`torch.linalg.solve` (one solve against the concatenated column blocks), never `inverse()`, and
caches the explicit bundle whenever gradients are off. `long_memory` is ported and is the
default, since random init yields fast-forgetting models.

---

## 3. Module layout

Follows the `phnn` package shape, since REN also grows private fused kernels. As built:

```
tsfast/models/architectures/ren/
    __init__.py       package docstring, re-exports for both models
    common.py         RENSpec, ExplicitREN + what both parameterizations share:
                      _ACTS, _EPS, _lecun_normal_, cayley_contraction, parameter_cache_key
    core.py           RENParameterization, RENCore, REN, custom ops, fused_rollout
    backend_c.py      generic scalar-templated C++ rollout, float and double
    backend_triton.py persistent per-trajectory CUDA rollout, float32
    lbdn.py           SandwichLayer, LBDN, lbdn_forward, folded_weights, cayley_blocks,
                      ExplicitSandwich
    r2dn.py           R2DNSpec, ExplicitR2DN, R2DNParameterization, R2DNCore, R2DN,
                      custom ops, fused_rollout, r2dn_param_grads
    r2dn_backend_triton.py  persistent per-trajectory CUDA rollout, float32 (Phase 7)
```

Registered in the `architectures/__init__.py` docstring (as "the certified-by-construction
models"), and `models/__init__.py` re-exports the package.

`common.py` did not grow the planned `flat_params`/`split_params`/`spec_caps` — the custom ops
take `ExplicitREN.tensors` in field order and each backend screens with the shared
`rollout_unsupported`, which covered what those were for.

**Public surface.**

- `RENSpec` — frozen dataclass: `n_state, n_input, n_output, n_nl, variant, alpha, act`.
  `variant ∈ {"contracting", "lipschitz", "dissipative"}`. `gamma` stays *out* of the spec
  for the same reason `dt` is out of `PHNNSpec`: it is a runtime scalar and must not
  trigger a kernel recompile.
- `ExplicitREN` — frozen dataclass of the twelve explicit tensors.
- `RENCore.explicit() -> ExplicitREN` — step (A), public and separately tested.
- `REN` — user-facing module. Signature and semantics track `NeuralStateSpace`:
  `forward(u, x0=None, state=None)`, `return_state`, `backend`.
- `R2DNSpec`, `ExplicitR2DN`, `R2DNCore.explicit()`, `R2DN` — the same four roles for the
  second model. `R2DNSpec` adds `hidden`, the 1-Lipschitz network's widths, which the REN has
  no analogue for; the user-facing `R2DN` takes `depth` and derives `hidden = (n_nl,) * depth`,
  leaving per-layer widths to a directly constructed `R2DNCore`.
  `variant ∈ {"contracting", "lipschitz"}`, and `n_h` is `2*n_state` rather than the REN's
  `2*n_state + n_nl` — the scalability claim, in one field.
- `LBDN`, `SandwichLayer` — exported in their own right: a certified-Lipschitz MLP is useful
  outside the recurrence, and `LBDN`'s `gamma` is a runtime scalar like the REN's.

**Initial state — no bespoke encoder, but not because none is needed.** `PHNN` carries a
`SubnetEncoder` (`n_init=50`) to estimate `x0`. Contraction makes `x0` self-correcting at
rate `ᾱ`, which looks like it removes the need — but the forgetting time `≈ 1/(1−ᾱ)` *is*
the longest time constant the model can represent. Same quantity, so no setting both
represents an integrator and forgets `x0`:

- `ᾱ < 1` — trajectory separation must decay, so a pure integrator (`x⁺ = x + u`, constant
  separation) is outside the class. A leaky approximation with `ᾱ = 1 − δ` costs
  `n_skip ≈ 1/δ`.
- `ᾱ = 1` — integrators admissible, forgetting exactly zero. (Check against the reference
  whether the parameterization stays well-posed there, given `P/ᾱ²` and `ε`.)

Integrating dynamics are common in the target benchmarks: position from velocity (`EMPS`,
the robot sets), tank level (`CascadedTanks`), thermal accumulation. Zero-`x0` plus
`n_skip` does not degrade gracefully there — it fails outright.

So `REN` takes `x0=None → zeros` like `NeuralStateSpace` and grows no encoder of its own.
Instead it must implement the stateful protocol exactly — `forward(u, x0=None, state=None)`
returning `(y, {"x": x_L})` under `return_state`. That makes `FranSys`
(`prediction/fransys.py`) usable as the state estimator for free: `discover_state_spec`
flattens a state dict through pytree, and the diagnosis model reads `x0` off an `(u, y)`
window with no forgetting required — separating estimation from dynamics instead of asking
the dynamics to solve it. **"REN under FranSys" is the supported answer for integrating
systems**; test the composition in Phase 1, not later.

Delivered as written, for both models: `x0=None → zeros`, no encoder, the stateful protocol
exact, and `test_fransys_composition` in each test file asserts `discover_state_spec` finds
the physical state through pytree and that gradients reach the prognosis. The Phase 4 gate
then measured what the composition is worth (see there), and both class docstrings carry the
warning that the Lipschitz bound is stated for a fixed initial state and does not survive it.

---

## 4. Phases

Effort is engineering judgment, in the ROADMAP's S/M/L vocabulary. All six are delivered;
each section keeps the original brief and records what actually shipped.

### Phase 1 — eager contracting REN (S) — **done**

`common.py` + `core.py` with `backend="eager"` only. Nested Python loop over `L` and the
`nv` sweep. Correct, slow, complete. Explicit matrices built once per `forward`, not per
step.

Ships with the Phase-1 tests from §5. **This is the deliverable that makes every later
phase optional** — a working certified model, before any kernel work.

Shipped as planned, plus two things the brief did not name: `equilibrium_sweep` as a public
function (each neuron a rank-1 update of the pending pre-activations, so the loop costs two
tensor ops per neuron rather than a growing matmul), and the `explicit()` cache keyed on
parameter versions, which only pays off under `inference_mode` and is skipped whenever
gradients or `torch.compile` are live.

### Phase 2 — numerical validation (S) — **done**

`comparisons/compare_ren.py`, following `compare_phnn.py`: float64, same parameters loaded
into both sides, print max relative deviation of outputs and all gradients, exit non-zero
past tolerance. Reference is `nic-barbara/R2DN` `robustnn/ren.py` (MIT, JAX).

The framework gap is a feature here — a JAX reference makes this an independent check
rather than a transcription — but it means the comparison runs weight transplant across
frameworks. Budget for that plumbing; it is the bulk of the phase. Add the row to
`comparisons/README.md`.

`DecodEPFL/SSM` `src/neural_ssm/rens/ren.py` has **no license file**. Read it for shape,
do not transcribe, and do not import it in the comparison.

Delivered in two layers rather than one, and the risk table's fallback was taken *as well as*
the transplant, not instead of it: the always-on section is an implicit-form reference written
from the paper's equations (per-step `solve` against `E` and `Λ`, equilibrium by fixed-point
iteration instead of forward substitution), which agrees across nine configurations to ≤7e-16
on outputs and ≤7e-15 on every gradient; the JAX transplant is an optional section, skipped
unless `robustnn` is installed, and agrees to ≤7e-16 on outputs across the six configurations
it can express. A third section reports the certificates themselves, since agreement between
two evaluations of a construction says nothing about whether the construction is right.

Two reference limitations are worked around rather than papered over: it builds `M` with an
`(ny,ny)` identity where it needs `(min(nu,ny))²`, so its non-square Lipschitz and dissipative
paths do not run and those configurations are skipped; and it applies its own `eps` adjustment
to `(Q,S,R)` internally, so the raw matrices are handed over.

### Phase 3 — Lipschitz and dissipative variants (S) — **done**

Step (A) only, per §2: the LMI gains the `(Q,S,R)` terms and `D22` stops being free.
`gamma` becomes a runtime argument. Adds the incremental-gain certificate test from §5.

Ordered *before* the gate deliberately — the gate's second criterion measures certificate
tightness, which needs a prescribed `γ` to measure against.

Both shipped. `gamma` is a property with a setter, so retuning the certificate needs no
rebuild; `qsr` is validated at construction (`Q ≺ 0`, `R - S Q⁻¹ Sᵀ ≻ 0`, shapes) and nudged
inside the definiteness boundary by `eps` so the Cholesky factors the construction takes stay
well conditioned. `variant="lipschitz"` is implemented as the `(Q,S,R)` special case
`Q = -I/γ, S = 0, R = γI` rather than as its own branch, which is why the dissipative code
path carries the Lipschitz one and both are covered by the same tests.

### Phase 4 — gate: accuracy and certificate tightness (S) — **done, decision: continue**

Settles the A1 open question. Two criteria, both required to proceed.

**Accuracy.** Train `REN` at matched parameter count against two baselines, through
`tsdata/benchmark.py`:

- `RNN` (`rnn_type="gru"`, the default) — **the honest opponent.** A GRU's state is a
  convex combination of `h_{t−1}` and a `tanh` output, so `‖h‖_∞ ≤ 1` for all time: it
  cannot diverge either. REN has to win on something other than boundedness.
- `NeuralStateSpace` — unconstrained, genuinely divergence-prone; the easier comparison.

Benchmarks: `EMPS` (friction/stiction *and* integrator modes — the two ways contraction is
most likely to bind), `CascadedTanks` (integrating), `Silverbox`, `WH`. Run the integrating
ones **both** standalone and under `FranSys`, reporting identified `ᾱ` next to NRMSE;
standalone-vs-FranSys decides whether integrating systems need the estimator or merely
prefer it.

**Certificate tightness.** Report `γ_empirical / γ_certified`, with `γ_empirical` a lower
bound from power iteration over input-perturbation pairs (§5). A model that fits well but
carries a vacuous certificate has failed at the only thing it was built for, so this is
pass/fail, not a diagnostic.

Do not read the certified-Lipschitz literature's reported 20×–50× gaps as a prediction
here: those are *post-hoc* bounds on freely-trained networks. REN prescribes `γ` as a
training constraint, so optimization pressure pushes the model to consume its gain budget.
The regimes differ, in both directions — which is why this is measured rather than assumed.

**Outcomes.**

- Competitive fit *and* tight certificate → continue.
- Contraction costs badly on `EMPS` only → continue; record the exclusion in the `REN`
  docstring so users do not reach for it on stick-slip systems.
- Good fit, vacuous certificate → **stop.** That accuracy is available from a GRU at a
  fraction of the cost and none of the maintenance.
- Contraction costs badly everywhere → **stop**; move A1 to *Considered and declined* with
  the numbers. Phases 1–3 remain useful on their own.

Deliverable regardless of outcome: the worked example in §6.

Do not start Phase 5 before this gate. Fused kernels are the expensive, hard-to-reverse
part of the entry.

**Outcome: the second branch — continue, with an exclusion recorded.** `benchmarks/gate_ren.py`
is the experiment; it grew two controls the brief did not ask for, because the
standalone-vs-FranSys column alone answers neither question it was meant to: a standalone REN
grown to the composition's parameter budget separates the estimator from the parameters it
brings, and `GRU+FranSys` is the honest opponent in the composed setting just as the bare GRU
is in the standalone one. The accuracy result is recorded in the `REN` docstring, which is
where a user meets it: a clear win on `CascadedTanks` (0.37 NRMSE against 0.60 for a GRU, and
0.10 against 0.25 with both under `FranSys`), a tie on `Silverbox`, a loss of roughly a
quarter on `WH`, and a 3.4× loss on `EMPS` under `FranSys` — friction-dominated, and stick-slip
is precisely what a contraction certificate excludes, since trajectories in a stick phase do
not converge. That exclusion is stated in the docstring as an instruction not to reach for the
model on stick-slip or hysteretic plants.

**Certificate tightness: not vacuous.** The figure recorded in the repo is
`compare_ren.py`'s — `γ_empirical/γ_certified = 0.998` in all three Lipschitz configurations at
parameters scaled ×20, so the bound is essentially exactly what the model achieves there.
Nothing like the 20×–50× gaps of post-hoc bounds on freely-trained networks, which is why the
plan insisted on measuring rather than assuming. The criterion as *stated* is over trained
models: `gate_ren.py`'s last line reports that ratio across benchmarks, but the number is not
committed anywhere — rerun the gate to reproduce it.
`test_ren.py::test_incremental_gain_below_certificate` prints the ratio per configuration so a
regression stays visible rather than merely asserted. Note the asymmetry this exposes with
R2DN, recorded under Phase 6: the REN saturates its budget through the free feedthrough `D22`,
and the R2DN's Lipschitz parameterization has no `D22` to saturate.

**Open item, carried since Phase 1 — settled, and deliberately not shipped.** Definition 2 of
the REN paper states contraction as `|x_t^a − x_t^b| ≤ K αᵗ |a − b|` in Euclidean norm with a
generic constant `K`; the definition does not supply `K`. It is `K = cond(M)^½` with
`M = Eᵀ P⁻¹ E` the Lyapunov metric — **not** `cond(P)^½` as the plan guessed: the metric in
which the rate is exactly `alpha` carries the `E` factor, and it does not cancel. The
derivation lives in `test_ren.py::_lyapunov_metric`'s docstring next to the code that uses it.
No API exposes it: `n_skip` is a training decision, and a bound this loose is a poor way to
make one.

### Phase 5 — fused backends (M) — **done**

`backend_c.py` and `backend_triton.py` behind `tsfast::ren_rollout{,_train,_bwd}` custom
ops, reusing `_core/dispatch.py` unchanged and the `rollout_unsupported` screen from
`ssm/core.py`.

The kernel signature takes `ExplicitREN` tensors, so `supports(spec, u, x0)` caps on
`n_state`, `n_nl`, dtype and device, and knows nothing about the certificate.

Parameter gradients follow `ssm/core.py:mlp_param_grads`: the state-adjoint recurrence is
the only sequential part; `∂L/∂A`, `∂L/∂B1` etc. are batched GEMMs over `B*L` step
samples. Within a step the sweep reverses over `nv`.

Write `MATH_REN.md` first, as `MATH.md` was written for PHNN. It is short here — the
recurrence is linear plus one elementwise nonlinearity.

Profile before writing anything: confirm the eager path is dispatch-bound at realistic
`(L, n_state, n_nl)`. Add `benchmarks/benchmark_ren.py` alongside `benchmark_ssm.py`.

Delivered as specified: `MATH_REN.md` first, then both backends behind
`tsfast::ren_rollout{,_train,_bwd}`, `ren_param_grads` for the batched half of BPTT, the shared
`rollout_unsupported` screen, `tests/test_ren_backends.py`, and `benchmarks/benchmark_ren.py`
with `--n-nl` as the axis to scan. Two things worth carrying forward: the saved-tensor set is
just `xs` and `ws`, since `σ'(v)` is recoverable from `w` alone for every admissible activation
(`MATH_REN.md` §1), and `dD11` is masked with `tril(·, -1)` because entries on and above the
diagonal are structurally absent from the forward — not required for correct training, since
the construction re-masks in its own backward, but required for the op to be the gradient of
the function it claims to compute.

### Phase 6 — R2DN parameterization (M) — **done**

Barbara, Wang & Manchester, arXiv:2504.01250. Replaces the single `(2nx+nv)²` construction
with a contracting linear core wrapped around a stack of Lipschitz-bounded (LBDN/sandwich)
layers — same certificates, and a certificate matrix that stays `(2nx)²` however large the
nonlinearity grows.

Landed as **its own model** (`ren/r2dn.py`: `R2DNSpec`, `ExplicitR2DN`,
`R2DNParameterization`, `R2DNCore`, `R2DN`) plus `ren/lbdn.py` (`SandwichLayer`, `LBDN`),
not as `variant="r2dn"` on `RENSpec`. The sketch above collapsed two orthogonal axes: R2DN
is a *parameterization*, and it carries the same `variant` axis the REN does
(`"contracting"`, `"lipschitz"`). Nothing of step (B) survives the swap either — `D11 = 0`,
so the sweep is gone rather than replaced, and the explicit realization has different fields.
What did get shared, in `common.py`: `_ACTS`, `_EPS`, `_lecun_normal_`, the non-square Cayley
transform (`cayley_contraction`, which the REN's `D22` also uses) and `parameter_cache_key`.

Deviations from the plan's assumptions, each forced by the sources:

- **`alpha` is derived, not transcribed.** The reference implements no contraction rate for
  the R2DN, and the paper's §V-B fixes `E + Eᵀ = H11 + P`. `E + Eᵀ = H11 + P/ᾱ²` generalizes
  it exactly as in the REN case (the identity `α²EᵀP⁻¹E ⪰ E + Eᵀ - P/α²` is what both rest
  on), and `ᾱ ≤ 1` only strengthens the IQC, so the Lipschitz variant is unaffected. Checked
  as an eigenvalue, not by sampling — see below.
- **Lipschitz R2DN has no reference implementation.** §V-C is derived from Prop. 2 instead:
  `D12` and `D21` stop being free (each is `√γ` times a Cayley contraction, since `R ≻ 0`
  needs both below `√γ` in norm) and `D22 = 0`, which is the subset the paper parameterizes
  and explicitly leaves open beyond. `(Q,S,R)`-dissipative R2DN is future work in the paper
  and is rejected with a `ValueError` here.
- ~~**No fused backend.**~~ **Reversed — see Phase 7.** The reasoning ("per-step work is
  `depth` GEMMs, not a sweep, so there is far less for a kernel to collapse") confused *work*
  with *launches*. Deleting the sweep makes the step cheap in FLOPs and leaves the launch
  count untouched, so the eager R2DN is dispatch-bound exactly as the eager REN is —
  measurably so: at `L=200, B=64` it was 1.74 ms per trajectory against the fused REN's
  0.03 ms. `benchmarks/benchmark_r2dn.py` is still the comparison that decides whether the
  architecture change or the kernel is the better answer, and the answer turned out to be
  *both*.

Validation went beyond the phase's brief, because two of the constructions have no reference
to check against. `comparisons/compare_r2dn.py` and `tests/test_r2dn.py` both assemble the
dissipation inequality as a single quadratic form in `(Δx, Δw, Δu)` from the *explicit*
realization and the Lyapunov metric, reading nothing of how `H` was built, and assert its
largest eigenvalue is negative — the guarantee itself rather than a sample of it. The JAX
reference then confirms the contracting `H`, `E`, `A`, `B1` and every sandwich-layer
realization to ~1e-16. One deliberate divergence from the released reference: its `LBDN`
never sets the flag that would make its last layer the norm-bounded linear map of the network
it cites, so its stack carries one extra nonlinear layer; tsfast follows the ICML'23
definition and `acfr/RobustNeuralNetworks.jl`. `compare_r2dn.py` documents this and composes
the JAX section's network from the reference's own `SandwichLayer` so no formula goes
unchecked.

**Two open items, both about the Lipschitz variant.** First, its certificate is measurably
*looser* than the REN's on the same measurement: `γ_empirical/γ_certified` comes out 0.02–0.13
at parameters scaled ×20, against the REN's 0.998. That is the expected consequence of
`D22 = 0` — the REN saturates its gain budget through a free feedthrough, and this
parameterization has none to saturate — and the paper flags the same limitation, but it means
Phase 4's tightness criterion has *not* been re-established for a **trained** Lipschitz R2DN,
only for a randomly perturbed one. Second, accuracy: R2DN inherits A1's answer from the REN's
gate and the paper reports parity at matched size, but `gate_ren.py` has not been run with
`R2DN` as a third arm. Both want a training run, not more implementation.

- `RENLearner` in `training/learners.py`, modelled on `SSMLearner` (Phase 1). Add to
  `__all__`. Docstring must state the `n_skip` guidance: with contraction the cold start
  is self-correcting, so `n_skip` follows from the contraction rate rather than from an
  unknown initial state. **Done**, and `R2DNLearner` alongside it in Phase 6 with the same
  `n_skip` guidance. Both carry a note the brief did not ask for: normalizing the output
  changes what `gamma` certifies, since the bound then applies to the normalized
  input/output pair rather than to engineering units.
- Docs: `docs/api/models/` has no page for `ssm` or `phnn`, so **do not add one for REN** —
  a page here would be inconsistent with both neighbours. Revisit only if those get pages.
  **Held**, for R2DN too.
- An `examples/` notebook is optional; defer past Phase 4. **Still deferred** — nothing under
  `examples/scripts/` covers either model.

### Phase 7 — fused R2DN backend (M) — **done**

Not in the original six. Phase 6 declined a fused R2DN backend on the grounds that a stack of
GEMMs leaves a kernel little to collapse; running the Phase 6 gate is what showed that
reasoning to be wrong, and expensively so — the R2DN arms cost 35 s per epoch against the
fused REN's 1.6 s, which put the four-benchmark gate at roughly 17 hours of GPU time. What a
kernel collapses is *launches*, and deleting the sweep does nothing about those: the eager
rollout still issues `L * (8 + 3*depth)` of them, every one on a matrix small enough that the
launch is the whole cost.

Delivered as the REN's Phase 5 was, and reusing all of its machinery: `MATH_R2DN.md` first,
then `r2dn_backend_triton.py` behind `tsfast::r2dn_rollout{,_train,_bwd}`, `r2dn_param_grads`
for the batched half of BPTT, the shared `rollout_unsupported` screen and `dispatch.resolve`
unchanged, `tests/test_r2dn_backends.py`, and a fused row in `benchmarks/benchmark_r2dn.py`.

Three things worth carrying forward:

- **Fold everything constant across the rollout into the weights.** `√2` and `Ψ` go first
  (`folded_weights`), and then each layer's output map `V_l` composes into the *next* layer's
  `W_{l+1}`, because `W_{l+1}(V_l a_l) = (W_{l+1} V_l) a_l`. One matrix per layer reaches the
  kernel. This is the same trick as the REN's explicit realization, applied one level down,
  and it halves both the register footprint and — what a sequential rollout is actually bound
  by — the dependent cross-lane reductions per timestep.
- **The op boundary carries no width list.** Layer widths are read off the weight shapes
  (`hidden_widths`). The first version passed `hidden: list[int]`, which also tripped a
  `torch.library` pytree quirk: an *empty* list argument is a node with no leaves, so the
  backward has to return `[]` for it where a non-empty one wants `None`. A redundant argument
  is one more thing a backend could disagree with; deleting it removed both problems.
- **Fusing the rollout exposed the construction.** With the kernel in place, `explicit()` was
  1.12 ms of a 3.94 ms training step, because each sandwich layer takes its own Cayley
  transform and each transform issued two `linalg.solve` calls. Both blocks are the same
  factorization acting on different right-hand sides, so they became one solve
  (`cayley_contraction`, which the REN's `D22` path also benefits from), and identically
  shaped hidden layers now resolve as one batched transform (`cayley_blocks`,
  `LBDN.explicit`). Step time 3.94 → 3.12 ms.

**Outcome, at matched parameter count** (`L=200`, `B=64`, RTX 4090, exclusive GPU; the eager
R2DN is flat at ~1.74 ms per trajectory throughout):

| train step, µs/traj | nv=8/6 | nv=16/10 | nv=32/18 | nv=64/31 |
|---|---|---|---|---|
| REN triton | 26.9 | 27.8 | 40.2 | 118.1 |
| R2DN triton | 47.4 | 47.6 | 54.4 | 52.0 |

Inference is the cleaner picture, being all rollout: R2DN flat at 6.1–6.3 µs against the REN's
3.9 → 32.6. So the architecture's scalability claim *does* hold — cost flat in nonlinear
capacity where the REN's grows — but only against an equally fused REN, and only above the
crossover at `n_nl ≈ 24`. Below it the REN's sweep is a handful of neurons and the R2DN's
fixed per-layer construction dominates. Neither the paper's "up to an order of magnitude" nor
Phase 6's "far less for a kernel to collapse" survives contact with a fused comparison.

---

## 5. Tests

`tests/test_ren.py`, conventions from `test_statespace.py` (`_run`, `_rel`,
`_assert_backend_parity`); `tests/test_ren_backends.py` at Phase 5, from
`test_phnn_backends.py`; `tests/test_r2dn.py` from Phase 6, which also covers the `LBDN` on
its own (Lipschitz bound by power iteration at badly-scaled parameters, the isometry identity
per layer, and that a depthless network is affine — i.e. that the output layer really is one).

**The certificate tests are the distinctive ones.** A direct parameterization claims the
guarantee holds at *every* parameter value, so test it at randomly drawn parameters — not
only at initialization, and not only after training:

- **Contraction, random parameters.** Draw `X, Y1, ...` from several distributions and
  scales. For each, roll two trajectories from different `x0` under identical `u`, assert
  `‖x_t − x̃_t‖` decays at least at rate `ᾱ`. Include deliberately badly-scaled draws
  (`X * 1e3`) — that is exactly what an unconstrained optimizer will eventually produce.
- **Contraction after a parameter step.** Perturb every parameter by large random noise,
  re-check. This is the property that distinguishes a direct parameterization from a
  projected one; nothing else in the suite covers it.
- **Incremental gain** (Phase 3). Estimate `γ_empirical = sup ‖y − ỹ‖_ℓ2 / ‖u − ũ‖_ℓ2` by
  power iteration over input-perturbation pairs from a **common initial state** (the bound
  is stated for a fixed `a`; see §6). Assert `γ_empirical ≤ γ` with margin, and *report* the
  ratio — Phase 4 gates on it, so the test must expose the number, not just the assertion.
- **Well-posedness.** `D11` strictly lower triangular; the sweep's result satisfies the
  equilibrium equation to tolerance.
- **`explicit()` invariants.** `E + Eᵀ ≻ 0`, `Λ` positive, shapes, and that `explicit()` is
  differentiable end-to-end.

Plus the conventional set: shapes/dtype/device, zero-`x0` default, `state` round-trip vs a
full-sequence rollout, float64 `gradcheck` on a tiny config, `torch.compile(fullgraph=True)`
parity, and backend parity at Phase 5.

All of the above shipped, with R2DN's analogues wherever they transfer — there is no sweep
whose well-posedness needs checking, so `D22 = 0` and the two `‖·‖ ≤ √γ` feedthrough bounds
take that slot. Two additions worth keeping in mind when extending the suite:

- **The dissipation inequality as an eigenvalue** (`test_r2dn.py::_dissipation_residual`,
  mirrored in `compare_r2dn.py`) supersedes trajectory sampling as the primary certificate
  test for R2DN. `V(Δx⁺) - ᾱ²V(Δx) - s(Δu, Δy) + ‖Δv‖² - ‖Δw‖² ≤ 0` for *all* `(Δx, Δw, Δu)` is
  a quadratic form, so one eigenvalue decides it — no sampling, and it reads only the explicit
  realization and the metric, never the construction under test. It exists because two of the
  R2DN constructions have no reference to compare against. The REN would benefit from the same
  treatment; its suite still samples.
- **Certificate survives training** (both files, `@pytest.mark.slow`): 30 Adam steps in
  float32, then re-check `H ≻ 0` and the empirical gain. This is what catches the float32
  conditioning risk from §7 in the regime that actually matters, rather than in float64.

---

## 6. Worked example: what the certificate actually buys

Ships with the Phase 4 report as a worked example — **not** as a `certified_bounds()` method
on `REN`. Of three quantities originally proposed, two survived checking against the paper's
definitions, and both carry preconditions that a public API would hide.

**Where it landed:** in the `REN` and `R2DN` class docstrings rather than in a separate
report — the first quantity, both preconditions and the *not backed* paragraph, stated where a
user reads them. No `certified_bounds()` exists, as intended. The envelope-around-a-reference
framing below is the one piece that did not make the cut, on space grounds; it is the same
inequality read differently.

**Backed** (Def. 3 / Eq. 5: `‖ℜ_a(u) − ℜ_a(v)‖_T ≤ γ‖u − v‖_T`):

1. **Input-perturbation sensitivity.** Input measurement error of energy `δ` moves the
   output by at most `γδ`. Immediately usable — you know your sensor spec.
2. **Envelope around a reference simulation.** `‖y(u) − y(u_ref)‖_ℓ2 ≤ γ‖u − u_ref‖_ℓ2`.

Two conditions must appear wherever these do:

- **Same initial state.** Def. 3 is stated for one fixed `a`; the general incremental IQC
  carries a penalty `d(a,b) ≥ 0` vanishing only when the states coincide. The zero-`x0`
  default satisfies it — **a `FranSys`-estimated start does not**, so the bound does not
  survive the composition §3 recommends for integrating systems.
- **Truncated ℓ₂ over the horizon**, so the width grows with sequence length. This is
  output energy per unit input energy, *not* a deployment envelope. Going pointwise via
  `‖·‖_∞ ≤ ‖·‖_ℓ2` is valid but close to vacuous on a 10⁵-sample test signal.

**Not backed.** Nothing here bounds model-vs-plant error. The certificate is a property of
the learned map, not of its fidelity — a REN with `γ = 1` can be an arbitrarily bad model,
certified smooth and stable rather than correct. Certifying prediction error would need a
validation-set bound *plus* an assumed incremental gain for the plant, which is what
identification was meant to establish. State this explicitly in the example; it is the claim
users will otherwise assume was made.

Where `γ` stops being modest and becomes load-bearing is closed-loop: small-gain gives
`γ_model · γ_controller < 1` ⟹ stable interconnection. Out of scope per §8 — a choice
recorded here rather than a gap left implicit.

---

## 7. Risks — how each one turned out

| Risk | Outcome |
|---|---|
| Contraction excludes the dynamics (stiction, limit cycles) | **Materialized, on `EMPS` only** (3.4× worse than a GRU under `FranSys`). Handled by the planned response: the `REN` docstring tells users not to reach for the model on stick-slip or hysteretic plants |
| Certificate valid but vacuous — `γ_certified` orders of magnitude above what the model achieves | **Did not materialize for the REN**: `γ_empirical/γ_certified = 0.998` at badly-scaled parameters, so the bound is what the model achieves. **Open for the Lipschitz R2DN**, where `D22 = 0` leaves nothing to saturate and the same measurement gives 0.02–0.13 (Phase 6) |
| Integrating dynamics: no `ᾱ` both represents the integrator and forgets `x0` | **Materialized as predicted.** `CascadedTanks` goes 0.37 → 0.10 NRMSE under `FranSys`, so the composition is not optional there. Documented in both class docstrings and in both learners; tested since Phase 1 |
| `Λ⁻¹ = 2/diag(H22)` ill-conditioned in float32 | **Contained by the `ε` floor alone**; the float64 mixed-precision solve was not needed. float64 `gradcheck` passes and the slow `test_certificate_survives_training` trains 30 float32 steps and re-checks `H ≻ 0` and the gain |
| Random init gives fast-forgetting models that will not fit long horizons | **Mitigated as planned**: `long_memory` is ported and is the default for both models. It writes the target realization down first (`E = P = I`, `A = I - diag(eigs)`, dead nonlinear coupling) and takes `X` from its Cholesky factor |
| `nv` sweep dominates runtime at useful widths | **Confirmed, and it drove three phases** — the fused kernels (Phase 5), then R2DN (Phase 6), which removes the sweep instead of accelerating it, and then Phase 7, which fused the R2DN too after the eager version turned out to be bound by launches rather than by the sweep it had deleted. `nv` is not capped; `benchmarks/benchmark_ren.py --n-nl` and `benchmark_r2dn.py` are how the trade is measured |
| Dispatch overhead outside the rollout becomes the floor | **Materialized in Phase 7, once the kernels landed.** `explicit()` was 1.12 ms of a 3.94 ms R2DN training step — a Cayley transform per sandwich layer, two `linalg.solve` calls each. Fixed by solving both blocks against one factorization and batching identically shaped layers into a single transform; both models are now construction-bound below `n_nl ≈ 24` rather than rollout-bound |
| Cross-framework weight transplant in Phase 2 costs more than the phase | **Did not materialize**; both fallback and transplant shipped. The transplants are ~20 lines each and cover both models' contracting paths. Note the environment cost is real but one-time: `uv pip install jax flax "robustnn @ git+…"`, and the sections skip when absent |

## 8. Out of scope

- **The general implicit REN** (dense `D11`, per-step fixed-point solve,
  implicit-function-theorem backward). Separate M. Only if the acyclic class demonstrably
  underfits in Phase 4 — which the gate did not show: its two losses are `EMPS`, attributable
  to contraction itself rather than to the equilibrium layer's structure, and `WH` by roughly a
  quarter, which the gate was not designed to attribute either way. Still out of scope, but the
  `WH` gap is the one result that would justify revisiting it.
- **`(Q,S,R)`-dissipative R2DN.** Future work in the paper itself; `variant="dissipative"`
  raises for `R2DN` while the REN supports it. Also **Lipschitz R2DN with `D22 ≠ 0`**, which
  the paper explicitly leaves open and which is what would tighten the bound noted in §7.
- **Continuous-time REN** (`DecodEPFL/NodeREN`). Belongs to ROADMAP A4, not here.
- **REN as a controller / Youla parameterization.** Control synthesis is outside what
  TSFast does.
