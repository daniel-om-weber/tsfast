# JIT compile cost in the generated-C++ backends

Internal engineering analysis. Not user documentation — this is a problem statement plus an
evaluation of candidate fixes, so the design can be decided on measured numbers.

All numbers below were measured on the development host (32-core x86-64, glibc, g++, torch
2.12.1 built with OpenMP, `at::get_num_threads() == 24`). Every claim is labelled
**measured**, **inferred**, or **assumed**. Timings are CPU/host-compiler only; no GPU
throughput numbers appear here.

---

## 1. Problem

`tests/test_statespace.py` runs **223.6 s** on a cold cache and **6.1 s** warm — a 36× gap on
identical hardware and identical tests (**measured**; both runs 61 passed / 2 skipped).

The gap is entirely host-compiler time. Isolating one spec's first call from its second:

| | |
|---|---|
| first call (builds the extension) | **9.38 s** |
| second call (pure execution) | **0.29 ms** |

**measured**, and reproduced across five configurations: 9.11 / 9.38 / 9.23 / 9.25 / 9.31 s to
build, 0.28–0.40 ms to run. The build costs ~32 000× what the kernel costs to execute.

### Why it recurs

`_gen_source` bakes the layer dims, activation and gate into the emitted C++ so the tiny GEMVs
unroll and vectorize. The source therefore changes with every distinct `SSMSpec`
(`n_state`, `n_input`, `hidden`, `act`, `gate`, `eps`), and `_get_extension` hashes the source
into the extension name — so each spec is a separate `load_inline`, i.e. a separate g++ run.
The dev machine's cache holds **118** built extensions, **88** of them `tsfast_ssm_c_*`
(**measured**).

Cold cost of `test_statespace.py`, by test (**measured**):

```
37.57s  test_c_parity_gated[gru]         (4 specs)
37.48s  test_c_parity_gated[residual]    (4 specs)
37.44s  test_c_parity_gated[leak]        (4 specs)
27.88s  test_c_parity_linear_and_acts    (3 specs)
18.70s  test_residual_eps_parity
18.55s  test_c_parity
 3.01s  test_triton_parity_gated[gru]    (3 specs)   <- triton ~10x cheaper per config
```

~178 s of the 221 s sits in the six C-backend tests. `is_available()` adds a further **9.11 s**
one-time: it builds a probe extension to verify the toolchain.

### Scope

Measured on `ssm` only. `ren`, `r2dn`, `narx`, `phnn`, `dynonet` and the diagonal/selective
scan backends all call `load_inline` through the same `kernel_c` toolkit with the same
`_build_flags`, so the same cost structure applies to them (**inferred** — not separately
timed; the 118 cached extensions across all six architectures corroborate it).

---

## 2. Where the 9.3 s actually goes

Compiling translation units of increasing header weight with the project's own flags
(`-O3 -march=native -ffast-math -std=c++17 -fPIC -shared -fopenmp`), each containing only a
no-op function (**measured**):

| translation unit | compile time |
|---|---|
| no headers | **0.02 s** |
| `#include <ATen/Parallel.h>` | **0.21 s** |
| `#include <ATen/ATen.h>` + `<ATen/Parallel.h>` | **2.45 s** |
| `#include <torch/extension.h>` *(what the generator emits)* | **8.26 s** |
| the above **+ the real generated SSM kernel** | **8.45 s** |

The generated kernel body — the thing that actually differs per spec — costs about **0.19 s**.
Everything else is `torch/extension.h`, which drags in the full torch C++ frontend and
pybind11. The remaining ~0.9 s up to the observed 9.3 s is `load_inline` overhead: ninja
invocation, the generated pybind module glue, and linking (**inferred** from the 8.45 s direct
compile vs the 9.38 s end-to-end `load_inline`).

**This is the central fact: ~98 % of per-spec compile time is header parsing that has nothing
to do with the spec.**

---

## 3. Options evaluated

### 3.1 Rejected by measurement

**Lower the optimization level.** With `torch/extension.h` present (**measured**):

| | `-O3` | `-O2` | `-O1` | `-O0` |
|---|---|---|---|---|
| compile | 8.45 s | 8.21 s | 6.83 s | 6.73 s |

Only 1.7 s is available even at `-O0`, and it would be paid for in kernel throughput on a
backend whose entire reason to exist is CPU speed. Dead end.

**Precompiled header of `torch/extension.h`** (**measured**): 12.40 s to build the PCH,
**719 MB** on disk, after which each compile takes **4.27 s**. A 2× win for a 719 MB artifact
that must be rebuilt whenever torch, the compiler or the flags change. Poor trade next to the
option below.

**ccache.** Not measured. Caches identical translation units, but distinct specs *are*
distinct source — it would only help re-runs that `load_inline`'s own disk cache already
covers (**inferred**).

### 3.2 Drop `torch/extension.h` — pass a C ABI

The kernel never needs a `torch::Tensor`. It reads `data_ptr<float>()` and the two sizes, then
works exclusively on `float*`. Passing raw pointers plus `B`/`L` across an `extern "C"`
boundary and loading the result with `ctypes` removes pybind11 and the torch C++ frontend
entirely.

Prototyped by mechanically rewriting the **real** `_gen_source` output (`torch::Tensor` →
`float*`, drop `.data_ptr<float>()`, hoist `B`/`L` to parameters, `extern "C"` on the two
entry points) and compiling it with plain g++ (**measured**):

| | compile | forward vs shipped backend | runtime |
|---|---|---|---|
| current (`torch/extension.h`, pybind) | 8.45 s | — | 0.335 ms |
| **C ABI, no torch headers** | **0.19 s** | **0.000e+00 — bit-exact** | 0.325 ms |
| C ABI + `<ATen/Parallel.h>` only | 0.39 s | — | — |

**44× faster to compile, bit-identical output, same runtime.**

The middle row matters for portability. Raw `#pragma omp parallel for` is only correct because
this torch build uses OpenMP; on an `AT_PARALLEL_NATIVE` build it would spawn a second thread
pool alongside torch's and oversubscribe (**inferred** — not reproduced, no native-threadpool
build to hand). Keeping `#include <ATen/Parallel.h>` preserves `at::parallel_for` semantics and
`torch.set_num_threads()` honouring for **0.39 s** instead of 0.19 s — still **21×**. That is
the safer default; the macOS/GCD path is unaffected either way.

Costs to weigh: hand-written `ctypes` signatures replace pybind's type checking, so argument
count/order/dtype/contiguity become our invariant to hold rather than the binding layer's. A
mismatch is a segfault, not a `TypeError`. Mitigation is to build the `argtypes` list from the
same `spec` that generates the source, so both sides derive from one description.

### 3.3 Batch several specs into one extension

*This is the proposal you asked me to expand.* Today `_get_extension` emits one translation
unit per spec, each re-parsing the same headers:

```
spec A -> source A -> g++ (8.4s headers + 0.1s body) -> ext_A.so
spec B -> source B -> g++ (8.4s headers + 0.1s body) -> ext_B.so
```

The fixed header cost is paid per spec. Batching emits *N* specs into **one** translation unit,
with per-spec function names (`ssm_fwd_0`, `ssm_bwd_0`, `ssm_fwd_1`, …) and per-spec constants
(`NX_0`, `NU_0`, …), so one g++ run pays the header cost once:

```
specs A..H -> one source -> g++ (8.4s headers + 8 x 0.1s bodies) -> ext_ABCDEFGH.so
```

Measured on 8 genuinely distinct specs, separate TUs vs one combined TU:

| headers | N | separate (sum) | combined (1 TU) | speedup |
|---|---|---|---|---|
| `torch/extension.h` | 1 | 8.44 s | 8.41 s | 1.0× |
| | 2 | 16.77 s | 8.51 s | 2.0× |
| | 4 | 33.47 s | 8.66 s | 3.9× |
| | 8 | **67.70 s** | **9.22 s** | **7.3×** |
| C ABI | 1 | 0.18 s | 0.18 s | 1.0× |
| | 2 | 0.34 s | 0.21 s | 1.6× |
| | 4 | 0.69 s | 0.29 s | 2.4× |
| | 8 | **1.50 s** | **0.56 s** | **2.7×** |

**measured.** The two levers compose: 8 specs cost 67.70 s today and **0.56 s** with C ABI +
batching — **121×**.

The catch, and it is a real one: **batching needs the spec set known before the first compile.**
A `Learner` that meets one model at a time cannot batch — the second spec arrives after the
first extension is already built. Batching therefore pays off exactly where the set *is* known
in advance:

- the test suite (the parametrize lists are the catalogue);
- an AOT prebuild step (§3.4);
- a user script that constructs several models before training any of them, if we expose a
  `prebuild([specs])` entry point.

Note also the diminishing return in the C-ABI regime: once headers are gone the fixed cost is
small, so batching only amortises the ~0.13 s process/link overhead. **Batching is worth most
in the current header regime, and is a secondary optimisation once §3.2 lands** — which is an
argument for doing §3.2 first and treating batching as opportunistic.

### 3.4 Ahead-of-time compilation

Two sub-strategies, and the blocking question for both is whether specialization is worth
paying for at all.

**Is `-march=native` load-bearing?** No (**measured**, one spec, 1 thread): native 2.877 ms,
`x86-64-v3` 2.967 ms, `x86-64-v2` 3.010 ms, baseline `x86-64` 2.939 ms. Within run-to-run
noise. The current docstring's warning that a shared cache "must not be reused" across CPU
generations is therefore a constraint we could simply drop by targeting a portable baseline —
**which is what makes shipping prebuilt binaries feasible at all.**

**Is dim specialization load-bearing?** Partly. A hand-written generic kernel taking dims as
runtime arguments — one compiled copy serving every spec, **0.19 s** to build, once, ever —
measured against the specialized kernels across all 8 specs:

| spec (nx/nu/hidden/act) | specialized | generic | ratio |
|---|---|---|---|
| 4/3/(48,32)/tanh | 1.977 ms | 3.287 ms | 1.66× |
| 4/3/()/tanh | 0.053 ms | 0.225 ms | **4.25×** |
| 4/3/(24,)/sigmoid | 0.554 ms | 1.027 ms | 1.85× |
| 4/3/(16,)/relu | 0.298 ms | 0.711 ms | 2.39× |
| 8/3/(64,64)/tanh | 3.911 ms | 6.623 ms | 1.69× |
| 8/5/(32,)/tanh | 1.149 ms | 1.547 ms | 1.35× |
| 16/2/(128,)/relu | 4.432 ms | 4.733 ms | **1.07×** |
| 2/1/(8,8)/sigmoid | 0.889 ms | 0.937 ms | 1.05× |

1 thread: **median 1.69×**, range 1.05–4.25×. At 24 threads: **median 1.47×**, range
0.74–2.94×. Numerically equivalent throughout (max rel err 2.7e-07). Reproduced across two runs
with ratios stable to ±0.02× (**measured**).

> Absolute times are only comparable *within* one harness. The `-march` comparison above and
> this table were taken by two different drivers and disagree on the same spec's absolute cost
> (2.877 ms vs 1.977 ms for 4/3/(48,32)), most likely code layout or turbo state — unexplained,
> and not chased because both conclusions rest on within-harness ratios. Anyone re-timing these
> should compare inside a single process, not across the two tables.

Worth flagging honestly: a first single-spec measurement gave 1.20×, which would have been a
misleadingly rosy headline. The spread only appeared at n=8. The penalty is worst exactly where
the kernel is cheapest — the tiny `hidden=()` spec, where specialization collapses the whole
rollout to straight-line code — and nearly vanishes on wide layers where real FLOPs dominate.
The generic prototype is also unoptimised (per-batch `std::vector` scratch, no blocking); a
tuned one would close part of the gap (**assumed**, not measured).

**Strategy A — generic AOT kernel only.** Compile one portable generic kernel at wheel-build
time. JIT disappears completely; cold start becomes zero. Costs ~1.5–1.7× median CPU
throughput, worst case ~4× on trivially small models.

**Strategy B — prebuilt catalogue + JIT fallback.** Ship specialized kernels for a catalogue of
common specs, built at wheel-build time against `x86-64-v2`/`v3` (portable at no measured
throughput cost), and JIT anything outside it. Full speed for the common path, but the
catalogue must be chosen, versioned, and kept honest, and it inflates the wheel.

**Strategy C — tiered, generic-first.** Serve the generic AOT kernel immediately on first call,
kick off the specialized build in a background thread, and swap the pointer in when it lands.
No cold-start stall, no throughput loss once warm, no catalogue to curate. Costs a swap
mechanism and makes the first N steps of a run non-deterministic in timing (not in numerics —
the two kernels agree to 2.7e-07). **This looks like the best trade** but is the most
implementation work, and the swap needs care around the autograd boundary.

---

## 4. Recommendation

Ordered by measured payoff per unit of work.

1. **Replace `torch/extension.h` with a C ABI + `ctypes` in `kernel_c`** (§3.2) — done for
   `ssm` (§5); §9 carries it to the rest. 21–44× on
   every generated backend, bit-exact, no runtime cost, and it is a change to one shared
   toolkit plus the six generators' signature emission. Prefer the `<ATen/Parallel.h>` variant
   (0.39 s) over the raw-OpenMP one (0.19 s) to keep `at::parallel_for` semantics.
2. **Compile the generic kernel AOT and serve it first** (§3.4 Strategy C), or ship Strategy A
   if the median 1.5–1.7× is acceptable for the CPU backend's use cases. This is what actually
   removes JIT from the critical path rather than just making it cheaper.
3. **Batch specs where the set is known** (§3.3) — chiefly a session-scoped prebuild in the test
   suite. Worth 7.3× today; worth ~2.7× after step 1, so it is opportunistic, not urgent.
4. **Drop `-march=native` for a portable baseline** (§3.4). Costs nothing measurable, and it is
   a precondition for any prebuilt-binary strategy.

### Test-suite-specific, independent of the above

- Warm caches are already a 36× lever — CI should persist `~/.cache/torch_extensions` between
  runs, and it should not be cleared locally without reason.
- `pytest-xdist` is **not** installed and the suite runs serially on 32 cores; the warm suite is
  416 s / 874 tests. Parallelising needs a check for GPU-test safety and for concurrent
  `load_inline` writes to the same cache directory first.
- Two other warm-suite hotspots, out of scope here but worth separate tickets: `test_onnx.py`
  at ~93 s across 8 tests, and the diagonal-scan triton tests at 107.6 s across 28 tests
  (triton compiles ~1.9 s per kernel and, unlike the C path, its disk cache does **not**
  eliminate the cost across processes — a fresh process pays 3.93 s again with a warm 4.9 GB
  `~/.triton/cache`) (**measured**).

---

## 5. Prototype: `ssm` on the C ABI

Implemented. `kernel_c` gained `load_cabi`, which builds a generated translation unit with the
host compiler and loads it through `ctypes`; the `ssm` generator emits `extern "C"` entry points
over `float*` with `B`/`L` as explicit parameters, and `_run_fwd`/`backward` marshal storage
pointers instead of tensors. `<ATen/Parallel.h>` stays, so `batch_parallel` is unchanged.

Results (**measured**):

| | before | after |
|---|---|---|
| `test_statespace.py`, cold caches | 223.6 s | **35.0 s** |
| `test_statespace.py`, warm | 6.06 s | **5.80 s** |
| `test_c_parity_gated[gru]`, cold | 37.57 s | **1.93 s** |
| `test_c_parity_linear_and_acts`, cold | 27.88 s | **1.27 s** |
| `is_available()` probe | 9.11 s | **0.29 s** |
| kernel cache on disk | 96 MB / 118 dirs | **664 KB / 22 objects** |
| `forward_train`, B=32 L=512 hidden=(48,32) | 0.335 ms | 0.324 ms |

Full suite: **874 passed, 7 skipped** — identical to the baseline count, in 414.9 s vs 416.1 s.
`torch.set_num_threads()` still governs the batch split (14.57 ms at 1 thread → 1.73 ms at 24),
confirming `at::parallel_for` semantics survived the ABI change.

What the cold 35 s now consists of is no longer the C backend: the top entries are
`test_compiled_parity` at 11.1 s (inductor) and the triton parity tests at ~3 s each.

Two consequences worth noting. The old `~/.cache/torch_extensions` entries for `ssm` (88 dirs,
the bulk of 96 MB) are now dead and can be deleted. And `ctypes` performs no type checking at
the boundary, so a wrong dtype, device or stride is a bad read rather than an exception —
`supports()` screening for contiguous float32 CPU input is now load-bearing in a way it was not
when pybind would have raised.

## 6. Transfer to the other C backends

The decisive structural fact is that **only `ssm` and `narx` generate spec-specialized source.**
`ren`, `phnn`, `diagonal_c` and `selective_c` take no spec at all — their dims are runtime
arguments, so they build once, not once per configuration. The build counts cached on this host
measure it directly (**measured**): `ssm` 88, `narx` 6, `diagonal` 3, `ren` 1, `phnn` 1,
`selective` 1.

So `ssm` was 88 of ~100 builds. The remaining five are worth ~45 s once on a fully cold cache,
plus whatever `narx` accumulates across distinct specs — a real prize, but a much smaller one.

Compiling each backend's real generated source, and decomposing against the 8.20 s
`torch/extension.h` cost measured on this host:

| backend | per-spec source | builds | current | body | minimal headers | C ABI | speedup |
|---|---|---|---|---|---|---|---|
| `ssm` | yes | 88 | 9.38 s | — | Parallel 0.20 | **0.45 s** ✔measured | **21×** |
| `narx` | yes | 6 | 8.43 s | 0.23 s | Parallel 0.20 | **0.43 s** ✔measured | **20×** |
| `selective_c` | no | 1 | 8.41 s | 0.21 s | Parallel 0.20 | ~0.41 s *inferred* | ~21× |
| `diagonal_c` | no | 3 | 9.35 s | 1.15 s | Parallel + complex 0.26 | ~1.6 s *inferred* | ~6× |
| `ren` | no | 1 | 9.16 s | 0.96 s | Parallel + Dispatch 0.40 | ~1.4 s *inferred* | ~7× |
| `phnn` | no | 1 | 10.02 s | 1.82 s | Parallel + Dispatch 0.40 | ~2.2 s *inferred* | ~4.5× |

`narx` was converted mechanically — same regex rewrite that worked on `ssm` — and landed at
0.43 s against a 0.43 s projection, which is what licenses the inferred rows. The rows are
inferred by linear decomposition (minimal header + measured body); they assume the body cost is
header-independent, which held exactly for the two cases tested.

### What each backend needs

All five use `torch::Tensor` only for `data_ptr` and `.size()` — none reaches for tensor
allocation, `TORCH_CHECK`, or ATen ops inside the kernel. There is no architectural obstacle
anywhere. The differences are in marshalling surface:

- **`narx`, `selective_c`** — structurally identical to `ssm`: `data_ptr<float>`, two or three
  sizes to hoist into parameters. `narx` already passes `int64_t washout` as a scalar, which
  `ctypes` handles unchanged. Mechanical.
- **`diagonal_c`** — already the closest to a C ABI: it `reinterpret_cast`s everything to
  `float*` internally and passes `M`, `L`, `N`, `has_x0`, `is_complex` explicitly. The
  `using cf = c10::complex<float>` alias exists only to spell `data_ptr<cf>()` and disappears
  with the tensors; `<c10/util/complex.h>` costs 0.26 s if it is wanted anyway.
- **`ren`, `phnn`** — the only ones with real work. Both dispatch on dtype via
  `AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), …)`, and with no tensor there is nothing to ask
  for a scalar type. **`<ATen/Dispatch.h>` costs 0.39 s, not 8 s** (**measured**), so the macro
  can simply be kept with the type code passed in as a parameter; alternatively both
  instantiations can be emitted and selected by an int flag, dropping the header entirely. The
  kernel bodies are already templated on `scalar_t`, so neither route touches the maths.
  `phnn` additionally passes `double` scalars and optional tensors
  (`ow.defined() ? … : nullptr`), which become `c_double` and a NULL pointer.

### Suggested order

1. **`narx`** — the only remaining backend whose cost recurs with spec count, mechanical, and
   the 20× is already measured rather than projected.
2. **`selective_c`, then `diagonal_c`** — mechanical, one-time ~8 s and ~9 s.
3. **`ren`, `phnn`** — largest diff for the smallest recurring benefit; worth doing for
   consistency and to retire `load_inline` (and with it the ninja requirement in `_probe`)
   rather than for the seconds.

Retiring `load_inline` everywhere is what lets the ninja check come out of `is_available`,
which is currently kept only because the un-migrated backends need it.

## 7. Would specialization help the other backends?

`ssm` and `narx` bake dims into the source; `ren`, `phnn`, `diagonal_c` and `selective_c` take
them at runtime. Since the C ABI makes a build ~20× cheaper, specializing the others is now
affordable where it was not — so the question is whether it would buy anything.

Mostly **no**. The measured answer does not follow the structural intuition.

| backend | dominant inner structure | specialization value |
|---|---|---|
| `ssm` | MLP layers, widths fixed per spec | **1.69× median, 1 thread** (§3.4) — already specialized |
| `narx` | MLP over the lagged window | same family as `ssm` — already specialized |
| `ren` | triangular solve, O(nv²) | **1.02× median** ✔measured on the real kernel |
| `diagonal_c` | `template <bool C, bool STORE, int KB>` + `if constexpr` | already compile-time specialized internally |
| `selective_c` | elementwise recurrence, `TILE = 64`, fixed-size stack arrays | little left to specialize |
| `phnn` | O(n³) `JR = Am·Amᵀ`, n runtime | **non-monotonic, 2.15× faster to 3× slower** — unreliable |

**`ren` (measured, real kernel).** Its `fwd_impl` already takes `nx/nu/ny/nv` as parameters, so
converting them to template parameters is a two-line change. Forward pass, 1 thread, B=32 L=512,
generic vs specialized: 1.17× at nx/nu/ny/nv = 8/2/2/8, and 1.01–1.02× at 8/2/2/32, 8/2/2/128
and 16/2/2/64. Outputs agree to 1e-07.

The reason is structural: the dominant term is the strictly-lower-triangular forward
substitution, `dot(D11 + i*nv, w, i)`, whose trip count is `i` — it varies per neuron and stays
runtime-valued no matter what `nv` is fixed to. Specializing the *other* dots leaves the hot
loop untouched. Only at nv=8, where the fixed-width dots are a larger share, does anything show.

> A first version of this A/B passed literal dims into the generic call, letting GCC
> constant-fold them and specialize both arms. The corrected version passes dims in as
> parameters of the `extern "C"` entry point, where nothing in the translation unit can fold
> them. The corrected numbers are the ones above; the flawed run gave 1.00–1.05×, so the
> conclusion did not change, but the first experiment did not test what it claimed to.

**`phnn` (measured, microbenchmark — treat as a lead, not a verdict).** Its dominant per-step
term is O(n³) with runtime `n`. Isolating that loop and comparing runtime-`n` against
compile-time-`n`, min of 5 runs, ns per evaluation:

| n | runtime-n | compile-time-n | ratio |
|---|---|---|---|
| 4 | 18.1 | 8.4 | **2.15×** |
| 8 | 51.4 | 97.3 | **0.53×** |
| 16 | 320.1 | 955.0 | **0.34×** |
| 32 | 1564.5 | 982.7 | **1.59×** |

Reproducible to the tenth of a nanosecond across runs, with a dependency chain preventing
hoisting and dead-code elimination, and outputs agreeing to 1e-08. Compile-time `n` is
genuinely ~3× *slower* at n=16.

The mechanism is unresolved. The obvious hypothesis — GCC over-unrolls at a known trip count —
is supported by code size (`jr_ct_16` is 5366 bytes against 232 for `jr_ct_32`, where GCC
evidently stops unrolling) but **falsified by the direct test**: `-fno-unroll-loops` left the
timing unchanged at 954 ns. That flag may not disable complete unrolling of known-small loops,
so the hypothesis is not cleanly dead either; it is simply untested. This is one extracted loop,
not the kernel in situ, so the honest reading is narrow: **do not assume specializing `phnn`
helps.** It would need per-`n` measurement on the real kernel, and blanket specialization could
regress it.

The practical consequence is that the C-ABI work stands on its own as a compile-time win, and
specialization is not a follow-on prize for the remaining backends — with the possible exception
of `phnn` at particular `n`, which is a measurement job rather than a design decision.

## 8. Defects found in the prototype by adversarial review, and their fixes

An adversarial pass over the C-ABI change found four real defects, two of them serious. All
four are fixed; each was reproduced before and after.

**1. `-ffast-math` on the link line flipped the whole process to flush-to-zero.** The build ran
as a single compile-and-link command, so `-ffast-math` reached the link step, where GCC links
`crtfastmath.o` — whose ELF constructor sets FTZ/DAZ in MXCSR when the object is `dlopen`ed.
`load_inline` never did this because ninja compiles and links in separate rules. The blast
radius was far wider than the SSM backend: `is_available()` alone builds and loads the probe,
and `DeepLRU`/`DeepMamba` reach it through `diagonal_c.supports()` with stock arguments and no
`backend="c"` anywhere — including on a `supports()` call that *declines*. After that, every
torch operation in the process flushed denormals, permanently and silently.

```
before: 1e-40 stays 1.00e-40
after is_available(): 1e-40 stays 0.00e+00
```

Fixed by splitting compile and link so `-ffast-math` stays off the link line. Verified: the
constructor is gone from the cached objects, denormals survive `is_available()`, and the kernel
output checksum is unchanged (297.500854), so fast-math codegen in the body was not lost.

**2. Unchecked dtype at the ABI boundary — heap over-read where pybind raised.**
`rollout_unsupported` screens `u` and `x0` only; `params` and `leak` were never checked, and
`backward` does not pass through `supports()` at all. pybind's `data_ptr<float>()` used to
throw on a wrong dtype. Under the C ABI, float64 params produced silent NaN, and any dtype
narrower than 4 bytes made the kernel read twice its allocation — the reviewer reproduced a
SIGSEGV with fp16 params at `hidden=(1<<21,)`. Fixed by validating dtype, device and
contiguity in `_call`, which runs once per rollout rather than per step; the cost is ~2 %
(0.324 → 0.330 ms on the reference case).

**3. The shared probe stopped covering the toolchain the other backends need.** The C-ABI probe
includes only `<ATen/Parallel.h>`, but `is_available()` also gates `diagonal_c`,
`selective_c`, `phnn` and `narx`, which still build through `load_inline` and therefore need
ninja and `Python.h`. On a Python without development headers the probe now passed and those
backends' failure moved from a graceful decline to a compile error thrown out of `forward`.
Fixed by gating on ninja plus the presence of `Python.h` — checked by file existence rather
than by compiling, which would cost back the seconds `load_cabi` exists to save. Only the
positive case is tested here; the negative was demonstrated by the reviewer at the compiler
level, not end to end.

**4. A corrupt cached object was permanent.** `if not path.exists()` then `CDLL`, with no
verification and no rebuild, so a truncated `.so` — interrupted write, partially copied cache —
failed for every future process, and via `is_available()` disabled the whole C family with a
warning. Fixed by catching `OSError`, unlinking and rebuilding once; the object is also
`fsync`ed before the rename, so the atomic rename now stands on durable content. Verified
across processes (build → truncate to 14 bytes → load recovers).

Also hardened: the cache key now folds in the compiler identity and version, `torch.__version__`,
`platform.machine()` and `sys.platform`. Previously a gcc upgrade, or a PATH change swapping
`c++` from gcc to clang, would silently reuse the old object.

### Accepted, not fixed

- **`-march=native` cannot be hashed.** It enters the key as the literal string, identical on
  every host, so two CPU generations sharing `~/.cache` over NFS collide and the loser takes
  SIGILL — which `_probe`'s `except Exception` cannot catch, so the process dies. Pre-existing
  with `load_inline`, not a regression, and not fixable by hashing; §3.4 argues for dropping
  `-march=native` anyway, which would also close this.
- **`ctypes` releases the GIL where pybind11 held it.** Two Python threads can now enter
  `at::parallel_for` concurrently, each opening its own OpenMP team, so a multi-threaded
  inference server could oversubscribe. Correctness is unaffected (ATen's `in_parallel_region`
  guard); throughput may not be. Reasoned, not tested.
- **Raw `PermissionError` from a read-only cache directory** escapes `_get_extension`
  unannotated instead of surfacing as the `RuntimeError("kernel build failed…")` the
  surrounding code implies, and `TemporaryDirectory` leaks a `tmpXXXX` inside the cache on
  SIGKILL.

### Checked and clean

Argument lifetime (every tensor stays bound in the calling frame, and `_call`'s list parameter
holds refs across the call); argument count and order, derived independently on both sides for
all 4 gates × `hidden ∈ {(), (8,), (8,6)}` — 24 signatures, all matching, numerics within
2.4e-7 of eager; `storage_offset` (`data_ptr()` accounts for it); concurrent builds (12 racing
processes produced one object and no leftover temp directories); `restype` on `void` entry
points; rpath resolution and the absence of a duplicate OpenMP runtime.

## 9. Roadmap: retiring `load_inline` and the ninja dependency

The goal is not dependency hygiene. Today a C backend needs **ninja and `Python.h`** on top of a
compiler, because `load_inline` builds a pybind11 module; on any host missing either — slim
containers, a system Python without `python3-dev` — every generated-C++ backend silently
declines. After migration they need only a C++ compiler. The build-time savings of §6 come
along, but the availability change is the larger prize.

Ninja cannot simply be dropped first. `load_inline` calls `verify_ninja_availability()`
unconditionally inside `_write_ninja_file_and_build_library`, confirmed by running it with ninja
off `PATH` (**measured**):

```
with ninja:     load_inline: OK
without ninja:  RuntimeError: Ninja is required to load C++ extensions
```

So the dependency comes out only after the last backend leaves `load_inline`.

### Surface per backend

| backend | `AT_DISPATCH` sites | optional tensors | `.size()` to hoist | notes |
|---|---|---|---|---|
| `diagonal_c` | 0 | 0 | **0** | already passes `M, L, N` and its flags explicitly; `using cf = c10::complex<float>` disappears with the tensors |
| `narx` | 0 | 0 | 4 | conversion already prototyped and measured at 0.43 s (§6) |
| `selective_c` | 0 | 0 | 6 | same shape as `ssm` |
| `ren` | 2 | 0 | 11 | dtype dispatch; bodies already templated on `scalar_t` |
| `phnn` | 2 | 4 | 4 | dtype dispatch, optional tensors, `double` scalar params |

### Order, one commit each

1. **`diagonal_c`** — least work of all (no sizes to hoist), and exercises the complex path.
2. **`narx`** — the only remaining backend whose build cost recurs with spec count, and its 20×
   is measured rather than projected.
3. **`selective_c`** — mechanical.
4. **`ren`** — first backend needing dtype dispatch. Either keep `AT_DISPATCH_FLOATING_TYPES`
   and pass the scalar type in as a code (`<ATen/Dispatch.h>` costs 0.39 s), or emit both
   `float`/`double` instantiations and select on an int flag. Neither touches the maths.
5. **`phnn`** — same dispatch question plus NULL pointers for the optional tensors and
   `c_double` for the scalar params.

### The invariant every step must hold

A C ABI cannot reject a wrong dtype, and the consequence is memory unsafety rather than a bad
number: a dtype narrower than the kernel assumes makes it read past the end of the allocation
(§8, defect 2). **Every migrated backend needs its own boundary validation before its first
call**, on the pattern of `ssm`'s `_call` — dtype, device and contiguity, once per rollout, not
per step. Costed at ~2 % on the `ssm` reference case. Screening in `supports()` is not
sufficient: it does not see the parameters, and `backward` does not pass through it.

Each step verifies with the full suite plus a cold-cache run of that backend's own test file,
compared against the numbers in §6.

### Cleanup once the last one lands

- Delete the ninja and `Python.h` gate from `_probe`; `is_available` becomes a compiler check
  plus the `load_cabi` probe.
- Remove `cpp = ["ninja"]` from `pyproject.toml` entirely and drop `ninja` from `dev`.
- Reword the 11 test skip reasons that read `"no C++ toolchain / ninja"`.
- Delete the stale `~/.cache/torch_extensions` entries (96 MB on this host); nothing writes
  there any more.

### Risks carried through the migration

- **Windows.** `load_inline` has a Windows branch (`.lib`, `/LIBPATH:`) that `load_cabi` lacks.
  Currently moot — `_compiler()` looks for `c++`/`g++`, so the C backends are already
  unavailable under MSVC — but the migration removes a latent path rather than a live one.
- **macOS.** Lower risk than it looks: torch's own `_prepare_ldflags` uses the same
  non-Windows branch for Linux and macOS (`-L{TORCH_LIB_PATH} -lc10 -ltorch_cpu
  -Wl,-rpath,{TORCH_LIB_PATH}`), which `_torch_flags` already mirrors. Untested here — no macOS
  host available — so it stays untested rather than verified.
- The coupling to torch is thin and worth keeping that way: the built kernels import exactly
  seven symbols — `at::get_num_threads`, `get_thread_num`, `init_num_threads`,
  `in_parallel_region`, `internal::set_thread_num`, and `c10::ParallelGuard`'s constructor and
  destructor. All ATen threading. If a future backend needs more than that, the cheap-header
  premise should be re-measured rather than assumed.

## 10. Open questions

- What does the spec distribution look like in real use? Strategy B's viability depends on
  whether a small catalogue covers most models — currently unknown. The cached build counts in
  §6 are a test-suite artefact, not a usage measurement.
- Is `supports()` a tight enough screen now that `ctypes` will not catch a bad tensor? Worth a
  deliberate look at every path reaching `forward_train`/`backward` from outside the custom op.
- Why is compile-time `n` ~3× slower than runtime `n` for `phnn`'s O(n³) term at n=16 (§7)? The
  over-unrolling explanation survives a code-size check but not a `-fno-unroll-loops` test.
  Worth resolving before anyone specializes that kernel — and worth checking whether the same
  trap applies at any of `ssm`'s widths, where specialization is currently assumed good.
- Does the generic kernel close the gap when tuned (preallocated scratch, blocked GEMV)? If it
  reaches ~1.2× median, Strategy A becomes clearly sufficient and Strategy C's complexity is
  unnecessary.
- What is the equivalent story for the triton backends? The 3.93 s per-process cost above
  suggests a separate, unrelated caching problem.

## 11. Reproducing

```bash
# the 36x cold/warm gap
mkdir -p /tmp/cold/{xdg,ext,ind,triton}
XDG_CACHE_HOME=/tmp/cold/xdg TORCH_EXTENSIONS_DIR=/tmp/cold/ext \
TORCHINDUCTOR_CACHE_DIR=/tmp/cold/ind TRITON_CACHE_DIR=/tmp/cold/triton \
  uv run pytest tests/test_statespace.py -q --durations=20   # 223s, then 6s on re-run

# the header decomposition
INC=$(uv run python -c "from torch.utils.cpp_extension import include_paths; import sysconfig; \
  print(' '.join('-I'+p for p in include_paths()), '-I'+sysconfig.get_paths()['include'])")
FLAGS="-O3 -march=native -ffast-math -std=c++17 -fPIC -shared"
printf '#include <torch/extension.h>\nint f(){return 0;}\n' > /tmp/h.cpp
printf '#include <ATen/Parallel.h>\nint f(){return 0;}\n'   > /tmp/a.cpp
time g++ $FLAGS $INC /tmp/h.cpp -o /tmp/h.so    # 8.42 s
time g++ $FLAGS $INC /tmp/a.cpp -o /tmp/a.so    # 0.20 s
```
