# `tsfast.models` layout

Two zones plus a stable facade. The public import surface is flat — `from tsfast.models
import Mamba, SimpleRNN, StandardScaler` — regardless of where a symbol physically lives;
`__init__.py` re-exports everything. The directory split is for *readers and maintainers*.

```
models/
  __init__.py          facade: re-exports the public API (stable; internal moves don't change it)
  _core/               shared machinery — rarely opened; nothing here imports an architecture
    layers.py scaling.py           sequence layers, scalers  (public, surfaced via the facade)
    state.py cudagraph.py          stateful-model helpers
    dispatch.py                    backend preference (set_backend/use_backend) + fused-kernel
                                     resolver + once-per-process fallback warning
    scan.py                        scan-recurrence library (tsfast::scan_* custom ops + the
                                     pure-PyTorch doubling reference)
    scan_backends/                 kernels reachable through scan.py's ops (shared by >1 model)
      diagonal_{c,triton}.py         used by lru + s5
      selective_{c,triton}.py        used by mamba's generic path
    kernel_c.py kernel_triton.py   shared C/Triton primitives (activation macros, toolchain probe,
                                     padding helpers) used by every generated backend
  architectures/       the models a user instantiates — one item each
    rnn.py cnn.py transformer.py subnet.py lru.py s5.py     single-file models
    mamba/    core.py + conv_triton.py + mamba_triton.py    model + its private fused kernels
    dynonet/  core.py + allpole_triton.py
    narx/     core.py + backend_{c,triton}.py
    ssm/      core.py + backend_{c,triton,metal}.py
    phnn/     core.py + backend_{c,triton}.py + common.py
```

## Backend selection

One knob, three scopes:

- **Process default**: `tsfast.models.set_backend("auto" | "triton" | "c" | "metal" |
  "reference")`. `"auto"` picks the fastest usable fused kernel for the input's device;
  `"reference"` disables fused kernels everywhere (pure-PyTorch paths only); an explicit
  family forces it where an op has such a kernel, with a once-per-process warning (and
  reference fallback) when it is unusable. A family an op simply doesn't have selects the
  reference path silently.
- **Scoped**: `with tsfast.models.use_backend("reference"): ...` — contextvar-based,
  thread- and async-safe, nests.
- **Per model instance**: the `backend` attribute (e.g. `NeuralStateSpace(backend="c")`,
  Mamba/dynonet `"scan"`/`"eager"`). An instance set to `"auto"` defers to the process
  preference; an explicit instance value overrides it.

## Kernel integration (`torch.library`)

Every fused execution path is a registered custom op (`tsfast::scan_diagonal`,
`tsfast::mamba_scan`, `tsfast::ssm_rollout`, ...) with a paired `_bwd` op, a fake impl,
and `register_autograd`. Consequences:

- `torch.compile(model, fullgraph=True)` works through every fused kernel — the op is
  opaque to Inductor, no graph breaks, and the backward is compile-clean too.
- Eager/compiled reference paths stay *outside* the ops so the compiler can optimize them.
- Backend modules expose a uniform protocol instead of `autograd.Function`s:
  `supports(...) -> str | None` (a reason when unusable, consumed by `dispatch.resolve`)
  plus raw `forward`/`backward` kernel entry points called from inside the ops.
- Op impls must `.contiguous()` every tensor they hand to raw kernels: op inputs (and
  tensors replayed from `setup_context` in backward) can be stride-0 broadcast views.
- Broadcasting/flattening happens in thin wrappers *outside* the ops, so plain autograd
  reduces lane gradients back to broadcast shapes — backends never `sum_to_size`.

## Where does new code go?

- **A new architecture** → `architectures/`. One `.py` if it has no private kernels; a package
  (`core.py` + kernels) if it does.
- **A compute backend for one model** → inside that model's package, named `backend_<impl>.py`
  (or `<op>_<impl>.py` when the model dispatches several ops directly, as mamba does).
- **A kernel used by two or more models** → `_core/scan_backends/` if it's a scan op routed through
  `scan.py`'s ops; otherwise a shared module under `_core/`.
- **A primitive every backend needs** (an activation macro, the toolchain probe, a padding helper)
  → `_core/kernel_c.py` / `_core/kernel_triton.py`.

## The rule that keeps `_core` honest

`_core` never imports from `architectures`. A kernel is "shared" only if it is reached through a
`_core` dispatcher (`scan.py`) or is a `_core` primitive — those live in `_core`. A kernel a single
model reaches for directly (through its own registry entry) is *private* and lives in that model's
package, even if it is a `triton`/`c` kernel. That line is what stopped the shared `scan_backends/`
folder from accreting single-model kernels.
