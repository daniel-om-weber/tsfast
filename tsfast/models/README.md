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
    dispatch.py                    once-per-process backend-fallback warning
    scan.py                        scan-recurrence library (dispatcher + reference implementations)
    scan_backends/                 kernels reachable through scan.py's dispatcher (shared by >1 model)
      diagonal_{c,triton}.py         used by lru + s5
      selective_{c,triton}.py        routed through the shared dispatcher
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

## Where does new code go?

- **A new architecture** → `architectures/`. One `.py` if it has no private kernels; a package
  (`core.py` + kernels) if it does.
- **A compute backend for one model** → inside that model's package, named `backend_<impl>.py`
  (or `<op>_<impl>.py` when the model dispatches several ops directly, as mamba does).
- **A kernel used by two or more models** → `_core/scan_backends/` if it's a scan op routed through
  `scan.py`'s dispatcher; otherwise a shared module under `_core/`.
- **A primitive every backend needs** (an activation macro, the toolchain probe, a padding helper)
  → `_core/kernel_c.py` / `_core/kernel_triton.py`.

## The rule that keeps `_core` honest

`_core` never imports from `architectures`. A kernel is "shared" only if it is reached through a
`_core` dispatcher (`scan.py`) or is a `_core` primitive — those live in `_core`. A kernel a single
model reaches for directly (`importlib`/`from .`) is *private* and lives in that model's package,
even if it is a `triton`/`c` kernel. That line is what stopped the shared `scan_backends/` folder
from accreting single-model kernels.
