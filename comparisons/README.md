# Reference-implementation comparisons

Each script in this directory validates one tsfast model family numerically against the
implementation published by the original authors (or the closest available reference).
They are documentation artifacts, not tests: run one and it prints the exact maximum
relative deviations of outputs and gradients across a range of configurations, so the
agreement claim is reproducible at a glance. The pytest suite (`tests/`) contains
tighter, faster regression versions of the same checks.

All comparisons run in float64 with identical parameters copied into both
implementations, so the printed deviations measure algorithmic agreement, not
initialization or precision noise (the one exception is noted inside `compare_mamba.py`).

| Script | tsfast model | Reference |
|---|---|---|
| `compare_dynonet.py` | `tsfast.models.dynonet` | authors' `dynonet` package (PyPI) |
| `compare_lru.py` | `tsfast.models.lru` | Forgione et al., `lru-reduction` (transcribed, MIT) |
| `compare_s5.py` | `tsfast.models.s5` | official JAX S5 (transcribed, MIT) + `s5-pytorch` (PyPI) |
| `compare_mamba.py` | `tsfast.models.mamba` | official `selective_scan_ref` (transcribed, Apache-2.0) + `mambapy` (PyPI) |

Run from the repository root:

```bash
uv pip install dynonet s5-pytorch mambapy   # reference packages (optional sections skip if absent)
uv run python comparisons/compare_dynonet.py
uv run python comparisons/compare_lru.py
uv run python comparisons/compare_s5.py
uv run python comparisons/compare_mamba.py
```

Every script exits non-zero if any deviation exceeds its stated tolerance.
