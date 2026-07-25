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
| `compare_dynonet.py` | `tsfast.models.architectures.dynonet` | authors' `dynonet` package (PyPI) |
| `compare_lru.py` | `tsfast.models.architectures.lru` | Forgione et al., `lru-reduction` (transcribed, MIT) |
| `compare_s5.py` | `tsfast.models.architectures.s5` | official JAX S5 (transcribed, MIT) + `s5-pytorch` (PyPI) |
| `compare_mamba.py` | `tsfast.models.architectures.mamba` | official `selective_scan_ref` (transcribed, Apache-2.0) + `mambapy` (PyPI) |
| `compare_subnet.py` | `tsfast.models.architectures.subnet` | deepSI `SS_encoder_general` (transcribed, BSD-3-Clause) |
| `compare_phnn.py` | `tsfast.models.architectures.phnn` | OE-pHNN paper equations, autograd-based (authors' repo is unlicensed; formulation separately verified against their released trained models) |
| `compare_ren.py` | `tsfast.models.architectures.ren` | REN paper equations in implicit form (per-step solves instead of the explicit realization), plus the authors' JAX `robustnn` in an optional section |
| `compare_r2dn.py` | `tsfast.models.architectures.ren.r2dn` | R2DN paper equations in implicit form, the dissipation inequality itself as an eigenvalue, plus the authors' JAX `robustnn` for the contracting case in an optional section |
| `compare_transformer.py` | `tsfast.models.architectures.transformer` | Rufolo et al., `sysid-prob-transformer` (unlicensed, so downloaded at run time from a pinned commit rather than transcribed) |

Run from the repository root:

```bash
uv pip install dynonet s5-pytorch mambapy   # reference packages (optional sections skip if absent)
uv pip install jax flax "robustnn @ git+https://github.com/nic-barbara/R2DN"   # for compare_ren.py, compare_r2dn.py
uv run python comparisons/compare_dynonet.py
uv run python comparisons/compare_lru.py
uv run python comparisons/compare_s5.py
uv run python comparisons/compare_mamba.py
uv run python comparisons/compare_subnet.py
uv run python comparisons/compare_phnn.py
uv run python comparisons/compare_ren.py
uv run python comparisons/compare_r2dn.py
uv run python comparisons/compare_transformer.py   # downloads the reference module (network required)
```

Every script exits non-zero if any deviation exceeds its stated tolerance.

## Deliberate divergences from a reference

Two, both in the R2DN family, both recorded here because a comparison that silently works
around a difference is worse than one that names it.

**The LBDN's last layer.** The released `robustnn` never sets the flag that would make its
final layer the norm-bounded linear map of the network it cites, so its stack carries one
extra nonlinear layer. tsfast follows the ICML'23 definition and `acfr/RobustNeuralNetworks.jl`;
`compare_r2dn.py` composes the JAX section's network from the reference's own `SandwichLayer`
so that no formula goes unchecked despite the difference.

**A slack margin on the Lipschitz feedthroughs.** `R2DNParameterization._bounded` holds its
Cayley contractions a further `_CONTRACTION_SLACK` inside the unit ball, which neither the
JAX nor the Julia reference does. The reason is numerical rather than aesthetic: the
construction inverts `gamma*I - D12ᵀD12`, and that matrix *is* the slack the Cayley transform
leaves, `4(I+m)⁻ᵀ(XᵀX + eps*I)(I+m)⁻¹` — at best `~4*eps`, shrinking quadratically as the free
skew part of `m` grows. Training walks straight into it, because consuming the gain budget is
useful, and in float32 the matrix rounds to singular (observed after ~20 epochs as a
`linalg.solve` failure). The certificate only ever needs the bound, never equality, so a
shrunken feedthrough certifies the same `gamma`.

Worth knowing when reading the references: the Lipschitz R2DN is Prop. 2 of the paper and
nothing more — `robustnn/r2dn.py` implements only `ContractingR2DN`, and none of the paper's
three experiments prescribes a `gamma`. The same latent degeneracy sits in the *REN* Lipschitz
path of all three codebases, where `R = gamma*(I - NᵀN)` is inverted the same way; tsfast
leaves that one untouched so the transplant in `compare_ren.py` keeps measuring agreement
rather than a deviation.
