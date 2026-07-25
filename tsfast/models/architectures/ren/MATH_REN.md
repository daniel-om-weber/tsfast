# REN fused rollout: forward and BPTT math

This is the correctness contract for the fused C++ and Triton rollout backends of
`tsfast.models.architectures.ren.RENCore`. It derives the per-step update and its full
backpropagation-through-time (BPTT) gradient.

It is much shorter than `MATH.md` because of where the model was cut. The certificate
construction — `H = XᵀX + εI`, its partition, the `E⁻¹`/`Λ⁻¹` solves — runs once per
forward in ordinary autograd and produces the **explicit realization**: twelve plain
tensors. The kernels see only those. A backend that gets `∂L/∂A` right needs no opinion
about `∂L/∂X`; autograd carries the rest. There is consequently no analogue of `MATH.md`'s
second-order Hamiltonian term here — the recurrence is linear plus one elementwise
nonlinearity.

Symbols: state `x ∈ R^nx`, input `u ∈ R^nu`, output `y ∈ R^ny`, equilibrium-layer
activations `w ∈ R^nv`. Sequence length `L`, batch `B`. The activation `σ` is monotone and
slope-restricted to `[0,1]` (`tanh`, `relu`, `sigmoid`).

## 1. Forward

Explicit parameters `A, B1, B2, C1, D11, D12, C2, D21, D22, bx, bv, by`, with `D11`
**strictly lower triangular**. For `t = 0 .. L-1`, starting from `x_0`:

    b_t     = C1 x_t + D12 u_t + bv                              in R^nv
    v_{t,i} = b_{t,i} + Σ_{j<i} D11[i,j] w_{t,j}                 (i = 0 .. nv-1)
    w_{t,i} = σ(v_{t,i})
    y_t     = C2 x_t + D21 w_t + D22 u_t + by                    in R^ny
    x_{t+1} = A x_t + B1 w_t + B2 u_t + bx                       in R^nx

The equilibrium `w = σ(D11 w + b)` is the fixed point of the layer; strict lower
triangularity makes neuron `i` depend only on `0..i-1`, so **one forward substitution over
`nv` resolves it exactly** — no iteration, no implicit-function-theorem backward. That is
the entire reason the acyclic REN class is cheap.

Note that `y_t` observes the state *before* the update. A rollout of length `L` from `x_0`
therefore emits `y_0 .. y_{L-1}` and carries `x_L`.

The `u`-dependent terms are loop-invariant: `D12 u_t + bv`, `B2 u_t + bx` and
`D22 u_t + by` are three batched GEMMs over all `B·L` samples, computed once before the
sequential loop. Only the `x`- and `w`-dependent parts stay inside it.

**Stored for backward:** `xs[t] = x_t` (`t = 0..L-1`) and `ws[t] = w_t`. Nothing else. The
slope `σ'(v_t)` is recoverable from `w_t` alone for every admissible activation:

    tanh:    σ' = 1 - w²
    sigmoid: σ' = w (1 - w)
    relu:    σ' = [w > 0]

(For `relu`, `w > 0 ⟺ v > 0` except at `v = 0`, where both conventions give `0`.)

## 2. Backward

Inputs: `gy_t = ∂L/∂y_t` and `gxL = ∂L/∂x_L` (the latter nonzero only when the carried
state is consumed downstream, e.g. under TBPTT).

Write `λ_t = ∂L/∂x_t` for the *total* state adjoint and `gv_t = ∂L/∂v_t` for the
pre-activation adjoint of the equilibrium layer.

### 2.1 The sequential part

Run `t = L-1 .. 0` with `λ_L = gxL`:

**(a) direct adjoint of `w_t`** — the two places `w_t` is consumed:

    ǧ_t = D21ᵀ gy_t + B1ᵀ λ_{t+1}                                in R^nv

**(b) reverse sweep**, `i = nv-1 .. 0`. Mirror of the forward substitution: `w_i` feeds
every `v_k` with `k > i`, so its adjoint collects from the neurons already processed.

    gv_{t,i} = σ'(v_{t,i}) · ( ǧ_{t,i} + Σ_{k>i} D11[k,i] gv_{t,k} )

`Σ_{k>i} D11[k,i] (·)` reads *column* `i` of `D11` below the diagonal, so the kernels take
a transposed copy and read it as a contiguous row.

**(c) state adjoint** — `x_t` is consumed by the equilibrium layer, the output map, and
the state update:

    λ_t = Aᵀ λ_{t+1} + C2ᵀ gy_t + C1ᵀ gv_t                       in R^nx

Finally `∂L/∂x_0 = λ_0`.

This recurrence is the *only* sequential work in the backward pass; a backend implements
exactly this and nothing else. Its outputs are `λ_{t+1}` and `gv_t` for every `t`, kept as
`[B, L, nx]` and `[B, L, nv]`.

### 2.2 The batched part

Every remaining gradient is a reduction over all `B·L` step samples — a batched GEMM,
shared by all backends and computed once in Python. With `Λ[t] := λ_{t+1}`, and all
tensors flattened to `[B·L, ·]`:

    dA   = Λᵀ xs        dB1  = Λᵀ ws        dB2  = Λᵀ u        dbx = Σ Λ
    dC1  = gvᵀ xs       dD11 = tril(gvᵀ ws, -1)                 dbv = Σ gv
    dD12 = gvᵀ u
    dC2  = gyᵀ xs       dD21 = gyᵀ ws       dD22 = gyᵀ u        dby = Σ gy

    du   = Λ B2 + gv D12 + gy D22                                in R^{B·L × nu}

The `tril(·, -1)` mask on `dD11` is not cosmetic: entries on and above the diagonal are
structurally absent from the forward, so their gradient is exactly zero. Leaving the raw
outer product there would still train correctly (the upstream construction re-applies the
mask in its own backward), but the op would no longer be the gradient of the function it
claims to compute.

## 3. What the kernels must agree on

- **Parameter order.** The custom ops flatten the explicit realization in the field order
  of `ExplicitREN`: `A, B1, B2, C1, D11, D12, C2, D21, D22, bx, bv, by`.
- **Saved tensors.** `xs = [x_0 .. x_{L-1}]` and `ws = [w_0 .. w_{L-1}]`, both `[B, L, ·]`.
- **Transposes.** The backward reads `Aᵀ, B1ᵀ, C1ᵀ, C2ᵀ, D21ᵀ, D11ᵀ`; the ops materialize
  them once per call rather than making each backend transpose in-kernel.
- **`D11` is trusted to be strictly lower triangular.** The kernels never read the upper
  part, and the parameterization guarantees it — a hand-built `ExplicitREN` that violates
  it silently gets its upper triangle ignored rather than a fixed-point solve.
