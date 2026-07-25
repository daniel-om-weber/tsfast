# R2DN fused rollout: forward and BPTT math

This is the correctness contract for the fused execution backends of
`tsfast.models.architectures.ren.r2dn.R2DNCore`. It derives the per-step update and its full
backpropagation-through-time (BPTT) gradient, and it is the R2DN's counterpart to
`MATH_REN.md`.

The cut is the same one: the certificate construction — `H = XᵀX + εI`, its partition, the
`E⁻¹` solve, the Cayley transforms of every sandwich layer — runs once per forward in
ordinary autograd and produces the **explicit realization**. The kernels see only plain
tensors. A backend that gets `∂L/∂A` and `∂L/∂W_l` right needs no opinion about `∂L/∂X`.

Symbols: state `x ∈ R^nx`, input `u ∈ R^nu`, output `y ∈ R^ny`, interconnection `v, w ∈ R^nv`.
The 1-Lipschitz network has `d` hidden layers of widths `m_0 .. m_{d-1}`. Sequence length
`L`, batch `B`. The activation `σ` is monotone and slope-restricted to `[0,1]` (`tanh`,
`relu`, `sigmoid`).

## 1. Folded layer weights

A sandwich layer evaluates `h ↦ √2 · A Ψ σ(√2 Ψ⁻¹ B h + c)`, with `Ψ = diag(ψ)` the free
per-unit scaling and `[Aᵀ; Bᵀ]` an isometry. The `√2` and the two `Ψ` factors are what make
the layer 1-Lipschitz, but they are *constant across the rollout*, so the kernels never see
them. Folding them gives

    W_l = √2 Ψ_l⁻¹ B_l          V_l = √2 A_l Ψ_l           layer l is  h ↦ V_l σ(W_l h + c_l)

and then `V_l` folds again, into the *following* layer's weight, because
`W_{l+1}(V_l a_l) = (W_{l+1} V_l) a_l`. What crosses the op boundary is therefore **one
matrix and one bias per layer**:

    W̃_0 = W_0        W̃_{l+1} = W_{l+1} V_l        W̃_out = W_out V_{d-1}

built once per rollout in ordinary autograd, which then carries `∂L/∂B_l`, `∂L/∂ψ_l` and
`∂L/∂A_l` back through the composition for free. Below, `W_l` means the folded `W̃_l`.

This matters more than it looks. A sequential rollout is bound by the *dependency chain* per
timestep, not by arithmetic: unfolded, each layer costs two dependent cross-lane reductions
(`p = W h`, then `h = V a`); folded, it costs one. It also halves the register footprint,
which is what the on-chip caps are spent on.

This is the only place the R2DN's op boundary differs in spirit from the REN's: there the
explicit realization is already what the kernel wants, here a cheap reparameterization stands
between them. The **output layer** carries no activation and no `Ψ` — it is the norm-bounded
linear map that closes the stack — so only the `V` composition applies to it.

## 2. Forward

Explicit parameters `A, B1, B2, C1, C2, D12, D21, D22, bx, bv, by` plus the folded network
`(W_l, c_l)` for `l = 0 .. d-1` and `(W_out, c_out)`. For `t = 0 .. L-1`, from `x_0`:

    v_t     = C1 x_t + D12 u_t + bv                        in R^nv
    a_t^0   = σ(W_0 v_t + c_0)                             in R^{m_0}
    a_t^l   = σ(W_l a_t^{l-1} + c_l)                            l = 1 .. d-1
    w_t     = W_out a_t^{d-1} + c_out                      in R^nv
    y_t     = C2 x_t + D21 w_t + D22 u_t + by              in R^ny
    x_{t+1} = A x_t + B1 w_t + B2 u_t + bx                 in R^nx

(with `d = 0` the network is `w_t = W_out v_t + c_out`, a plain affine map.)

There is no equilibrium layer and no `D11`: `w_t` is a feedforward function of `v_t` alone.
That is the whole architectural difference from the REN, and it is why the per-step work is
`d` small GEMVs instead of a sequential sweep over `nv` neurons. It is *not* why the fused
backend exists — with `L` steps of ~`8 + 3d` tiny kernels each, the eager rollout is
dispatch-bound exactly as the REN's is, and one persistent kernel per trajectory removes the
launches from both.

`y_t` observes the state *before* the update, so a rollout of length `L` from `x_0` emits
`y_0 .. y_{L-1}` and carries `x_L`.

The `u`-dependent terms `D12 u_t + bv`, `B2 u_t + bx` and `D22 u_t + by` are loop-invariant
batched GEMMs in the eager path. The fused kernels keep `u_t` in registers instead and fold
these into the per-step accumulation, which costs the same FMAs and saves three `B·L`-sized
temporaries.

**Stored for backward:** `xs[t] = x_t`, `vs[t] = v_t`, `ws[t] = w_t`, and the activations
`as^l[t] = a_t^l` for each hidden layer. Nothing else — the folding leaves every layer's
input equal to the previous layer's stored activation. As in the REN, the slope `σ'(p)` is
recoverable from the post-activation alone:

    tanh:    σ' = 1 - a²
    sigmoid: σ' = a (1 - a)
    relu:    σ' = [a > 0]

## 3. Backward

Inputs: `gy_t = ∂L/∂y_t` and `gxL = ∂L/∂x_L` (nonzero only when the carried state is consumed
downstream, e.g. under TBPTT).

### 3.1 The sequential part

Write `λ_t = ∂L/∂x_t`. Run `t = L-1 .. 0` with `λ_L = gxL`:

    gw_t     = D21ᵀ gy_t + B1ᵀ λ_{t+1}                     in R^nv
    g_t      = W_outᵀ gw_t                                 in R^{m_{d-1}}
    gp_t^l   = σ'(a_t^l) ⊙ g_t                                  l = d-1 .. 0
    g_t      = W_lᵀ gp_t^l                                      (rebound each layer)
    gv_t     = g_t
    λ_t      = Aᵀ λ_{t+1} + C2ᵀ gy_t + C1ᵀ gv_t            in R^nx

Finally `∂L/∂x_0 = λ_0`. Note the folded form costs one matrix-vector product per layer here
too — the `V_lᵀ` that a layer's adjoint would otherwise apply is already inside `W_{l+1}ᵀ`.

The network's backward sits *inside* the sequential loop and cannot be lifted out of it,
even though the network itself is feedforward: `gv_t` depends on `λ_{t+1}` through `gw_t`,
and `λ_t` depends on `gv_t`. That single chain is what forces a reverse kernel rather than a
batched pass, and it is the mirror of the REN's reverse sweep.

**Emitted per step:** `λ_{t+1}`, `gv_t`, and `gp_t^l` for each hidden layer. `gw` is *not*
emitted, being one batched GEMM away from what already is:

    gw = gy D21 + Λ B1                     (Λ[t] := λ_{t+1})

so emitting it would spend bandwidth to save arithmetic that is already free.

### 3.2 The batched part

Every remaining gradient is a reduction over all `B·L` step samples — a batched GEMM shared
by all backends and computed once in Python. All tensors flattened to `[B·L, ·]`:

    dA     = Λᵀ xs        dB1  = Λᵀ ws        dB2  = Λᵀ u        dbx = Σ Λ
    dC1    = gvᵀ xs                           dD12 = gvᵀ u       dbv = Σ gv
    dC2    = gyᵀ xs       dD21 = gyᵀ ws       dD22 = gyᵀ u       dby = Σ gy

    dW_0   = gp^0ᵀ vs     dW_l = gp^lᵀ a^{l-1}                  dc_l   = Σ gp^l
    dW_out = gwᵀ a^{d-1}                                        dc_out = Σ gw

    du     = Λ B2 + gv D12 + gy D22                            in R^{B·L × nu}

Every layer gradient is a plain outer product of its pre-activation adjoint with its input,
and the folding is what makes the input simply the previous layer's stored activation.

## 4. What the kernels must agree on

- **Parameter order.** The custom ops flatten the realization as `ExplicitR2DN.tensors`
  (field order: `A, B1, B2, C1, C2, D12, D21, D22, bx, bv, by`) followed by the folded
  network, two tensors per layer: `W_0, c_0, .., W_{d-1}, c_{d-1}, W_out, c_out`.
- **Layer widths are read off the shapes.** Each `W` is `[out, in]`, so nothing has to carry
  a width list across the boundary — a redundant argument is one more thing a backend could
  disagree with. `d = 0` is legal and means the network is the single affine output layer.
- **Saved tensors.** `xs`, `vs`, `ws` (`[B, L, ·]`) and one `as^l` per hidden layer.
- **Transposes.** The backward reads `Aᵀ, B1ᵀ, C1ᵀ, C2ᵀ, D21ᵀ, W_lᵀ, W_outᵀ`; the ops
  materialize them once per call rather than making each backend transpose in-kernel.
- **`D22` may be structurally zero.** The Lipschitz parameterization sets it so, but the
  kernels read it like any other matrix — no branch, since a zeroed tile costs the same FMAs
  and specializing on it would double the kernel count for no measurable gain.
