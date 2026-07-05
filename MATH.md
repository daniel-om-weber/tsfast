# PHNN fused rollout: forward and BPTT math

This is the correctness contract for the fused C++ and Triton rollout backends of
`tsfast.models.phnn.PHNNCore`. It derives the exact per-step update and its full
backpropagation-through-time (BPTT) gradient, **including** the second-order terms
that arise because the forward step already contains the state gradient `dH/dx` of the
Hamiltonian network (computed in closed form, not by autograd — so it is ordinary
forward arithmetic and reverse-mode through it is mechanical).

Symbols: state `x in R^n` (`n = n_state`), input `u in R^m` (`m = n_input`), all
component nets have hidden width `h` and `K = num_layers + 1` linear layers.
The fused backends support `rk4_steps == 1` (one RK4 step, 4 stages).

## 1. Component nets and the field evaluation `fields(x)`

### 1.1 Hamiltonian MLP: value and closed-form gradient
A tanh MLP with a scalar output. Layers `W_l, b_l`, `l = 0..K-1`, `z_{-1} := x`:

    p_l   = W_l z_{l-1} + b_l          (l = 0..K-1)
    z_l   = tanh(p_l)                  (l = 0..K-2)      d_l := 1 - z_l^2
    H_raw = p_{K-1}   (scalar; output layer is linear)

Closed-form state gradient (plain backprop with seed `dH/dp_{K-1} = 1`, gradient tape):

    gz_{K-2}  = W_{K-1}^T · 1  = row 0 of W_{K-1}          in R^h
    gp_l      = gz_l  ⊙ d_l                                 (l = K-2 .. 0)
    gz_{l-1}  = W_l^T gp_l    (l = K-2 .. 1)                in R^h
    dHdx_raw  = W_0^T gp_0                                   in R^n

Optional ELU lower bound (`h_lower_bound = c`, `b = c + 1`):

    s   = H_raw - b
    m   = elu'(s) = (s > 0 ? 1 : exp(s))     (scalar per lane)
    dHdx = m · dHdx_raw

`H_raw` (the scalar) is needed only for the `m` multiplier; the rollout uses only
`dHdx`. If the bound is disabled, `m = 1` and `H_raw` is irrelevant.

### 1.2 Structure matrices and drift
With `jr_scale = ((2+n)n)^{-1/4}`, `g_scale = m^{-1/2}` and plain-MLP outputs
reshaped row-major:

    B = jr_scale · reshape(j_net(x), n×n)      J = B - B^T          (skew)
    A = jr_scale · reshape(r_net(x), n×n)      R = A A^T            (PSD)
    G = g_scale  · reshape(g_net(x), n×m)      (g_net is a ResMLP: linear + MLP)
    JR    = J - R
    drift = JR · dHdx

`fields(x)` returns `(G, dHdx, drift)` and internally `J,R,JR,A,B,H_raw,s,m,dHdx_raw`.

## 2. Per-step update

    (G1, dHdx1, drift1) = fields(x)                 # shared: output + RK4 stage 1
    rhs(x',u) = fields(x').drift + fields(x').G · u
    h = dt / rk4_steps                              # = dt for rk4_steps = 1

    k1 = h·(drift1 + G1 u)
    k2 = h·rhs(x + k1/2, u)
    k3 = h·rhs(x + k2/2, u)
    k4 = h·rhs(x + k3,   u)
    x_next = x + (k1 + 2k2 + 2k3 + k4)/6

    y = G1^T dHdx1        (output="ph",  requires m == n)
    y = W_out x + b_out   (output="linear")

The four field evaluations are at `q1=x, q2=x+k1/2, q3=x+k2/2, q4=x+k3`; the output
and stage 1 share `fields(q1)`.

## 3. BPTT

The only cross-time dependency is `x_t -> x_{t+1}`. The reverse sweep carries the state
adjoint `bx = dL/dx_{t+1}` and, at each step, backprops the step map with output
adjoints `(gy = dL/dy_t, gxn = bx)` to produce `bx' = dL/dx_t` (the carry for step t-1),
`du_t`, and parameter-gradient contributions. Forward stores only the per-step input
states `x_t` (B×L×n); the backward recomputes each step's intra-step activations.

### 3.1 Step reverse (rk4_steps = 1)
From `x_next`:  `bx += gxn`; `bk1 = gxn/6, bk2 = bk4 = gxn/6? ` — exactly
`bk1 = bk4 = gxn/6`, `bk2 = bk3 = gxn/3`.

Output: ph → `bG1[i,j] += gy[j]·dHdx1[i]`, `bdHdx1 += G1 gy`.
linear → `bx += W_out^T gy`, `dW_out += gy⊗x`, `db_out += gy`.

Stages 4→1 (each `k_s = h·rhs_s`, `rhs_s = drift_s + G_s u`):

    brhs_s = h·bk_s ;  bdrift_s = brhs_s
    bG_s  += brhs_s ⊗ u ;  du += G_s^T brhs_s
    bq_s   = fields_bwd(stage_s, bG_s, bdHdx_ext_s, bdrift_s)   # bdHdx_ext = 0 except stage 1
    stage 4: bx += bq4 ; bk3 += bq4
    stage 3: bx += bq3 ; bk2 += bq3/2
    stage 2: bx += bq2 ; bk1 += bq2/2
    stage 1: bG1 already holds the output term; bdHdx_ext_1 = bdHdx1 (output); bx += bq1

After all stages, `bx = dL/dx_t` is the carry.

### 3.2 `fields_bwd(bG, bdHdx_ext, bdrift) -> bq`
    bJR[i,j] = bdrift[i]·dHdx[j]                     # from drift = JR·dHdx
    bdHdx    = bdHdx_ext + JR^T bdrift
    bB[i,j]  = bJR[i,j] - bJR[j,i]                   # J = B - B^T,  bJ = bJR
    bA       = -(bJR + bJR^T) · A                    # R = A A^T, bR = -bJR
    b_jout   = jr_scale · bB     (flattened, adjoint of j_net output)
    b_rout   = jr_scale · bA
    b_gout   = g_scale  · bG
    # ELU / Hamiltonian gradient split:
    b_dHdx_raw = m · bdHdx
    b_Hraw     = (has_bound ? (Σ_i dHdx_raw[i]·bdHdx[i]) · (s>0 ? 0 : exp(s)) : 0)
    bq = hnet_vjp(b_dHdx_raw, b_Hraw) + mlp_vjp(J,b_jout) + mlp_vjp(R,b_rout)
       + resmlp_vjp(G,b_gout)

### 3.3 Hamiltonian VJP (the second-order term) `hnet_vjp(b_dHdx_raw, b_Hraw)`
Reverse of §1.1 as one tape (gradient tape + value tape share `W_0..W_{K-1}`, which is
why weight gradients receive contributions from both — the "second order" coupling).

Gradient tape (reverse):

    dW_0    += b_dHdx_raw ⊗ gp_0 ;  bgp_0 = W_0 b_dHdx_raw
    bgz_0    = bgp_0 ⊙ d_0 ;  bd_0 = bgp_0 ⊙ gz_0
    for l = 1..K-2:                                  # ascending
        dW_l  += bgz_{l-1} ⊗ gp_l ;  bgp_l = W_l bgz_{l-1}
        bgz_l  = bgp_l ⊙ d_l ;  bd_l = bgp_l ⊙ gz_l
    dW_{K-1}[0,:] += bgz_{K-2}                       # gz_{K-2} = W_{K-1}^T·1

Value tape (reverse), with `bz_l` seeded from the gradient tape via `bd_l`:

    bz_{K-2}  = W_{K-1}[0,:]·b_Hraw  +  (-2 z_{K-2} ⊙ bd_{K-2})
    dW_{K-1}[0,:] += b_Hraw · z_{K-2} ;  db_{K-1} += b_Hraw
    for l = K-2 .. 0:
        bp_l   = bz_l ⊙ d_l
        dW_l  += bp_l ⊗ inp_l ;  db_l += bp_l         # inp_l = z_{l-1} (l>0) else x
        binp   = W_l^T bp_l
        if l>0: bz_{l-1} = binp + (-2 z_{l-1} ⊙ bd_{l-1})
        else:   bq += binp

### 3.4 Plain MLP VJP `mlp_vjp` (j_net, r_net, g_net.mlp; linear output, dim `n_o`)
    bz_{K-2}  = W_{K-1}^T b_out
    dW_{K-1} += b_out ⊗ z_{K-2} ;  db_{K-1} += b_out
    for l = K-2 .. 0:
        bp_l   = bz_l ⊙ d_l
        dW_l  += bp_l ⊗ inp_l ;  db_l += bp_l
        binp   = W_l^T bp_l
        if l>0: bz_{l-1} = binp   else: bq += binp

`resmlp_vjp` adds the linear bypass: `dW_lin += b_out ⊗ x`, `db_lin += b_out`,
`bq += W_lin^T b_out`, then `mlp_vjp` with the same `b_out`.

## 4. Validation
- fp64 `torch.autograd.gradcheck` of the C rollout for all parameters and inputs.
- fp32 parameter-gradient equivalence vs the eager backend (< 1e-3 relative).
- Triton validated against the C backend (fp32) and eager.
