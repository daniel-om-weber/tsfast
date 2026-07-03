"""Compare tsfast's Mamba against the official implementation of Gu & Dao.

Reference: ``selective_scan_ref`` from github.com/state-spaces/mamba
(``mamba_ssm/ops/selective_scan_interface.py``, Apache-2.0), the pure-PyTorch function
the official CUDA kernels are themselves tested against, published with "Mamba:
Linear-Time Sequence Modeling with Selective State Spaces" (COLM 2024,
arXiv:2312.00752). The official package cannot be imported without its compiled CUDA
extension, so the function is transcribed below (without its fp32 casts, so the
comparison dtype is preserved), together with the surrounding block computation of
``mamba_simple.py`` (in_proj -> causal depthwise conv -> SiLU -> selective scan ->
SiLU-gating -> out_proj), using the tsfast layer's own parameters.

For every configuration the script reports the maximum relative deviation of the output
and of all parameter/input gradients, in float64. Expected agreement: < 1e-12.

If the ``mambapy`` package (github.com/alxndrTL/mamba.py, an independent pure-PyTorch
implementation) is installed, a second section cross-checks the forward pass against its
``MambaBlock`` with copied weights. That comparison runs in float32 because mambapy
casts ``A_log``/``D`` to float32 internally; agreement is then bounded by float32
round-off (~1e-6), not by algorithmic differences.
"""

import sys

import torch
import torch.nn.functional as F

from tsfast.models.mamba import MambaLayer

TOL = 1e-12


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True):
    """Transcribed from state-spaces/mamba, mamba_ssm/ops/selective_scan_interface.py
    (Apache-2.0), without the fp32 casts so the comparison dtype is preserved.

    Shapes: u/delta/z (B, D, L), A (D, N), B/C (B, N, L), D (D,), delta_bias (D,).
    """
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, L = u.shape
    dstate = A.shape[1]
    x = A.new_zeros((batch, dim, dstate))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y + u * D.unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z)
    return out


def reference_mixer_forward(layer, u):
    """The official Mamba block forward (mamba_simple.py reference path) on top of
    ``selective_scan_ref``, using the tsfast layer's own parameters."""
    L = u.shape[1]
    x, z = layer.in_proj(u).chunk(2, dim=-1)
    x = x.transpose(1, 2)
    x = F.silu(
        F.conv1d(x, layer.conv1d.weight, layer.conv1d.bias, padding=layer.d_conv - 1, groups=layer.d_inner)[..., :L]
    )
    x_dbl = layer.x_proj(x.transpose(1, 2))
    dt, B, C = x_dbl.split([layer.dt_rank, layer.d_state, layer.d_state], dim=-1)
    delta = (dt @ layer.dt_proj.weight.mT).transpose(1, 2)  # bias enters the scan separately
    A = -torch.exp(layer.A_log)
    out = selective_scan_ref(
        x, delta, A, B.transpose(1, 2), C.transpose(1, 2), layer.D, z.transpose(1, 2), layer.dt_proj.bias
    )
    return layer.out_proj(out.transpose(1, 2))


def compare(d_model, d_state, L, seed):
    torch.manual_seed(seed)
    layer = MambaLayer(d_model, d_state=d_state).double()
    u = torch.randn(3, L, d_model, dtype=torch.float64)
    u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
    y_ref = reference_mixer_forward(layer, u_ref)
    w = torch.randn_like(y_ref)  # random adjoint probes all gradient components
    (y_ref * w).sum().backward()
    g_ref = [p.grad.clone() for p in layer.parameters()]
    for p in layer.parameters():
        p.grad = None
    y_ours = layer(u_ours)
    (y_ours * w).sum().backward()
    g_err = max(rel(p.grad, g) for p, g in zip(layer.parameters(), g_ref))
    return {"output": rel(y_ours, y_ref), "grad params": g_err, "grad u": rel(u_ours.grad, u_ref.grad)}


def compare_mambapy():
    try:
        from mambapy.mamba import MambaBlock, MambaConfig
    except ImportError:
        print("\nmambapy not installed - skipping the cross-check (pip install mambapy)")
        return 0.0
    torch.manual_seed(0)
    d_model = 4
    ours = MambaLayer(d_model, d_state=8)
    ref = MambaBlock(MambaConfig(d_model=d_model, n_layers=1, d_state=8))
    with torch.no_grad():
        for name in ("in_proj", "x_proj", "dt_proj", "out_proj", "conv1d"):
            getattr(ref, name).load_state_dict(getattr(ours, name).state_dict())
        ref.A_log.copy_(ours.A_log)
        ref.D.copy_(ours.D)
    u = torch.randn(2, 200, d_model)
    err = rel(ours(u), ref(u))
    print(f"\nmambapy cross-check (float32, see module docstring): output deviation {err:.1e}")
    return err


def main():
    configs = [
        ("small d=4 N=8", dict(d_model=4, d_state=8, L=200)),
        ("paper constants d=16 N=16", dict(d_model=16, d_state=16, L=500)),
        ("wide d=64 N=16", dict(d_model=64, d_state=16, L=1000)),
    ]
    keys = ["output", "grad params", "grad u"]
    print("Mamba mixer forward/backward vs official selective_scan_ref semantics (float64)")
    print(f"{'configuration':<30}" + "".join(f"{k:>14}" for k in keys))
    worst = 0.0
    for name, cfg in configs:
        errs = compare(**cfg, seed=0)
        worst = max(worst, *errs.values())
        print(f"{name:<30}" + "".join(f"{errs[k]:>14.1e}" for k in keys))

    cross = compare_mambapy()
    print(f"\nworst relative deviation vs official semantics: {worst:.2e} (tolerance {TOL:.0e})")
    if worst > TOL or cross > 1e-4:
        sys.exit("FAIL: deviation exceeds tolerance")
    print("PASS: tsfast Mamba matches the official implementation of Gu & Dao")


if __name__ == "__main__":
    main()
