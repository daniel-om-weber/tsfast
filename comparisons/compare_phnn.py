"""Compare tsfast's OE-pHNN against a direct implementation of the published equations.

Reference: Moradi, Beintema, Jaensson, Tóth & Schoukens, "Port-Hamiltonian Neural
Networks with Output Error Noise Models" (Automatica 2026, arXiv:2502.14432), eqs.
(21)-(30); reference implementation github.com/sarvin90/OE-pHNN. That repository carries
no license file, so its code is not transcribed here; the reference below implements the
paper's equations directly the way the released code does — dH/dx via
``torch.autograd.grad(create_graph=True)``, matrix nets reshaped and scaled per step,
``y = G(x)^T dH/dx`` evaluated before the RK4 update, input held constant over the RK4
stages (ZOH, ``dt/n_steps`` substeps). In a separate analysis this formulation was
verified against the released trained 2MSD models (weight transplant: encoder bit-exact,
single steps ~1e-7, 200-step float32 rollouts ~1e-6).

tsfast's implementation computes the identical function without per-step autograd: dH/dx
in closed form (explicit backpropagation through the Hamiltonian MLP, itself an autograd-
differentiable expression) and the first RK4 stage sharing its network evaluations with
the output map. Both implementations here operate on the *same* parameter tensors — the
reference wraps the tsfast submodules and only replaces the forward algorithm — so the
printed deviations measure algorithmic agreement only.

For every configuration the script reports the maximum relative deviation of the rollout
output and of all parameter and input gradients, in float64, including the ELU-bounded
and unbounded Hamiltonian, multi-substep RK4, MIMO, an na != nb encoder window, and the
linear-output variant for non-square systems. Expected agreement: < 1e-12.
"""

import sys

import torch
from torch import nn

from tsfast.models.phnn import PHNN

TOL = 1e-12


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class ReferencePHNN(nn.Module):
    """Paper-equation forward pass over the tsfast model's own parameter tensors."""

    def __init__(self, model: PHNN):
        super().__init__()
        self.m = model

    def dhdx(self, x):
        # like the reference code: differentiate w.r.t. the (non-leaf) state so the
        # create_graph result keeps gradients flowing through the whole trajectory
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            h = self.m.core.hamiltonian.net(x)[:, 0]
            lb = self.m.core.hamiltonian.lower_bound
            if lb is not None:
                b = lb + 1.0
                h = torch.nn.functional.elu(h - b) + b
            return torch.autograd.grad(h.sum(), x, create_graph=True)[0]

    def matrices(self, x):
        core = self.m.core
        n = core.n_state
        b = core.j_net(x).view(x.shape[0], n, n) * core.jr_scale
        a = core.r_net(x).view(x.shape[0], n, n) * core.jr_scale
        j = b - b.permute(0, 2, 1)
        r = torch.einsum("bik,bjk->bij", a, a)
        g = core.g_net(x).view(x.shape[0], n, core.n_input) * core.g_scale
        return j, r, g

    def rhs(self, x, u):
        j, r, g = self.matrices(x)
        dhdx = self.dhdx(x)
        return torch.einsum("bij,bj->bi", j - r, dhdx) + torch.einsum("bij,bj->bi", g, u)

    def step(self, x, u):
        core = self.m.core
        _, _, g = self.matrices(x)
        if core.output_map is not None:
            y = core.output_map(x)
        else:
            y = torch.einsum("bij,bi->bj", g, self.dhdx(x))
        dt = core.dt / core.rk4_steps
        for _ in range(core.rk4_steps):
            k1 = dt * self.rhs(x, u)
            k2 = dt * self.rhs(x + k1 / 2, u)
            k3 = dt * self.rhs(x + k2 / 2, u)
            k4 = dt * self.rhs(x + k3, u)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y, x

    def forward(self, xin):
        m = self.m
        u, ymeas = xin[..., : m.n_input], xin[..., m.n_input :]
        n0 = m.n_init
        x = m.encoder(u[:, n0 - m.encoder.nb : n0], ymeas[:, n0 - m.encoder.na : n0])
        outs = []
        for t in range(n0, xin.shape[1]):
            y, x = self.step(x, u[:, t])
            outs.append(y)
        warmup = xin.new_zeros(xin.shape[0], n0, m.n_output)
        return torch.cat((warmup, torch.stack(outs, dim=1)), dim=1)


def run(model, forward, xin):
    for p in model.parameters():
        p.grad = None
    xin = xin.clone().requires_grad_()
    out = forward(xin)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    # without the ELU bound the Hamiltonian's final bias is a constant offset that
    # cannot receive gradient in either implementation; compare it as zero
    grads = [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in model.parameters()]
    return out.detach(), grads, xin.grad.clone()


CONFIGS = [
    dict(n_input=1, n_output=1, n_state=4, num_layers=2, h_lower_bound=0.0),
    dict(n_input=1, n_output=1, n_state=2, num_layers=2, h_lower_bound=None),
    dict(n_input=1, n_output=1, n_state=4, num_layers=1, h_lower_bound=0.0, rk4_steps=3),
    dict(n_input=3, n_output=3, n_state=6, num_layers=2, h_lower_bound=0.0),
    dict(n_input=1, n_output=1, n_state=4, num_layers=2, h_lower_bound=0.0, na=7, nb=3),
    dict(n_input=2, n_output=5, n_state=4, num_layers=2, h_lower_bound=0.0, output="linear"),
]

torch.manual_seed(0)
failed = False
for cfg in CONFIGS:
    model = PHNN(hidden_size=16, dt=0.13, n_init=8, backend="eager", **cfg).double()
    for p in model.parameters():
        nn.init.normal_(p, std=0.4)
    reference = ReferencePHNN(model)

    B, L = 5, 8 + 40
    xin = torch.randn(B, L, cfg["n_input"] + cfg["n_output"], dtype=torch.float64)
    out_t, grads_t, du_t = run(model, model, xin)
    out_r, grads_r, du_r = run(model, reference, xin)

    devs = [rel(out_t, out_r)] + [rel(a, b) for a, b in zip(grads_t, grads_r)] + [rel(du_t, du_r)]
    worst = max(devs)
    status = "OK  " if worst < TOL else "FAIL"
    failed |= worst >= TOL
    print(f"{status} {cfg}: output {devs[0]:.2e}, worst grad {max(devs[1:]):.2e}")

print("\nall configurations within tolerance" if not failed else "\nTOLERANCE EXCEEDED")
sys.exit(1 if failed else 0)
