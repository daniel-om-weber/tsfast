"""Compare tsfast's LRU against the reference implementation of Forgione et al.

Reference: the ``LRU`` class from ``lru/linear.py`` of github.com/forgi86/lru-reduction
(MIT license), the implementation published with "Model order reduction of deep
structured state-space models: A system-theoretic approach" (IEEE CDC 2024,
arXiv:2403.14833), which itself follows the parameterization of Orvieto et al.,
"Resurrecting Recurrent Neural Networks for Long Sequences" (ICML 2023,
arXiv:2303.06349), Appendix A. The repository is not packaged for PyPI, so the class is
transcribed verbatim below (sequential-loop forward only).

Parameter mapping: ``nu_log``/``theta_log``/``gamma_log``/``D`` copy 1:1; the reference
stores ``B`` and ``C`` as complex parameters while tsfast stores real/imaginary pairs, so
``B_ref = B_re + i B_im`` (same for ``C``). Both use the ring initialization
``|lambda|^2 ~ U[r_min^2, r_max^2]`` and the full feedthrough matrix ``D`` of the
system-identification variant.

For every configuration the script reports the maximum relative deviation of the output
and of all parameter/input gradients, in float64, with and without a nonzero initial
state. Expected agreement: < 1e-12.
"""

import math
import sys

import torch
from torch import nn

from tsfast.models.architectures.lru import LRU

TOL = 1e-12


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class ReferenceLRU(nn.Module):
    """Transcribed from github.com/forgi86/lru-reduction, lru/linear.py (MIT license)."""

    def __init__(self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, max_phase=math.pi):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))))
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        lambdas = torch.complex(lambda_abs * torch.cos(lambda_phase), lambda_abs * torch.sin(lambda_phase))
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        return lambdas, gammas * self.B, self.C, self.D

    def forward(self, input, state=None):
        lambdas, B, C, D = self.ss_params()
        if state is None:
            state = torch.zeros(self.nu_log.shape[0], dtype=B.dtype, device=input.device)
        states = []
        for u_step in input.split(1, dim=1):
            u_step = u_step.squeeze(1)
            state = lambdas * state + u_step.to(B.dtype) @ B.T
            states.append(state)
        states = torch.stack(states, 1)
        return (states @ C.mT).real + input @ D.T


def compare(in_f, out_f, N, L, seed, with_state):
    torch.manual_seed(seed)
    ours = LRU(in_f, out_f, N, r_min=0.2, r_max=0.99).double()
    ref = ReferenceLRU(in_f, out_f, N).double()
    with torch.no_grad():
        for name in ("nu_log", "theta_log", "gamma_log", "D"):
            getattr(ref, name).copy_(getattr(ours, name))
        # Module.double() leaves complex params at complex64; replace them outright
        ref.B = nn.Parameter(torch.complex(ours.B_re, ours.B_im).detach().clone())
        ref.C = nn.Parameter(torch.complex(ours.C_re, ours.C_im).detach().clone())

    u = torch.randn(4, L, in_f, dtype=torch.float64)
    x0 = torch.randn(4, N, dtype=torch.complex128) if with_state else None
    u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
    y_ref, y_ours = ref(u_ref, state=x0), ours(u_ours, state=x0)
    w = torch.randn_like(y_ref)  # random adjoint probes all gradient components
    (y_ref * w).sum().backward()
    (y_ours * w).sum().backward()
    errs = {"output": rel(y_ours, y_ref), "grad u": rel(u_ours.grad, u_ref.grad)}
    for name in ("nu_log", "theta_log", "gamma_log", "D"):
        errs[f"grad {name}"] = rel(getattr(ours, name).grad, getattr(ref, name).grad)
    # torch stores complex grads as complex(dL/dRe, dL/dIm), matching the real-pair grads
    errs["grad B"] = rel(torch.complex(ours.B_re.grad, ours.B_im.grad), ref.B.grad)
    errs["grad C"] = rel(torch.complex(ours.C_re.grad, ours.C_im.grad), ref.C.grad)
    return errs


def main():
    configs = [
        ("SISO N=8", dict(in_f=1, out_f=1, N=8, L=500, with_state=False)),
        ("MIMO 3->2 N=16", dict(in_f=3, out_f=2, N=16, L=500, with_state=False)),
        ("MIMO + initial state", dict(in_f=3, out_f=2, N=16, L=500, with_state=True)),
        ("wide, long", dict(in_f=8, out_f=8, N=64, L=4000, with_state=False)),
    ]
    keys = ["output", "grad u", "grad nu_log", "grad theta_log", "grad gamma_log", "grad D", "grad B", "grad C"]
    print(f"{'configuration':<24}" + "".join(f"{k.replace('grad ', 'd'):>10}" for k in keys))
    worst = 0.0
    for name, cfg in configs:
        errs = compare(**cfg, seed=0)
        worst = max(worst, *errs.values())
        print(f"{name:<24}" + "".join(f"{errs[k]:>10.1e}" for k in keys))
    print(f"\nworst relative deviation: {worst:.2e} (tolerance {TOL:.0e})")
    if worst > TOL:
        sys.exit("FAIL: deviation exceeds tolerance")
    print("PASS: tsfast LRU matches the reference implementation of Forgione et al.")


if __name__ == "__main__":
    main()
