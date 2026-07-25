"""Compare tsfast's REN against the paper equations in implicit form, and check its certificates.

Reference: Revay, Wang & Manchester, "Recurrent Equilibrium Networks: Flexible Dynamic
Models with Guaranteed Stability and Robustness" (IEEE TAC 2023, arXiv:2104.05942),
eqs. (19), (28)-(33); reference implementation github.com/nic-barbara/R2DN
``robustnn/ren.py`` (MIT, JAX).

tsfast folds the implicit realization into an explicit one *once per forward*: it inverts
``E`` and ``Λ`` up front, so the rollout is a plain linear recurrence plus one forward
substitution over the ``nv`` neurons. The reference here does neither. It reads the
certificate matrix ``H`` off the model, partitions it independently, and then runs the
*implicit* system as written in the paper — a general ``torch.linalg.solve`` against ``E``
and ``Λ`` at every step, with the equilibrium reached by fixed-point iteration instead of
by forward substitution. Both implementations operate on the same parameter tensors, so
the printed deviations measure the explicit folding and the sweep, not initialization.

One convention is worth stating because it is invisible in the deviations: tsfast (like
the reference implementation) keeps the biases *explicit*, i.e. ``x⁺ = A x + ... + bx``
rather than ``E x⁺ = F x + ... + bx``. The two differ only by a reparameterization of an
unconstrained parameter, and the implicit reference below multiplies them back through
``E`` and ``Λ`` so the comparison is exact rather than approximate.

Deviations alone cannot show that the construction is *right*, only that two evaluations
of it agree — so the script also measures the certificates themselves: positive
definiteness of ``H``, the per-step contraction rate in the Lyapunov metric it induces,
and the empirical incremental gain against the certified ``gamma``, all at deliberately
badly-scaled parameters.

The optional final section transplants the weights into the authors' JAX implementation,
which is the only fully independent check of the ``H`` construction available. It is
skipped unless the reference is installed:

    uv pip install jax flax "robustnn @ git+https://github.com/nic-barbara/R2DN"
"""

import sys

import numpy as np
import torch
from torch import nn

from tsfast.models.architectures.ren import REN

TOL = 1e-11
ACTS = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}


def rel(a, b):
    # Z3 is empty whenever n_input == n_output, and an empty tensor cannot deviate
    if b.numel() == 0:
        return 0.0
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class ImplicitREN(nn.Module):
    """Paper-equation forward pass in implicit form over the tsfast model's own parameters."""

    def __init__(self, model: REN):
        super().__init__()
        self.m = model

    def implicit(self):
        p = self.m.core.parameterization
        nx, nv = self.m.spec.n_state, self.m.spec.n_nl
        h = p.hmatrix()
        h11, h21, h22 = h[:nx, :nx], h[nx : nx + nv, :nx], h[nx : nx + nv, nx : nx + nv]
        h31, h32, h33 = h[nx + nv :, :nx], h[nx + nv :, nx : nx + nv], h[nx + nv :, nx + nv :]
        return {
            "E": (h11 + h33 / self.m.spec.alpha**2 + p.Y1 - p.Y1.mH) / 2,
            "F": h31,
            "B1": h32,
            "C1": -h21,
            "D11": -torch.tril(h22, -1),
            "Lambda": torch.diag(torch.diagonal(h22) / 2),
        }

    def forward(self, u, x0):
        p = self.m.core.parameterization
        i = self.implicit()
        act = ACTS[self.m.spec.act]
        d22 = self.m.core.explicit().D22
        # The explicit biases, pushed back through the maps the explicit form divided out.
        bx = p.bx @ i["E"].mH
        bv = p.bv @ i["Lambda"].mH
        x, outs = x0, []
        for t in range(u.shape[1]):
            const = x @ i["C1"].mH + u[:, t] @ p.D12.mH + bv
            w = torch.zeros_like(const)
            # Strictly lower triangular D11 fixes one more neuron per pass, so nv passes
            # reach the exact equilibrium (and its exact derivative).
            for _ in range(self.m.spec.n_nl):
                w = act(torch.linalg.solve(i["Lambda"], (const + w @ i["D11"].mH).mH).mH)
            outs.append(x @ p.C2.mH + w @ p.D21.mH + u[:, t] @ d22.mH + p.by)
            rhs = x @ i["F"].mH + w @ i["B1"].mH + u[:, t] @ p.B2.mH + bx
            x = torch.linalg.solve(i["E"], rhs.mH).mH
        return torch.stack(outs, dim=1)


def run(model, forward, u, x0):
    for p in model.parameters():
        p.grad = None
    u, x0 = u.clone().requires_grad_(), x0.clone().requires_grad_()
    out = forward(u, x0)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    grads = [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in model.parameters()]
    return out.detach(), grads, u.grad.clone(), x0.grad.clone()


def make_qsr(nu, ny):
    a, b = torch.randn(ny, ny, dtype=torch.float64), torch.randn(nu, nu, dtype=torch.float64)
    s = torch.randn(nu, ny, dtype=torch.float64) * 0.5
    q = -(a.mH @ a + 0.3 * torch.eye(ny, dtype=torch.float64))
    r = s @ torch.linalg.solve(q, s.mH) + b.mH @ b + 0.5 * torch.eye(nu, dtype=torch.float64)
    return q, s, r


CONFIGS = [
    dict(n_input=1, n_output=1, n_state=4, n_nl=8),
    dict(n_input=3, n_output=2, n_state=5, n_nl=6, act="relu", alpha=0.9),
    dict(n_input=2, n_output=4, n_state=3, n_nl=12, act="sigmoid", polar=False),
    dict(n_input=2, n_output=2, n_state=4, n_nl=8, init="random", alpha=0.5),
    dict(n_input=2, n_output=3, n_state=4, n_nl=8, variant="lipschitz", gamma=0.7),
    dict(n_input=3, n_output=2, n_state=4, n_nl=8, variant="lipschitz", gamma=5.0),
    dict(n_input=2, n_output=2, n_state=4, n_nl=8, variant="lipschitz", gamma=1.5),
    dict(n_input=3, n_output=2, n_state=4, n_nl=8, variant="dissipative"),
    dict(n_input=2, n_output=2, n_state=4, n_nl=6, variant="dissipative"),
]


def build(cfg, seed):
    torch.manual_seed(seed)
    cfg = dict(cfg)
    if cfg.get("variant") == "dissipative":
        cfg["qsr"] = make_qsr(cfg["n_input"], cfg["n_output"])
    model = REN(backend="eager", **cfg).double()
    with torch.no_grad():  # move off the initialization, where much of H is structurally zero
        for p in model.core.parameterization.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    return model


def lyapunov_metric(model):
    """``M = Eᵀ P⁻¹ E``: the metric in which the certified contraction rate is exactly alpha."""
    i = ImplicitREN(model).implicit()
    p = model.core.parameterization
    nx, nv = model.spec.n_state, model.spec.n_nl
    return i["E"].mH @ torch.linalg.solve(p.hmatrix()[nx + nv :, nx + nv :], i["E"])


def contraction_ratio(model, steps=40, batch=8):
    with torch.no_grad():
        metric, e = lyapunov_metric(model), model.core.explicit()
        u = torch.randn(batch, steps, model.spec.n_input, dtype=torch.float64)
        xa = torch.randn(batch, model.spec.n_state, dtype=torch.float64) * 3
        xb = torch.randn(batch, model.spec.n_state, dtype=torch.float64) * 3
        v = lambda d: torch.einsum("bi,ij,bj->b", d, metric, d)  # noqa: E731
        floor, worst = v(xa - xb).max() * 1e-18, 0.0
        for t in range(steps):
            _, xa_next = model.core.rollout(e, u[:, t : t + 1], xa)
            _, xb_next = model.core.rollout(e, u[:, t : t + 1], xb)
            v0, v1 = v(xa - xb), v(xa_next - xb_next)
            live = v0 > floor
            if live.any():
                worst = max(worst, (v1[live] / v0[live]).max().sqrt().item() / model.spec.alpha)
            xa, xb = xa_next, xb_next
        return worst


def empirical_gain(model, length=60, iters=30, batch=8):
    """Power iteration on the input-to-output Jacobian, both trajectories from x0 = 0."""
    e = model.core.explicit()
    x0 = torch.zeros(batch, model.spec.n_state, dtype=torch.float64)
    u = torch.randn(batch, length, model.spec.n_input, dtype=torch.float64)
    d = torch.randn(batch, length, model.spec.n_input, dtype=torch.float64)
    with torch.no_grad():
        y_ref, _ = model.core.rollout(e, u, x0)
    best = 0.0
    for _ in range(iters):
        d = d / (d.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-30)
        ud = (u + d).requires_grad_()
        y_pert, _ = model.core.rollout(e, ud, x0)
        dy = y_pert - y_ref
        norm = dy.detach().flatten(1).norm(dim=1)
        best = max(best, norm.max().item())
        d = torch.autograd.grad((dy * (dy / (norm.view(-1, 1, 1) + 1e-30)).detach()).sum(), ud)[0].detach()
    return best


def jax_reference(model, u, x0):
    """Outputs of the authors' JAX implementation with the tsfast weights transplanted."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from robustnn import ren as jren

    spec, p = model.spec, model.core.parameterization
    common = dict(
        input_size=spec.n_input,
        state_size=spec.n_state,
        features=spec.n_nl,
        output_size=spec.n_output,
        activation=getattr(jax.nn, spec.act) if spec.act != "tanh" else jnp.tanh,
        abar=spec.alpha,
        eps=p.eps,
        do_polar_param=p.polar,
        param_dtype=jnp.float64,
    )
    if spec.variant == "contracting":
        jm = jren.ContractingREN(**common)
    elif spec.variant == "lipschitz":
        jm = jren.LipschitzREN(gamma=p.gamma, **common)
    else:
        # the raw matrices: the reference applies the same eps adjustment internally
        jm = jren.GeneralREN(Q=_j(p.Q), S=_j(p.S), R=_j(p.R), **common)

    d = min(spec.n_input, spec.n_output)
    free = {n: _j(t) for n, t in p.named_parameters()}
    free.setdefault("D22", jnp.zeros((spec.n_output, spec.n_input)))
    free.setdefault("X3", jnp.eye(d))
    free.setdefault("Y3", jnp.zeros((d, d)))
    free.setdefault("Z3", jnp.zeros((abs(spec.n_output - spec.n_input), d)))
    _, y = jm.simulate_sequence({"params": free}, _j(x0), _j(u).transpose(1, 0, 2))
    return torch.from_numpy(np.array(y)).permute(1, 0, 2)


def _j(t):
    import jax.numpy as jnp

    return jnp.asarray(t.detach().cpu().numpy())


print("implicit-form reference: max relative deviation of outputs and all gradients (float64)\n")
failed = False
for cfg in CONFIGS:
    model = build(cfg, seed=0)
    reference = ImplicitREN(model)
    u = torch.randn(5, 30, cfg["n_input"], dtype=torch.float64)
    x0 = torch.randn(5, cfg["n_state"], dtype=torch.float64)

    out_t, grads_t, du_t, dx0_t = run(model, model, u, x0)
    out_r, grads_r, du_r, dx0_r = run(model, reference, u, x0)
    devs = [rel(out_t, out_r)] + [rel(a, b) for a, b in zip(grads_t, grads_r)] + [rel(du_t, du_r), rel(dx0_t, dx0_r)]
    worst = max(devs)
    failed |= worst >= TOL
    label = ", ".join(f"{k}={v}" for k, v in cfg.items())
    print(f"{'OK  ' if worst < TOL else 'FAIL'} {label}: output {devs[0]:.2e}, worst grad {max(devs[1:]):.2e}")

print("\ncertificates at parameters perturbed far off the initialization\n")
for cfg in CONFIGS:
    model = build(cfg, seed=1)
    with torch.no_grad():
        for p in model.core.parameterization.parameters():
            p.mul_(20.0)
    min_eig = torch.linalg.eigvalsh(model.core.parameterization.hmatrix()).min().item()
    ratio = contraction_ratio(model)
    line = f"minEig(H) {min_eig:.2e}, contraction {ratio:.4f} of alpha"
    if cfg.get("variant") == "lipschitz":
        line += f", incremental gain {empirical_gain(model) / cfg['gamma']:.3f} of gamma"
        failed |= empirical_gain(model) > cfg["gamma"] * (1 + 1e-9)
    failed |= min_eig <= 0 or ratio > 1 + 1e-9
    print(f"{'OK  ' if not failed else 'FAIL'} {cfg.get('variant', 'contracting')} nx={cfg['n_state']}: {line}")

try:
    import robustnn  # noqa: F401
except ImportError:
    print("\nJAX reference (robustnn) not installed; skipping the cross-framework section.")
else:
    print("\nJAX reference robustnn.ren, weights transplanted (float64)\n")
    for cfg in CONFIGS:
        # the reference builds M with an (ny, ny) identity where it needs (min(nu,ny),)²,
        # so its non-square Lipschitz/dissipative path does not run at all
        if cfg.get("variant", "contracting") != "contracting" and cfg["n_input"] != cfg["n_output"]:
            continue
        model = build(cfg, seed=0)
        u = torch.randn(5, 30, cfg["n_input"], dtype=torch.float64)
        x0 = torch.randn(5, cfg["n_state"], dtype=torch.float64)
        with torch.no_grad():
            dev = rel(model(u, x0), jax_reference(model, u, x0))
        failed |= dev >= TOL
        print(f"{'OK  ' if dev < TOL else 'FAIL'} {cfg}: output {dev:.2e}")

print("\nall checks within tolerance" if not failed else "\nTOLERANCE EXCEEDED")
sys.exit(1 if failed else 0)
