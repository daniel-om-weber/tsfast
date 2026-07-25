"""Compare tsfast's R2DN against the paper equations in implicit form, and check its certificates.

Reference: Barbara, Wang & Manchester, "R2DN: Scalable Parameterization of Contracting and
Lipschitz Recurrent Deep Networks" (arXiv:2504.01250), eqs. (10), (16)-(18), (20)-(26);
reference implementation github.com/nic-barbara/R2DN ``robustnn/r2dn.py`` (MIT, JAX).

tsfast folds the implicit realization into an explicit one *once per forward*: it inverts
``E`` up front, so the rollout is a plain linear recurrence around one feedforward network.
The reference here does neither. It reads the certificate matrix ``H`` off the model,
partitions it independently, then runs the *implicit* system as written in the paper — a
general ``torch.linalg.solve`` against ``E`` at every step — and rebuilds each layer of the
1-Lipschitz network from an explicit matrix inverse in the opposite factor order. Both
implementations operate on the same parameter tensors and share the certificate
construction, so the printed deviations measure the explicit folding and the network's
realization; whether the construction itself is right is what the two sections below are for.

Deviations alone cannot show that a construction is *right*, only that two evaluations of it
agree, so the second section checks the certificate itself. It assembles the dissipation
inequality as a single quadratic form in ``(Δx, Δw, Δu)`` from the explicit realization and
the Lyapunov metric — reading nothing of how ``H`` was built — and reports its largest
eigenvalue, which must be negative. That is the guarantee, not a sample of it.

The optional final section transplants the weights into the authors' JAX implementation,
which is the only fully independent check of the ``H`` construction available. It is skipped
unless the reference is installed:

    uv pip install jax flax "robustnn @ git+https://github.com/nic-barbara/R2DN"

Two scope notes on that section. The reference implements the contracting R2DN at a rate of
exactly one, so the Lipschitz construction of §V-C and every ``alpha < 1`` configuration are
covered by the certificate section above instead. And its
``LBDN`` never sets the flag that would make its last layer the norm-bounded linear map of
the Lipschitz-bounded network it cites (Wang & Manchester, ICML 2023, as implemented in
``acfr/RobustNeuralNetworks.jl``); its stack therefore carries one more nonlinear layer than
tsfast's. The comparison follows the Julia reference and composes the JAX section's network
from the JAX ``SandwichLayer`` directly, so every formula is still checked against the
reference — only the layer-count bookkeeping differs.
"""

import sys

import numpy as np
import torch
from torch import nn

from tsfast.models.architectures.ren import R2DN
from tsfast.models.architectures.ren.common import _ACTS, _EPS
from tsfast.models.architectures.ren.lbdn import lbdn_forward

TOL = 1e-11


def rel(a, b):
    if b.numel() == 0:  # the Cayley Z block is empty whenever the matrix is square
        return 0.0
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class ImplicitR2DN(nn.Module):
    """Paper-equation forward pass in implicit form over the tsfast model's own parameters."""

    def __init__(self, model: R2DN):
        super().__init__()
        self.m = model

    def implicit(self):
        """``E``, ``𝒜`` and ``𝒫`` read off ``H`` (eqs. 18, 21-22)."""
        p = self.m.core.parameterization
        nx = self.m.spec.n_state
        h = p.hmatrix()
        return {
            "E": (h[:nx, :nx] + h[nx:, nx:] / self.m.spec.alpha**2 + p.Y - p.Y.mH) / 2,
            "A": h[nx:, :nx],
            "P": h[nx:, nx:],
        }

    def network(self, v):
        """The 1-Lipschitz network, with each layer's Cayley transform taken independently."""
        p = self.m.core.parameterization
        act = _ACTS[self.m.spec.act]
        h = v
        for layer in p.net.layers:
            w = layer.a / layer.XY.pow(2).sum().add(_EPS).sqrt() * layer.XY
            u, y = w[: layer.n_out], w[layer.n_out :]
            eye = torch.eye(layer.n_out, dtype=w.dtype)
            z = u - u.mH + y.mH @ y
            # the factors of the explicit form, in the reverse order and through an explicit
            # inverse: (I - Z)(I + Z)⁻¹ rather than solving (I + Z) against (I - Z)
            inv = torch.linalg.inv(eye + z)
            a_t, b = (eye - z) @ inv, (-2 * y @ inv).mH
            assert rel(a_t.mH @ a_t + b @ b.mH, eye) < 1e-10, "the stacked blocks must be an isometry"
            if layer.is_output:
                return h @ b.mH + layer.b
            psi = layer.d.exp()
            h = 2**0.5 * (act(2**0.5 * (h @ b.mH) / psi + layer.b) * psi) @ a_t.mH
        raise AssertionError("the network has no output layer")

    def forward(self, u, x0):
        p = self.m.core.parameterization
        i = self.implicit()
        d12, d21, d22 = (getattr(self.m.core.explicit(), n) for n in ("D12", "D21", "D22"))
        # The explicit bias, pushed back through the map the explicit form divided out.
        bx = p.bx @ i["E"].mH
        x, outs = x0, []
        for t in range(u.shape[1]):
            w = self.network(x @ p.C1.mH + u[:, t] @ d12.mH + p.bv)
            outs.append(x @ p.C2.mH + w @ d21.mH + u[:, t] @ d22.mH + p.by)
            rhs = x @ i["A"].mH + w @ p.B1.mH + u[:, t] @ p.B2.mH + bx
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


CONFIGS = [
    dict(n_input=1, n_output=1, n_state=4, n_nl=8, depth=1),
    dict(n_input=3, n_output=2, n_state=5, n_nl=6, depth=2, act="tanh", alpha=0.9),
    dict(n_input=2, n_output=4, n_state=3, n_nl=5, depth=3, act="sigmoid", polar=False),
    dict(n_input=2, n_output=2, n_state=4, n_nl=8, depth=2, init="random"),
    dict(n_input=2, n_output=2, n_state=3, n_nl=6, depth=2, act="tanh", alpha=0.5),
    dict(n_input=2, n_output=3, n_state=4, n_nl=6, depth=2, variant="lipschitz", gamma=0.7),
    dict(n_input=3, n_output=2, n_state=4, n_nl=6, depth=1, variant="lipschitz", gamma=5.0),
    dict(n_input=2, n_output=2, n_state=4, n_nl=6, depth=2, variant="lipschitz", gamma=1.5),
    dict(n_input=2, n_output=2, n_state=4, n_nl=6, depth=2, variant="lipschitz", gamma=2.0, alpha=0.8),
]


def build(cfg, seed):
    torch.manual_seed(seed)
    model = R2DN(backend="eager", **cfg).double()
    with torch.no_grad():  # move off the initialization, where much of H is structurally zero
        for p in model.core.parameterization.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    return model


def lyapunov_metric(model):
    """``M = Eᵀ P⁻¹ E``: the metric in which the certified contraction rate is exactly alpha."""
    i = ImplicitR2DN(model).implicit()
    return i["E"].mH @ torch.linalg.solve(i["P"], i["E"])


def dissipation_residual(model):
    """Largest eigenvalue of the form the certificate needs negative semidefinite.

    ``V(Δx⁺) - alpha²V(Δx) - s(Δu, Δy) + ‖Δv‖² - ‖Δw‖² ≤ 0`` for all ``(Δx, Δw, Δu)``, with
    ``V`` the Lyapunov function, ``s`` the supply rate and ``‖Δw‖ ≤ ‖Δv‖`` the only thing
    assumed of the network. Built from the explicit realization, so it is independent of the
    construction under test.
    """
    spec, e = model.spec, model.core.explicit()
    nx, nv, nu = spec.n_state, spec.n_nl, spec.n_input
    metric = lyapunov_metric(model)
    zeros = lambda r, c: torch.zeros(r, c, dtype=metric.dtype)  # noqa: E731
    eye_v, eye_u = torch.eye(nv, dtype=metric.dtype), torch.eye(nu, dtype=metric.dtype)
    step = torch.cat((e.A, e.B1, e.B2), dim=1)
    to_v = torch.cat((e.C1, zeros(nv, nv), e.D12), dim=1)
    to_w = torch.cat((zeros(nv, nx), eye_v, zeros(nv, nu)), dim=1)
    to_y = torch.cat((e.C2, e.D21, e.D22), dim=1)
    to_u = torch.cat((zeros(nu, nx), zeros(nu, nv), eye_u), dim=1)
    pad = torch.block_diag(metric, zeros(nv, nv), zeros(nu, nu))
    form = step.mH @ metric @ step - spec.alpha**2 * pad + to_v.mH @ to_v - to_w.mH @ to_w
    if spec.variant == "lipschitz":
        form = form + to_y.mH @ to_y / model.gamma - model.gamma * to_u.mH @ to_u
    else:
        # a contracting model is only claimed for trajectories under the same input
        keep = nx + nv
        form = form[:keep, :keep]
    return torch.linalg.eigvalsh((form + form.mH) / 2).max().item()


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


def jax_reference(model, x):
    """One step of the authors' JAX implementation with the tsfast weights transplanted.

    Returns the explicit LTI parameters, the per-layer network realizations and the network's
    output, so the comparison can name which part of the construction disagrees.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from robustnn import lbdn as jlbdn
    from robustnn import r2dn as jr2dn

    spec, p = model.spec, model.core.parameterization
    e = model.core.explicit()
    jm = jr2dn.ContractingR2DN(
        input_size=spec.n_input,
        state_size=spec.n_state,
        features=spec.n_nl,
        output_size=spec.n_output,
        hidden=spec.hidden,
        activation=getattr(jax.nn, spec.act) if spec.act != "tanh" else jnp.tanh,
        eps=p.eps,
        do_polar_param=p.polar,
        param_dtype=jnp.float64,
        init_method="random",
    )
    # identical parameter names, except that tsfast's free B2 is the implicit E·B2, so the
    # reference gets the explicit one, and the network's parameters sit in a nested dict
    free = {n: _j(t) for n, t in p.named_parameters() if not n.startswith("net.")}
    free["B2"] = _j(e.B2)
    free["network"] = {
        f"layers_{k}": {
            "XY": _j(layer.XY),
            "a": _j(layer.a),
            "b": _j(layer.b),
            # the reference builds this scaling for its output layer too, where it is unused
            "d": _j(layer.d) if not layer.is_output else jnp.zeros(layer.n_out),
        }
        for k, layer in enumerate(p.net.layers)
    }

    explicit = jm.direct_to_explicit({"params": free})
    h = _j(x @ e.C1.mH)
    for k, layer in enumerate(p.net.layers):
        jl = jlbdn.SandwichLayer(
            input_size=layer.n_in,
            features=layer.n_out,
            activation=jm.activation,
            is_output=layer.is_output,
            param_dtype=jnp.float64,
        )
        h = jl.apply(
            {"params": free["network"][f"layers_{k}"]},
            h,
            explicit.network_params.layers[k],
            method="_explicit_call",
        )
    return explicit, h


def _j(t):
    import jax.numpy as jnp

    return jnp.asarray(t.detach().cpu().numpy())


def _t(a):
    return torch.from_numpy(np.array(a))


print("implicit-form reference: max relative deviation of outputs and all gradients (float64)\n")
failed = False
for cfg in CONFIGS:
    model = build(cfg, seed=0)
    reference = ImplicitR2DN(model)
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
    metric_scale = max(1.0, lyapunov_metric(model).abs().max().item())
    residual = dissipation_residual(model)
    ratio = contraction_ratio(model)
    line = f"minEig(H) {min_eig:.2e}, dissipation {residual / metric_scale:+.2e}, contraction {ratio:.4f} of alpha"
    if cfg.get("variant") == "lipschitz":
        gain = empirical_gain(model)
        line += f", incremental gain {gain / cfg['gamma']:.3f} of gamma"
        failed |= gain > cfg["gamma"] * (1 + 1e-9)
    failed |= min_eig <= 0 or residual > 1e-9 * metric_scale or ratio > 1 + 1e-9
    print(f"{'OK  ' if not failed else 'FAIL'} {cfg.get('variant', 'contracting')} nx={cfg['n_state']}: {line}")

try:
    import robustnn  # noqa: F401
except ImportError:
    print("\nJAX reference (robustnn) not installed; skipping the cross-framework section.")
else:
    print("\nJAX reference robustnn.r2dn, weights transplanted (float64)\n")
    for cfg in CONFIGS:
        # the reference parameterizes the contracting case at a rate of exactly one, so the
        # other configurations are covered by the sections above instead
        if cfg.get("variant", "contracting") != "contracting" or cfg.get("alpha", 1.0) != 1.0:
            continue
        model = build(cfg, seed=0)
        e = model.core.explicit()
        x = torch.randn(5, cfg["n_state"], dtype=torch.float64)
        with torch.no_grad():
            explicit, net = jax_reference(model, x)
            devs = {n: rel(getattr(e, n), _t(getattr(explicit, n))) for n in ("A", "B1", "B2", "C1", "C2", "D12", "D21", "D22")}  # fmt: skip
            layers = [
                max(rel(mine.B, _t(theirs.B)), 0.0 if mine.A is None else rel(mine.A, _t(theirs.A_T)))
                for mine, theirs in zip(e.net, explicit.network_params.layers)
            ]
            mine_net = lbdn_forward(e.net, x @ e.C1.mH, _ACTS[model.spec.act])
            dev = max(max(devs.values()), max(layers), rel(mine_net, _t(net)))
        failed |= dev >= TOL
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{'OK  ' if dev < TOL else 'FAIL'} {label}: LTI {max(devs.values()):.2e}, network {rel(mine_net, _t(net)):.2e}")  # fmt: skip

print("\nall checks within tolerance" if not failed else "\nTOLERANCE EXCEEDED")
sys.exit(1 if failed else 0)
