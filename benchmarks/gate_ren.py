#!/usr/bin/env python
"""Gate: does a contraction certificate cost accuracy, and is the certificate tight?

This is a *training* experiment, not a kernel timing script. A certificate that holds at
every parameter value is only worth its maintenance if it survives contact with real
identification data, which asks two questions at once — whether constraining the model class
costs accuracy, and whether the guarantee that buys is quantitatively meaningful. Both are
measured here, and both have to pass:

**Accuracy.** ``REN`` and ``R2DN`` trained at matched parameter count against two baselines
on the identibench simulation benchmarks. The honest opponent is the GRU: its state is a
convex combination of ``h_{t-1}`` and a ``tanh`` output, so ``‖h‖_inf <= 1`` for all time
and it cannot diverge either — a certified model has to win on something other than
boundedness. ``NeuralStateSpace`` is unconstrained and genuinely divergence-prone, the
easier comparison. The two certified models carry the same guarantees and differ only in
where nonlinear capacity lives (an equilibrium layer of ``n_nl`` neurons against a deep
1-Lipschitz network), so running them side by side at matched size is what says whether the
cheaper parameterization also fits as well. ``EMPS`` and ``CascadedTanks`` are integrating,
and are additionally run under ``FranSys``: contraction makes the initial state
self-correcting at rate ``alpha``, but the forgetting time ``1/(1-alpha)`` *is* the longest
time constant the model can represent, so no single ``alpha`` both forgets ``x0`` and
represents an integrator. Whether those plants *need* the state estimator or merely prefer
it is what the standalone-vs-FranSys column decides — against two controls, since that
column alone answers neither question: a standalone REN grown to the composition's budget
separates the estimator from the parameters it brings, and ``GRU+FranSys`` (gated units with
a state estimate, same budget and same diagnosis model) is the honest opponent in the
composed setting just as the bare GRU is in the standalone one.

**Certificate tightness.** A Lipschitz REN and a Lipschitz R2DN are trained with a
prescribed ``gamma`` and the report gives ``gamma_empirical / gamma_certified``, the
empirical value being a lower bound from power iteration over input-perturbation pairs. A
model that fits well but carries a vacuous certificate has failed at the only thing it was
built for, so this is pass/fail, not a diagnostic. Do not read the certified-Lipschitz
literature's 20x-50x gaps as a prediction: those are post-hoc bounds on freely-trained
networks, whereas these models prescribe ``gamma`` as a training constraint and optimization
pressure pushes them to consume the gain budget. The two are expected to differ: the REN
saturates its budget through the free feedthrough ``D22``, and the Lipschitz R2DN has
``D22 = 0`` and so nothing to saturate — which is why the ratio is reported per model rather
than pooled.

The contraction column reports what the *trained* model actually does — the worst per-step
decay in the Lyapunov metric of its own certificate, as a fraction of ``alpha``. A value
near 1 means contraction is binding; well below 1 means the model chose to forget faster
than it had to.

Usage:
    uv run python benchmarks/gate_ren.py
    uv run python benchmarks/gate_ren.py --benchmarks EMPS --epochs 40
    uv run python benchmarks/gate_ren.py --n-nl 16 --epochs 10   # quick pass
"""

import argparse
import copy
import time

import torch
from torch import nn

from tsfast.models._core.scaling import StandardScaler
from tsfast.models.architectures.ren import REN, R2DN, RENSpec
from tsfast.models.architectures.ren.backend_triton import fits
from tsfast.models.architectures.rnn import RNN
from tsfast.prediction import FranSysLearner
from tsfast.training import R2DNLearner, RENLearner, RNNLearner, SSMLearner, nrmse
from tsfast.tsdata import (
    create_dls_cascaded_tanks,
    create_dls_emps,
    create_dls_silverbox,
    create_dls_wh,
)

#: name -> (dataloader factory, n_skip, integrating)
BENCHMARKS = {
    "WH": (create_dls_wh, 20, False),
    "Silverbox": (create_dls_silverbox, 20, False),
    "CascadedTanks": (create_dls_cascaded_tanks, 20, True),
    "EMPS": (create_dls_emps, 100, True),
}

N_STATE = 8
GAMMA = 5.0
COMMON = dict(loss_func=nn.MSELoss(), metrics=[nrmse], input_norm=StandardScaler, output_norm=StandardScaler)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def match_width(build, target: int, lo: int = 4, hi: int = 512) -> int:
    """Smallest width whose model has at least ``target`` parameters."""
    while lo < hi:
        mid = (lo + hi) // 2
        if n_params(build(mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def inner(learner) -> REN | R2DN | None:
    """The certified model inside whatever the learner wrapped it in, if there is one."""
    return next((m for m in learner.model.modules() if isinstance(m, REN | R2DN)), None)


def cpu64(model: REN | R2DN) -> REN | R2DN:
    """A float64 CPU copy: the certificate margins are far below float32 resolution, and
    the trained model must come out of the diagnostics unchanged."""
    return copy.deepcopy(model).double().cpu()


def lyapunov_metric(model: REN | R2DN):
    """``M = Eᵀ P⁻¹ E``: the metric in which the certified rate is exactly ``alpha``."""
    p = model.core.parameterization
    h = p.hmatrix().double()
    nx = model.spec.n_state
    match model:
        case REN():
            # H is (2nx+nv) square, and the storage block sits past the nv neuron rows
            off, y = nx + model.spec.n_nl, p.Y1.double()
        case _:
            off, y = nx, p.Y.double()
    e = (h[:nx, :nx] + h[off:, off:] / model.spec.alpha**2 + y - y.mH) / 2
    return e.mH @ torch.linalg.solve(h[off:, off:], e)


@torch.no_grad()
def contraction_ratio(model: REN | R2DN, steps=60, batch=16):
    """Worst per-step decay in the Lyapunov metric, as a fraction of ``alpha``."""
    model = cpu64(model)
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


def empirical_gain(model: REN | R2DN, length=200, iters=40, batch=8):
    """Power iteration on the input-to-output Jacobian, both trajectories from ``x0 = 0``.

    The certificate is stated for a fixed initial state (Def. 3 of the paper), so this
    measures exactly the quantity it bounds and nothing wider.
    """
    model = cpu64(model)
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
        dy = model.core.rollout(e, ud, x0)[0] - y_ref
        norm = dy.detach().flatten(1).norm(dim=1)
        best = max(best, norm.max().item())
        d = torch.autograd.grad((dy * (dy / (norm.view(-1, 1, 1) + 1e-30)).detach()).sum(), ud)[0].detach()
    return best


def train(make_learner, epochs, lrs):
    """Best over a small learning-rate sweep: ``(NRMSE, its learner, lr, wall-clock seconds)``.

    One shared learning rate is not a fair matched-capacity comparison: at 3e-3 the
    unconstrained ``NeuralStateSpace`` sits on the constant-predictor plateau on three of
    the four benchmarks and never leaves it, while at 1e-3 it trains normally. Each
    architecture gets its own best rate so the accuracy column measures the architecture.

    A learning rate that drives the model into a numerically degenerate corner is reported
    and dropped rather than raised: for a certified model that corner is reachable (the
    optimizer is pulled toward the edge of the gain budget), and losing the other twenty runs
    to it would say nothing about the architecture.
    """
    t0 = time.perf_counter()
    best = (float("inf"), None, None)
    for lr in lrs:
        torch.manual_seed(0)
        learner = make_learner()
        try:
            learner.fit(epochs, lr)
        except (torch._C._LinAlgError, RuntimeError) as e:
            print(f"    lr {lr:g} abandoned: {type(e).__name__}: {str(e).splitlines()[0][:90]}", flush=True)
            continue
        score = min(row[2] for row in learner.recorder)
        if score < best[0]:
            best = (score, learner, lr)
    return (*best, time.perf_counter() - t0)


def make_fransys(dls, prognosis, args):
    """FranSys around a given prognosis, with the diagnosis model held fixed across runs."""
    return FranSysLearner(
        dls,
        init_sz=args.init_sz,
        attach_output=True,
        prognosis=prognosis,
        hidden_size=args.diag_hidden,
        loss_func=nn.MSELoss(),
        metrics=[nrmse],
        show_bar=args.bar,
    )


def ren_prognosis(inp, out, n_nl, args):
    return REN(inp, out, n_state=N_STATE, n_nl=n_nl, backend=args.backend, return_state=True)


def r2dn_prognosis(inp, out, n_nl, args):
    return R2DN(inp, out, n_state=N_STATE, n_nl=n_nl, depth=args.depth, backend=args.backend, return_state=True)


def gru_prognosis(inp, hidden):
    return RNN(inp, hidden_size=hidden, num_layers=1, ret_full_hidden=True)


def run_benchmark(name, args):
    factory, n_skip, integrating = BENCHMARKS[name]
    dls = factory()
    inp, out = dls.one_batch()[0].shape[-1], dls.one_batch()[1].shape[-1]

    ren_kwargs = dict(n_state=N_STATE, n_nl=args.n_nl, backend=args.backend, **COMMON)
    target = n_params(REN(inp, out, n_state=N_STATE, n_nl=args.n_nl))
    # the R2DN's width is what varies to reach the REN's budget: its depth is the axis the
    # architecture is *for*, so holding it fixed keeps the comparison about the parameterization
    r2dn_nl = match_width(lambda nl: R2DN(inp, out, n_state=N_STATE, n_nl=nl, depth=args.depth), target, hi=256)
    r2dn_kwargs = dict(n_state=N_STATE, n_nl=r2dn_nl, depth=args.depth, backend=args.backend, **COMMON)
    gru_h = match_width(lambda h: RNNLearner(dls, hidden_size=h, show_bar=False, **COMMON).model, target)
    ssm_h = match_width(
        lambda h: SSMLearner(dls, n_state=N_STATE, hidden_size=h, show_bar=False, **COMMON).model, target
    )

    runs = [
        ("REN", lambda: RENLearner(dls, n_skip=n_skip, show_bar=args.bar, **ren_kwargs)),
        ("R2DN", lambda: R2DNLearner(dls, n_skip=n_skip, show_bar=args.bar, **r2dn_kwargs)),
        ("GRU", lambda: RNNLearner(dls, hidden_size=gru_h, n_skip=n_skip, show_bar=args.bar, **COMMON)),
        (
            "NeuralStateSpace",
            lambda: SSMLearner(dls, n_state=N_STATE, hidden_size=ssm_h, n_skip=n_skip, show_bar=args.bar, **COMMON),
        ),
    ]
    note = f", R2DN n_nl={r2dn_nl}"
    if integrating:
        # Two controls, because "REN+FranSys beats REN" answers neither question on its own.
        # The estimator brings its own parameters, so the capacity control is a standalone
        # REN grown to the composition's total budget. And FranSys around a GRU is the
        # honest opponent in the composed setting exactly as a bare GRU is in the standalone
        # one — gated units with a state estimate — so it gets the same total budget and the
        # same diagnosis model, leaving the prognosis as the only difference.
        fs_total = n_params(make_fransys(dls, ren_prognosis(inp, out, args.n_nl, args), args).model)
        big_nl = match_width(lambda nl: REN(inp, out, n_state=N_STATE, n_nl=nl), fs_total, hi=256)
        gru_fs_h = match_width(lambda h: make_fransys(dls, gru_prognosis(inp, h), args).model, fs_total)
        runs += [
            ("REN+FranSys", lambda: make_fransys(dls, ren_prognosis(inp, out, args.n_nl, args), args)),
            ("R2DN+FranSys", lambda: make_fransys(dls, r2dn_prognosis(inp, out, r2dn_nl, args), args)),
            ("GRU+FranSys", lambda: make_fransys(dls, gru_prognosis(inp, gru_fs_h), args)),
            (
                f"REN n_nl={big_nl}",
                lambda: RENLearner(
                    dls, n_state=N_STATE, n_nl=big_nl, backend=args.backend, n_skip=n_skip, show_bar=args.bar, **COMMON
                ),
            ),
        ]
        note += f", FranSys budget {fs_total} matched by n_nl={big_nl} / GRU h={gru_fs_h}"
        if not fits(RENSpec(N_STATE, inp, out, big_nl, "contracting", 1.0, "tanh")):
            note += " (off the fused envelope, eager)"
    runs += [
        (
            f"REN lipschitz(g={GAMMA})",
            lambda: RENLearner(dls, n_skip=n_skip, variant="lipschitz", gamma=GAMMA, show_bar=args.bar, **ren_kwargs),
        ),
        (
            f"R2DN lipschitz(g={GAMMA})",
            lambda: R2DNLearner(dls, n_skip=n_skip, variant="lipschitz", gamma=GAMMA, show_bar=args.bar, **r2dn_kwargs),
        ),
    ]

    rows, ratios = [], {}
    for label, make in runs:
        best, learner, lr, secs = train(make, args.epochs, args.lrs)
        if learner is None:
            rows.append((label, 0, best, secs, "every learning rate abandoned"))
            continue
        extra = f"lr {lr:g}"
        # the diagnostics follow the model, not the label: a certified model reports what its
        # own certificate does whether it was trained bare or under FranSys
        if (m := inner(learner)) is not None:
            rho = torch.linalg.eigvals(m.core.explicit().A.detach().double()).abs().max().item()
            extra += f", contraction {contraction_ratio(m):.3f} of alpha, rho(A) {rho:.4f}"
            if m.spec.variant == "lipschitz":
                ratios[type(m).__name__] = ratio = empirical_gain(m) / GAMMA
                extra += f", gamma_emp/gamma_cert {ratio:.3f}"
        rows.append((label, n_params(learner.model), best, secs, extra))

    print(f"\n=== {name} ===  n_skip={n_skip}  matched params ~{target}  (GRU h={gru_h}, NSS h={ssm_h}{note})")
    print(f"{'model':>24s}{'params':>9s}{'NRMSE':>10s}{'train s':>9s}   notes")
    for label, params, best, secs, extra in rows:
        print(f"{label:>24s}{params:>9d}{best:>10.4f}{secs:>9.1f}   {extra}", flush=True)
    return ratios


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4], help="swept per architecture")
    p.add_argument("--n-nl", type=int, default=32, help="REN equilibrium-layer width")
    p.add_argument("--depth", type=int, default=2, help="R2DN nonlinear depth; its width matches the REN's budget")
    p.add_argument("--init-sz", type=int, default=50, help="FranSys diagnosis window")
    # kept small on purpose: the diagnosis model is pure overhead against the parameter
    # budget, and at the library default (100) it alone outweighs the REN more than tenfold
    p.add_argument("--diag-hidden", type=int, default=32, help="FranSys diagnosis model width")
    p.add_argument("--backend", default="auto")
    p.add_argument("--bar", action="store_true", help="show per-epoch progress bars")
    args = p.parse_args()
    ratios = [run_benchmark(name, args) for name in args.benchmarks]
    print()
    for kind in ("REN", "R2DN"):
        seen = [r[kind] for r in ratios if kind in r]
        print(f"certificate tightness across benchmarks, {kind}: {min(seen):.3f} .. {max(seen):.3f} of gamma")


if __name__ == "__main__":
    main()
