#!/usr/bin/env python
"""Gate: does gating the NeuralStateSpace transition earn its kernel work?

This is a *training* experiment, not a kernel timing script. It settles whether
``NeuralStateSpace(gate=...)`` is worth fusing, with two criteria that must both pass.

**Mechanism.** ``‖∂L/∂x_0‖`` with the loss restricted to the last few steps, so the adjoint
has to traverse the whole rollout. Reported in float64, since the ungated decay runs far
below float32 resolution and would otherwise read as a flat zero.

The bar is *not* a flat gradient-versus-horizon curve. Only ``residual``'s exactly-unit
carry achieves that, and an adjoint that never decays is the wrong prior for a dissipative
plant — the model would be unable to express that a real system forgets its initial
condition. What a principled gate must deliver instead is *control*: the per-step decay
rate implies a gradient time constant, and that time constant has to track ``gate_tmax``.
The ``tmax`` sweep is therefore the criterion, and the horizon table is context for it.

**Accuracy.** Simulation NRMSE at matched parameter count against two baselines. The
ungated ``NeuralStateSpace`` is the thing being improved. The GRU is the honest opponent:
its state is already a convex combination of ``h_{t-1}`` and a ``tanh`` output, so it has
had this exact gate since 2014 and a gated SSM that does not reach it has bought nothing
the library did not already offer. What a gated SSM keeps and a GRU does not is the
physical-state contract — ``n_state`` is the system order and chunked rollouts are exactly
equivalent to the full sequence — so closing the gap is the whole claim.

CascadedTanks and EMPS are integrating: a per-channel pole near 1 is exactly what ``leak``
and ``gru`` can express directly and the ungated model has to discover. They are the
informative columns, and ``--benchmarks CascadedTanks EMPS`` is the quick pass.

Usage:
    uv run python benchmarks/gate_ssm_gating.py
    uv run python benchmarks/gate_ssm_gating.py --benchmarks CascadedTanks EMPS --epochs 20
    uv run python benchmarks/gate_ssm_gating.py --skip-accuracy     # mechanism only, seconds
"""

import argparse
import time

import torch
from torch import nn

from tsfast.models._core.scaling import StandardScaler
from tsfast.models.architectures.ssm import NeuralStateSpace
from tsfast.training import RNNLearner, SSMLearner, nrmse
from tsfast.tsdata import (
    create_dls_cascaded_tanks,
    create_dls_emps,
    create_dls_silverbox,
    create_dls_wh,
)

#: name -> (dataloader factory, n_skip)
BENCHMARKS = {
    "WH": (create_dls_wh, 20),
    "Silverbox": (create_dls_silverbox, 20),
    "CascadedTanks": (create_dls_cascaded_tanks, 20),
    "EMPS": (create_dls_emps, 100),
}

GATES = ("none", "leak", "gru", "residual")
N_STATE = 8
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


def x0_grad(gate: str, length: int, tmax: float, args, tail: int = 10) -> float:
    """``‖∂L/∂x_0‖`` for a loss on the last ``tail`` steps of a length-``length`` rollout."""
    torch.manual_seed(0)
    model = NeuralStateSpace(1, 1, n_state=N_STATE, hidden_size=args.hidden, gate=gate, gate_tmax=tmax, backend="eager")
    model = model.to(args.device).double()
    u = torch.randn(4, length, 1, device=args.device, dtype=torch.float64)
    x0 = torch.randn(4, N_STATE, device=args.device, dtype=torch.float64, requires_grad=True)
    model(u, x0)[:, -tail:].pow(2).mean().backward()
    return x0.grad.norm().item()


def gradient_time_constant(gate: str, tmax: float, args, lo: int = 200, hi: int = 1000) -> float:
    """Steps over which the state adjoint decays by ``1/e``, fitted between two horizons."""
    g_lo, g_hi = x0_grad(gate, lo, tmax, args), x0_grad(gate, hi, tmax, args)
    if g_lo <= 0.0 or g_hi <= 0.0:
        return 0.0  # underflowed: dead long before the fit window
    rate = (g_hi / g_lo) ** (1.0 / (hi - lo))
    return float("inf") if rate >= 1.0 else -1.0 / torch.log(torch.tensor(rate)).item()


def report_mechanism(args):
    """Horizon table plus the ``gate_tmax`` sweep that is the actual criterion."""
    lengths = [10, 20, 50, 100, 250, 500, 1000, 2000]
    print(f"\n=== mechanism: ‖dL/dx0‖, float64, loss on the last 10 steps (n_state={N_STATE}) ===")
    print(f"{'L':>6}" + "".join(f"{g:>13s}" for g in GATES))
    for length in lengths:
        cells = "".join(f"{x0_grad(g, length, args.gate_tmax, args):>13.3e}" for g in GATES)
        print(f"{length:>6}" + cells, flush=True)

    print("\n=== criterion: does gate_tmax control the gradient horizon? (fitted over L=200..1000) ===")
    print(f"{'gate_tmax':>10}" + "".join(f"{g:>13s}" for g in GATES))
    fitted = {}
    for tmax in args.tmax_sweep:
        cells = ""
        for gate in GATES:
            t = gradient_time_constant(gate, tmax, args)
            fitted.setdefault(gate, []).append(t)
            cells += f"{t:>13.1f}" if t not in (0.0, float("inf")) else f"{'dead' if t == 0.0 else 'inf':>13s}"
        print(f"{tmax:>10.0f}" + cells, flush=True)

    print("\nverdict per variant:")
    for gate in GATES:
        ts = fitted[gate]
        monotone = all(b >= a for a, b in zip(ts, ts[1:]))
        within = [t / m for t, m in zip(ts, args.tmax_sweep) if m <= 300 and t not in (0.0, float("inf"))]
        tracks = bool(within) and all(0.5 <= r <= 2.0 for r in within)
        status = "PASS" if monotone and tracks else "fail"
        detail = "no finite horizon" if not within else f"T/tmax in [{min(within):.2f}, {max(within):.2f}] up to 300"
        print(f"  {gate:>9s}  {status}  monotone={monotone}  {detail}")


def train(make_learner, epochs, lrs):
    """Best over a small learning-rate sweep: ``(NRMSE, params, lr, wall-clock seconds)``.

    One shared rate is not a matched-capacity comparison: chrono initialization changes the
    scale of the effective update, so the gated variants tolerate rates the ungated model
    diverges at. Each variant gets its own best rate so the column measures the gate.
    """
    t0 = time.perf_counter()
    best = (float("inf"), 0, None)
    for lr in lrs:
        torch.manual_seed(0)
        learner = make_learner()
        learner.fit(epochs, lr)
        score = min(row[2] for row in learner.recorder)
        if score < best[0]:
            best = (score, n_params(learner.model), lr)
    return (*best, time.perf_counter() - t0)


def run_benchmark(name, args):
    factory, n_skip = BENCHMARKS[name]
    dls = factory()

    # Match every variant to the ungated model at the reference width: the input-dependent
    # gates double the final layer, which is otherwise a free capacity bump.
    target = n_params(SSMLearner(dls, n_state=N_STATE, hidden_size=args.hidden, show_bar=False, **COMMON).model)
    widths = {
        g: args.hidden
        if g in ("none", "leak")
        else match_width(
            lambda h, g=g: SSMLearner(dls, n_state=N_STATE, hidden_size=h, gate=g, show_bar=False, **COMMON).model,
            target,
        )
        for g in GATES
    }
    gru_h = match_width(lambda h: RNNLearner(dls, hidden_size=h, show_bar=False, **COMMON).model, target)

    runs = [
        (
            f"SSM gate={g}",
            lambda g=g: SSMLearner(
                dls,
                n_state=N_STATE,
                hidden_size=widths[g],
                gate=g,
                gate_tmax=args.gate_tmax,
                n_skip=n_skip,
                show_bar=args.bar,
                **COMMON,
            ),
        )
        for g in GATES
    ]
    runs.append(("GRU", lambda: RNNLearner(dls, hidden_size=gru_h, n_skip=n_skip, show_bar=args.bar, **COMMON)))

    rows = []
    for label, make in runs:
        best, params, lr, secs = train(make, args.epochs, args.lrs)
        rows.append((label, params, best, secs, f"lr {lr:g}"))

    widths_note = ", ".join(f"{g} h={widths[g]}" for g in GATES)
    print(f"\n=== {name} ===  n_skip={n_skip}  matched params ~{target}  ({widths_note}, GRU h={gru_h})")
    print(f"{'model':>20s}{'params':>9s}{'NRMSE':>10s}{'train s':>9s}   notes")
    for label, params, best, secs, extra in rows:
        print(f"{label:>20s}{params:>9d}{best:>10.4f}{secs:>9.1f}   {extra}", flush=True)

    scores = {label: best for label, _, best, _, _ in rows}
    ungated, gru = scores["SSM gate=none"], scores["GRU"]
    gap = ungated - gru
    if gap <= 0:
        print(f"  ungated already at or past the GRU ({ungated:.4f} vs {gru:.4f}); gap criterion not applicable")
        return None
    closed = {g: (ungated - scores[f"SSM gate={g}"]) / gap for g in GATES if g != "none"}
    print("  fraction of the ungated->GRU gap closed: " + ", ".join(f"{g} {f:.2f}" for g, f in closed.items()))
    return closed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4], help="swept per variant")
    p.add_argument("--hidden", type=int, default=64, help="reference transition width, sets the parameter budget")
    p.add_argument("--gate-tmax", type=float, default=100.0, help="chrono-init longest time constant")
    p.add_argument("--tmax-sweep", type=float, nargs="+", default=[10, 30, 100, 300, 1000])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-accuracy", action="store_true", help="mechanism criterion only")
    p.add_argument("--bar", action="store_true", help="show per-epoch progress bars")
    args = p.parse_args()

    report_mechanism(args)
    if args.skip_accuracy:
        return
    closed = [c for c in (run_benchmark(name, args) for name in args.benchmarks) if c is not None]
    if not closed:
        print("\nno benchmark produced an ungated->GRU gap to close")
        return
    print("\nmedian fraction of the ungated->GRU gap closed, across benchmarks:")
    for gate in GATES[1:]:
        fracs = sorted(c[gate] for c in closed)
        print(f"  {gate:>9s} {fracs[len(fracs) // 2]:.2f}   (per benchmark: {', '.join(f'{f:.2f}' for f in fracs)})")


if __name__ == "__main__":
    main()
