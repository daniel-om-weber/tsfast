#!/usr/bin/env python
"""Silverbox: does a certificate help where the model has to extrapolate?

The benchmark's test set is not homogeneous. ``test_1`` is the arrow-head section, driven to
amplitudes about 47% beyond anything in the training data, while ``test_0`` and ``test_2``
stay inside it. The aggregate NRMSE that ``gate_ren.py`` reports pools all three, so it cannot
separate interpolation from extrapolation — this script rolls each trained model over each
test file on its own.

What the certificates can and cannot do here is worth stating before the numbers, since the
two halves come apart:

- **Boundedness is guaranteed.** A Lipschitz model driven outside its training range moves its
  output by at most ``gamma`` times the input perturbation, whatever the input. An
  unconstrained model has no such protection, and that is the failure mode the arrow section
  is built to provoke.
- **Fidelity is not.** The certificate bounds the learned map, never its agreement with the
  plant. Silverbox is a Duffing oscillator with a *hardening* cubic spring, so its incremental
  gain is highest inside the training range and lower on the arrow (the amplitude ratio
  ``max|y|/max|u|`` falls from 2.13 to 2.01). A ``gamma`` fitted where the data is therefore
  has slack exactly where the extrapolation happens: the constraint is inactive where it would
  have to bind to help. On a *softening* plant the same constraint would instead forbid
  correct extrapolation.

So the expectation is graceful degradation rather than better degradation, and the point of
the script is to measure which.

Both metrics are reported per file, because they answer different questions. ``NRMSE``
normalizes by each file's own target variance and so is comparable with the gate's numbers;
absolute ``RMSE`` does not, and is what shows that a wider-swinging signal is genuinely
harder rather than merely differently scaled.

Usage:
    uv run python benchmarks/silverbox_extrapolation.py
    uv run python benchmarks/silverbox_extrapolation.py --epochs 5      # quick pass
"""

import argparse

import h5py
import identibench as idb
import numpy as np
import torch

from gate_ren import COMMON, GAMMA, N_STATE, inner, match_width, n_params, train

from tsfast.models.architectures.ren import REN, R2DN
from tsfast.training import R2DNLearner, RENLearner, RNNLearner, SSMLearner
from tsfast.tsdata import create_dls_silverbox

N_SKIP = 20


def test_files() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(name, u, y)`` for each Silverbox test file, in benchmark order."""
    spec = idb.BenchmarkSilverbox_Simulation
    spec.ensure_datasets_exist()
    out = []
    for path in spec.test_files():
        with h5py.File(path, "r") as h:
            out.append((path.stem, np.array(h[spec.u_cols[0]]).ravel(), np.array(h[spec.y_cols[0]]).ravel()))
    return out


@torch.no_grad()
def simulate(model, u: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Free-running simulation over a whole test file: ``(RMSE, NRMSE)``.

    One rollout from a zero initial state — no state estimator and no teacher forcing, which
    is what a simulation benchmark asks for and what the contraction certificate makes
    admissible. The first ``N_SKIP`` samples are dropped as the cold-start transient, matching
    the training protocol.
    """
    device = next(model.parameters()).device
    ut = torch.tensor(u, dtype=torch.float32, device=device).reshape(1, -1, 1)
    yt = torch.tensor(y, dtype=torch.float32, device=device).reshape(1, -1, 1)
    pred = model(ut)
    if isinstance(pred, tuple):
        pred = pred[0]
    err, targ = (pred - yt)[:, N_SKIP:], yt[:, N_SKIP:]
    rmse = err.pow(2).mean().sqrt().item()
    return rmse, (err.pow(2).mean() / targ.var()).sqrt().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4], help="swept per architecture")
    p.add_argument("--n-nl", type=int, default=32, help="REN equilibrium-layer width")
    p.add_argument("--depth", type=int, default=2, help="R2DN nonlinear depth")
    p.add_argument("--backend", default="auto")
    args = p.parse_args()

    dls = create_dls_silverbox()
    files = test_files()
    ren_kwargs = dict(n_state=N_STATE, n_nl=args.n_nl, backend=args.backend, **COMMON)
    target = n_params(REN(1, 1, n_state=N_STATE, n_nl=args.n_nl))
    r2dn_nl = match_width(lambda nl: R2DN(1, 1, n_state=N_STATE, n_nl=nl, depth=args.depth), target, hi=256)
    r2dn_kwargs = dict(n_state=N_STATE, n_nl=r2dn_nl, depth=args.depth, backend=args.backend, **COMMON)
    gru_h = match_width(lambda h: RNNLearner(dls, hidden_size=h, show_bar=False, **COMMON).model, target)
    ssm_h = match_width(
        lambda h: SSMLearner(dls, n_state=N_STATE, hidden_size=h, show_bar=False, **COMMON).model, target
    )

    runs = [
        ("REN", lambda: RENLearner(dls, n_skip=N_SKIP, show_bar=False, **ren_kwargs)),
        (
            f"REN lipschitz(g={GAMMA})",
            lambda: RENLearner(dls, n_skip=N_SKIP, variant="lipschitz", gamma=GAMMA, show_bar=False, **ren_kwargs),
        ),
        ("R2DN", lambda: R2DNLearner(dls, n_skip=N_SKIP, show_bar=False, **r2dn_kwargs)),
        (
            f"R2DN lipschitz(g={GAMMA})",
            lambda: R2DNLearner(dls, n_skip=N_SKIP, variant="lipschitz", gamma=GAMMA, show_bar=False, **r2dn_kwargs),
        ),
        ("GRU", lambda: RNNLearner(dls, hidden_size=gru_h, n_skip=N_SKIP, show_bar=False, **COMMON)),
        (
            "NeuralStateSpace",
            lambda: SSMLearner(dls, n_state=N_STATE, hidden_size=ssm_h, n_skip=N_SKIP, show_bar=False, **COMMON),
        ),
    ]

    print(f"matched params ~{target}  (GRU h={gru_h}, NSS h={ssm_h}, R2DN n_nl={r2dn_nl})\n")
    print(f"{'file':>10s}{'N':>9s}{'max|u|':>9s}{'max|y|':>9s}   (train max|u| 0.1014)")
    for name, u, y in files:
        print(f"{name:>10s}{len(u):>9d}{np.abs(u).max():>9.4f}{np.abs(y).max():>9.4f}")

    rows = []
    for label, make in runs:
        _, learner, lr, _ = train(make, args.epochs, args.lrs)
        if learner is None:
            continue
        model = learner.model.eval()
        scores = [simulate(model, u, y) for _, u, y in files]
        gain = ""
        if (m := inner(learner)) is not None and m.spec.variant == "lipschitz":
            gain = f"  gamma_emp/gamma_cert {empirical_ratio(m):.3f}"
        rows.append((label, lr, scores, gain))

    for metric, idx in (("RMSE (absolute)", 0), ("NRMSE (per-file variance)", 1)):
        print(f"\n{metric}")
        header = f"{'model':>24s}" + "".join(f"{n:>12s}" for n, _, _ in files) + f"{'arrow/interp':>14s}"
        print(header)
        print("-" * len(header))
        for label, lr, scores, gain in rows:
            vals = [s[idx] for s in scores]
            # test_1 is the arrow; the other two stay inside the training amplitude range
            ratio = vals[1] / ((vals[0] + vals[2]) / 2)
            print(f"{label:>24s}" + "".join(f"{v:>12.4f}" for v in vals) + f"{ratio:>14.2f}x" + (gain if idx else ""))
        print(f"{'':>24s}   lr " + ", ".join(f"{lr:g}" for _, lr, _, _ in rows))


def empirical_ratio(model) -> float:
    from gate_ren import empirical_gain

    return empirical_gain(model) / GAMMA


if __name__ == "__main__":
    main()
