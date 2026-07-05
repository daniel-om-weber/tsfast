"""Compare tsfast's dynoNet blocks against the authors' reference implementation.

Reference: the ``dynonet`` package by M. Forgione and D. Piga (``pip install dynonet``),
the implementation published with "dynoNet: A neural network architecture for learning
dynamical systems" (Int. J. Adapt. Control Signal Process., 2021, arXiv:2006.02250).

Compared: ``tsfast.models.architectures.dynonet.LinearDynamicalOperator`` (the G-block, a MIMO bank of
rational transfer functions) against ``dynonet.lti.MimoLinearDynamicalOperator``. The
parameterizations are identical (``b_coeff`` numerator taps, ``a_coeff`` monic-denominator
coefficients), so parameters are copied 1:1 and the comparison isolates the execution
path: the reference filters sequentially per pair (scipy-style lfilter semantics in
PyTorch), tsfast runs the same recurrence sequence-parallel via a log-doubling scan in
companion form.

For every configuration the script reports the maximum relative deviation of the output
and of the gradients w.r.t. ``b_coeff``, ``a_coeff``, and the input, in float64 with
denominator coefficients drawn from stable pole pairs. Low-order blocks agree at
~1e-15; for higher denominator orders the (mathematically equivalent) execution orders
accumulate float64 round-off differently because companion matrices of high-order
polynomials are ill-conditioned and the scan squares them repeatedly, so the deviation
grows to ~1e-11 at ``na=6``. Tolerance: 1e-10.
"""

import sys

import numpy as np
import torch

from tsfast.models.architectures.dynonet import LinearDynamicalOperator

TOL = 1e-10


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def stable_a_coeffs(out_ch, in_ch, na, rng):
    """Monic-denominator coefficients from well-damped poles inside the unit circle
    (radius below 0.7, so the recurrences themselves stay well-conditioned)."""
    a = np.zeros((out_ch, in_ch, na))
    for i in range(out_ch):
        for j in range(in_ch):
            poles = []
            while na - len(poles) >= 2:
                p = rng.uniform(0.3, 0.7) * np.exp(1j * rng.uniform(0, np.pi))
                poles += [p, p.conjugate()]
            if len(poles) < na:
                poles.append(rng.uniform(-0.7, 0.7))
            a[i, j] = np.real(np.poly(poles))[1:]
    return torch.tensor(a)


def compare(in_ch, out_ch, nb, na, L, seed):
    from dynonet.lti import MimoLinearDynamicalOperator

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ours = LinearDynamicalOperator(in_ch, out_ch, nb=nb, na=na).double()
    ref = MimoLinearDynamicalOperator(in_ch, out_ch, n_b=nb, n_a=na).double()
    with torch.no_grad():
        b = torch.randn(out_ch, in_ch, nb, dtype=torch.float64) * 0.5
        a = stable_a_coeffs(out_ch, in_ch, na, rng)
        for m in (ours, ref):
            m.b_coeff.copy_(b)
            m.a_coeff.copy_(a)

    u = torch.randn(4, L, in_ch, dtype=torch.float64)
    u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
    y_ref, y_ours = ref(u_ref), ours(u_ours)
    w = torch.randn_like(y_ref)  # random adjoint probes all gradient components
    (y_ref * w).sum().backward()
    (y_ours * w).sum().backward()
    return {
        "output": rel(y_ours, y_ref),
        "grad b": rel(ours.b_coeff.grad, ref.b_coeff.grad),
        "grad a": rel(ours.a_coeff.grad, ref.a_coeff.grad),
        "grad u": rel(u_ours.grad, u_ref.grad),
    }


def main():
    try:
        import dynonet  # noqa: F401
    except ImportError:
        sys.exit("reference package missing: pip install dynonet")

    configs = [
        ("SISO short", dict(in_ch=1, out_ch=1, nb=4, na=3, L=200)),
        ("SISO long", dict(in_ch=1, out_ch=1, nb=8, na=2, L=5000)),
        ("MIMO 2x3", dict(in_ch=2, out_ch=3, nb=3, na=2, L=500)),
        ("MIMO 4x4 high order", dict(in_ch=4, out_ch=4, nb=16, na=6, L=1000)),
    ]
    print(f"{'configuration':<22}" + "".join(f"{k:>12}" for k in ("output", "grad b", "grad a", "grad u")))
    worst = 0.0
    for name, cfg in configs:
        errs = compare(**cfg, seed=0)
        worst = max(worst, *errs.values())
        print(f"{name:<22}" + "".join(f"{v:>12.2e}" for v in errs.values()))
    print(f"\nworst relative deviation: {worst:.2e} (tolerance {TOL:.0e})")
    if worst > TOL:
        sys.exit("FAIL: deviation exceeds tolerance")
    print("PASS: tsfast dynoNet matches the authors' implementation")


if __name__ == "__main__":
    main()
