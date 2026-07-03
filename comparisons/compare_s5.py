"""Compare tsfast's S5 against the official implementation of Smith et al.

Reference: the official JAX repository github.com/lindermanlab/S5 (MIT license),
published with "Simplified State Space Layers for Sequence Modeling" (ICLR 2023,
arXiv:2208.04933). Two comparisons are run:

1. Initialization: ``tsfast.models.s5.make_dplr_hippo`` against a verbatim NumPy
   transcription of the official ``make_DPLR_HiPPO`` (``s5/ssm_init.py``) — the diagonal
   approximation of HiPPO-LegS whose eigenvalue real parts are exactly -0.5.
2. Forward/backward: the tsfast ``S5`` layer against a transcription of the official
   forward semantics (``s5/ssm.py``: ZOH/bilinear discretization, complex diagonal
   recurrence, conjugate-symmetric ``2 Re(C x)`` output), run as a sequential loop with
   identical parameters. Both discretizations and both conjugate-symmetry modes are
   covered. Expected agreement: < 1e-12 in float64.

If the ``s5-pytorch`` package (github.com/i404788/s5-pytorch, an independent PyTorch
port) is installed, a third section cross-checks the forward pass against it in its
supported configuration (``conj_sym=False``, dense B/C initialization).
"""

import sys

import numpy as np
import torch

from tsfast.models.s5 import S5, make_dplr_hippo

TOL = 1e-12


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def reference_dplr_hippo(N):
    """make_DPLR_HiPPO transcribed from lindermanlab/S5, s5/ssm_init.py (MIT license)."""
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = -(np.tril(A) - np.diag(np.arange(N)))
    P = np.sqrt(np.arange(N) + 0.5)
    S = A + P[:, np.newaxis] * P[np.newaxis, :]
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    return Lambda_real + 1j * Lambda_imag, V


def reference_forward(layer, u):
    """Official S5 forward transcribed from lindermanlab/S5, s5/ssm.py (MIT license)."""
    lam_re = torch.clamp(layer.Lambda_re, max=-1e-4) if layer.clip_eigs else layer.Lambda_re
    Lambda = torch.complex(lam_re, layer.Lambda_im)
    B_tilde = torch.complex(layer.B_re, layer.B_im)
    C_tilde = torch.complex(layer.C_re, layer.C_im)
    Delta = layer.step_rescale * torch.exp(layer.log_step)
    if layer.discretization == "zoh":
        Lambda_bar = torch.exp(Lambda * Delta)
        B_bar = (1 / Lambda * (Lambda_bar - 1))[..., None] * B_tilde
    else:
        BL = 1 / (1 - (Delta / 2.0) * Lambda)
        Lambda_bar = BL * (1 + (Delta / 2.0) * Lambda)
        B_bar = (BL * Delta)[..., None] * B_tilde
    x = torch.zeros(u.shape[0], Lambda.shape[0], dtype=Lambda_bar.dtype)
    xs = []
    for k in range(u.shape[1]):
        x = Lambda_bar * x + u[:, k].to(B_bar.dtype) @ B_bar.T
        xs.append(x)
    xs = torch.stack(xs, dim=1)
    ys = (xs @ C_tilde.mT).real
    if layer.conj_sym:
        ys = 2 * ys
    return ys + layer.D * u


def compare_init():
    print("HiPPO-LegS diagonal initialization vs official make_DPLR_HiPPO")
    print(f"{'state size':<24}{'Lambda':>12}{'V':>12}")
    worst = 0.0
    for N in (4, 8, 32, 64):
        lam, V = make_dplr_hippo(N)
        lam_ref, V_ref = reference_dplr_hippo(N)
        e_lam = np.abs(lam - lam_ref).max()
        e_v = np.abs(V - V_ref).max()
        worst = max(worst, e_lam, e_v)
        print(f"N={N:<22}{e_lam:>12.1e}{e_v:>12.1e}")
    return worst


def compare_forward(d_model, d_state, conj_sym, discretization, L, seed):
    torch.manual_seed(seed)
    layer = S5(d_model, d_state, conj_sym=conj_sym, discretization=discretization).double()
    u = torch.randn(4, L, d_model, dtype=torch.float64)
    u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
    y_ref = reference_forward(layer, u_ref)
    w = torch.randn_like(y_ref)  # random adjoint probes all gradient components
    (y_ref * w).sum().backward()
    g_ref = [p.grad.clone() for p in layer.parameters()]
    for p in layer.parameters():
        p.grad = None
    y_ours = layer(u_ours)
    (y_ours * w).sum().backward()
    g_err = max(rel(p.grad, g) for p, g in zip(layer.parameters(), g_ref))
    return {"output": rel(y_ours, y_ref), "grad params": g_err, "grad u": rel(u_ours.grad, u_ref.grad)}


def compare_s5_pytorch():
    try:
        from s5.s5_model import S5SSM
    except ImportError:
        print("\ns5-pytorch not installed - skipping the cross-check (pip install s5-pytorch)")
        return 0.0
    torch.manual_seed(0)
    H, P, L = 3, 8, 200
    ours = S5(H, P, conj_sym=False, C_init="lecun_normal").double()
    lam, V = make_dplr_hippo(P)
    # constructor dtype only affects the initial values, which are all overwritten below
    ref = S5SSM(
        torch.tensor(lam, dtype=torch.complex64),
        torch.tensor(V, dtype=torch.complex64),
        torch.tensor(V.conj().T, dtype=torch.complex64),
        h=H,
        p=P,
        dt_min=0.001,
        dt_max=0.1,
        bcInit="dense",
    )
    with torch.no_grad():
        ref.Lambda = torch.nn.Parameter(torch.complex(ours.Lambda_re, ours.Lambda_im).detach().clone())
        ref.B = torch.nn.Parameter(torch.stack((ours.B_re, ours.B_im), dim=-1).detach().clone())
        ref.C = torch.nn.Parameter(torch.complex(ours.C_re, ours.C_im).detach().clone())
        ref.D = torch.nn.Parameter(ours.D.detach().clone())
        ref.log_step = torch.nn.Parameter(ours.log_step.detach().clone())
    u = torch.randn(L, H, dtype=torch.float64)
    err = rel(ours(u.unsqueeze(0)).squeeze(0), ref(u))  # s5-pytorch operates on unbatched [L, H]
    print(f"\ns5-pytorch cross-check (conj_sym=False): output deviation {err:.1e}")
    return err


def main():
    worst = compare_init()

    print("\nS5 layer forward/backward vs official semantics (float64)")
    keys = ["output", "grad params", "grad u"]
    print(f"{'configuration':<38}" + "".join(f"{k:>14}" for k in keys))
    for conj_sym in (True, False):
        for disc in ("zoh", "bilinear"):
            errs = compare_forward(3, 16, conj_sym, disc, L=500, seed=0)
            worst = max(worst, *errs.values())
            name = f"conj_sym={conj_sym!s:<6} {disc}"
            print(f"{name:<38}" + "".join(f"{errs[k]:>14.1e}" for k in keys))

    cross = compare_s5_pytorch()  # port accumulates float32-level noise internally
    print(f"\nworst relative deviation vs official semantics: {worst:.2e} (tolerance {TOL:.0e})")
    if worst > TOL or cross > 1e-9:
        sys.exit("FAIL: deviation exceeds tolerance")
    print("PASS: tsfast S5 matches the official implementation of Smith et al.")


if __name__ == "__main__":
    main()
