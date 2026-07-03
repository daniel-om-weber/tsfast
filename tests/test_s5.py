"""Tests for tsfast.models.s5 (make_dplr_hippo, S5, DeepS5)."""

import numpy as np
import pytest
import torch

from tsfast.models.s5 import S5, DeepS5, make_dplr_hippo


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _run(m, backend, u):
    """Forward + backward on a cloned leaf; returns (out, param grads, du)."""
    m.backend = backend
    for p in m.parameters():
        p.grad = None
    u = u.clone().requires_grad_()
    out = m(u)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    return out, [p.grad.clone() for p in m.parameters()], u.grad.clone()


def _reference_dplr_hippo(N):
    """make_DPLR_HiPPO transcribed from the official S5 repository (lindermanlab/S5,
    s5/ssm_init.py, MIT license), diagonal part only."""
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = -(np.tril(A) - np.diag(np.arange(N)))
    P = np.sqrt(np.arange(N) + 0.5)
    S = A + P[:, np.newaxis] * P[np.newaxis, :]
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    return Lambda_real + 1j * Lambda_imag, V


def _reference_forward(layer, u):
    """The official S5 forward transcribed from lindermanlab/S5 (s5/ssm.py, MIT license):
    ZOH/bilinear discretization, sequential complex recurrence, conjugate-symmetric output."""
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


class TestMakeDplrHippo:
    def test_matches_reference(self):
        for N in (4, 8, 16):
            lam, V = make_dplr_hippo(N)
            lam_ref, V_ref = _reference_dplr_hippo(N)
            assert np.abs(lam - lam_ref).max() < 1e-12
            assert np.abs(V - V_ref).max() < 1e-12

    def test_eigendecomposition_properties(self):
        N = 16
        lam, V = make_dplr_hippo(N)
        assert np.allclose(lam.real, -0.5)  # mean diagonal of the normal part is exactly -1/2
        assert np.abs(V @ V.conj().T - np.eye(N)).max() < 1e-12  # unitary
        # first half of the spectrum holds one member of each conjugate pair
        assert (lam.imag[: N // 2] < 0).all() and (lam.imag[N // 2 :] > 0).all()


class TestS5:
    @pytest.mark.parametrize("conj_sym", [True, False])
    @pytest.mark.parametrize("discretization", ["zoh", "bilinear"])
    def test_matches_reference(self, conj_sym, discretization):
        """Outputs and gradients agree with the official S5 forward semantics."""
        torch.manual_seed(0)
        layer = S5(3, 8, conj_sym=conj_sym, discretization=discretization).double()
        u = torch.randn(4, 100, 3, dtype=torch.float64)
        u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
        y_ref = _reference_forward(layer, u_ref)
        w = torch.randn_like(y_ref)
        (y_ref * w).sum().backward()
        g_ref = [p.grad.clone() for p in layer.parameters()]
        for p in layer.parameters():
            p.grad = None
        y_ours = layer(u_ours)
        (y_ours * w).sum().backward()
        assert _rel(y_ours, y_ref) < 1e-12
        assert _rel(u_ours.grad, u_ref.grad) < 1e-12
        for a, b in zip([p.grad for p in layer.parameters()], g_ref):
            assert _rel(a, b) < 1e-11

    def test_matches_s5_pytorch_package(self):
        """Forward agrees with the s5-pytorch port of the official implementation (if installed)."""
        pytest.importorskip("s5")
        from s5.s5_model import S5SSM

        torch.manual_seed(0)
        H, P, L = 3, 8, 50
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
        y_ref = ref(u)  # s5-pytorch operates on unbatched [L, H]
        y_ours = ours(u.unsqueeze(0)).squeeze(0)
        assert _rel(y_ours, y_ref) < 1e-10

    def test_clip_eigs(self):
        torch.manual_seed(0)
        layer = S5(2, 4, clip_eigs=True)
        with torch.no_grad():
            layer.Lambda_re.fill_(1.0)  # unstable continuous-time system
        lam_bar, _ = layer.discretize()
        assert lam_bar.abs().max() < 1.0

    def test_backend_parity(self):
        torch.manual_seed(0)
        m = S5(3, 16)
        u = torch.randn(5, 64, 3)
        out_e, g_e, du_e = _run(m, "eager", u)
        out_s, g_s, du_s = _run(m, "scan", u)
        assert _rel(out_s, out_e) < 5e-5
        assert max(_rel(a, b) for a, b in zip(g_s, g_e)) < 5e-5
        assert _rel(du_s, du_e) < 5e-5

    def test_blocks_and_validation(self):
        torch.manual_seed(0)
        assert S5(2, 16, blocks=4)(torch.randn(2, 20, 2)).shape == (2, 20, 2)
        with pytest.raises(ValueError):
            S5(2, 16, blocks=3)  # not divisible
        with pytest.raises(ValueError):
            S5(2, 6, blocks=2)  # odd block size with conj_sym
        with pytest.raises(ValueError):
            S5(2, 8, C_init="bogus")


class TestDeepS5:
    def test_shapes(self):
        u = torch.randn(4, 25, 3)
        assert DeepS5(3, 2)(u).shape == (4, 25, 2)
        for act in ("full_glu", "half_glu1", "half_glu2", "gelu"):
            assert DeepS5(3, 2, n_layers=1, d_model=8, d_state=4, activation=act)(u).shape == (4, 25, 2)
        y, state = DeepS5(3, 2, return_state=True)(u)
        assert y.shape == (4, 25, 2) and len(state) == 3
        assert state[0].shape == (4, 32) and state[0].is_complex()

    def test_stateful_chunked_equivalence(self):
        torch.manual_seed(0)
        m = DeepS5(2, 1, d_model=8, d_state=8, n_layers=2, return_state=True).double()
        u = torch.randn(4, 30, 2, dtype=torch.float64)
        full, _ = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:12], state=state)
        out3, _ = m(u[:, 12:], state=state)
        chunked = torch.cat((out1, out2, out3), dim=1)
        assert _rel(chunked, full) < 1e-12

    def test_cuda_parity(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(0)
            m = DeepS5(3, 2, d_model=8, d_state=8, n_layers=2)
            u = torch.randn(5, 64, 3)
            out_c, g_c, du_c = _run(m, "scan", u)
            m = m.cuda()
            out_g, g_g, du_g = _run(m, "scan", u.cuda())
            assert _rel(out_g.cpu(), out_c) < 5e-5
            assert max(_rel(a.cpu(), b) for a, b in zip(g_g, g_c)) < 5e-5
            assert _rel(du_g.cpu(), du_c) < 5e-5
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    @pytest.mark.slow
    def test_s5_learner_fit(self, dls_simulation):
        from tsfast.training import S5Learner

        lrn = S5Learner(dls_simulation, d_model=8, d_state=8, n_layers=2, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))

    @pytest.mark.slow
    def test_s5_learner_tbptt_fit(self, dls_simulation):
        from tsfast.training import S5Learner

        lrn = S5Learner(dls_simulation, d_model=8, d_state=8, n_layers=2, sub_seq_len=50, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))
