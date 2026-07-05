"""Tests for tsfast.models.architectures.lru (LRU, DeepLRU)."""

import math

import pytest
import torch
from torch import nn

from tsfast.models.architectures.lru import LRU, DeepLRU


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


class _ReferenceLRU(nn.Module):
    """The LRU layer of Forgione et al., transcribed from github.com/forgi86/lru-reduction
    (lru/linear.py, MIT license): complex B/C parameters and a sequential state loop."""

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


class TestLRU:
    def test_matches_reference(self):
        """Outputs and gradients agree with the reference implementation of Forgione et al."""
        torch.manual_seed(0)
        in_f, out_f, N, L = 3, 2, 8, 200
        ours = LRU(in_f, out_f, N, r_min=0.4, r_max=0.99).double()
        ref = _ReferenceLRU(in_f, out_f, N).double()
        with torch.no_grad():
            for name in ("nu_log", "theta_log", "gamma_log", "D"):
                getattr(ref, name).copy_(getattr(ours, name))
            # Module.double() leaves complex params at complex64; replace them outright
            ref.B = nn.Parameter(torch.complex(ours.B_re, ours.B_im).detach().clone())
            ref.C = nn.Parameter(torch.complex(ours.C_re, ours.C_im).detach().clone())

        u = torch.randn(4, L, in_f, dtype=torch.float64)
        u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
        y_ref, y_ours = ref(u_ref), ours(u_ours)
        w = torch.randn_like(y_ref)
        (y_ref * w).sum().backward()
        (y_ours * w).sum().backward()
        assert _rel(y_ours, y_ref) < 1e-12
        assert _rel(u_ours.grad, u_ref.grad) < 1e-12
        for name in ("nu_log", "theta_log", "gamma_log", "D"):
            assert _rel(getattr(ours, name).grad, getattr(ref, name).grad) < 1e-12, name
        # torch stores complex grads as complex(dL/dRe, dL/dIm), matching the real-pair grads
        assert _rel(torch.complex(ours.B_re.grad, ours.B_im.grad), ref.B.grad) < 1e-12
        assert _rel(torch.complex(ours.C_re.grad, ours.C_im.grad), ref.C.grad) < 1e-12

    def test_matches_reference_with_state(self):
        torch.manual_seed(1)
        ours = LRU(2, 2, 6).double()
        ref = _ReferenceLRU(2, 2, 6).double()
        with torch.no_grad():
            for name in ("nu_log", "theta_log", "gamma_log", "D"):
                getattr(ref, name).copy_(getattr(ours, name))
            ref.B = nn.Parameter(torch.complex(ours.B_re, ours.B_im).detach().clone())
            ref.C = nn.Parameter(torch.complex(ours.C_re, ours.C_im).detach().clone())
        u = torch.randn(3, 50, 2, dtype=torch.float64)
        x0 = torch.randn(3, 6, dtype=torch.complex128)
        assert _rel(ours(u, state=x0), ref(u, state=x0)) < 1e-12

    def test_stability_by_construction(self):
        torch.manual_seed(0)
        lru = LRU(1, 1, 32, r_min=0.0, r_max=1.0)
        with torch.no_grad():
            lru.nu_log.uniform_(-10, 10)
            lru.theta_log.uniform_(-10, 10)
        assert lru.eigenvalues().abs().max() < 1.0

    def test_backend_parity(self):
        torch.manual_seed(0)
        m = LRU(3, 2, 16)
        u = torch.randn(5, 64, 3)
        out_e, g_e, du_e = _run(m, "eager", u)
        out_s, g_s, du_s = _run(m, "scan", u)
        assert _rel(out_s, out_e) < 5e-5
        assert max(_rel(a, b) for a, b in zip(g_s, g_e)) < 5e-5
        assert _rel(du_s, du_e) < 5e-5

    def test_unknown_backend_raises(self):
        lru = LRU(1, 1, 4, backend="fft")
        with pytest.raises(ValueError):
            lru(torch.randn(1, 10, 1))


class TestDeepLRU:
    def test_shapes(self):
        u = torch.randn(4, 25, 3)
        assert DeepLRU(3, 2)(u).shape == (4, 25, 2)
        assert DeepLRU(3, 2, n_layers=1, d_model=8, d_state=4)(u).shape == (4, 25, 2)
        y, state = DeepLRU(3, 2, return_state=True)(u)
        assert y.shape == (4, 25, 2) and len(state) == 3
        assert state[0].shape == (4, 64) and state[0].is_complex()

    def test_stateful_chunked_equivalence(self):
        torch.manual_seed(0)
        m = DeepLRU(2, 1, d_model=8, d_state=8, n_layers=2, return_state=True).double()
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
            m = DeepLRU(3, 2, d_model=8, d_state=8, n_layers=2)
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
    def test_lru_learner_fit(self, dls_simulation):
        from tsfast.training import LRULearner

        lrn = LRULearner(dls_simulation, d_model=8, d_state=8, n_layers=2, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))

    @pytest.mark.slow
    def test_lru_learner_tbptt_fit(self, dls_simulation):
        from tsfast.training import LRULearner

        lrn = LRULearner(dls_simulation, d_model=8, d_state=8, n_layers=2, sub_seq_len=50, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))
