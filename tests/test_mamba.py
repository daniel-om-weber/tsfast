"""Tests for tsfast.models.architectures.mamba (MambaLayer, DeepMamba)."""

import pytest
import torch
import torch.nn.functional as F

from tsfast.models.architectures.mamba import DeepMamba, MambaLayer


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


def _selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True):
    """selective_scan_ref transcribed from the official Mamba repository
    (state-spaces/mamba, mamba_ssm/ops/selective_scan_interface.py, Apache-2.0),
    without the fp32 casts so the test dtype is preserved.

    Shapes: u/delta/z (B, D, L), A (D, N), B/C (B, N, L), D (D,), delta_bias (D,).
    """
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, L = u.shape
    dstate = A.shape[1]
    x = A.new_zeros((batch, dim, dstate))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y + u * D.unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z)
    return out


def _reference_mixer_forward(layer, u):
    """The official Mamba block forward (mamba_simple.py reference path) on top of
    ``_selective_scan_ref``, using the layer's own parameters."""
    L = u.shape[1]
    x, z = layer.in_proj(u).chunk(2, dim=-1)
    x = x.transpose(1, 2)
    x = F.silu(
        F.conv1d(x, layer.conv1d.weight, layer.conv1d.bias, padding=layer.d_conv - 1, groups=layer.d_inner)[..., :L]
    )
    x_dbl = layer.x_proj(x.transpose(1, 2))
    dt, B, C = x_dbl.split([layer.dt_rank, layer.d_state, layer.d_state], dim=-1)
    delta = (dt @ layer.dt_proj.weight.mT).transpose(1, 2)  # bias enters the scan separately
    A = -torch.exp(layer.A_log)
    out = _selective_scan_ref(
        x,
        delta,
        A,
        B.transpose(1, 2),
        C.transpose(1, 2),
        layer.D,
        z.transpose(1, 2),
        layer.dt_proj.bias,
    )
    return layer.out_proj(out.transpose(1, 2))


class TestMambaLayer:
    def test_matches_reference(self):
        """Outputs and gradients agree with the official selective_scan_ref semantics."""
        torch.manual_seed(0)
        layer = MambaLayer(4, d_state=8).double()
        u = torch.randn(3, 60, 4, dtype=torch.float64)
        u_ref, u_ours = u.clone().requires_grad_(), u.clone().requires_grad_()
        y_ref = _reference_mixer_forward(layer, u_ref)
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

    def test_matches_mambapy_package(self):
        """Forward agrees with the mambapy reference implementation (if installed)."""
        pytest.importorskip("mambapy")
        from mambapy.mamba import MambaBlock, MambaConfig

        torch.manual_seed(0)
        d_model, L = 4, 40
        # float32 comparison: mambapy casts A_log/D to float32 internally, so float64
        # parity is unreachable; its parallel scan also reorders the reductions
        ours = MambaLayer(d_model, d_state=8)
        ref = MambaBlock(MambaConfig(d_model=d_model, n_layers=1, d_state=8))
        with torch.no_grad():
            for name in ("in_proj", "x_proj", "dt_proj", "out_proj", "conv1d"):
                getattr(ref, name).load_state_dict(getattr(ours, name).state_dict())
            ref.A_log.copy_(ours.A_log)
            ref.D.copy_(ours.D)
        u = torch.randn(2, L, d_model)
        assert _rel(ours(u), ref(u)) < 5e-5

    def test_dt_bias_softplus_inverse(self):
        torch.manual_seed(0)
        layer = MambaLayer(8, dt_min=0.001, dt_max=0.1)
        dt = F.softplus(layer.dt_proj.bias)
        assert (dt >= 1e-4 - 1e-9).all() and (dt <= 0.1 + 1e-6).all()

    def test_backend_parity(self):
        torch.manual_seed(0)
        m = MambaLayer(4, d_state=8)
        u = torch.randn(5, 64, 4)
        out_e, g_e, du_e = _run(m, "eager", u)
        out_s, g_s, du_s = _run(m, "scan", u)
        assert _rel(out_s, out_e) < 5e-5
        assert max(_rel(a, b) for a, b in zip(g_s, g_e)) < 5e-5
        assert _rel(du_s, du_e) < 5e-5

    def test_unknown_backend_raises(self):
        m = MambaLayer(4, backend="fft")
        with pytest.raises(ValueError):
            m(torch.randn(1, 10, 4))


class TestDeepMamba:
    def test_shapes(self):
        u = torch.randn(4, 25, 3)
        assert DeepMamba(3, 2)(u).shape == (4, 25, 2)
        assert DeepMamba(3, 2, n_layers=1, d_model=8, d_state=4)(u).shape == (4, 25, 2)
        y, state = DeepMamba(3, 2, return_state=True)(u)
        assert y.shape == (4, 25, 2) and len(state) == 3
        assert state[0]["conv"].shape == (4, 64, 3) and state[0]["ssm"].shape == (4, 64 * 16)

    def test_stateful_chunked_equivalence(self):
        torch.manual_seed(0)
        # chunk lengths indivisible by d_conv, one chunk (2) shorter than the conv tail
        m = DeepMamba(2, 1, d_model=8, d_state=4, n_layers=2, return_state=True).double()
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
            m = DeepMamba(3, 2, d_model=8, d_state=4, n_layers=2)
            u = torch.randn(5, 64, 3)
            out_c, g_c, du_c = _run(m, "scan", u)
            m = m.cuda()
            out_g, g_g, du_g = _run(m, "scan", u.cuda())
            assert _rel(out_g.cpu(), out_c) < 5e-5
            assert max(_rel(a.cpu(), b) for a, b in zip(g_g, g_c)) < 5e-5
            assert _rel(du_g.cpu(), du_c) < 5e-5
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def test_fused_stateful_chunked_cuda(self):
        """Chunked rollouts through the fused kernel's h0/h_last path match the full pass."""
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        torch.manual_seed(0)
        m = DeepMamba(2, 1, d_model=8, d_state=4, n_layers=2, return_state=True).cuda()
        u = torch.randn(4, 70, 2, device="cuda")
        with torch.no_grad():
            full, _ = m(u)
            out1, state = m(u[:, :17])
            out2, state = m(u[:, 17:19], state=state)
            out3, _ = m(u[:, 19:], state=state)
        assert _rel(torch.cat((out1, out2, out3), dim=1), full) < 1e-5

    def test_fused_grad_through_state_cuda(self):
        """Gradients through the carried state (conv tail and h0 in, h_last out) match the generic path."""
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        import tsfast.models._core.scan as scan

        torch.manual_seed(0)
        layer = MambaLayer(4, d_state=8).cuda()
        u = torch.randn(3, 40, 4, device="cuda")
        state0 = {
            "conv": torch.randn(3, 8, layer.d_conv - 1, device="cuda"),
            "ssm": torch.randn(3, 8 * 8, device="cuda"),
        }
        results = {}
        try:
            for backend in ("auto", "doubling"):
                scan.backend = backend
                for p in layer.parameters():
                    p.grad = None
                state = {
                    "conv": state0["conv"].clone().requires_grad_(),
                    "ssm": state0["ssm"].clone().requires_grad_(),
                }
                out, new_state = layer(u, state=state, return_state=True)
                (out.square().mean() + new_state["ssm"].square().mean()).backward()
                results[backend] = (
                    out.detach(),
                    new_state["ssm"].detach(),
                    state["ssm"].grad.clone(),
                    state["conv"].grad.clone(),
                    [p.grad.clone() for p in layer.parameters()],
                )
        finally:
            scan.backend = "auto"
        out_f, hl_f, gh0_f, gc0_f, g_f = results["auto"]
        out_d, hl_d, gh0_d, gc0_d, g_d = results["doubling"]
        assert _rel(out_f, out_d) < 5e-5
        assert _rel(hl_f, hl_d) < 5e-5
        assert _rel(gh0_f, gh0_d) < 5e-5
        assert _rel(gc0_f, gc0_d) < 5e-5
        assert max(_rel(a, b) for a, b in zip(g_f, g_d)) < 5e-5

    @pytest.mark.slow
    def test_mamba_learner_fit(self, dls_simulation):
        from tsfast.training import MambaLearner

        lrn = MambaLearner(dls_simulation, d_model=8, d_state=4, n_layers=2, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))

    @pytest.mark.slow
    def test_mamba_learner_tbptt_fit(self, dls_simulation):
        from tsfast.training import MambaLearner

        lrn = MambaLearner(dls_simulation, d_model=8, d_state=4, n_layers=2, sub_seq_len=50, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))
