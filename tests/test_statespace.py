"""Tests for tsfast.models.architectures.ssm (NeuralStateSpace and its execution backends)."""

import pytest
import torch


def _run(m, backend, u, x0):
    """Forward + backward on cloned leaves; returns (out, param grads, du, dx0)."""
    m.backend = backend
    for p in m.parameters():
        p.grad = None
    u = u.clone().requires_grad_()
    x0 = x0.clone().requires_grad_()
    out = m(u, x0)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    return out, [p.grad.clone() for p in m.parameters()], u.grad.clone(), x0.grad.clone()


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _run_compiled(m, backend, u, x0):
    """Like ``_run`` but through ``torch.compile(m, fullgraph=True)``."""
    m.backend = backend
    cm = torch.compile(m, fullgraph=True)
    for p in m.parameters():
        p.grad = None
    u = u.clone().requires_grad_()
    x0 = x0.clone().requires_grad_()
    out = cm(u, x0)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    return out, [p.grad.clone() for p in m.parameters()], u.grad.clone(), x0.grad.clone()


def _assert_backend_parity(backend, device, hidden=(48, 32), act="tanh", tol=5e-4):
    from tsfast.models.architectures.ssm import NeuralStateSpace

    torch.manual_seed(0)
    m = NeuralStateSpace(3, 2, n_state=4, hidden_size=list(hidden), act=act, backend="eager").to(device)
    u = torch.randn(5, 40, 3, device=device)
    x0 = torch.randn(5, 4, device=device)
    out_e, g_e, du_e, dx0_e = _run(m, "eager", u, x0)
    out_b, g_b, du_b, dx0_b = _run(m, backend, u, x0)
    assert _rel(out_b, out_e) < tol
    assert max(_rel(a, b) for a, b in zip(g_b, g_e)) < tol
    assert _rel(du_b, du_e) < tol and _rel(dx0_b, dx0_e) < tol
    # inference path (no autograd graph)
    m.backend = backend
    with torch.no_grad():
        out_i = m(u, x0)
    assert _rel(out_i, out_e.detach()) < tol


class TestNeuralStateSpace:
    def test_eager_shapes(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        m = NeuralStateSpace(3, 2, n_state=5, hidden_size=16, num_layers=1, backend="eager")
        u = torch.randn(4, 25, 3)
        assert m(u).shape == (4, 25, 2)
        assert m(u, torch.randn(4, 5)).shape == (4, 25, 2)
        assert m(u, torch.randn(4, 1, 5)).shape == (4, 25, 2)  # [B,1,NX] x0 accepted

    def test_arbitrary_layers(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        m = NeuralStateSpace(1, 2, n_state=3, hidden_size=[8, 16, 8], act="relu", backend="eager")
        assert m(torch.randn(2, 10, 1)).shape == (2, 10, 2)
        linear = NeuralStateSpace(1, 2, n_state=3, hidden_size=[], backend="eager")  # linear state space
        assert linear(torch.randn(2, 10, 1)).shape == (2, 10, 2)

    def test_unknown_activation_raises(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        with pytest.raises(ValueError):
            NeuralStateSpace(1, 2, act="gelu")

    @pytest.mark.slow
    def test_compiled_parity(self):
        _assert_backend_parity("compiled", "cpu", hidden=(16,))

    def test_c_parity(self):
        from tsfast.models.architectures.ssm import backend_c as ssm_c

        if not ssm_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        _assert_backend_parity("c", "cpu")

    def test_c_parity_linear_and_acts(self):
        from tsfast.models.architectures.ssm import backend_c as ssm_c

        if not ssm_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        _assert_backend_parity("c", "cpu", hidden=(), act="tanh")
        _assert_backend_parity("c", "cpu", hidden=(24,), act="sigmoid")
        _assert_backend_parity("c", "cpu", hidden=(24,), act="relu")

    def test_triton_parity(self):
        from tsfast.models.architectures.ssm import backend_triton as ssm_triton

        if not ssm_triton.is_available():
            pytest.skip("no CUDA/triton")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            _assert_backend_parity("triton", "cuda")
            _assert_backend_parity("triton", "cuda", hidden=(), act="tanh")
            _assert_backend_parity("triton", "cuda", hidden=(64, 64), act="sigmoid")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def test_metal_parity(self):
        from tsfast.models.architectures.ssm import backend_metal as ssm_metal

        if not ssm_metal.is_available():
            pytest.skip("no MPS / shader compilation")
        _assert_backend_parity("metal", "mps")
        _assert_backend_parity("metal", "mps", hidden=(), act="tanh")
        _assert_backend_parity("metal", "mps", hidden=(24,), act="sigmoid")
        _assert_backend_parity("metal", "mps", hidden=(64, 64), act="relu")

    def test_metal_scan_backward_parity(self):
        # long sequence at small batch engages the sequence-parallel adjoint scan
        from tsfast.models.architectures.ssm import backend_metal as ssm_metal

        if not ssm_metal.is_available():
            pytest.skip("no MPS / shader compilation")
        from tsfast.models.architectures.ssm import NeuralStateSpace

        torch.manual_seed(0)
        m = NeuralStateSpace(3, 2, n_state=4, hidden_size=[48, 32], backend="eager").to("mps")
        u = torch.randn(4, 400, 3, device="mps")
        x0 = torch.randn(4, 4, device="mps")
        assert ssm_metal._scan_chunks(m.spec, 4, 400) > 1
        out_e, g_e, du_e, dx0_e = _run(m, "eager", u, x0)
        out_b, g_b, du_b, dx0_b = _run(m, "metal", u, x0)
        assert _rel(out_b, out_e) < 5e-4
        assert max(_rel(a, b) for a, b in zip(g_b, g_e)) < 5e-4
        assert _rel(du_b, du_e) < 5e-4 and _rel(dx0_b, dx0_e) < 5e-4

    def test_metal_fit_envelope(self):
        from tsfast.models.architectures.ssm.backend_metal import fits
        from tsfast.models.architectures.ssm import SSMSpec

        assert fits(SSMSpec(10, 10, (128, 128), "tanh"))
        assert not fits(SSMSpec(10, 10, (256,), "tanh"))
        assert not fits(SSMSpec(120, 10, (64,), "tanh"))

    def test_triton_fit_envelope(self):
        from tsfast.models.architectures.ssm.backend_triton import fits
        from tsfast.models.architectures.ssm import SSMSpec

        assert fits(SSMSpec(10, 10, (128, 128), "tanh"))
        assert not fits(SSMSpec(10, 10, (256,), "tanh"))
        assert not fits(SSMSpec(200, 10, (64,), "tanh"))

    def test_compile_fullgraph_c_parity(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace, backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        torch.manual_seed(0)
        m = NeuralStateSpace(3, 2, n_state=4, hidden_size=[48, 32], backend="eager")
        u = torch.randn(5, 40, 3)
        x0 = torch.randn(5, 4)
        out_e, g_e, du_e, dx0_e = _run(m, "eager", u, x0)
        out_c, g_c, du_c, dx0_c = _run_compiled(m, "c", u, x0)
        assert _rel(out_c, out_e) < 1e-4
        assert max(_rel(a, b) for a, b in zip(g_c, g_e)) < 1e-4
        assert _rel(du_c, du_e) < 1e-4 and _rel(dx0_c, dx0_e) < 1e-4

    def test_compile_fullgraph_triton_parity(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace, backend_triton

        if not backend_triton.is_available():
            pytest.skip("no CUDA/triton")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(0)
            m = NeuralStateSpace(3, 2, n_state=4, hidden_size=[48, 32], backend="eager").to("cuda")
            u = torch.randn(5, 40, 3, device="cuda")
            x0 = torch.randn(5, 4, device="cuda")
            out_e, g_e, du_e, dx0_e = _run(m, "eager", u, x0)
            out_t, g_t, du_t, dx0_t = _run_compiled(m, "triton", u, x0)
            assert _rel(out_t, out_e) < 1e-4
            assert max(_rel(a, b) for a, b in zip(g_t, g_e)) < 1e-4
            assert _rel(du_t, du_e) < 1e-4 and _rel(dx0_t, dx0_e) < 1e-4
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def test_use_backend_scoping(self):
        from tsfast.models import use_backend
        from tsfast.models.architectures.ssm import NeuralStateSpace, backend_c
        from tsfast.models.architectures.ssm import core as ssm_core

        torch.manual_seed(0)
        m = NeuralStateSpace(3, 2, n_state=4, hidden_size=[16], backend="auto")
        u = torch.randn(2, 12, 3)
        x0 = torch.randn(2, 4)
        calls = []
        orig = ssm_core.fused_rollout

        def spy(*args, **kwargs):
            calls.append("fused")
            return orig(*args, **kwargs)

        ssm_core.fused_rollout = spy
        try:
            with use_backend("reference"):
                out_ref = m(u, x0)
            assert calls == []  # reference scope keeps auto models off the fused kernels
            if backend_c.is_available():
                with use_backend("c"):
                    out_c = m(u, x0)
                assert calls == ["fused"]
                assert _rel(out_c, out_ref) < 5e-4
            if torch.cuda.is_available():
                from tsfast.models.architectures.ssm import backend_triton

                if backend_triton.is_available():
                    mc = NeuralStateSpace(3, 2, n_state=4, hidden_size=[16], backend="auto").to("cuda")
                    calls.clear()
                    with use_backend("triton"):
                        mc(u.to("cuda"), x0.to("cuda"))
                    assert calls == ["fused"]
                    calls.clear()
                    mc(u.to("cuda"), x0.to("cuda"))  # outside the scope: auto still picks triton
                    assert calls == ["fused"]
        finally:
            ssm_core.fused_rollout = orig

    def test_stateful_chunked_equivalence(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        torch.manual_seed(0)
        m = NeuralStateSpace(2, 1, n_state=3, hidden_size=16, num_layers=1, backend="eager", return_state=True)
        u = torch.randn(4, 30, 2)
        full, _ = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:25], state=state)
        out3, _ = m(u[:, 25:], state=state)
        chunked = torch.cat((out1, out2, out3), dim=1)
        assert _rel(chunked, full) < 1e-6  # the physical state fully captures the dynamics

    def test_graphed_stateful_model(self):
        from tsfast.models._core.cudagraph import GraphedStatefulModel
        from tsfast.models.architectures.ssm import NeuralStateSpace

        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        torch.manual_seed(0)
        m = NeuralStateSpace(3, 2, n_state=4, hidden_size=32, num_layers=2, backend="triton", return_state=True).to(
            "cuda"
        )
        graphed = GraphedStatefulModel(m, num_warmup_iters=3)
        u = torch.randn(8, 40, 3, device="cuda")
        out_g, state_g = graphed(u)
        out_e, state_e = m(u)
        assert _rel(out_g, out_e) < 5e-4
        assert _rel(state_g["x"], state_e["x"]) < 5e-4
        # captured backward produces usable gradients
        (out_g**2).mean().backward()
        assert all(p.grad is not None for p in m.parameters())
        # carried state replays through the same graph
        out2_g, _ = graphed(u, state=state_g)
        out2_e, _ = m(u, state=state_e)
        assert _rel(out2_g, out2_e) < 5e-4

    @pytest.mark.slow
    def test_ssm_learner_fit(self, dls_simulation):
        from tsfast.training import SSMLearner

        lrn = SSMLearner(dls_simulation, hidden_size=16, num_layers=1, backend="eager", n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))

    @pytest.mark.slow
    def test_ssm_learner_tbptt_cuda_graph(self, dls_simulation):
        from tsfast.training import SSMLearner

        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        lrn = SSMLearner(dls_simulation, hidden_size=16, num_layers=1, sub_seq_len=50, cuda_graph=True, n_skip=5)
        lrn.fit(1, 1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not torch.isnan(torch.tensor(final_valid_loss))
