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


def _assert_backend_parity(backend, device, hidden=(48, 32), act="tanh", tol=5e-4, gate="none", n_state=4, eps=1.0):
    from tsfast.models.architectures.ssm import NeuralStateSpace

    torch.manual_seed(0)
    m = NeuralStateSpace(
        3, 2, n_state=n_state, hidden_size=list(hidden), act=act, gate=gate, eps=eps, backend="eager"
    ).to(device)
    u = torch.randn(5, 40, 3, device=device)
    x0 = torch.randn(5, n_state, device=device)
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

    def test_unknown_gate_raises(self):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        with pytest.raises(ValueError):
            NeuralStateSpace(1, 2, gate="lstm")

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


GATES = ("none", "leak", "gru", "residual")


class TestGatedNeuralStateSpace:
    """The gate variants of the state update (``NeuralStateSpace(gate=...)``)."""

    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("hidden", [64, [], [8, 16]])
    def test_shapes(self, gate, hidden):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        m = NeuralStateSpace(3, 2, n_state=4, hidden_size=hidden, gate=gate, backend="eager")
        assert m(torch.randn(5, 20, 3)).shape == (5, 20, 2)
        assert m(torch.randn(5, 20, 3), torch.randn(5, 4)).shape == (5, 20, 2)
        # the gate pre-activation widens only the final layer
        assert m.linears[-1].out_features == (8 if gate in ("gru", "residual") else 4)

    def test_ungated_default_unchanged(self):
        """``gate="none"`` must be bit-identical to the ungated model: it is the default path."""
        from tsfast.models.architectures.ssm import NeuralStateSpace

        u = torch.randn(2, 30, 3)
        torch.manual_seed(0)
        plain = NeuralStateSpace(3, 2, n_state=4, backend="eager")
        torch.manual_seed(0)
        gated = NeuralStateSpace(3, 2, n_state=4, gate="none", backend="eager")
        assert torch.equal(plain(u), gated(u))

    @pytest.mark.parametrize("gate", ["leak", "gru", "residual"])
    def test_chrono_init_band(self, gate):
        """Initial retention ``1 - z`` spans ``[1/2, (tmax-1)/tmax]`` (arXiv:1804.11188)."""
        from tsfast.models.architectures.ssm import NeuralStateSpace

        tmax = 100.0
        torch.manual_seed(0)
        m = NeuralStateSpace(1, 1, n_state=4096, hidden_size=8, gate=gate, gate_tmax=tmax, backend="eager")
        logit = m.leak_logit if gate == "leak" else m.linears[-1].bias[4096:]
        retention = 1.0 - torch.sigmoid(logit)
        assert retention.min() >= 0.5
        assert retention.max() <= (tmax - 1.0) / tmax
        assert retention.max() > 0.95  # the long-time-constant tail is populated

    def test_leak_and_gru_start_identical(self):
        """Zeroed gate-weight rows make ``gru`` open on exactly ``leak``'s dynamics.

        This is what makes the two comparable: at step zero they differ in nothing, so any
        later difference is the input dependence and not the initialization.
        """
        from tsfast.models.architectures.ssm import NeuralStateSpace

        nx = 4
        torch.manual_seed(1)
        ml = NeuralStateSpace(3, 2, n_state=nx, hidden_size=16, num_layers=1, gate="leak", backend="eager")
        mg = NeuralStateSpace(3, 2, n_state=nx, hidden_size=16, num_layers=1, gate="gru", backend="eager")
        with torch.no_grad():
            for src, dst in zip(ml.linears[:-1], mg.linears[:-1]):
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
            mg.linears[-1].weight[:nx].copy_(ml.linears[-1].weight)
            mg.linears[-1].bias[:nx].copy_(ml.linears[-1].bias)
            mg.linears[-1].bias[nx:].copy_(ml.leak_logit)
            mg.output_map.weight.copy_(ml.output_map.weight)
            mg.output_map.bias.copy_(ml.output_map.bias)
        u = torch.randn(3, 12, 3)
        assert torch.allclose(ml(u), mg(u), atol=1e-6)

    @pytest.mark.parametrize("gate", GATES)
    def test_step_matches_closed_form(self, gate):
        """One step against the update equations, independent of the rollout implementation."""
        from tsfast.models.architectures.ssm import NeuralStateSpace

        nx, eps = 4, 0.7
        torch.manual_seed(0)
        m = NeuralStateSpace(3, 2, n_state=nx, hidden_size=16, gate=gate, eps=eps, backend="eager").double()
        u = torch.randn(2, 1, 3, dtype=torch.float64)
        x0 = torch.randn(2, nx, dtype=torch.float64)
        y = m.net(torch.cat((x0, u[:, 0]), dim=1))
        match gate:
            case "none":
                expect = y
            case "leak":
                expect = x0 + torch.sigmoid(m.leak_logit) * (y - x0)
            case "gru":
                expect = x0 + torch.sigmoid(y[:, nx:]) * (y[:, :nx] - x0)
            case "residual":
                expect = x0 + eps * torch.sigmoid(y[:, nx:]) * y[:, :nx]
        got = m(u, x0, state=None)
        assert torch.allclose(got, m.output_map(expect).unsqueeze(1), atol=1e-12)

    @pytest.mark.parametrize("gate", GATES)
    def test_gradcheck(self, gate):
        from tsfast.models.architectures.ssm import NeuralStateSpace

        torch.manual_seed(0)
        m = NeuralStateSpace(2, 1, n_state=3, hidden_size=8, num_layers=1, gate=gate, backend="eager").double()
        u = torch.randn(2, 6, 2, dtype=torch.float64, requires_grad=True)
        x0 = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b: m(a, b), (u, x0), atol=1e-8)

    @pytest.mark.parametrize("gate", GATES)
    def test_stateful_chunked_equivalence(self, gate):
        """The gate must not break exact chunking — that contract is why this is not a GRU."""
        from tsfast.models.architectures.ssm import NeuralStateSpace

        torch.manual_seed(0)
        m = NeuralStateSpace(
            2, 1, n_state=3, hidden_size=16, num_layers=1, gate=gate, backend="eager", return_state=True
        )
        u = torch.randn(4, 30, 2)
        full, _ = m(u)
        out1, state = m(u[:, :10])
        out2, state = m(u[:, 10:25], state=state)
        out3, _ = m(u[:, 25:], state=state)
        assert _rel(torch.cat((out1, out2, out3), dim=1), full) < 1e-6

    def test_auto_fallback_is_not_warned(self):
        """Falling off the fused kernels is the designed route for a gated model, not a misuse.

        An explicitly named fused family is still a request the library cannot serve, so it
        keeps warning — only ``"auto"`` goes quiet.
        """
        import warnings

        from tsfast.models.architectures.ssm import NeuralStateSpace

        u, x0 = torch.randn(2, 8, 3), torch.randn(2, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for gate in ("leak", "gru", "residual"):
                NeuralStateSpace(3, 2, n_state=4, hidden_size=16, gate=gate, backend="auto")(u, x0)

        # metal is the one backend that serves no gate at all, and its gate screen runs
        # before its device screen, so this reaches the warning without an MPS device.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NeuralStateSpace(3, 2, n_state=4, hidden_size=16, gate="gru", backend="metal")(u, x0)
        assert any("kernel" in str(w.message) for w in caught)

    @pytest.mark.slow
    @pytest.mark.parametrize("gate", ["leak", "gru"])
    def test_learner_fit_tbptt(self, gate, dls_simulation):
        """``gate`` reaches the model through ``SSMLearner(**kwargs)`` and survives TBPTT.

        State carrying across chunks is where a gate could plausibly break training rather
        than the forward pass, so this fits with ``sub_seq_len`` set.
        """
        from tsfast.training import SSMLearner

        lrn = SSMLearner(
            dls_simulation, hidden_size=16, num_layers=1, gate=gate, backend="eager", sub_seq_len=25, n_skip=5
        )
        assert lrn.model.model.spec.gate == gate
        lrn.fit(1, 1e-3)
        assert not torch.isnan(torch.tensor(lrn.recorder[-1][1]))

    @pytest.mark.parametrize("gate", ["leak", "gru", "residual"])
    def test_c_parity_gated(self, gate):
        from tsfast.models.architectures.ssm import backend_c as ssm_c

        if not ssm_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        for hidden, act in (((48, 32), "tanh"), ((), "tanh"), ((24,), "sigmoid"), ((16,), "relu")):
            _assert_backend_parity("c", "cpu", hidden=hidden, act=act, gate=gate)

    @pytest.mark.parametrize("gate", ["leak", "gru", "residual"])
    def test_triton_parity_gated(self, gate):
        from tsfast.models.architectures.ssm import backend_triton as ssm_triton

        if not ssm_triton.is_available():
            pytest.skip("no CUDA/triton")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for hidden, act, n_state in (((48, 32), "tanh", 4), ((), "tanh", 4), ((64, 64), "sigmoid", 10)):
                _assert_backend_parity("triton", "cuda", hidden=hidden, act=act, gate=gate, n_state=n_state)
            # n_state off a power of two exercises the padding masks on both gate blocks
            _assert_backend_parity("triton", "cuda", hidden=(16,), act="tanh", gate=gate, n_state=7)
            # widest spec fits() still admits: the gate adds on-chip vectors the ungated
            # envelope was sized without, so the boundary is worth pinning
            _assert_backend_parity("triton", "cuda", hidden=(128,), act="tanh", gate=gate, n_state=128)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def test_compile_fullgraph_gated_parity(self):
        """The widened final layer and the extra saved tensors must not break the op boundary."""
        from tsfast.models.architectures.ssm import NeuralStateSpace, backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        torch.manual_seed(0)
        m = NeuralStateSpace(3, 2, n_state=4, hidden_size=[48, 32], gate="gru", backend="eager")
        u, x0 = torch.randn(5, 40, 3), torch.randn(5, 4)
        out_e, g_e, du_e, dx0_e = _run(m, "eager", u, x0)
        out_c, g_c, du_c, dx0_c = _run_compiled(m, "c", u, x0)
        assert _rel(out_c, out_e) < 1e-4
        assert max(_rel(a, b) for a, b in zip(g_c, g_e)) < 1e-4
        assert _rel(du_c, du_e) < 1e-4 and _rel(dx0_c, dx0_e) < 1e-4

    def test_fused_gate_matches_a_frozen_leak(self):
        """A ``gru`` gate with zeroed weight rows is a per-state leak, and the kernel agrees.

        This is the claim that justified fusing ``gru`` rather than ``leak``: the cheaper
        variant is a special case of the fused one, not a separate architecture.
        """
        from tsfast.models.architectures.ssm import NeuralStateSpace, backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        nx = 4
        torch.manual_seed(3)
        leak = NeuralStateSpace(3, 2, n_state=nx, hidden_size=16, num_layers=1, gate="leak", backend="eager")
        gru = NeuralStateSpace(3, 2, n_state=nx, hidden_size=16, num_layers=1, gate="gru", backend="c")
        with torch.no_grad():
            for src, dst in zip(leak.linears[:-1], gru.linears[:-1]):
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
            gru.linears[-1].weight[:nx].copy_(leak.linears[-1].weight)
            gru.linears[-1].bias[:nx].copy_(leak.linears[-1].bias)
            gru.linears[-1].weight[nx:].zero_()
            gru.linears[-1].bias[nx:].copy_(leak.leak_logit)
            gru.output_map.weight.copy_(leak.output_map.weight)
            gru.output_map.bias.copy_(leak.output_map.bias)
        u = torch.randn(3, 30, 3)
        assert _rel(gru(u), leak(u)) < 5e-6

    def test_metal_declines_a_gated_spec(self):
        """Metal has no gated generator, so it must decline rather than run the ungated kernel.

        The gate widens the final layer to ``2 * n_state``; a backend that accepted the spec
        without emitting the gate would read past its own layout and return wrong results.
        Checked through ``supports`` on any device — the gate screen runs before the device
        screen, so this needs no MPS.
        """
        import torch as _torch

        from tsfast.models.architectures.ssm import SSMSpec, backend_metal

        spec = SSMSpec(4, 3, (16,), "tanh", "gru")
        reason = backend_metal.supports(spec, _torch.zeros(1, 2, 3), _torch.zeros(1, 4))
        assert reason is not None and "gate" in reason

    def test_gate_keys_the_kernel_cache(self):
        """``SSMSpec`` carries the gate, so gated and ungated specs cannot share a compiled kernel."""
        from tsfast.models.architectures.ssm import SSMSpec

        plain = SSMSpec(4, 3, (16,), "tanh")
        assert plain.gate == "none" and plain.out_width == 4
        assert SSMSpec(4, 3, (16,), "tanh", "gru").out_width == 8
        assert plain != SSMSpec(4, 3, (16,), "tanh", "gru")
        assert len({plain, SSMSpec(4, 3, (16,), "tanh", "leak"), SSMSpec(4, 3, (16,), "tanh", "gru")}) == 3

    def test_eps_keys_the_kernel_cache(self):
        """``residual`` bakes ``eps`` into the generated source, so it must key the spec too."""
        from tsfast.models.architectures.ssm import SSMSpec, backend_c

        half, one = SSMSpec(4, 3, (16,), "tanh", "residual", 0.5), SSMSpec(4, 3, (16,), "tanh", "residual", 1.0)
        assert half != one and len({half, one}) == 2
        assert backend_c._gen_source(half) != backend_c._gen_source(one)
        # every other gate pins eps at 1.0, so they cannot fork the cache on a field they ignore
        assert "EPS" not in backend_c._gen_source(SSMSpec(4, 3, (16,), "tanh", "gru"))

    def test_residual_eps_parity(self):
        """``eps`` reaches the kernel across the custom-op boundary, not just the eager path."""
        from tsfast.models.architectures.ssm import backend_c as ssm_c

        if not ssm_c.is_available():
            pytest.skip("no C++ toolchain / ninja")
        for eps in (0.25, 2.0):
            _assert_backend_parity("c", "cpu", hidden=(24,), gate="residual", eps=eps)
