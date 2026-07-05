"""Tests for the NarxMLP model and NarxMLPLearner."""

import math

import pytest

import numpy as np
import torch

from tsfast.models.architectures.narx import NarxMLP
from tsfast.training import NarxMLPLearner


class TestNarxMLP:
    def test_teacher_forced_shape(self):
        model = NarxMLP(2, 3, na=4, nb=6, hidden_size=16)
        x = torch.randn(5, 50, 5)
        out = model.forward(x, ar=False)
        assert out.shape == (5, 50, 3)

    def test_free_run_shape(self):
        model = NarxMLP(2, 3, na=4, nb=6, hidden_size=16)
        x = torch.randn(5, 50, 5)
        out = model.forward(x, ar=True)
        assert out.shape == (5, 50, 3)

    def test_free_run_matches_teacher_forcing_under_full_washout(self):
        """With washout covering the whole sequence, both paths compute the same function."""
        torch.manual_seed(0)
        model = NarxMLP(1, 1, na=3, nb=5, hidden_size=8, washout=50)
        x = torch.randn(2, 50, 2)
        tf = model.forward(x, ar=False)
        ar = model.forward(x, ar=True)
        assert torch.allclose(tf, ar, atol=1e-6)

    def test_free_run_ignores_y_after_washout(self):
        """Beyond the washout window the y channel must not influence predictions."""
        torch.manual_seed(0)
        model = NarxMLP(1, 1, na=3, nb=3, hidden_size=8, washout=10)
        x = torch.randn(2, 40, 2)
        corrupted = x.clone()
        corrupted[:, 10:, 1:] = 1e6
        assert torch.allclose(model.forward(x, ar=True), model.forward(corrupted, ar=True))

    def test_teacher_forcing_is_causal(self):
        torch.manual_seed(0)
        model = NarxMLP(1, 1, na=3, nb=3, hidden_size=8)
        x = torch.randn(1, 40, 2)
        perturbed = x.clone()
        perturbed[:, 20:] += 1.0
        out, out_p = model.forward(x, ar=False), model.forward(perturbed, ar=False)
        assert torch.allclose(out[:, :20], out_p[:, :20])
        assert not torch.allclose(out[:, 20:], out_p[:, 20:])

    def test_gradients_flow_through_feedback(self):
        """A loss on late timesteps must reach the weights via the fed-back predictions."""
        torch.manual_seed(0)
        model = NarxMLP(1, 1, na=2, nb=2, hidden_size=8, washout=0)
        x = torch.randn(2, 30, 2)
        x[:, :, 1:] = 0  # washout=0: predictions must not depend on the y channel at all
        model.forward(x, ar=True)[:, -1].sum().backward()
        assert model.conv_y.weight.grad is not None
        assert model.conv_y.weight.grad.abs().sum() > 0

    def test_eval_mode_free_runs(self):
        """In eval mode the forward default must be free run even when trained one-step."""
        torch.manual_seed(0)
        model = NarxMLP(1, 1, na=3, nb=3, hidden_size=8, teacher_forcing=True, washout=5)
        x = torch.randn(2, 30, 2)
        model.train()
        assert torch.allclose(model(x), model.forward(x, ar=False))
        model.eval()
        assert torch.allclose(model(x), model.forward(x, ar=True))


class TestNarxMLPLearner:
    def test_fit_free_run(self, dls_simulation):
        lrn = NarxMLPLearner(dls_simulation, na=4, nb=4, hidden_size=16, show_bar=False)
        lrn.fit(1, lr=1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float("inf")

    def test_fit_one_step(self, dls_simulation):
        lrn = NarxMLPLearner(dls_simulation, na=4, nb=4, hidden_size=16, train_mode="one_step", show_bar=False)
        lrn.fit(1, lr=1e-3)
        final_valid_loss = lrn.recorder[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float("inf")

    def test_inference_wrapper_free_runs(self, dls_simulation):
        from tsfast.inference import InferenceWrapper

        lrn = NarxMLPLearner(dls_simulation, na=4, nb=4, hidden_size=16, washout=10, show_bar=False)
        lrn.fit(1, lr=1e-3)
        wrapper = InferenceWrapper(lrn)
        u = np.random.randn(200, 1).astype(np.float32)
        y_init = np.random.randn(10, 1).astype(np.float32)
        out = wrapper(u, y_init)
        assert out.shape == (200, 1)
        assert np.isfinite(out).all()


def _run_backend(m, backend, x):
    """Free-run forward + backward on a cloned leaf; returns (out, param grads, dx)."""
    m.backend = backend
    for p in m.parameters():
        p.grad = None
    x = x.clone().requires_grad_()
    out = m.forward(x, ar=True)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    return out, [p.grad.clone() for p in m.parameters()], x.grad.clone()


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _assert_backend_parity(backend, device, act="tanh", num_layers=2, washout=7, tol=2e-3):
    from tsfast.models.architectures.narx import NarxMLP

    torch.manual_seed(0)
    m = NarxMLP(2, 3, na=3, nb=4, hidden_size=24, num_layers=num_layers, act=act, washout=washout).to(device)
    x = torch.randn(5, 40, 5, device=device)
    out_e, g_e, dx_e = _run_backend(m, "eager", x)
    out_b, g_b, dx_b = _run_backend(m, backend, x)
    assert _rel(out_b, out_e) < tol
    assert max(_rel(a, b) for a, b in zip(g_b, g_e)) < tol
    assert _rel(dx_b, dx_e) < tol
    # inference path (no autograd graph)
    m.backend = backend
    with torch.no_grad():
        out_i = m.forward(x, ar=True)
    assert _rel(out_i, out_e.detach()) < tol


class TestNarxBackends:
    def test_c_parity(self):
        from tsfast.models.architectures.narx import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain")
        _assert_backend_parity("c", "cpu")

    @pytest.mark.parametrize("act", ["tanh", "sigmoid", "relu"])
    @pytest.mark.parametrize("washout", [0, 7, 100])
    def test_c_parity_acts_and_washouts(self, act, washout):
        from tsfast.models.architectures.narx import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain")
        _assert_backend_parity("c", "cpu", act=act, washout=washout)

    def test_c_parity_single_layer(self):
        from tsfast.models.architectures.narx import backend_c

        if not backend_c.is_available():
            pytest.skip("no C++ toolchain")
        _assert_backend_parity("c", "cpu", num_layers=1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_triton_parity(self):
        from tsfast.models.architectures.narx import backend_triton

        if not backend_triton.is_available():
            pytest.skip("triton unavailable")
        _assert_backend_parity("triton", "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("act", ["tanh", "sigmoid", "relu"])
    @pytest.mark.parametrize("washout", [0, 7, 100])
    def test_triton_parity_acts_and_washouts(self, act, washout):
        from tsfast.models.architectures.narx import backend_triton

        if not backend_triton.is_available():
            pytest.skip("triton unavailable")
        _assert_backend_parity("triton", "cuda", act=act, washout=washout)

    def test_triton_fit_envelope(self):
        from tsfast.models.architectures.narx import NarxSpec, backend_triton

        assert backend_triton.fits(NarxSpec(1, 32, (64, 64), "tanh"))
        assert not backend_triton.fits(NarxSpec(6, 32, (64, 64), "tanh"))  # padded buffer 256
