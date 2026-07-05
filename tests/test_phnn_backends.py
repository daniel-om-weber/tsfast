"""Tests for the fused PHNN rollout backends (c, triton).

Correctness contract in ``MATH.md``. The C backend is fp64-gradcheckable and is the
reference the Triton backend is validated against on GPU.
"""

import pytest
import torch
from torch import nn

from tsfast.models.phnn import PHNN, PHNNCore
from tsfast.models.phnn_backends import spec_of, supports
from tsfast.models.phnn_backends.backend_c import c_rollout, is_available
from tsfast.models.phnn_backends.common import flat_params

pytestmark = pytest.mark.skipif(not is_available(), reason="no C++ toolchain")


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def _randn_core(n_state, n_input, n_output, hidden, num_layers, output, bound, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    core = PHNNCore(n_state, n_input, n_output, hidden_size=hidden, num_layers=num_layers,
                    dt=0.12, h_lower_bound=bound, output=output).to(dtype)
    for p in core.parameters():
        nn.init.normal_(p, std=0.3)
    return core


def _eager_roll(core, u, x0):
    x = x0
    outs = []
    for t in range(u.shape[1]):
        y, x = core.step(x, u[:, t])
        outs.append(y)
    return torch.stack(outs, dim=1)


CONFIGS = [
    (2, 4, 1, 1, "ph", 0.0),
    (1, 4, 1, 1, "ph", 0.0),
    (2, 5, 2, 2, "ph", None),
    (2, 4, 2, 3, "linear", 0.0),
    (2, 4, 1, 1, "ph", 2.5),
]


class TestCBackend:
    @pytest.mark.parametrize("nl,n,m,ny,out,bnd", CONFIGS)
    def test_forward_matches_eager(self, nl, n, m, ny, out, bnd):
        core = _randn_core(n, m, ny, 8, nl, out, bnd)
        spec = spec_of(core)
        x0 = torch.randn(3, n, dtype=torch.float64)
        u = torch.randn(3, 7, m, dtype=torch.float64)
        with torch.no_grad():
            ref = _eager_roll(core, u, x0)
            got = c_rollout(core, spec, u, x0)
        assert _rel(got, ref) < 1e-12

    @pytest.mark.parametrize("nl,n,m,ny,out,bnd", CONFIGS)
    def test_gradcheck(self, nl, n, m, ny, out, bnd):
        core = _randn_core(n, m, ny, 6, nl, out, bnd, seed=1)
        spec = spec_of(core)
        x0 = torch.randn(2, n, dtype=torch.float64, requires_grad=True)
        u = torch.randn(2, 5, m, dtype=torch.float64, requires_grad=True)
        params = flat_params(core)
        f = lambda u, x0, *ps: c_rollout(core, spec, u, x0)  # noqa: E731
        assert torch.autograd.gradcheck(f, (u, x0, *params), eps=1e-6, atol=1e-5, rtol=1e-3)

    def test_param_grad_equivalence_fp32(self):
        torch.manual_seed(2)
        m = PHNN(1, 1, n_state=6, hidden_size=32, num_layers=2, dt=0.1, n_init=10, backend="eager")
        for p in m.parameters():
            nn.init.normal_(p, std=0.1)
        x = torch.randn(4, 50, 2)
        m.backend = "eager"
        ge = torch.autograd.grad(m(x)[:, 10:].pow(2).mean(), list(m.parameters()))
        m.backend = "c"
        gc = torch.autograd.grad(m(x)[:, 10:].pow(2).mean(), list(m.parameters()))
        assert max(_rel(a, b) for a, b in zip(gc, ge)) < 1e-3

    def test_supports_caps(self):
        # single RK4 step and >=1 hidden layer required
        core2 = PHNNCore(4, 1, rk4_steps=2)
        assert not supports(spec_of(core2), "c")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
class TestTritonBackend:
    @pytest.mark.parametrize("nl,n,m,ny,out,bnd", CONFIGS)
    def test_matches_c_forward_and_grads(self, nl, n, m, ny, out, bnd):
        core = _randn_core(n, m, ny, 16, nl, out, bnd, seed=3, dtype=torch.float64)
        spec = spec_of(core)
        from tsfast.models.phnn_backends.backend_triton import triton_rollout

        x0 = torch.randn(3, n, dtype=torch.float64)
        u = torch.randn(3, 12, m, dtype=torch.float64)
        x0d = x0.clone().requires_grad_(True)
        ud = u.clone().requires_grad_(True)
        gc = torch.autograd.grad(c_rollout(core, spec, ud, x0d).pow(2).mean(), [ud, x0d, *flat_params(core)])
        ct = core.float().cuda()
        x0t = x0.float().cuda().requires_grad_(True)
        ut = u.float().cuda().requires_grad_(True)
        outt = triton_rollout(ct, spec, ut, x0t)
        gt = torch.autograd.grad(outt.pow(2).mean(), [ut, x0t, *flat_params(ct)])
        assert max(_rel(a.cpu().float(), b.float()) for a, b in zip(gt, gc)) < 1e-3
