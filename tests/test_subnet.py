"""Tests for tsfast.models.subnet (ResMLP, SubnetEncoder, SubnetSSM)."""

import pytest
import torch
from torch import nn

from tsfast.models.subnet import ResMLP, SubnetEncoder, SubnetSSM


def _rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


class TestSubnetEncoder:
    def test_deepsi_ordering_and_structure(self):
        """Encoder equals deepSI's default_encoder_net: linear + zero-bias MLP on [u_hist, y_hist]."""
        torch.manual_seed(0)
        enc = SubnetEncoder(n_input=2, n_output=3, n_state=4, na=5, nb=7, hidden_size=8, num_layers=2)
        u_hist, y_hist = torch.randn(6, 7, 2), torch.randn(6, 5, 3)
        flat = torch.cat((u_hist.flatten(1), y_hist.flatten(1)), dim=1)
        expected = enc.net.lin(flat) + enc.net.mlp(flat)
        assert torch.equal(enc(u_hist, y_hist), expected)
        for m in enc.net.mlp:
            if isinstance(m, nn.Linear):
                assert m.bias.abs().max() == 0  # fresh init: deepSI zero-bias convention

    def test_res_mlp_linear_only(self):
        m = ResMLP(3, 2, num_layers=0)
        x = torch.randn(4, 3)
        assert torch.equal(m(x), m.lin(x))


class TestSubnetSSM:
    def test_matches_manual_rollout(self):
        """Full forward equals encoder + observe-before-update state rollout (deepSI semantics)."""
        torch.manual_seed(1)
        n_init, T = 6, 15
        m = SubnetSSM(n_input=2, n_output=3, n_state=4, hidden_size=8, n_init=n_init, backend="eager").double()
        xin = torch.randn(5, n_init + T, 5, dtype=torch.float64)
        out = m(xin)

        u = xin[..., :2]
        x = m.encode(xin)
        ys = []
        for t in range(n_init, xin.shape[1]):
            ys.append(m.core.output_map(x))
            x = m.core.net(torch.cat((x, u[:, t]), dim=1))
        manual = torch.stack(ys, dim=1)
        assert out[:, :n_init].abs().max() == 0
        assert _rel(out[:, n_init:], manual) < 1e-14

    def test_gradients_flow_everywhere(self):
        torch.manual_seed(2)
        m = SubnetSSM(n_input=1, n_output=1, n_state=3, hidden_size=8, n_init=4)
        m(torch.randn(3, 20, 2)).pow(2).mean().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

    def test_encoder_window_validation(self):
        with pytest.raises(ValueError, match="cannot exceed n_init"):
            SubnetSSM(1, 1, n_init=4, na=5)

    def test_short_sequence_rejected(self):
        m = SubnetSSM(1, 1, n_init=8)
        with pytest.raises(ValueError, match="too short"):
            m(torch.randn(2, 8, 2))

    def test_minimal_sequence_single_prediction(self):
        m = SubnetSSM(1, 1, n_init=8)
        assert m(torch.randn(2, 9, 2)).shape == (2, 9, 1)

    def test_only_warmup_y_is_read(self):
        """Measured outputs beyond n_init must not influence predictions (no leakage)."""
        torch.manual_seed(3)
        m = SubnetSSM(n_input=1, n_output=1, n_state=3, n_init=5)
        xin = torch.randn(2, 25, 2)
        perturbed = xin.clone()
        perturbed[:, 5:, 1] = torch.randn(2, 20)
        assert torch.equal(m(xin), m(perturbed))
