"""Tests for tsfast.pinn module."""
import math
import pytest
import torch
import numpy as np


class TestDifferencing:
    def test_diff1_forward_shape(self):
        from tsfast.pinn.core import diff1_forward
        x = torch.rand(4, 100, 2)
        dx = diff1_forward(x, 0.01)
        assert dx.shape == x.shape

    def test_diff2_central_shape(self):
        from tsfast.pinn.core import diff2_central
        x = torch.rand(4, 100, 2)
        d2x = diff2_central(x, 0.01)
        assert d2x.shape == x.shape

    def test_diff1_central_shape(self):
        from tsfast.pinn.core import diff1_central
        x = torch.rand(4, 100, 2)
        dx = diff1_central(x, 0.01)
        assert dx.shape == x.shape


class TestExcitationSignals:
    def test_generate_excitation_signals_shape(self):
        from tsfast.pinn.core import generate_excitation_signals
        u = generate_excitation_signals(8, 200, n_inputs=2, dt=0.01)
        assert u.shape == (8, 200, 2)

    def test_generate_random_states_shape(self):
        from tsfast.pinn.core import generate_random_states
        s = generate_random_states(16, 2, [(-1, 1), (0, 10)])
        assert s.shape == (16, 2)


@pytest.mark.pinn
class TestPhysicsCallbacks:
    @pytest.mark.slow
    def test_physics_loss_callback(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.pinn.core import PhysicsLossCallback, diff1_forward

        def simple_physics(u, y_pred, y_ref):
            dy = diff1_forward(y_pred, 0.01)
            return {"physics": (dy ** 2).mean()}

        lrn = RNNLearner(dls_simulation, rnn_type="gru", num_layers=1, hidden_size=10)
        lrn.add_cb(PhysicsLossCallback(
            physics_loss_func=simple_physics,
            weight=0.1,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])


@pytest.mark.pinn
class TestHigherOrderDerivatives:
    def test_diff2_forward_shape(self):
        from tsfast.pinn.core import diff2_forward
        x = torch.rand(4, 100, 2)
        d2x = diff2_forward(x, 0.01)
        assert d2x.shape == x.shape

    def test_diff3_forward_shape(self):
        from tsfast.pinn.core import diff3_forward
        x = torch.rand(4, 100, 2)
        d3x = diff3_forward(x, 0.01)
        assert d3x.shape == x.shape

    def test_diff3_central_shape(self):
        from tsfast.pinn.core import diff3_central
        x = torch.rand(4, 100, 2)
        d3x = diff3_central(x, 0.01)
        assert d3x.shape == x.shape

    def test_diff1_central4_double_shape(self):
        from tsfast.pinn.core import diff1_central4_double
        x = torch.rand(4, 100, 2)
        dx = diff1_central4_double(x, 0.01)
        assert dx.shape == x.shape

    def test_diff2_forward_quadratic_accuracy(self):
        from tsfast.pinn.core import diff2_forward
        dt = 0.01
        t = torch.arange(0, 1.0, dt).unsqueeze(0).unsqueeze(-1)  # (1, 100, 1)
        f = t ** 2  # f(t) = t^2, f''(t) = 2.0
        d2f = diff2_forward(f, dt)
        # Interior points (excluding boundary artifacts) should be close to 2.0
        interior = d2f[0, 5:-5, 0]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.1)


@pytest.mark.pinn
class TestPIRNN:
    def test_pirnn_forward_shape(self, dls_pinn_prediction):
        from tsfast.pinn.pirnn import PIRNN
        batch = dls_pinn_prediction.one_batch()
        device = batch[0].device
        model = PIRNN(n_u=1, n_y=2, init_sz=20, hidden_size=20, rnn_layer=1).to(device)
        out = model(batch[0])
        assert out.shape[0] == batch[1].shape[0]  # batch size
        assert out.shape[2] == 2  # n_y outputs

    def test_pirnn_state_encoder(self):
        from tsfast.pinn.pirnn import PIRNN
        model = PIRNN(n_u=1, n_y=2, init_sz=20, hidden_size=20, rnn_layer=2)
        states = model.encode_single_state(torch.randn(4, 2))
        assert len(states) == 2  # rnn_layer
        assert states[0].shape == (1, 4, 20)  # (1, batch, hidden_size)

    @pytest.mark.slow
    def test_pirnn_learner_fit(self, dls_pinn_prediction):
        from tsfast.pinn.pirnn import PIRNNLearner
        lrn = PIRNNLearner(dls_pinn_prediction, init_sz=20, hidden_size=20, rnn_layer=1)
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_pirnn_learner_attach_output(self, dls_pinn):
        from tsfast.pinn.pirnn import PIRNNLearner
        lrn = PIRNNLearner(dls_pinn, init_sz=20, attach_output=True, hidden_size=20, rnn_layer=1)
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])


@pytest.mark.pinn
class TestCollocationPointsCB:
    @pytest.mark.slow
    def test_collocation_points_training(self, dls_pinn):
        from tsfast.models.rnn import RNNLearner
        from tsfast.pinn.core import CollocationPointsCB, diff1_forward, generate_excitation_signals

        def simple_physics(u, y_pred, y_ref):
            dy = diff1_forward(y_pred, 0.01)
            return {"physics": (dy ** 2).mean()}

        def gen_fn(batch_size, seq_len, device):
            return generate_excitation_signals(batch_size, seq_len, n_inputs=1, dt=0.01)

        lrn = RNNLearner(dls_pinn, rnn_type="gru", num_layers=1, hidden_size=10)
        lrn.add_cb(CollocationPointsCB(
            generate_pinn_input=gen_fn,
            physics_loss_func=simple_physics,
            weight=0.1,
            num_workers=1,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])


@pytest.mark.pinn
class TestPINNCallbacks:
    @pytest.mark.slow
    def test_transition_smoothness(self, dls_pinn_prediction):
        from tsfast.prediction.fransys import FranSysLearner
        from tsfast.pinn.core import TransitionSmoothnessCallback
        lrn = FranSysLearner(dls_pinn_prediction, init_sz=20, hidden_size=10, rnn_layer=1)
        lrn.add_cb(TransitionSmoothnessCallback(init_sz=20, weight=0.1))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_alternating_encoder(self, dls_pinn_prediction):
        from tsfast.pinn.pirnn import PIRNNLearner
        from tsfast.pinn.core import AlternatingEncoderCB
        lrn = PIRNNLearner(dls_pinn_prediction, init_sz=20, hidden_size=20, rnn_layer=1)
        lrn.add_cb(AlternatingEncoderCB(p_state=0.3))
        lrn.fit(1, 3e-3)
        assert lrn.model.default_encoder_mode == 'sequence'
