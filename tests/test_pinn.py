"""Tests for tsfast.pinn module."""
import pytest
import torch


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
    def test_physics_loss_callback(self, wh_path):
        from tsfast.datasets.core import create_dls
        from tsfast.models.rnn import RNNLearner
        from tsfast.pinn.core import PhysicsLossCallback, diff1_forward

        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5,
        )

        def simple_physics(u, y_pred, y_ref):
            dy = diff1_forward(y_pred, 0.01)
            return {"physics": (dy ** 2).mean()}

        lrn = RNNLearner(dls, rnn_type="gru", num_layers=1, hidden_size=10)
        lrn.add_cb(PhysicsLossCallback(
            norm_input=dls.train.after_batch[0],
            physics_loss_func=simple_physics,
            weight=0.1,
        ))
        lrn.fit(1, 3e-3)
