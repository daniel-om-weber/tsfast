"""Tests for tsfast.models module."""
import pytest
import torch


class TestRNN:
    def test_simple_rnn_gru_shape(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = SimpleRNN(1, 1, num_layers=2, rnn_type="gru").to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_simple_rnn_lstm_shape(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = SimpleRNN(1, 1, num_layers=1, rnn_type="lstm").to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_residual_rnn_shape(self, dls_simulation):
        from tsfast.models.rnn import SimpleResidualRNN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = SimpleResidualRNN(1, 1, num_blocks=1).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_dense_rnn_shape(self, dls_simulation):
        from tsfast.models.rnn import DenseNet_RNN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = DenseNet_RNN(1, 1, growth_rate=10, block_config=(1, 1), num_init_features=2).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    @pytest.mark.slow
    def test_rnn_learner_fit(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        lrn = RNNLearner(dls_simulation, rnn_type="gru")
        lrn.fit(1, 1e-4)


class TestCNN:
    def test_tcn_shape(self, dls_simulation):
        from tsfast.models.cnn import TCN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = TCN(1, 1, hl_depth=2, hl_width=10).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    @pytest.mark.slow
    def test_tcn_learner_fit(self, dls_simulation):
        from tsfast.models.cnn import TCNLearner
        lrn = TCNLearner(dls_simulation)
        lrn.fit(1, 1e-4)

    @pytest.mark.slow
    def test_crnn_learner_fit(self, dls_simulation):
        from tsfast.models.cnn import CRNNLearner
        lrn = CRNNLearner(dls_simulation)
        lrn.fit(1, 1e-4)


class TestLayers:
    def test_seq_linear_shape(self):
        from tsfast.models.layers import SeqLinear
        m = SeqLinear(10, 3, hidden_layer=1)
        x = torch.rand(2, 50, 10)
        assert m(x).shape == (2, 50, 3)
