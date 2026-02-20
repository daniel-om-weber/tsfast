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


class TestSeperateModels:
    def test_seperate_rnn_shape(self, dls_simulation):
        from tsfast.models.rnn import SeperateRNN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = SeperateRNN(input_list=[[0]], output_size=1, hidden_size=20).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_seperate_tcn_shape(self, dls_simulation):
        from tsfast.models.cnn import SeperateTCN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = SeperateTCN(input_list=[1], output_size=1, hl_depth=2, hl_width=10).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_seperate_crnn_shape(self, dls_simulation):
        from tsfast.models.cnn import SeperateCRNN
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = SeperateCRNN(
            input_list=[1], output_size=1, num_ft=4,
            num_cnn_layers=2, num_rnn_layers=1, hs_cnn=4, hs_rnn=4,
        ).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    @pytest.mark.slow
    def test_ar_tcn_learner_fit(self, dls_prediction):
        from tsfast.models.cnn import AR_TCNLearner
        lrn = AR_TCNLearner(dls_prediction, hl_depth=2, hl_width=10)
        lrn.fit(1, 1e-4)


class TestModelWrappers:
    def test_normalized_model_forward(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.models.layers import NormalizedModel
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = NormalizedModel.from_dls(SimpleRNN(1, 1), dls_simulation).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_ar_model_teacher_forcing(self, dls_prediction):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.models.layers import AR_Model
        batch = dls_prediction.one_batch()
        device = batch[0].device
        # prediction mode: input has u+y concatenated (2 channels), output is y (1 channel)
        model = AR_Model(SimpleRNN(2, 1), ar=False).to(device)
        u = batch[0][:, :, :1]  # u only
        y = batch[0][:, :, 1:]  # y from input
        out = model(u, y=y)
        assert out.shape == batch[1].shape

    def test_batch_norm_1d_stateful_shape(self):
        from tsfast.models.layers import BatchNorm_1D_Stateful
        bn = BatchNorm_1D_Stateful(hidden_size=10, stateful=True, batch_first=True)
        x = torch.rand(4, 100, 10)
        out = bn(x)
        assert out.shape == x.shape
        bn.reset_state()
        out2 = bn(x)
        assert out2.shape == x.shape

    def test_seq_aggregation_last(self):
        from tsfast.models.layers import SeqAggregation
        agg = SeqAggregation()
        x = torch.rand(4, 100, 10)
        out = agg(x)
        assert out.shape == (4, 10)

    def test_seq_aggregation_mean(self):
        from tsfast.models.layers import SeqAggregation
        agg = SeqAggregation(func=lambda x, dim: x.mean(dim=dim))
        x = torch.rand(4, 100, 10)
        out = agg(x)
        assert out.shape == (4, 10)
