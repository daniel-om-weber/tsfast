"""Tests for tsfast.models module."""
import math
import pytest
import torch
import numpy as np


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
        final_valid_loss = lrn.recorder.values[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')


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
        final_valid_loss = lrn.recorder.values[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')

    @pytest.mark.slow
    def test_crnn_learner_fit(self, dls_simulation):
        from tsfast.models.cnn import CRNNLearner
        lrn = CRNNLearner(dls_simulation)
        lrn.fit(1, 1e-4)
        final_valid_loss = lrn.recorder.values[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')


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
        final_valid_loss = lrn.recorder.values[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')


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


class TestScalers:
    @pytest.fixture
    def norm_pair(self):
        from tsfast.datasets.core import NormPair
        return NormPair(
            mean=np.array([1.0, 2.0], dtype=np.float32),
            std=np.array([0.5, 1.0], dtype=np.float32),
            min=np.array([0.0, 0.0], dtype=np.float32),
            max=np.array([2.0, 4.0], dtype=np.float32),
        )

    def test_standard_roundtrip(self, norm_pair):
        from tsfast.models.layers import StandardScaler1D
        scaler = StandardScaler1D(norm_pair.mean, norm_pair.std)
        x = torch.rand(2, 10, 2)
        out = scaler.denormalize(scaler.normalize(x))
        torch.testing.assert_close(out, x)

    def test_minmax_roundtrip(self, norm_pair):
        from tsfast.models.layers import MinMaxScaler1D
        scaler = MinMaxScaler1D(norm_pair.min, norm_pair.max)
        x = torch.rand(2, 10, 2)
        out = scaler.denormalize(scaler.normalize(x))
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_maxabs_roundtrip(self, norm_pair):
        from tsfast.models.layers import MaxAbsScaler1D
        scaler = MaxAbsScaler1D(norm_pair.min, norm_pair.max)
        x = torch.rand(2, 10, 2)
        out = scaler.denormalize(scaler.normalize(x))
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_from_stats(self, norm_pair):
        from tsfast.models.layers import Scaler, StandardScaler1D, MinMaxScaler1D, MaxAbsScaler1D
        assert isinstance(Scaler.from_stats(norm_pair, 'standard'), StandardScaler1D)
        assert isinstance(Scaler.from_stats(norm_pair, 'minmax'), MinMaxScaler1D)
        assert isinstance(Scaler.from_stats(norm_pair, 'maxabs'), MaxAbsScaler1D)

    def test_from_stats_invalid(self, norm_pair):
        from tsfast.models.layers import Scaler
        with pytest.raises(ValueError, match="Unknown"):
            Scaler.from_stats(norm_pair, 'invalid')

    def test_unnormalize_alias(self, norm_pair):
        from tsfast.models.layers import StandardScaler1D
        scaler = StandardScaler1D(norm_pair.mean, norm_pair.std)
        x = torch.rand(2, 10, 2)
        norm = scaler.normalize(x)
        torch.testing.assert_close(scaler.unnormalize(norm), scaler.denormalize(norm))

    def test_normalizer1d_alias(self):
        from tsfast.models.layers import Normalizer1D, StandardScaler1D
        assert Normalizer1D is StandardScaler1D

    def test_normalized_model_from_stats(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.models.layers import NormalizedModel
        batch = dls_simulation.one_batch()
        device = batch[0].device
        norm_u, _, norm_y = dls_simulation.norm_stats
        model = NormalizedModel.from_stats(SimpleRNN(1, 1), norm_u, norm_y, method='minmax').to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_rnn_learner_input_norm_false(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.models.layers import NormalizedModel
        lrn = RNNLearner(dls_simulation, rnn_type="gru", input_norm=False)
        assert not isinstance(lrn.model, NormalizedModel)

    def test_rnn_learner_input_norm_minmax(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.models.layers import NormalizedModel, MinMaxScaler1D
        lrn = RNNLearner(dls_simulation, rnn_type="gru", input_norm='minmax')
        assert isinstance(lrn.model, NormalizedModel)
        assert isinstance(lrn.model.input_norm, MinMaxScaler1D)

    def test_rnn_learner_output_norm(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.models.layers import NormalizedModel, StandardScaler1D
        lrn = RNNLearner(dls_simulation, rnn_type="gru", output_norm='standard')
        assert isinstance(lrn.model, NormalizedModel)
        assert isinstance(lrn.model.output_norm, StandardScaler1D)
