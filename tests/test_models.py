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
        from tsfast.training import RNNLearner
        lrn = RNNLearner(dls_simulation, rnn_type="gru")
        lrn.fit(1, 1e-4)
        final_valid_loss = lrn.recorder[-1][1]
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
        from tsfast.training import TCNLearner
        lrn = TCNLearner(dls_simulation)
        lrn.fit(1, 1e-4)
        final_valid_loss = lrn.recorder[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')

    @pytest.mark.slow
    def test_crnn_learner_fit(self, dls_simulation):
        from tsfast.training import CRNNLearner
        lrn = CRNNLearner(dls_simulation)
        lrn.fit(1, 1e-4)
        final_valid_loss = lrn.recorder[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')


class TestLayers:
    def test_seq_linear_shape(self):
        from tsfast.models.layers import SeqLinear
        m = SeqLinear(10, 3, hidden_layer=1)
        x = torch.rand(2, 50, 10)
        assert m(x).shape == (2, 50, 3)

    @pytest.mark.parametrize("hidden_layer", [0, 1, 3])
    def test_seq_linear_matches_conv1d(self, hidden_layer):
        """nn.Linear SeqLinear is numerically equal to the old Conv1d(1x1) formulation."""
        from tsfast.models.layers import SeqLinear
        nn = torch.nn

        torch.manual_seed(0)
        m = SeqLinear(7, 3, hidden_size=11, hidden_layer=hidden_layer)

        # Build the old Conv1d(1x1) reference, then copy the Linear weights into it.
        def conv_act(inp, out):
            return nn.Sequential(nn.Conv1d(inp, out, 1), nn.Mish())
        if hidden_layer < 1:
            ref_lin = nn.Conv1d(7, 3, 1)
        else:
            ref_lin = nn.Sequential(
                conv_act(7, 11),
                *[conv_act(11, 11) for _ in range(hidden_layer - 1)],
                nn.Conv1d(11, 3, 1),
            )
        linears = [mod for mod in m.lin.modules() if isinstance(mod, nn.Linear)]
        convs = [mod for mod in ref_lin.modules() if isinstance(mod, nn.Conv1d)]
        for lin, conv in zip(linears, convs):
            conv.weight.data.copy_(lin.weight.data.unsqueeze(-1))
            conv.bias.data.copy_(lin.bias.data)

        x = torch.rand(2, 50, 7)
        ref_out = ref_lin(x.transpose(1, 2)).transpose(1, 2)  # old forward: [B,S,C]->[B,C,S]->...
        torch.testing.assert_close(m(x), ref_out, atol=1e-5, rtol=1e-5)

    def test_seq_linear_preserves_leading_dims(self):
        """nn.Linear maps the last dim and keeps all leading dims (2-D / 3-D / 4-D)."""
        from tsfast.models.layers import SeqLinear
        m = SeqLinear(6, 4, hidden_size=8, hidden_layer=2)
        assert m(torch.rand(5, 6)).shape == (5, 4)
        assert m(torch.rand(2, 5, 6)).shape == (2, 5, 4)
        assert m(torch.rand(2, 3, 5, 6)).shape == (2, 3, 5, 4)

    @pytest.mark.parametrize("hidden_layer", [0, 2])
    def test_seq_linear_loads_old_conv_checkpoint(self, hidden_layer):
        """Old Conv1d-layout checkpoints (3-D ``*.weight``) load into the nn.Linear SeqLinear."""
        from tsfast.models.layers import SeqLinear
        torch.manual_seed(1)
        m = SeqLinear(5, 2, hidden_size=8, hidden_layer=hidden_layer)
        x = torch.rand(2, 10, 5)
        ref = m(x)

        # Emulate the old conv layout: every ``*.weight`` gains a trailing singleton dim.
        old_sd = {
            k: (v.unsqueeze(-1) if k.endswith("weight") else v)
            for k, v in m.state_dict().items()
        }
        assert any(v.dim() == 3 for k, v in old_sd.items() if k.endswith("weight"))

        m2 = SeqLinear(5, 2, hidden_size=8, hidden_layer=hidden_layer)
        m2.load_state_dict(old_sd)  # must not raise on the 3-D weights
        torch.testing.assert_close(m2(x), ref, atol=1e-6, rtol=1e-6)


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
    def test_ar_tcn_learner_fit(self, dls_simulation):
        from tsfast.training import AR_TCNLearner
        lrn = AR_TCNLearner(dls_simulation, hl_depth=2, hl_width=10)
        lrn.fit(1, 1e-4)
        final_valid_loss = lrn.recorder[-1][1]
        assert not math.isnan(final_valid_loss)
        assert final_valid_loss < float('inf')


class TestModelWrappers:
    def test_normalized_model_forward(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.models.scaling import ScaledModel
        batch = dls_simulation.one_batch()
        device = batch[0].device
        model = ScaledModel.from_dls(SimpleRNN(1, 1), dls_simulation).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_ar_model_teacher_forcing(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.models.layers import AR_Model
        batch = dls_simulation.one_batch()
        device = batch[0].device
        u = batch[0]  # (batch, seq, 1)
        y = batch[1]  # (batch, seq, 1)
        # AR_Model with teacher forcing: input is [u, y] concatenated (2 channels)
        model = AR_Model(SimpleRNN(2, 1), ar=False).to(device)
        out = model(torch.cat([u, y], dim=-1))
        assert out.shape == batch[1].shape

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
        from tsfast.tsdata import NormPair
        return NormPair(
            mean=np.array([1.0, 2.0], dtype=np.float32),
            std=np.array([0.5, 1.0], dtype=np.float32),
            min=np.array([0.0, 0.0], dtype=np.float32),
            max=np.array([2.0, 4.0], dtype=np.float32),
        )

    def test_standard_roundtrip(self, norm_pair):
        from tsfast.models.scaling import StandardScaler
        scaler = StandardScaler(norm_pair.mean, norm_pair.std)
        x = torch.rand(2, 10, 2)
        out = scaler.denormalize(scaler.normalize(x))
        torch.testing.assert_close(out, x)

    def test_minmax_roundtrip(self, norm_pair):
        from tsfast.models.scaling import MinMaxScaler
        scaler = MinMaxScaler(norm_pair.min, norm_pair.max)
        x = torch.rand(2, 10, 2)
        out = scaler.denormalize(scaler.normalize(x))
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_maxabs_roundtrip(self, norm_pair):
        from tsfast.models.scaling import MaxAbsScaler
        scaler = MaxAbsScaler(norm_pair.min, norm_pair.max)
        x = torch.rand(2, 10, 2)
        out = scaler.denormalize(scaler.normalize(x))
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_from_stats_classmethod(self, norm_pair):
        from tsfast.models.scaling import StandardScaler, MinMaxScaler, MaxAbsScaler
        assert isinstance(StandardScaler.from_stats(norm_pair), StandardScaler)
        assert isinstance(MinMaxScaler.from_stats(norm_pair), MinMaxScaler)
        assert isinstance(MaxAbsScaler.from_stats(norm_pair), MaxAbsScaler)

    def test_unnormalize_alias(self, norm_pair):
        from tsfast.models.scaling import StandardScaler
        scaler = StandardScaler(norm_pair.mean, norm_pair.std)
        x = torch.rand(2, 10, 2)
        norm = scaler.normalize(x)
        torch.testing.assert_close(scaler.unnormalize(norm), scaler.denormalize(norm))

    def test_normalized_model_from_stats(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.models.scaling import ScaledModel, MinMaxScaler
        batch = dls_simulation.one_batch()
        device = batch[0].device
        norm_u, norm_y = dls_simulation.norm_stats
        model = ScaledModel.from_stats(SimpleRNN(1, 1), norm_u, norm_y, scaler_cls=MinMaxScaler).to(device)
        out = model(batch[0])
        assert out.shape == batch[1].shape

    def test_rnn_learner_input_norm_none(self, dls_simulation):
        from tsfast.training import RNNLearner
        from tsfast.models.scaling import ScaledModel
        lrn = RNNLearner(dls_simulation, rnn_type="gru", input_norm=None)
        assert not isinstance(lrn.model, ScaledModel)

    def test_rnn_learner_input_norm_minmax(self, dls_simulation):
        from tsfast.training import RNNLearner
        from tsfast.models.scaling import ScaledModel, MinMaxScaler
        lrn = RNNLearner(dls_simulation, rnn_type="gru", input_norm=MinMaxScaler)
        assert isinstance(lrn.model, ScaledModel)
        assert isinstance(lrn.model.input_norm, MinMaxScaler)

    def test_rnn_learner_output_norm(self, dls_simulation):
        from tsfast.training import RNNLearner
        from tsfast.models.scaling import ScaledModel, StandardScaler
        lrn = RNNLearner(dls_simulation, rnn_type="gru", output_norm=StandardScaler)
        assert isinstance(lrn.model, ScaledModel)
        assert isinstance(lrn.model.output_norm, StandardScaler)

