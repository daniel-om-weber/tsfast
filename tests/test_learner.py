"""Tests for tsfast.learner module."""
import math
import pytest
import torch
from torch import nn


class TestLosses:
    def test_fun_rmse(self):
        from tsfast.learner.losses import fun_rmse
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = fun_rmse(x, y)
        assert loss.item() > 0

    def test_nrmse(self):
        from tsfast.learner.losses import nrmse
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = nrmse(x, y)
        assert loss.item() > 0

    def test_skip_n_loss(self):
        from tsfast.learner.losses import SkipNLoss
        fn = SkipNLoss(nn.MSELoss(), n_skip=10)
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_zero_loss(self):
        from tsfast.learner.losses import zero_loss
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        assert zero_loss(x, y).item() == 0.0

    def test_mse_nan(self):
        from tsfast.learner.losses import mse_nan
        x = torch.rand(4, 10, 2)
        y = torch.rand(4, 10, 2)
        y[0, :, :] = float("nan")
        loss = mse_nan(x, y)
        assert not torch.isnan(loss)

    def test_cos_sim_loss_same_vectors(self):
        from tsfast.learner.losses import cos_sim_loss
        x = torch.rand(4, 100, 2)
        loss = cos_sim_loss(x, x)
        assert loss.item() < 1e-5

    def test_cut_loss(self):
        from tsfast.learner.losses import CutLoss
        fn = CutLoss(nn.MSELoss(), l_cut=5, r_cut=-5)
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_weighted_mae(self):
        from tsfast.learner.losses import weighted_mae
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = weighted_mae(x, y)
        assert loss.item() > 0

    def test_rand_seq_len_loss(self):
        from tsfast.learner.losses import RandSeqLenLoss
        fn = RandSeqLenLoss(nn.MSELoss(), min_idx=10)
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_cos_sim_loss_pow(self):
        from tsfast.learner.losses import cos_sim_loss_pow
        x = torch.rand(4, 100, 2)
        loss_same = cos_sim_loss_pow(x, x)
        assert loss_same.item() < 1e-5
        y = torch.rand(4, 100, 2)
        loss_diff = cos_sim_loss_pow(x, y)
        assert loss_diff.item() > 0

    def test_nrmse_std(self):
        from tsfast.learner.losses import nrmse_std
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = nrmse_std(x, y)
        assert loss.item() > 0

    def test_mean_vaf_perfect(self):
        from tsfast.learner.losses import mean_vaf
        x = torch.rand(4, 100, 1)
        vaf = mean_vaf(x, x)
        assert vaf.item() == pytest.approx(100.0, abs=0.1)


class TestCallbacks:
    @pytest.mark.slow
    @pytest.mark.parametrize("make_cb", [
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['GradientClipping']).GradientClipping(10), id="gradient_clipping"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['WeightClipping']).WeightClipping(m, clip_limit=1), id="weight_clipping"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['VarySeqLen']).VarySeqLen(10), id="vary_seq_len"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['SkipFirstNCallback']).SkipFirstNCallback(10), id="skip_first_n"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['SkipNaNCallback']).SkipNaNCallback(), id="skip_nan"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['CB_TruncateSequence']).CB_TruncateSequence(30), id="truncate_sequence"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['CB_AddLoss']).CB_AddLoss(nn.L1Loss(), alpha=0.5), id="add_loss"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['BatchLossFilter']).BatchLossFilter(loss_perc=0.5), id="batch_loss_filter"),
        pytest.param(lambda m: __import__('tsfast.learner.callbacks', fromlist=['GradientBatchFiltering']).GradientBatchFiltering(filter_val=100), id="gradient_batch_filtering"),
    ])
    def test_callback_smoke(self, dls_simulation, make_cb):
        from tsfast.models.rnn import SimpleRNN
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        lrn = Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=make_cb(model))
        lrn.fit(1)
        final_valid_loss = lrn.recorder.values[-1][1]
        assert final_valid_loss < float('inf')
        assert not math.isnan(final_valid_loss)

    @pytest.mark.slow
    def test_gradient_batch_filtering_triggers(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import GradientBatchFiltering
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        # Very low threshold: every batch should be filtered
        params_before = [p.clone() for p in model.parameters()]
        lrn = Learner(dls_simulation, model, loss_func=nn.MSELoss(),
                      cbs=GradientBatchFiltering(filter_val=1e-10))
        lrn.fit(1)
        # Model params should be unchanged since all gradients were zeroed
        for p_before, p_after in zip(params_before, model.parameters()):
            assert torch.allclose(p_before, p_after.data)

    @pytest.mark.slow
    def test_time_series_regularizer(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import TimeSeriesRegularizer
        from fastai.basics import Learner
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(dls_simulation, model, loss_func=nn.MSELoss(),
                cbs=TimeSeriesRegularizer(alpha=0.1, beta=0.1, modules=[model.rnn]))
        lrn.fit(1)
        final_valid_loss = lrn.recorder.values[-1][1]
        assert final_valid_loss < float('inf')
        assert not math.isnan(final_valid_loss)
