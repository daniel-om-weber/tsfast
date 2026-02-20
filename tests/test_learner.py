"""Tests for tsfast.learner module."""
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
    def test_gradient_clipping(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import GradientClipping
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=GradientClipping(10)).fit(1)

    @pytest.mark.slow
    def test_weight_clipping(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import WeightClipping
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=WeightClipping(model, clip_limit=1)).fit(1)

    @pytest.mark.slow
    def test_vary_seq_len(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import VarySeqLen
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=VarySeqLen(10)).fit(1)

    @pytest.mark.slow
    def test_skip_first_n_callback(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import SkipFirstNCallback
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=SkipFirstNCallback(10)).fit(1)

    @pytest.mark.slow
    def test_skip_nan_callback(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import SkipNaNCallback
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=SkipNaNCallback()).fit(1)

    @pytest.mark.slow
    def test_cb_truncate_sequence(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import CB_TruncateSequence
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(), cbs=CB_TruncateSequence(30)).fit(1)

    @pytest.mark.slow
    def test_cb_add_loss(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import CB_AddLoss
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(),
                cbs=CB_AddLoss(nn.L1Loss(), alpha=0.5)).fit(1)

    @pytest.mark.slow
    def test_batch_loss_filter(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import BatchLossFilter
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(),
                cbs=BatchLossFilter(loss_perc=0.5)).fit(1)

    @pytest.mark.slow
    def test_time_series_regularizer(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import TimeSeriesRegularizer
        from fastai.basics import Learner
        model = SimpleRNN(1, 1, hidden_size=20)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(),
                cbs=TimeSeriesRegularizer(alpha=0.1, beta=0.1, modules=[model.rnn])).fit(1)

    @pytest.mark.slow
    def test_gradient_batch_filtering(self, dls_simulation):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import GradientBatchFiltering
        from fastai.basics import Learner
        model = SimpleRNN(1, 1)
        Learner(dls_simulation, model, loss_func=nn.MSELoss(),
                cbs=GradientBatchFiltering(filter_val=100)).fit(1)
