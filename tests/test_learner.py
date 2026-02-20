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
