"""Tests for tsfast.training losses."""
import pytest
import torch
from torch import nn


class TestLosses:
    def test_fun_rmse(self):
        from tsfast.training import fun_rmse
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = fun_rmse(x, y)
        assert loss.item() > 0

    def test_nrmse(self):
        from tsfast.training import nrmse
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = nrmse(x, y)
        assert loss.item() > 0

    def test_skip_n_loss(self):
        from tsfast.training import SkipNLoss
        fn = SkipNLoss(nn.MSELoss(), n_skip=10)
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_zero_loss(self):
        from tsfast.training import zero_loss
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        assert zero_loss(x, y).item() == 0.0

    def test_mse_nan(self):
        from tsfast.training import mse_nan
        x = torch.rand(4, 10, 2)
        y = torch.rand(4, 10, 2)
        y[0, :, :] = float("nan")
        loss = mse_nan(x, y)
        assert not torch.isnan(loss)

    def test_cos_sim_loss_same_vectors(self):
        from tsfast.training import cos_sim_loss
        x = torch.rand(4, 100, 2)
        loss = cos_sim_loss(x, x)
        assert loss.item() < 1e-5

    def test_cut_loss(self):
        from tsfast.training import CutLoss
        fn = CutLoss(nn.MSELoss(), l_cut=5, r_cut=-5)
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_weighted_mae(self):
        from tsfast.training import weighted_mae
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = weighted_mae(x, y)
        assert loss.item() > 0

    def test_rand_seq_len_loss(self):
        from tsfast.training import RandSeqLenLoss
        fn = RandSeqLenLoss(nn.MSELoss(), min_idx=10)
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_cos_sim_loss_pow(self):
        from tsfast.training import cos_sim_loss_pow
        x = torch.rand(4, 100, 2)
        loss_same = cos_sim_loss_pow(x, x)
        assert loss_same.item() < 1e-5
        y = torch.rand(4, 100, 2)
        loss_diff = cos_sim_loss_pow(x, y)
        assert loss_diff.item() > 0

    def test_nrmse_std(self):
        from tsfast.training import nrmse_std
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = nrmse_std(x, y)
        assert loss.item() > 0

    def test_mean_vaf_perfect(self):
        from tsfast.training import mean_vaf
        x = torch.rand(4, 100, 1)
        vaf = mean_vaf(x, x)
        assert vaf.item() == pytest.approx(100.0, abs=0.1)

    def test_norm_loss_equalizes_scales(self):
        """NormLoss should equalize the contribution of variables with different scales."""
        import numpy as np
        from tsfast.training import NormLoss
        from tsfast.tsdata import NormPair

        # Two variables: var0 ~ O(1), var1 ~ O(1000)
        stats = NormPair(
            mean=np.array([0.0, 500.0]),
            std=np.array([1.0, 500.0]),
            min=np.array([-2.0, 0.0]),
            max=np.array([2.0, 1000.0]),
        )
        pred = torch.zeros(2, 50, 2)
        targ = torch.zeros(2, 50, 2)
        # Same absolute error (1.0) in both variables
        targ[..., 0] = 1.0
        targ[..., 1] = 1.0

        # Raw MSE: both contribute equally (error=1 for both)
        raw_loss = nn.MSELoss()
        raw = raw_loss(pred, targ)

        # NormLoss: var1 error is 1/500 after normalization, var0 stays 1/1
        norm_loss = NormLoss(nn.MSELoss(), stats)
        normed = norm_loss(pred, targ)

        # var0 dominates → normed loss is almost entirely from var0
        # so normed ≈ 0.5 * (1/1)^2 + 0.5 * (1/500)^2 ≈ 0.5
        # while raw = 1.0 (equal contribution)
        assert normed.item() < raw.item()

        # Verify it composes with SkipNLoss
        from tsfast.training import SkipNLoss
        composed = SkipNLoss(NormLoss(nn.MSELoss(), stats), n_skip=10)
        loss = composed(pred, targ)
        assert loss.item() >= 0
