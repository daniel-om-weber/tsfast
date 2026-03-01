"""Tests for tsfast.training — pure-PyTorch training framework."""

import math

import matplotlib
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _SyntheticDls:
    """Minimal DataLoaders-like object for unit tests (no HDF5 dependency)."""

    def __init__(self, n_u=1, n_y=1, seq_len=100, n_train=16, n_valid=8, bs=4):
        x_train = torch.randn(n_train, seq_len, n_u)
        y_train = torch.randn(n_train, seq_len, n_y)
        x_valid = torch.randn(n_valid, seq_len, n_u)
        y_valid = torch.randn(n_valid, seq_len, n_y)
        self.train = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)
        self.valid = DataLoader(TensorDataset(x_valid, y_valid), batch_size=bs)
        self.test = None


# ──────────────────────────────────────────────────────────────────────────────
#  Unit tests — losses & metrics
# ──────────────────────────────────────────────────────────────────────────────


class TestLosses:
    def test_mse_local(self):
        from tsfast.training.losses import mse

        pred = torch.rand(4, 100, 1)
        targ = torch.rand(4, 100, 1)
        assert torch.allclose(mse(pred, targ), F.mse_loss(pred, targ))

    def test_mse_nan(self):
        from tsfast.training.losses import mse_nan

        x = torch.rand(4, 10, 2)
        y = torch.rand(4, 10, 2)
        y[0, :, :] = float("nan")
        loss = mse_nan(x, y)
        assert not torch.isnan(loss)

    def test_nan_reduce_mean(self):
        from tsfast.training.losses import nan_reduce

        loss = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        result = nan_reduce(loss)
        assert torch.allclose(result, torch.tensor(7.0 / 3.0))

    def test_nan_reduce_rms(self):
        from tsfast.training.losses import nan_reduce

        loss = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        result = nan_reduce(loss, reduction="rms")
        # sqrt((1 + 4 + 16) / 3) = sqrt(7)
        assert torch.allclose(result, torch.tensor(7.0).sqrt())

    def test_nan_reduce_no_nans(self):
        from tsfast.training.losses import nan_reduce

        loss = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(nan_reduce(loss), torch.tensor(2.0))

    def test_nan_reduce_all_nans(self):
        from tsfast.training.losses import nan_reduce

        loss = torch.full((4,), float("nan"))
        assert nan_reduce(loss).item() == 0.0

    def test_nan_reduce_multidim(self):
        from tsfast.training.losses import nan_reduce

        loss = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
        assert torch.allclose(nan_reduce(loss), torch.tensor(8.0 / 3.0))

    def test_nan_reduce_composable(self):
        from tsfast.training.losses import nan_reduce

        pred = torch.rand(4, 10, 2)
        targ = torch.rand(4, 10, 2)
        targ[0, :, :] = float("nan")
        loss = nan_reduce(F.l1_loss(pred, targ, reduction="none"))
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_mse_nan_safe(self):
        from tsfast.training.losses import mse

        pred = torch.rand(4, 10, 2)
        targ = torch.rand(4, 10, 2)
        targ[0, :, :] = float("nan")
        loss = mse(pred, targ)
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_cos_sim_loss_nan_safe(self):
        from tsfast.training.losses import cos_sim_loss

        pred = torch.rand(4, 10, 2)
        targ = torch.rand(4, 10, 2)
        targ[0, :, :] = float("nan")
        loss = cos_sim_loss(pred, targ)
        assert not torch.isnan(loss)

    def test_zero_loss(self):
        from tsfast.training.losses import zero_loss

        x = torch.rand(2, 50, 1)
        y = torch.rand(2, 50, 1)
        assert zero_loss(x, y).item() == 0.0

    def test_fun_rmse(self):
        from tsfast.training.losses import fun_rmse

        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        assert fun_rmse(x, y).item() > 0

    def test_cos_sim_loss_same(self):
        from tsfast.training.losses import cos_sim_loss

        x = torch.rand(4, 100, 2)
        assert cos_sim_loss(x, x).item() < 1e-5

    def test_nrmse(self):
        from tsfast.training.losses import nrmse

        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        assert nrmse(x, y).item() > 0

    def test_sched_lin_p(self):
        from tsfast.training.schedulers import sched_lin_p

        assert sched_lin_p(0, 1, 0.0) == 0.0
        assert sched_lin_p(0, 1, 0.75) == 1.0
        assert sched_lin_p(0, 1, 1.0) == 1.0

    def test_sched_ramp(self):
        from tsfast.training.schedulers import sched_ramp

        assert sched_ramp(0, 1, 0.0) == 0.0
        assert sched_ramp(0, 1, 1.0) == 1.0
        assert 0 < sched_ramp(0, 1, 0.4) < 1

    def test_cut_loss(self):
        from tsfast.training.losses import cut_loss

        fn = cut_loss(nn.MSELoss(), l_cut=5, r_cut=-5)
        x = torch.rand(2, 100, 1)
        y = torch.rand(2, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_weighted_mae(self):
        from tsfast.training.losses import weighted_mae

        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = weighted_mae(x, y)
        assert loss.item() > 0

    def test_rand_seq_len_loss(self):
        from tsfast.training.losses import rand_seq_len_loss

        fn = rand_seq_len_loss(nn.MSELoss(), min_idx=10)
        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = fn(x, y)
        assert loss.item() >= 0

    def test_cos_sim_loss_pow(self):
        from tsfast.training.losses import cos_sim_loss_pow

        x = torch.rand(4, 100, 2)
        loss_same = cos_sim_loss_pow(x, x)
        assert loss_same.item() < 1e-5
        y = torch.rand(4, 100, 2)
        loss_diff = cos_sim_loss_pow(x, y)
        assert loss_diff.item() > 0

    def test_nrmse_std(self):
        from tsfast.training.losses import nrmse_std

        x = torch.rand(4, 100, 1)
        y = torch.rand(4, 100, 1)
        loss = nrmse_std(x, y)
        assert loss.item() > 0

    def test_mean_vaf_perfect(self):
        from tsfast.training.losses import mean_vaf

        x = torch.rand(4, 100, 1)
        vaf = mean_vaf(x, x)
        assert vaf.item() == pytest.approx(100.0, abs=0.1)


# ──────────────────────────────────────────────────────────────────────────────
#  Unit tests — transforms
# ──────────────────────────────────────────────────────────────────────────────


class TestTransforms:
    def test_prediction_concat(self):
        from tsfast.training.transforms import prediction_concat

        t = prediction_concat(t_offset=1)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == (4, 99, 3)  # 2 + 1 features, 100 - 1 steps
        assert yb_out.shape == (4, 99, 1)

    def test_prediction_concat_no_offset(self):
        from tsfast.training.transforms import prediction_concat

        t = prediction_concat(t_offset=0)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == (4, 100, 3)
        assert yb_out.shape == (4, 100, 1)

    def test_prediction_concat_no_offset(self):
        from tsfast.training.transforms import prediction_concat

        t = prediction_concat(t_offset=0)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == (4, 100, 3)
        assert torch.equal(yb_out, yb)

    def test_vary_seq_len(self):
        from tsfast.training.transforms import vary_seq_len

        t = vary_seq_len(min_len=20)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert 20 <= yb_out.shape[1] <= 100
        assert xb_out.shape[1] == yb_out.shape[1]

    def test_truncate_sequence(self):
        from tsfast.training.transforms import truncate_sequence

        t = truncate_sequence(truncate_length=30)

        # Mock trainer
        class _MockTrainer:
            pct_train = 0.0

        trainer = _MockTrainer()
        t.setup(trainer)

        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)

        # At pct_train=0 with sched_ramp defaults, truncation is at start value
        xb_out, yb_out = t(xb, yb)
        assert yb_out.shape[1] <= 100

        # At pct_train=1.0, truncation should reach 0 (no truncation)
        trainer.pct_train = 1.0
        xb_out2, yb_out2 = t(xb, yb)
        assert yb_out2.shape[1] == 100

        t.teardown(trainer)

    def test_noise_shape_preserved(self):
        from tsfast.training.transforms import noise

        t = noise(std=0.1)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == xb.shape
        assert torch.equal(yb_out, yb)  # yb unchanged

    def test_noise_changes_values(self):
        from tsfast.training.transforms import noise

        t = noise(std=1.0)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, _ = t(xb, yb)
        assert not torch.equal(xb_out, xb)

    def test_bias_shape_preserved(self):
        from tsfast.training.transforms import bias

        t = bias(std=0.1)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == xb.shape
        assert torch.equal(yb_out, yb)

    def test_noise_varying(self):
        from tsfast.training.transforms import noise_varying

        t = noise_varying(std_std=0.1)
        xb = torch.randn(4, 100, 2)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == xb.shape

    def test_noise_grouped(self):
        from tsfast.training.transforms import noise_grouped

        t = noise_grouped(std_std=[0.1, 0.2], std_idx=[0, 0, 1])
        xb = torch.randn(4, 100, 3)
        yb = torch.randn(4, 100, 1)
        xb_out, yb_out = t(xb, yb)
        assert xb_out.shape == xb.shape


# ──────────────────────────────────────────────────────────────────────────────
#  Unit tests — visualization
# ──────────────────────────────────────────────────────────────────────────────


class TestViz:
    def test_plot_sequence(self):
        import matplotlib.pyplot as plt

        from tsfast.training.viz import plot_sequence

        fig, axs = plt.subplots(3, 1)
        in_sig = torch.randn(100, 2)
        targ_sig = torch.randn(100, 2)
        out_sig = torch.randn(100, 2)
        plot_sequence(axs, in_sig, targ_sig, out_sig=out_sig)
        # No signal names: no titles on target subplots, no legend on input subplot
        assert axs[0].get_title() == ""
        assert axs[-1].get_legend() is None
        plt.close(fig)

    def test_plot_sequence_with_signal_names(self):
        import matplotlib.pyplot as plt

        from tsfast.training.viz import plot_sequence

        fig, axs = plt.subplots(3, 1)
        in_sig = torch.randn(100, 2)
        targ_sig = torch.randn(100, 2)
        out_sig = torch.randn(100, 2)
        names = (["u1", "u2"], ["y1", "y2"])
        plot_sequence(axs, in_sig, targ_sig, out_sig=out_sig, signal_names=names)
        assert axs[0].get_title() == "y1"
        assert axs[1].get_title() == "y2"
        legend_texts = [t.get_text() for t in axs[-1].get_legend().get_texts()]
        assert legend_texts == ["u1", "u2"]
        plt.close(fig)

    def test_grad_norm(self):
        from tsfast.training.viz import grad_norm

        model = nn.Linear(10, 1)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        norm = grad_norm(model.parameters())
        assert norm > 0

    def test_grad_norm_no_grads(self):
        from tsfast.training.viz import grad_norm

        model = nn.Linear(10, 1)
        assert grad_norm(model.parameters()) == 0.0

    def test_plot_grad_flow(self):
        import matplotlib.pyplot as plt

        from tsfast.training.viz import plot_grad_flow

        model = nn.Linear(10, 1)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        plt.figure()
        plot_grad_flow(model.named_parameters())
        plt.close("all")


# ──────────────────────────────────────────────────────────────────────────────
#  Unit tests — aux losses
# ──────────────────────────────────────────────────────────────────────────────


class TestAuxLosses:
    def test_auxiliary_loss(self):
        from tsfast.training.aux_losses import AuxiliaryLoss

        aux = AuxiliaryLoss(nn.MSELoss(), alpha=0.5)
        pred = torch.rand(4, 100, 1)
        yb = torch.rand(4, 100, 1)
        xb = torch.rand(4, 100, 1)
        loss = aux(pred, yb, xb)
        expected = 0.5 * F.mse_loss(pred, yb)
        assert torch.allclose(loss, expected)

    def test_transition_smoothness_loss(self):
        from tsfast.pinn import TransitionSmoothnessLoss

        ts = TransitionSmoothnessLoss(init_sz=20, weight=1.0, window=3, dt=0.01)
        pred = torch.rand(4, 50, 2, requires_grad=True)
        yb = torch.rand(4, 50, 2)
        xb = torch.rand(4, 50, 1)
        loss = ts(pred, yb, xb)
        assert loss.item() >= 0
        # Verify it's differentiable
        loss.backward()
        assert pred.grad is not None


# ──────────────────────────────────────────────────────────────────────────────
#  Integration tests — Learner
# ──────────────────────────────────────────────────────────────────────────────


class TestLearner:
    def test_learner_smoke(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))
        lrn.fit(1)
        final_valid_loss = lrn.recorder.values[-1][1]
        assert math.isfinite(final_valid_loss)

    def test_learner_with_metrics(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner
        from tsfast.training.losses import fun_rmse

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), metrics=[fun_rmse], device=torch.device("cpu"))
        lrn.fit(1)
        assert len(lrn.recorder.values[-1]) == 3  # train_loss, val_loss, metric

    def test_learner_with_transforms(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner, prediction_concat

        dls = _SyntheticDls(n_u=1, n_y=1)
        # prediction_concat adds n_y=1 to input, so model sees n_u=2
        model = SimpleRNN(2, 1, hidden_size=20)
        lrn = Learner(
            model,
            dls,
            loss_func=nn.MSELoss(),
            transforms=[prediction_concat(t_offset=1)],
            device=torch.device("cpu"),
        )
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_learner_augmentations_train_only(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner
        from tsfast.training.transforms import noise

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)

        aug = noise(std=0.1)
        lrn = Learner(
            model,
            dls,
            loss_func=nn.MSELoss(),
            augmentations=[aug],
            device=torch.device("cpu"),
        )
        lrn.fit(1)

        # Validate calls should not apply augmentations — verify by running
        # validate twice and getting the same result
        val1, _ = lrn.validate()
        val2, _ = lrn.validate()
        assert val1 == val2

    def test_learner_aux_losses(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import AuxiliaryLoss, Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)

        # Train without aux loss
        lrn_base = Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))
        lrn_base.fit(1)

        # Train with aux loss (AuxiliaryLoss should make total > primary)
        model2 = SimpleRNN(1, 1, hidden_size=20)
        aux = AuxiliaryLoss(nn.L1Loss(), alpha=1.0)
        lrn_aux = Learner(model2, dls, loss_func=nn.MSELoss(), aux_losses=[aux], device=torch.device("cpu"))
        lrn_aux.fit(1)

        # Both should have finite losses
        assert math.isfinite(lrn_base.recorder.values[-1][1])
        assert math.isfinite(lrn_aux.recorder.values[-1][1])

    def test_learner_n_skip(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), n_skip=10, device=torch.device("cpu"))
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_learner_grad_clip(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), grad_clip=1.0, device=torch.device("cpu"))
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_learner_fit_flat_cos(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))
        lrn.fit_flat_cos(2, pct_start=0.5)
        assert len(lrn.recorder.values) == 2

    def test_get_preds(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1, n_valid=8, bs=4)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))
        preds, targs = lrn.get_preds(ds_idx=1)
        assert preds.shape[0] == 8
        assert targs.shape[0] == 8
        assert preds.shape[1] == 100
        assert preds.shape[2] == 1

    def test_show_batch_show_results(self):
        import matplotlib.pyplot as plt

        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))
        lrn.show_batch(max_n=2)
        plt.close("all")
        lrn.show_results(max_n=2)
        plt.close("all")

    def test_no_bar(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import Learner

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))
        with lrn.no_bar():
            lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_activation_regularizers(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import ActivationRegularizer, Learner, TemporalActivationRegularizer

        dls = _SyntheticDls(n_u=1, n_y=1)
        model = SimpleRNN(1, 1, hidden_size=20)
        ar = ActivationRegularizer(modules=[model.rnn], alpha=0.1)
        tar = TemporalActivationRegularizer(modules=[model.rnn], beta=0.1)
        lrn = Learner(model, dls, loss_func=nn.MSELoss(), aux_losses=[ar, tar], device=torch.device("cpu"))
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])


# ──────────────────────────────────────────────────────────────────────────────
#  Integration tests — TbpttLearner
# ──────────────────────────────────────────────────────────────────────────────


class TestTbpttLearner:
    def test_tbptt_smoke(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import TbpttLearner

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = TbpttLearner(model, dls, loss_func=nn.MSELoss(), sub_seq_len=25, device=torch.device("cpu"))
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_tbptt_n_skip_first_chunk_only(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import TbpttLearner

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = TbpttLearner(
            model, dls, loss_func=nn.MSELoss(), sub_seq_len=25, n_skip=10, device=torch.device("cpu")
        )
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])
        # n_skip must be restored after _train_one_batch
        assert lrn.n_skip == 10

    def test_tbptt_with_augmentations(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import TbpttLearner
        from tsfast.training.transforms import noise

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = TbpttLearner(
            model,
            dls,
            loss_func=nn.MSELoss(),
            sub_seq_len=25,
            augmentations=[noise(std=0.05)],
            device=torch.device("cpu"),
        )
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])


# ──────────────────────────────────────────────────────────────────────────────
#  CudaGraphTbpttLearner
# ──────────────────────────────────────────────────────────────────────────────


@requires_cuda
class TestCudaGraphTbpttLearner:
    def test_cuda_graph_tbptt_smoke(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import CudaGraphTbpttLearner

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = CudaGraphTbpttLearner(model, dls, loss_func=nn.MSELoss(), sub_seq_len=25)
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_cuda_graph_tbptt_n_skip(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import CudaGraphTbpttLearner

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = CudaGraphTbpttLearner(
            model, dls, loss_func=nn.MSELoss(), sub_seq_len=25, n_skip=10
        )
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])
        assert lrn.n_skip == 10

    def test_cuda_graph_tbptt_with_augmentations(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import CudaGraphTbpttLearner
        from tsfast.training.transforms import noise

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = CudaGraphTbpttLearner(
            model, dls, loss_func=nn.MSELoss(), sub_seq_len=25, augmentations=[noise(std=0.05)]
        )
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_cuda_graph_tbptt_aux_losses(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import CudaGraphTbpttLearner
        from tsfast.training.aux_losses import AuxiliaryLoss

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = CudaGraphTbpttLearner(
            model, dls, loss_func=nn.MSELoss(), sub_seq_len=25,
            aux_losses=[AuxiliaryLoss(nn.MSELoss(), alpha=0.1)],
        )
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])

    def test_cuda_graph_tbptt_graph_reset(self):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.training import CudaGraphTbpttLearner

        dls = _SyntheticDls(n_u=1, n_y=1, seq_len=100)
        model = SimpleRNN(1, 1, hidden_size=20, return_state=True)
        lrn = CudaGraphTbpttLearner(model, dls, loss_func=nn.MSELoss(), sub_seq_len=25)
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])
        # Second fit should re-capture the graph and succeed
        lrn.fit(1)
        assert math.isfinite(lrn.recorder.values[-1][1])
