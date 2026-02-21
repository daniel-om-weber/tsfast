"""Tests for tsfast.datasets module."""
import math
import pytest
import numpy as np
import torch


class TestCreateDls:
    def test_simulation_mode(self, dls_simulation):
        batch = dls_simulation.one_batch()
        assert batch[0].shape[-1] == 1  # u only
        assert batch[1].shape[-1] == 1  # y only

    def test_prediction_mode(self, dls_prediction):
        batch = dls_prediction.one_batch()
        assert batch[0].shape[-1] == 2  # u + y concatenated
        assert batch[1].shape[-1] == 1  # y only

    def test_test_dataloader_appended(self, dls_simulation):
        assert len(dls_simulation.loaders) == 3  # train, valid, test


class TestNormalization:
    def test_norm_stats_on_dls(self, dls_simulation):
        from tsfast.datasets.core import NormPair, NormStats
        assert hasattr(dls_simulation, 'norm_stats')
        stats = dls_simulation.norm_stats
        assert isinstance(stats, NormStats)
        assert isinstance(stats.u, NormPair)
        assert stats.x is None
        assert isinstance(stats.y, NormPair)
        # Named access
        assert stats.u.mean.shape == (1,)
        assert stats.u.std.shape == (1,)
        assert stats.u.min.shape == (1,)
        assert stats.u.max.shape == (1,)
        assert stats.y.mean.shape == (1,)
        assert stats.y.std.shape == (1,)
        # Tuple destructuring still works
        norm_u, norm_x, norm_y = stats
        assert norm_u.mean.shape == (1,)

    def test_extract_mean_std_from_dls(self, dls_simulation):
        from tsfast.datasets.core import extract_mean_std_from_dls
        norm_u, norm_x, norm_y = extract_mean_std_from_dls(dls_simulation)
        assert norm_u.mean.shape == (1,)
        assert norm_u.std.shape == (1,)

    def test_extract_mean_std_from_hdffiles(self, hdf_files):
        from tsfast.datasets.core import extract_mean_std_from_hdffiles
        means, stds = extract_mean_std_from_hdffiles(hdf_files, ["u", "y"])
        assert means.shape == (2,)
        assert stds.shape == (2,)
        assert all(stds > 0)

    def test_estimate_norm_stats(self, dls_simulation):
        from tsfast.datasets.core import estimate_norm_stats, NormPair
        input_stats, output_stats = estimate_norm_stats(dls_simulation, n_batches=3)
        assert isinstance(input_stats, NormPair)
        assert isinstance(output_stats, NormPair)
        assert input_stats.mean.shape == (1,)
        assert input_stats.std.shape == (1,)
        assert input_stats.min.shape == (1,)
        assert input_stats.max.shape == (1,)
        assert all(input_stats.std > 0)
        assert all(input_stats.min <= input_stats.mean)
        assert all(input_stats.max >= input_stats.mean)

    def test_normpair_add(self, dls_simulation):
        from tsfast.datasets.core import NormPair
        stats = dls_simulation.norm_stats
        combined = stats.u + stats.y
        assert combined.mean.shape == (2,)
        assert combined.std.shape == (2,)
        assert combined.min.shape == (2,)
        assert combined.max.shape == (2,)
        np.testing.assert_array_equal(combined.mean, np.hstack([stats.u.mean, stats.y.mean]))
        np.testing.assert_array_equal(combined.std, np.hstack([stats.u.std, stats.y.std]))

    def test_normpair_backward_compat(self, dls_simulation):
        from tsfast.datasets.core import NormPair
        stats = dls_simulation.norm_stats
        # Indexing
        assert np.array_equal(stats.u[0], stats.u.mean)
        assert np.array_equal(stats.u[1], stats.u.std)
        # Iteration
        items = list(stats.u)
        assert len(items) == 4
        # Destructuring
        mean, std, mn, mx = stats.u
        assert np.array_equal(mean, stats.u.mean)

    def test_is_dataset_directory(self, wh_path):
        from tsfast.datasets.core import is_dataset_directory
        assert is_dataset_directory(wh_path) is True
        assert is_dataset_directory(wh_path.parent) is False


class TestTbpttDataLoader:
    def test_tbptt_dl_sub_sequence_shape(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5, sub_seq_len=50,
        )
        batch = dls.one_batch()
        assert batch[0].shape[1] == 50  # sub_seq_len truncation

    def test_tbptt_dl_n_sub_seq(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5, sub_seq_len=50,
        )
        # win_sz=100 / sub_seq_len=50 = 2 sub-sequences per base batch
        assert dls.train.n_sub_seq == 2

    @pytest.mark.slow
    def test_tbptt_rnn_training(self, wh_path):
        from tsfast.datasets.core import create_dls
        from tsfast.models.rnn import RNNLearner
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2, sub_seq_len=25,
        )
        lrn = RNNLearner(dls, rnn_type="gru", num_layers=1, hidden_size=10, stateful=True)
        lrn.fit(1, 1e-4)
        assert not math.isnan(lrn.recorder.values[-1][1])

    def test_batch_limit_factory(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=None, max_batches_training=3,
        )
        assert len(dls.train) <= 3
