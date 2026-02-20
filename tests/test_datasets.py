"""Tests for tsfast.datasets module."""
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
    def test_extract_mean_std_from_dls(self, dls_simulation):
        from tsfast.datasets.core import extract_mean_std_from_dls
        mean, std = extract_mean_std_from_dls(dls_simulation)
        assert mean.shape[-1] == 1
        assert std.shape[-1] == 1

    def test_extract_mean_std_from_hdffiles(self, hdf_files):
        from tsfast.datasets.core import extract_mean_std_from_hdffiles
        means, stds = extract_mean_std_from_hdffiles(hdf_files, ["u", "y"])
        assert means.shape == (2,)
        assert stds.shape == (2,)
        assert all(stds > 0)

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
            n_batches_train=5, sub_seq_len=25,
        )
        lrn = RNNLearner(dls, rnn_type="gru", num_layers=1, hidden_size=10, stateful=True)
        lrn.fit(1, 1e-4)

    def test_batch_limit_factory(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=None, max_batches_training=3,
        )
        assert len(dls.train) <= 3
