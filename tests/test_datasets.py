"""Tests for tsfast.datasets module."""
import pytest
import numpy as np


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
