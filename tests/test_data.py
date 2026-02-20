"""Tests for tsfast.data module."""
import pytest
import torch
import numpy as np
from pathlib import Path


class TestHDFFiles:
    def test_get_hdf_files(self, wh_path):
        from tsfast.data.core import get_hdf_files
        files = get_hdf_files(wh_path)
        assert len(files) == 3

    def test_apply_df_tfms_idempotent(self, hdf_files):
        from tsfast.data.core import apply_df_tfms
        r1 = apply_df_tfms(hdf_files)
        r2 = apply_df_tfms(apply_df_tfms(hdf_files))
        assert r1.equals(r2)


class TestCreateDict:
    def test_basic(self, hdf_files):
        from tsfast.data.core import CreateDict
        result = CreateDict()(hdf_files)
        assert len(result) == 3
        assert "path" in result[0]

    def test_valid_clm_contains(self, hdf_files):
        from tsfast.data.core import CreateDict, ValidClmContains
        result = CreateDict([ValidClmContains(["valid"])])(hdf_files)
        valid_count = sum(1 for r in result if r["valid"])
        assert valid_count == 1

    def test_windowing(self, hdf_files):
        from tsfast.data.core import CreateDict, DfHDFCreateWindows
        win = CreateDict([DfHDFCreateWindows(win_sz=20_000, stp_sz=1000, clm="u")])
        result = win(hdf_files)
        assert len(result) > 10


class TestResampling:
    def test_resampling_factor(self, hdf_files):
        from tsfast.data.core import DfResamplingFactor, apply_df_tfms
        df = apply_df_tfms(hdf_files)
        result = DfResamplingFactor(100, [50, 100, 300])(df)
        assert len(result) == 9

    def test_resample_interp_shape(self):
        from tsfast.data.core import resample_interp
        x = np.random.normal(size=(100000, 9))
        assert resample_interp(x, 0.3).shape[0] == 30000


class TestHDF2Sequence:
    def _valid_file(self, hdf_files):
        return next(f for f in hdf_files if "valid" in str(f))

    def test_basic_extraction(self, hdf_files):
        from tsfast.data.core import HDF2Sequence
        seq = HDF2Sequence(["u", "y"], cached=False)
        result = seq(self._valid_file(hdf_files))
        assert result.shape == (20000, 2)

    def test_single_column(self, hdf_files):
        from tsfast.data.core import HDF2Sequence
        seq = HDF2Sequence(["u"])
        assert seq(self._valid_file(hdf_files)).shape == (20000, 1)

    def test_caching_consistent(self, hdf_files):
        from tsfast.data.core import HDF2Sequence
        f = self._valid_file(hdf_files)
        uncached = HDF2Sequence(["u", "y"], cached=False)(f)
        cached = HDF2Sequence(["u", "y"], cached=True)(f)
        np.testing.assert_array_equal(uncached, cached)


class TestTensorTypes:
    def test_mse_preserves_type(self):
        from tsfast.data.core import TensorSequencesInput, TensorSequencesOutput
        x1 = TensorSequencesInput(torch.rand(10, 10))
        x2 = TensorSequencesOutput(torch.rand(10, 10))
        result = torch.nn.functional.mse_loss(x1, x2)
        assert isinstance(result, TensorSequencesInput)

    def test_scalar_format(self):
        from tsfast.data.core import TensorScalars
        s = TensorScalars(torch.tensor([1.234, 5.678]))
        formatted = format(s, ".2f")
        assert "1.23" in formatted


class TestDataLoader:
    def test_get_inp_out_size(self, dls_simulation):
        from tsfast.data.loader import get_inp_out_size
        inp, out = get_inp_out_size(dls_simulation)
        assert inp == 1
        assert out == 1

    def test_nbatches_factory_length(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5,
        )
        assert len(dls.train) == 5


class TestDataTransforms:
    def test_seq_noise_injection(self, dls_simulation):
        from tsfast.data.transforms import SeqNoiseInjection
        from tsfast.data.core import TensorSequencesInput
        batch = dls_simulation.one_batch()
        x = batch[0]
        tfm = SeqNoiseInjection(std=0.1, p=1.0)
        noisy = tfm(x, split_idx=0)  # split_idx=0 to trigger training mode
        assert noisy.shape == x.shape
        assert isinstance(noisy, TensorSequencesInput)
        assert not torch.allclose(noisy, x)

    def test_seq_noise_injection_varying(self, dls_simulation):
        from tsfast.data.transforms import SeqNoiseInjection_Varying
        from tsfast.data.core import TensorSequencesInput
        batch = dls_simulation.one_batch()
        x = batch[0]
        tfm = SeqNoiseInjection_Varying(std_std=0.1, p=1.0)
        noisy = tfm(x, split_idx=0)
        assert noisy.shape == x.shape
        assert isinstance(noisy, TensorSequencesInput)

    def test_seq_bias_injection(self, dls_simulation):
        from tsfast.data.transforms import SeqBiasInjection
        from tsfast.data.core import TensorSequencesInput
        batch = dls_simulation.one_batch()
        x = batch[0]
        tfm = SeqBiasInjection(std=0.1, mean=0.0, p=1.0)
        biased = tfm(x, split_idx=0)
        assert biased.shape == x.shape
        assert isinstance(biased, TensorSequencesInput)

    def test_seq_slice_truncates(self):
        from tsfast.data.transforms import SeqSlice
        x = torch.rand(100, 3)
        tfm = SeqSlice(l_slc=10, r_slc=-10)
        sliced = tfm(x)
        assert sliced.shape == (80, 3)

    @pytest.mark.slow
    def test_noise_injection_in_training_pipeline(self, wh_path):
        from tsfast.datasets.core import create_dls
        from tsfast.data.transforms import SeqNoiseInjection
        from tsfast.models.rnn import RNNLearner
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5,
        )
        dls.add_tfms([SeqNoiseInjection(std=0.05)], 'after_batch')
        lrn = RNNLearner(dls, rnn_type="gru", num_layers=1, hidden_size=10)
        lrn.fit(1, 1e-4)


class TestDataSplitting:
    def test_parent_splitter(self, pinn_path):
        from tsfast.data.split import ParentSplitter
        from tsfast.data.core import get_hdf_files
        files = get_hdf_files(pinn_path)
        splitter = ParentSplitter(train_name="train", valid_name="valid")
        train_idxs, valid_idxs = splitter(files)
        assert len(train_idxs) == 2  # 2 train files
        assert len(valid_idxs) == 1  # 1 valid file
        assert all(Path(files[i]).parent.name == "train" for i in train_idxs)
        assert all(Path(files[i]).parent.name == "valid" for i in valid_idxs)

    def test_percentage_splitter(self):
        from tsfast.data.split import PercentageSplitter
        items = list(range(10))
        splitter = PercentageSplitter(pct=0.7)
        train_idxs, valid_idxs = splitter(items)
        assert len(train_idxs) == 7
        assert len(valid_idxs) == 3


class TestDataUtilities:
    def test_running_mean_shape(self):
        from tsfast.data.core import running_mean
        x = np.random.normal(size=(100, 2))
        result = running_mean(x, 10)
        assert result.shape == (91, 2)

    def test_downsample_mean_shape(self):
        from tsfast.data.core import downsample_mean
        x = np.random.normal(size=(100, 3))
        result = downsample_mean(x, 5)
        assert result.shape == (20, 3)
