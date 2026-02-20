"""Tests for tsfast.data module."""
import pytest
import torch
import numpy as np


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
    def test_basic_extraction(self, hdf_files):
        from tsfast.data.core import HDF2Sequence
        seq = HDF2Sequence(["u", "y"], cached=False)
        result = seq(hdf_files[0])
        assert result.shape == (20000, 2)

    def test_single_column(self, hdf_files):
        from tsfast.data.core import HDF2Sequence
        seq = HDF2Sequence(["u"])
        assert seq(hdf_files[0]).shape == (20000, 1)

    def test_caching_consistent(self, hdf_files):
        from tsfast.data.core import HDF2Sequence
        uncached = HDF2Sequence(["u", "y"], cached=False)(hdf_files[0])
        cached = HDF2Sequence(["u", "y"], cached=True)(hdf_files[0])
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
