"""Tests for tsfast.tsdata — pure PyTorch data pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
WH_PATH = PROJECT_ROOT / "test_data" / "WienerHammerstein"
PINN_PATH = PROJECT_ROOT / "test_data" / "pinn"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "data_pipeline.json"


# ──────────────────────────────────────────────────────────────────────────────
#  signal.py
# ──────────────────────────────────────────────────────────────────────────────


class TestSignal:
    def test_running_mean_shape(self):
        from tsfast.tsdata.signal import running_mean

        x = np.random.randn(100)
        result = running_mean(x, 5)
        assert result.shape == (96,)

    def test_downsample_mean_shape(self):
        from tsfast.tsdata.signal import downsample_mean

        x = np.random.randn(100, 3)
        result = downsample_mean(x, 10)
        assert result.shape == (10, 3)

    def test_downsample_mean_truncation(self):
        from tsfast.tsdata.signal import downsample_mean

        x = np.random.randn(105, 2)
        result = downsample_mean(x, 10)
        assert result.shape == (10, 2)

    def test_resample_interp_upsample(self):
        from tsfast.tsdata.signal import resample_interp

        x = np.random.randn(100, 2)
        result = resample_interp(x, 2.0)
        assert result.shape[0] == 200
        assert result.shape[1] == 2

    def test_resample_interp_downsample(self):
        from tsfast.tsdata.signal import resample_interp

        x = np.random.randn(100, 2)
        result = resample_interp(x, 0.5)
        assert result.shape[0] == 50
        assert result.shape[1] == 2

    def test_resample_interp_identity(self):
        from tsfast.tsdata.signal import resample_interp

        x = np.random.randn(100, 1)
        result = resample_interp(x, 1.0)
        assert result.shape == (100, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  blocks.py
# ──────────────────────────────────────────────────────────────────────────────


class TestBlocks:
    def test_hdf5signals_read(self):
        from tsfast.tsdata.blocks import HDF5Signals

        block = HDF5Signals(["u", "y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 2)

    def test_hdf5signals_file_len(self):
        from tsfast.tsdata.blocks import HDF5Signals

        block = HDF5Signals(["u"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        length = block.file_len(path)
        assert length == 80000

    def test_hdf5signals_len_cache(self):
        from tsfast.tsdata.blocks import HDF5Signals

        block = HDF5Signals(["u"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        block.file_len(path)
        assert path in block._len_cache
        # Second call should use cache
        assert block.file_len(path) == 80000

    def test_hdf5attrs_read(self):
        from tsfast.tsdata.blocks import HDF5Attrs

        block = HDF5Attrs(["mass", "spring_constant"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr = block.read(path)
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr, [1.0, 1.0], rtol=1e-5)

    def test_resampled_identity(self):
        from tsfast.tsdata.blocks import HDF5Signals, Resampled

        block = HDF5Signals(["u"])
        resampled = Resampled(block)
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = resampled.read(path, 0, 100, factor=1.0)
        arr_direct = block.read(path, 0, 100)
        np.testing.assert_array_equal(arr, arr_direct)

    def test_resampled_upsample(self):
        from tsfast.tsdata.blocks import HDF5Signals, Resampled

        block = HDF5Signals(["u"])
        resampled = Resampled(block)
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = resampled.read(path, 0, 200, factor=2.0)
        assert arr.shape == (200, 1)

    def test_hdf5signals_mmap_contiguous(self):
        """Contiguous datasets use mmap path and produce correct results."""
        import h5py as h5

        from tsfast.tsdata.blocks import HDF5Signals

        block = HDF5Signals(["u", "x", "v"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 3)
        # Verify mmap info was populated with actual offsets
        assert path in block._mmap_info
        for name in ["u", "x", "v"]:
            assert block._mmap_info[path][name] is not None
        # Verify values match direct h5py read
        with h5.File(path, "r") as f:
            expected = np.stack([f["u"][:100], f["x"][:100], f["v"][:100]], axis=-1)
        np.testing.assert_array_equal(arr, expected)

    def test_hdf5signals_fallback_chunked(self):
        """Chunked datasets fall back to h5py and still return correct results."""
        import h5py as h5

        from tsfast.tsdata.blocks import HDF5Signals

        block = HDF5Signals(["u", "y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 2)
        # Chunked datasets should have None in mmap_info
        assert path in block._mmap_info
        for name in ["u", "y"]:
            assert block._mmap_info[path][name] is None
        # Verify values match direct h5py read
        with h5.File(path, "r") as f:
            expected = np.stack([f["u"][:100], f["y"][:100]], axis=-1)
        np.testing.assert_array_equal(arr, expected)

    def test_hdf5signals_pickle_roundtrip(self):
        """Block survives pickle (simulating multiprocessing worker spawn)."""
        import pickle

        from tsfast.tsdata.blocks import HDF5Signals

        block = HDF5Signals(["u", "x"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr1 = block.read(path, 0, 50)
        block2 = pickle.loads(pickle.dumps(block))
        arr2 = block2.read(path, 0, 50)
        np.testing.assert_array_equal(arr1, arr2)


# ──────────────────────────────────────────────────────────────────────────────
#  Alternative blocks: CSVSignals, FilenameScalar
# ──────────────────────────────────────────────────────────────────────────────


class TestAltBlocks:
    def test_csv_signals_read(self, tmp_path):
        from tsfast.tsdata.blocks import CSVSignals

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("voltage,current\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        block = CSVSignals(["voltage", "current"])
        arr = block.read(str(csv_file), 0, 2)
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_csv_signals_file_len(self, tmp_path):
        from tsfast.tsdata.blocks import CSVSignals

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")
        block = CSVSignals(["a", "b"])
        assert block.file_len(str(csv_file)) == 4
        # Second call uses cache
        assert str(csv_file) in block._len_cache
        assert block.file_len(str(csv_file)) == 4

    def test_csv_signals_n_features(self):
        from tsfast.tsdata.blocks import CSVSignals

        block = CSVSignals(["x", "y", "z"])
        assert block.n_features == 3

    def test_csv_signals_with_dataset(self, tmp_path):
        from tsfast.tsdata.blocks import CSVSignals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        csv_file = tmp_path / "signals.csv"
        lines = ["u,y"] + [f"{i * 0.1},{i * 0.2}" for i in range(100)]
        csv_file.write_text("\n".join(lines) + "\n")

        block_u = CSVSignals(["u"])
        block_y = CSVSignals(["y"])
        entries = [FileEntry(path=str(csv_file))]
        ds = WindowedDataset(entries, block_u, block_y, win_sz=10, stp_sz=10)
        assert len(ds) == 10  # (100 - 10) // 10 + 1
        xb, yb = ds[0]
        assert xb.shape == (10, 1)
        assert yb.shape == (10, 1)
        assert isinstance(xb, torch.Tensor)

    def test_csv_signals_custom_delimiter(self, tmp_path):
        from tsfast.tsdata.blocks import CSVSignals

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a;b\n1.5;2.5\n3.5;4.5\n")
        block = CSVSignals(["a", "b"], delimiter=";")
        arr = block.read(str(csv_file), 0, 2)
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr, [[1.5, 2.5], [3.5, 4.5]])

    def test_filename_scalar_single_group(self, tmp_path):
        from tsfast.tsdata.blocks import FilenameScalar

        f = tmp_path / "test_25C.csv"
        f.touch()
        block = FilenameScalar(r"(\d+)C")
        arr = block.read(str(f))
        assert arr.shape == (1,)
        assert arr[0] == 25.0

    def test_filename_scalar_multi_group(self, tmp_path):
        from tsfast.tsdata.blocks import FilenameScalar

        f = tmp_path / "test_25C_100Hz.csv"
        f.touch()
        block = FilenameScalar(r"(\d+)C_(\d+)Hz")
        arr = block.read(str(f))
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr, [25.0, 100.0])

    def test_filename_scalar_no_match(self, tmp_path):
        from tsfast.tsdata.blocks import FilenameScalar

        f = tmp_path / "nodata.csv"
        f.touch()
        block = FilenameScalar(r"(\d+)C")
        with pytest.raises(ValueError, match="did not match"):
            block.read(str(f))

    def test_filename_scalar_n_features(self):
        from tsfast.tsdata.blocks import FilenameScalar

        assert FilenameScalar(r"(\d+)C").n_features == 1
        assert FilenameScalar(r"(\d+)C_(\d+)Hz").n_features == 2
        assert FilenameScalar(r"(\d+)_(\d+)_(\d+)").n_features == 3

    def test_mixed_csv_filename_dataset(self, tmp_path):
        from tsfast.tsdata.blocks import CSVSignals, FilenameScalar
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        csv_file = tmp_path / "trial_25C.csv"
        lines = ["u,y"] + [f"{i * 0.1},{i * 0.2}" for i in range(50)]
        csv_file.write_text("\n".join(lines) + "\n")

        block_u = CSVSignals(["u"])
        block_y = CSVSignals(["y"])
        block_temp = FilenameScalar(r"(\d+)C")
        entries = [FileEntry(path=str(csv_file))]
        ds = WindowedDataset(entries, block_u, (block_y, block_temp), win_sz=10, stp_sz=10)
        assert len(ds) == 5  # (50 - 10) // 10 + 1
        xb, (yb, temp) = ds[0]
        assert xb.shape == (10, 1)
        assert yb.shape == (10, 1)
        assert temp.shape == (1,)
        assert temp.item() == 25.0


# ──────────────────────────────────────────────────────────────────────────────
#  dataset.py
# ──────────────────────────────────────────────────────────────────────────────


class TestDataset:
    def test_windowed_dataset_window_count(self):
        from tsfast.tsdata.blocks import HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        block_u = HDF5Signals(["u"])
        block_y = HDF5Signals(["y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, block_u, block_y, win_sz=100, stp_sz=100)
        # 80000 samples, win_sz=100, stp_sz=100 → (80000 - 100) // 100 + 1 = 800
        assert len(ds) == 800

    def test_windowed_dataset_getitem(self):
        from tsfast.tsdata.blocks import HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        block_u = HDF5Signals(["u"])
        block_y = HDF5Signals(["y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, block_u, block_y, win_sz=100, stp_sz=100)
        xb, yb = ds[0]
        assert xb.shape == (100, 1)
        assert yb.shape == (100, 1)
        assert isinstance(xb, torch.Tensor)
        assert isinstance(yb, torch.Tensor)

    def test_windowed_dataset_fullfile(self):
        from tsfast.tsdata.blocks import HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        block_u = HDF5Signals(["u"])
        block_y = HDF5Signals(["y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, block_u, block_y, win_sz=None)
        assert len(ds) == 1
        xb, yb = ds[0]
        assert xb.shape == (80000, 1)
        assert yb.shape == (80000, 1)

    def test_windowed_dataset_multi_file(self):
        from tsfast.tsdata.blocks import HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        block_u = HDF5Signals(["u"])
        block_y = HDF5Signals(["y"])
        entries = [
            FileEntry(path=str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")),
            FileEntry(path=str(WH_PATH / "valid" / "WienerHammerstein_valid.hdf5")),
        ]
        ds = WindowedDataset(entries, block_u, block_y, win_sz=100, stp_sz=100)
        # Both files accessible
        assert len(ds) > 800

    def test_windowed_dataset_multi_block_tuple(self):
        from tsfast.tsdata.blocks import HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        block_u = HDF5Signals(["u"])
        block_x = HDF5Signals(["x"])
        block_v = HDF5Signals(["v"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, block_u, (block_x, block_v), win_sz=100, stp_sz=100)
        xb, (yb_x, yb_v) = ds[0]
        assert xb.shape == (100, 1)
        assert yb_x.shape == (100, 1)
        assert yb_v.shape == (100, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  norm.py
# ──────────────────────────────────────────────────────────────────────────────


class TestNorm:
    def test_normpair_add(self):
        from tsfast.tsdata.norm import NormPair

        a = NormPair(np.array([1.0]), np.array([2.0]), np.array([0.0]), np.array([3.0]))
        b = NormPair(np.array([4.0]), np.array([5.0]), np.array([1.0]), np.array([6.0]))
        c = a + b
        np.testing.assert_array_equal(c.mean, [1.0, 4.0])
        np.testing.assert_array_equal(c.std, [2.0, 5.0])

    def test_normpair_iter(self):
        from tsfast.tsdata.norm import NormPair

        a = NormPair(np.array([1.0]), np.array([2.0]), np.array([0.0]), np.array([3.0]))
        mean, std, mn, mx = a
        np.testing.assert_array_equal(mean, [1.0])
        np.testing.assert_array_equal(std, [2.0])

    def test_normpair_getitem(self):
        from tsfast.tsdata.norm import NormPair

        a = NormPair(np.array([1.0]), np.array([2.0]), np.array([0.0]), np.array([3.0]))
        np.testing.assert_array_equal(a[0], [1.0])
        np.testing.assert_array_equal(a[1], [2.0])

    def test_compute_stats_from_files_wh(self):
        from tsfast.tsdata.norm import compute_stats_from_files
        from tsfast.tsdata.split import get_hdf_files

        train_files = get_hdf_files(WH_PATH / "train")
        stats = compute_stats_from_files(train_files, ["u"])
        assert stats is not None
        assert stats.mean.shape == (1,)
        assert stats.std.shape == (1,)

    def test_compute_stats_from_files_empty(self):
        from tsfast.tsdata.norm import compute_stats_from_files

        assert compute_stats_from_files([], []) is None

    def test_compute_stats_from_files_pinn(self):
        from tsfast.tsdata.norm import compute_stats_from_files
        from tsfast.tsdata.split import get_hdf_files

        train_files = get_hdf_files(PINN_PATH / "train")
        stats = compute_stats_from_files(train_files, ["u"])
        assert stats is not None
        assert len(stats.mean) == 1

    def test_compute_stats_from_dl(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert dls.norm_stats is not None
        assert len(dls.norm_stats.u.mean) == 1
        assert len(dls.norm_stats.y.mean) == 1


# ──────────────────────────────────────────────────────────────────────────────
#  split.py
# ──────────────────────────────────────────────────────────────────────────────


class TestSplit:
    def test_get_hdf_files(self):
        from tsfast.tsdata.split import get_hdf_files

        files = get_hdf_files(WH_PATH)
        assert len(files) == 3
        assert all(f.suffix in (".hdf5", ".h5") for f in files)

    def test_discover_split_files(self):
        from tsfast.tsdata.split import discover_split_files

        result = discover_split_files(WH_PATH)
        assert len(result["train"]) == 1
        assert len(result["valid"]) == 1
        assert len(result["test"]) == 1

    def test_split_by_parent(self):
        from tsfast.tsdata.split import get_hdf_files, split_by_parent

        files = get_hdf_files(WH_PATH)
        train_idxs, valid_idxs = split_by_parent(files)
        assert len(train_idxs) == 1
        assert len(valid_idxs) == 1

    def test_split_by_percentage(self):
        from tsfast.tsdata.split import split_by_percentage

        items = list(range(10))
        train_idxs, valid_idxs = split_by_percentage(items, 0.8)
        assert len(train_idxs) == 8
        assert len(valid_idxs) == 2

    def test_is_dataset_directory(self):
        from tsfast.tsdata.split import is_dataset_directory

        assert is_dataset_directory(WH_PATH)
        assert not is_dataset_directory(WH_PATH / "train")


# ──────────────────────────────────────────────────────────────────────────────
#  pipeline.py — end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestPipeline:
    def test_create_dls_wh_simulation(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        batch = dls.one_batch()
        assert list(batch[0].shape) == [64, 100, 1]
        assert list(batch[1].shape) == [64, 100, 1]
        assert len(dls.train) == 2

    def test_create_dls_pinn_simulation(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(
            u=["u"], y=["x", "v"], dataset=PINN_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2
        )
        batch = dls.one_batch()
        assert list(batch[0].shape) == [64, 100, 1]
        assert list(batch[1].shape) == [64, 100, 2]
        assert len(dls.train) == 2

    def test_create_dls_dict_input(self):
        from tsfast.tsdata import create_dls

        dataset_dict = {
            "train": [WH_PATH / "train" / "WienerHammerstein_train.hdf5"],
            "valid": [WH_PATH / "valid" / "WienerHammerstein_valid.hdf5"],
            "test": [WH_PATH / "test" / "WienerHammerstein_test.hdf5"],
        }
        dls = create_dls(u=["u"], y=["y"], dataset=dataset_dict, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert dls.test is not None

    def test_create_dls_norm_stats_structure(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert dls.norm_stats is not None
        assert len(dls.norm_stats.u.mean) == 1
        assert len(dls.norm_stats.y.mean) == 1

    def test_create_dls_with_dls_id(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(
            u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2,
            dls_id="test_tsdata_wh"
        )
        assert dls.norm_stats is not None
        # Clean up cache
        cache_path = Path(".tsfast_cache/test_tsdata_wh.pkl")
        if cache_path.exists():
            cache_path.unlink()

    def test_create_dls_test_dl_fullfile(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert dls.test is not None
        batch = next(iter(dls.test))
        # Full-file test: bs=1, full sequence length
        assert batch[0].shape[0] == 1
        assert batch[0].shape[2] == 1

    def test_create_dls_n_batches_valid(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(
            u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100,
            num_workers=0, n_batches_train=2, n_batches_valid=3
        )
        # With n_batches_valid=3, we should get exactly 3 batches (via sampler)
        count = sum(1 for _ in dls.valid)
        assert count == 3

    def test_create_dls_loaders_property(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert len(dls.loaders) >= 2  # train + valid, possibly + test


# ──────────────────────────────────────────────────────────────────────────────
#  Golden baseline verification (simulation mode only)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="Golden baselines not found")
class TestGoldenBaselines:
    @pytest.fixture(scope="class")
    def golden(self):
        with open(GOLDEN_PATH) as f:
            return json.load(f)

    def test_wh_simulation_shapes(self, golden):
        from tsfast.tsdata import create_dls

        g = golden["wh_simulation"]
        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        batch = dls.one_batch()
        assert list(batch[0].shape) == g["batch_xb_shape"]
        assert list(batch[1].shape) == g["batch_yb_shape"]

    def test_wh_simulation_dl_lengths(self, golden):
        from tsfast.tsdata import create_dls

        g = golden["wh_simulation"]
        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert len(dls.train) == g["train_len"]
        assert len(dls.valid) == g["valid_len"]

    def test_wh_simulation_norm_stats(self, golden):
        from tsfast.tsdata import create_dls

        g = golden["wh_simulation"]
        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        # Feature counts must match
        assert len(dls.norm_stats.u.mean) == len(g["norm_u"]["mean"])
        assert len(dls.norm_stats.y.mean) == len(g["norm_y"]["mean"])
        # Loose tolerance — stats are estimated from random batches
        np.testing.assert_allclose(dls.norm_stats.u.mean, g["norm_u"]["mean"], atol=0.5)
        np.testing.assert_allclose(dls.norm_stats.y.mean, g["norm_y"]["mean"], atol=0.5)

    def test_pinn_simulation_shapes(self, golden):
        from tsfast.tsdata import create_dls

        g = golden["pinn_simulation"]
        dls = create_dls(
            u=["u"], y=["x", "v"], dataset=PINN_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2
        )
        batch = dls.one_batch()
        assert list(batch[0].shape) == g["batch_xb_shape"]
        assert list(batch[1].shape) == g["batch_yb_shape"]

    def test_pinn_simulation_dl_lengths(self, golden):
        from tsfast.tsdata import create_dls

        g = golden["pinn_simulation"]
        dls = create_dls(
            u=["u"], y=["x", "v"], dataset=PINN_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2
        )
        assert len(dls.train) == g["train_len"]
        assert len(dls.valid) == g["valid_len"]

    def test_pinn_simulation_norm_stats(self, golden):
        from tsfast.tsdata import create_dls

        g = golden["pinn_simulation"]
        dls = create_dls(
            u=["u"], y=["x", "v"], dataset=PINN_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2
        )
        assert len(dls.norm_stats.u.mean) == len(g["norm_u"]["mean"])
        assert len(dls.norm_stats.y.mean) == len(g["norm_y"]["mean"])
        np.testing.assert_allclose(dls.norm_stats.u.mean, g["norm_u"]["mean"], atol=0.5)
        np.testing.assert_allclose(dls.norm_stats.y.mean, g["norm_y"]["mean"], atol=0.5)


