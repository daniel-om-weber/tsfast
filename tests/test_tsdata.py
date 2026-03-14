"""Tests for tsfast.tsdata — pure PyTorch data pipeline."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
WH_PATH = PROJECT_ROOT / "test_data" / "WienerHammerstein"
PINN_PATH = PROJECT_ROOT / "test_data" / "pinn"


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
#  readers.py
# ──────────────────────────────────────────────────────────────────────────────


class TestReaders:
    def test_hdf5signals_read(self):
        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u", "y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 2)

    def test_hdf5signals_file_len(self):
        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        length = block.file_len(path)
        assert length == 80000

    def test_hdf5signals_len_cache(self):
        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        block.file_len(path)
        assert path in block._len_cache
        # Second call should use cache
        assert block.file_len(path) == 80000

    def test_hdf5attrs_read(self):
        from tsfast.tsdata.readers import HDF5Attrs

        block = HDF5Attrs(["mass", "spring_constant"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr = block.read(path)
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr, [1.0, 1.0], rtol=1e-5)

    def test_hdf5attrs_array(self, tmp_path):
        from tsfast.tsdata.readers import HDF5Attrs

        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.attrs["dt"] = np.float32(0.01)
            f.attrs["ja_rr"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            f.attrs["ja_rsaddle"] = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

        block = HDF5Attrs(["dt", "ja_rr", "ja_rsaddle"])
        arr = block.read(path)
        assert arr.shape == (10,)
        assert block.n_features == 10
        np.testing.assert_allclose(arr, [0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9], rtol=1e-5)

    def test_hdf5attrs_probe(self, tmp_path):
        from tsfast.tsdata.readers import HDF5Attrs

        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.attrs["dt"] = np.float32(0.01)
            f.attrs["ja_rr"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        block = HDF5Attrs(["dt", "ja_rr"])
        with pytest.warns(UserWarning, match="n_features accessed before probing"):
            assert block.n_features == 2  # fallback before probe
        block.probe(path)
        assert block.n_features == 4  # 1 scalar + 3 array elements

    def test_hdf5signals_2d(self, tmp_path):
        from tsfast.tsdata.readers import HDF5Signals

        T = 200
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("scalar_sig", data=np.arange(T, dtype=np.float32))
            f.create_dataset(
                "vector_sig",
                data=np.arange(T * 3, dtype=np.float32).reshape(T, 3),
            )

        block = HDF5Signals(["scalar_sig", "vector_sig"])
        arr = block.read(path, 0, T)
        assert arr.shape == (T, 4)
        assert block.n_features == 4
        np.testing.assert_array_equal(arr[:, 0], np.arange(T, dtype=np.float32))
        expected_2d = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
        np.testing.assert_array_equal(arr[:, 1:4], expected_2d)

    def test_hdf5signals_2d_no_mmap(self, tmp_path):
        """use_mmap=False with 2D datasets uses seek+read path."""
        from tsfast.tsdata.readers import HDF5Signals

        T = 200
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("scalar_sig", data=np.arange(T, dtype=np.float32))
            f.create_dataset(
                "vector_sig",
                data=np.arange(T * 3, dtype=np.float32).reshape(T, 3),
            )

        block = HDF5Signals(["scalar_sig", "vector_sig"], use_mmap=False)
        arr = block.read(path, 0, T)
        assert arr.shape == (T, 4)
        assert block.n_features == 4
        np.testing.assert_array_equal(arr[:, 0], np.arange(T, dtype=np.float32))
        expected_2d = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
        np.testing.assert_array_equal(arr[:, 1:4], expected_2d)
        # Probe cache should store int offsets (not memmap objects)
        assert path in block._probe_cache
        for name in ["scalar_sig", "vector_sig"]:
            assert isinstance(block._probe_cache[path][name], int)

    def test_hdf5signals_2d_chunked(self, tmp_path):
        """Chunked 2D datasets fall back to h5py and still return correct results."""
        from tsfast.tsdata.readers import HDF5Signals

        T = 200
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "scalar_sig",
                data=np.arange(T, dtype=np.float32),
                chunks=(50,),
            )
            f.create_dataset(
                "vector_sig",
                data=np.arange(T * 3, dtype=np.float32).reshape(T, 3),
                chunks=(50, 3),
            )

        block = HDF5Signals(["scalar_sig", "vector_sig"])
        arr = block.read(path, 0, T)
        assert arr.shape == (T, 4)
        assert block.n_features == 4
        np.testing.assert_array_equal(arr[:, 0], np.arange(T, dtype=np.float32))
        expected_2d = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
        np.testing.assert_array_equal(arr[:, 1:4], expected_2d)
        # Chunked datasets should have None in probe cache
        assert path in block._probe_cache
        for name in ["scalar_sig", "vector_sig"]:
            assert block._probe_cache[path][name] is None

    def test_resampled_identity(self):
        from tsfast.tsdata.readers import HDF5Signals, Resampled

        block = HDF5Signals(["u"])
        resampled = Resampled(block)
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = resampled.read(path, 0, 100, factor=1.0)
        arr_direct = block.read(path, 0, 100)
        np.testing.assert_array_equal(arr, arr_direct)

    def test_resampled_upsample(self):
        from tsfast.tsdata.readers import HDF5Signals, Resampled

        block = HDF5Signals(["u"])
        resampled = Resampled(block)
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = resampled.read(path, 0, 200, factor=2.0)
        assert arr.shape == (200, 1)

    def test_hdf5signals_mmap_contiguous(self):
        """Contiguous datasets use mmap path and produce correct results."""
        import h5py as h5

        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u", "x", "v"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 3)
        # Verify probe cache was populated with memmap views
        assert path in block._probe_cache
        for name in ["u", "x", "v"]:
            assert block._probe_cache[path][name] is not None
        # Verify values match direct h5py read
        with h5.File(path, "r") as f:
            expected = np.stack([f["u"][:100], f["x"][:100], f["v"][:100]], axis=-1)
        np.testing.assert_array_equal(arr, expected)

    def test_hdf5signals_fallback_chunked(self):
        """Chunked datasets fall back to h5py and still return correct results."""
        import h5py as h5

        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u", "y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 2)
        # Chunked datasets should have None in probe cache
        assert path in block._probe_cache
        for name in ["u", "y"]:
            assert block._probe_cache[path][name] is None
        # Verify values match direct h5py read
        with h5.File(path, "r") as f:
            expected = np.stack([f["u"][:100], f["y"][:100]], axis=-1)
        np.testing.assert_array_equal(arr, expected)

    def test_hdf5signals_no_mmap_contiguous(self):
        """use_mmap=False stores byte offsets and reads via seek+read."""
        import h5py as h5

        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u", "x", "v"], use_mmap=False)
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr = block.read(path, 0, 100)
        assert arr.shape == (100, 3)
        # Verify probe cache stores int offsets, not memmap objects
        assert path in block._probe_cache
        for name in ["u", "x", "v"]:
            assert isinstance(block._probe_cache[path][name], int)
        # Verify values match direct h5py read
        with h5.File(path, "r") as f:
            expected = np.stack([f["u"][:100], f["x"][:100], f["v"][:100]], axis=-1)
        np.testing.assert_array_equal(arr, expected)

    def test_hdf5signals_no_mmap_matches_mmap(self):
        """use_mmap=False produces identical results to use_mmap=True."""
        from tsfast.tsdata.readers import HDF5Signals

        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        block_mmap = HDF5Signals(["u", "x", "v"], use_mmap=True)
        block_direct = HDF5Signals(["u", "x", "v"], use_mmap=False)
        arr_mmap = block_mmap.read(path, 10, 200)
        arr_direct = block_direct.read(path, 10, 200)
        np.testing.assert_array_equal(arr_mmap, arr_direct)

    def test_hdf5signals_pickle_roundtrip(self):
        """Block survives pickle (simulating multiprocessing worker spawn)."""
        import pickle

        from tsfast.tsdata.readers import HDF5Signals

        block = HDF5Signals(["u", "x"])
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr1 = block.read(path, 0, 50)
        block2 = pickle.loads(pickle.dumps(block))
        arr2 = block2.read(path, 0, 50)
        np.testing.assert_array_equal(arr1, arr2)


# ──────────────────────────────────────────────────────────────────────────────
#  Cached wrapper
# ──────────────────────────────────────────────────────────────────────────────


class TestCached:
    def test_cached_hdf5signals_matches_uncached(self):
        from tsfast.tsdata.readers import Cached, HDF5Signals

        block = HDF5Signals(["u", "y"])
        cached = Cached(HDF5Signals(["u", "y"]))
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = block.read(path, 10, 110)
        arr_cached = cached.read(path, 10, 110)
        np.testing.assert_array_equal(arr, arr_cached)

    def test_cached_populates_data_cache(self):
        from tsfast.tsdata.readers import Cached, HDF5Signals

        cached = Cached(HDF5Signals(["u"]))
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        cached.read(path, 0, 100)
        assert path in cached._data_cache
        assert cached._data_cache[path].shape == (80000, 1)

    def test_cached_delegates_n_features(self):
        from tsfast.tsdata.readers import Cached, HDF5Signals

        cached = Cached(HDF5Signals(["u", "y"]))
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        cached.read(path, 0, 10)  # probe before accessing n_features
        assert cached.n_features == 2

    def test_cached_delegates_file_len(self):
        from tsfast.tsdata.readers import Cached, HDF5Signals

        cached = Cached(HDF5Signals(["u"]))
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        assert cached.file_len(path) == 80000

    def test_cached_hdf5attrs(self):
        from tsfast.tsdata.readers import Cached, HDF5Attrs

        cached = Cached(HDF5Attrs(["mass", "spring_constant"]))
        path = str(PINN_PATH / "train" / "trajectory_sine_1hz.h5")
        arr = cached.read(path)
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr, [1.0, 1.0], rtol=1e-5)
        assert path in cached._data_cache

    def test_cached_scalar_no_file_len(self):
        from tsfast.tsdata.readers import Cached, HDF5Attrs

        cached = Cached(HDF5Attrs(["mass"]))
        assert not hasattr(cached, "file_len")

    def test_cached_with_resampled(self):
        from tsfast.tsdata.readers import Cached, HDF5Signals, Resampled

        cached = Cached(HDF5Signals(["u"]))
        resampled = Resampled(cached)
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        arr = resampled.read(path, 0, 100, factor=1.0)
        assert arr.shape == (100, 1)
        # Inner block should be cached
        assert path in cached._data_cache

    def test_cached_with_windowed_dataset(self):
        from tsfast.tsdata.readers import Cached, HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        cached_u = Cached(HDF5Signals(["u"]))
        cached_y = Cached(HDF5Signals(["y"]))
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, cached_u, cached_y, win_sz=100, stp_sz=100)
        xb, yb = ds[0]
        assert xb.shape == (100, 1)
        assert yb.shape == (100, 1)
        # Both blocks should be cached after first access
        assert path in cached_u._data_cache
        assert path in cached_y._data_cache

    def test_create_dls_with_cache(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(
            u=["u"],
            y=["y"],
            dataset=WH_PATH,
            win_sz=100,
            stp_sz=100,
            num_workers=0,
            n_batches_train=2,
            cache=True,
        )
        batch = dls.one_batch()
        assert list(batch[0].shape) == [64, 100, 1]
        assert list(batch[1].shape) == [64, 100, 1]


# ──────────────────────────────────────────────────────────────────────────────
#  Alternative readers: CSVSignals, FilenameScalar
# ──────────────────────────────────────────────────────────────────────────────


class TestAltReaders:
    def test_csv_signals_read(self, tmp_path):
        from tsfast.tsdata.readers import CSVSignals

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("voltage,current\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        block = CSVSignals(["voltage", "current"])
        arr = block.read(str(csv_file), 0, 2)
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_csv_signals_file_len(self, tmp_path):
        from tsfast.tsdata.readers import CSVSignals

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")
        block = CSVSignals(["a", "b"])
        assert block.file_len(str(csv_file)) == 4
        # Second call uses cache
        assert str(csv_file) in block._len_cache
        assert block.file_len(str(csv_file)) == 4

    def test_csv_signals_n_features(self):
        from tsfast.tsdata.readers import CSVSignals

        block = CSVSignals(["x", "y", "z"])
        assert block.n_features == 3

    def test_csv_signals_with_dataset(self, tmp_path):
        from tsfast.tsdata.readers import CSVSignals
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
        from tsfast.tsdata.readers import CSVSignals

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a;b\n1.5;2.5\n3.5;4.5\n")
        block = CSVSignals(["a", "b"], delimiter=";")
        arr = block.read(str(csv_file), 0, 2)
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr, [[1.5, 2.5], [3.5, 4.5]])

    def test_filename_scalar_single_group(self, tmp_path):
        from tsfast.tsdata.readers import FilenameScalar

        f = tmp_path / "test_25C.csv"
        f.touch()
        block = FilenameScalar(r"(\d+)C")
        arr = block.read(str(f))
        assert arr.shape == (1,)
        assert arr[0] == 25.0

    def test_filename_scalar_multi_group(self, tmp_path):
        from tsfast.tsdata.readers import FilenameScalar

        f = tmp_path / "test_25C_100Hz.csv"
        f.touch()
        block = FilenameScalar(r"(\d+)C_(\d+)Hz")
        arr = block.read(str(f))
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr, [25.0, 100.0])

    def test_filename_scalar_no_match(self, tmp_path):
        from tsfast.tsdata.readers import FilenameScalar

        f = tmp_path / "nodata.csv"
        f.touch()
        block = FilenameScalar(r"(\d+)C")
        with pytest.raises(ValueError, match="did not match"):
            block.read(str(f))

    def test_filename_scalar_n_features(self):
        from tsfast.tsdata.readers import FilenameScalar

        assert FilenameScalar(r"(\d+)C").n_features == 1
        assert FilenameScalar(r"(\d+)C_(\d+)Hz").n_features == 2
        assert FilenameScalar(r"(\d+)_(\d+)_(\d+)").n_features == 3

    def test_mixed_csv_filename_dataset(self, tmp_path):
        from tsfast.tsdata.readers import CSVSignals, FilenameScalar
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
        from tsfast.tsdata.readers import HDF5Signals
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        block_u = HDF5Signals(["u"])
        block_y = HDF5Signals(["y"])
        path = str(WH_PATH / "train" / "WienerHammerstein_train.hdf5")
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, block_u, block_y, win_sz=100, stp_sz=100)
        # 80000 samples, win_sz=100, stp_sz=100 → (80000 - 100) // 100 + 1 = 800
        assert len(ds) == 800

    def test_windowed_dataset_getitem(self):
        from tsfast.tsdata.readers import HDF5Signals
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
        from tsfast.tsdata.readers import HDF5Signals
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
        from tsfast.tsdata.readers import HDF5Signals
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

    def test_windowed_dataset_probes_attrs(self, tmp_path):
        from tsfast.tsdata.readers import HDF5Signals, HDF5Attrs
        from tsfast.tsdata.dataset import FileEntry, WindowedDataset

        T = 200
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("u", data=np.zeros(T, dtype=np.float32))
            f.create_dataset("y", data=np.zeros(T, dtype=np.float32))
            f.attrs["dt"] = np.float32(0.01)
            f.attrs["ja_rr"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        block_u = HDF5Signals(["u"])
        block_y = HDF5Signals(["y"])
        block_attrs = HDF5Attrs(["dt", "ja_rr"])
        entries = [FileEntry(path=path)]
        ds = WindowedDataset(entries, block_u, (block_y, block_attrs), win_sz=10, stp_sz=10)
        # n_features should be correct immediately after construction
        assert block_attrs.n_features == 4

    def test_windowed_dataset_multi_block_tuple(self):
        from tsfast.tsdata.readers import HDF5Signals
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

    def test_compute_stats_from_files_with_nans(self, tmp_path):
        from tsfast.tsdata.norm import compute_stats_from_files

        # Create two HDF5 files with some NaN values
        data1 = np.array([1.0, 2.0, np.nan, 4.0])
        data2 = np.array([np.nan, 3.0, 5.0, 6.0])
        for name, data in [("a.h5", data1), ("b.h5", data2)]:
            with h5py.File(tmp_path / name, "w") as f:
                f.create_dataset("sig", data=data)

        stats = compute_stats_from_files([tmp_path / "a.h5", tmp_path / "b.h5"], ["sig"])
        valid = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0])
        np.testing.assert_allclose(stats.mean, [valid.mean()], rtol=1e-5)
        np.testing.assert_allclose(stats.std, [valid.std()], rtol=1e-5)
        np.testing.assert_allclose(stats.min, [1.0])
        np.testing.assert_allclose(stats.max, [6.0])

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
        dls = create_dls(
            u=["u"], y=["y"], dataset=dataset_dict, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2
        )
        assert dls.test is not None

    def test_create_dls_norm_stats_structure(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert dls.norm_stats is not None
        assert len(dls.norm_stats.u.mean) == 1
        assert len(dls.norm_stats.y.mean) == 1

    def test_create_dls_with_dls_id(self, monkeypatch, tmp_path):
        from tsfast.tsdata import norm as norm_mod
        from tsfast.tsdata import create_dls, NormStats, NormPair

        monkeypatch.setattr(norm_mod, "_cache_path", lambda dls_id: tmp_path / f"{dls_id}.pkl")
        dls = create_dls(
            u=["u"],
            y=["y"],
            dataset=WH_PATH,
            win_sz=100,
            stp_sz=100,
            num_workers=0,
            n_batches_train=2,
            dls_id="test_cache",
        )
        stats1 = dls.norm_stats
        assert (tmp_path / "test_cache.pkl").exists()
        assert isinstance(stats1, NormStats)
        assert isinstance(stats1.u, NormPair)
        # Second call loads from cache
        dls2 = create_dls(
            u=["u"],
            y=["y"],
            dataset=WH_PATH,
            win_sz=100,
            stp_sz=100,
            num_workers=0,
            n_batches_train=2,
            dls_id="test_cache",
        )
        np.testing.assert_array_equal(dls2.norm_stats.u.mean, stats1.u.mean)
        np.testing.assert_array_equal(dls2.norm_stats.u.std, stats1.u.std)
        np.testing.assert_array_equal(dls2.norm_stats.y.mean, stats1.y.mean)

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
            u=["u"],
            y=["y"],
            dataset=WH_PATH,
            win_sz=100,
            stp_sz=100,
            num_workers=0,
            n_batches_train=2,
            n_batches_valid=3,
        )
        # With n_batches_valid=3, we should get exactly 3 batches (via sampler)
        count = sum(1 for _ in dls.valid)
        assert count == 3

    def test_create_dls_loaders_property(self):
        from tsfast.tsdata import create_dls

        dls = create_dls(u=["u"], y=["y"], dataset=WH_PATH, win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
        assert len(dls.loaders) >= 2  # train + valid, possibly + test

    def test_empty_path_raises_file_not_found(self, tmp_path):
        from tsfast.tsdata import create_dls

        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No HDF5 files found"):
            create_dls(u=["u"], y=["y"], dataset=empty_dir, win_sz=100, num_workers=0)

    def test_dict_dataset_no_test(self):
        from tsfast.tsdata import create_dls
        from tsfast.tsdata.split import discover_split_files

        splits = discover_split_files(WH_PATH)
        del splits["test"]
        dls = create_dls(
            u=["u"],
            y=["y"],
            dataset=splits,
            win_sz=100,
            stp_sz=100,
            num_workers=0,
            n_batches_train=2,
        )
        assert len(dls.loaders) == 2  # train, valid only

    def test_dict_dataset_missing_train_raises(self):
        from tsfast.tsdata import create_dls

        with pytest.raises(ValueError, match="'train'"):
            create_dls(u=["u"], y=["y"], dataset={"valid": ["a.h5"]}, win_sz=100, num_workers=0)

    def test_dict_dataset_missing_valid_raises(self):
        from tsfast.tsdata import create_dls

        with pytest.raises(ValueError, match="'valid'"):
            create_dls(u=["u"], y=["y"], dataset={"train": ["a.h5"]}, win_sz=100, num_workers=0)
